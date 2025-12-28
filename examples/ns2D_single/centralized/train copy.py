import argparse
import json
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import flax.serialization
from flax import linen as nn
from tqdm import trange
from tesseract_core import Tesseract

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))
from dynamics import PDEDynamics, sample_initial_vorticity
from data_utils import generate_shape_pair, make_actuator_grid

# --- LOSS FUNCTIONS ---
def compute_iou(z_curr, z_target, epsilon=1e-8):
    intersection = jnp.sum(z_curr * z_target)
    union = jnp.sum(z_curr + z_target - z_curr * z_target)
    iou = intersection / (union + epsilon)
    return 1.0 - iou

def compute_smooth_loss(z_curr, z_target):
    """MSE + Center-of-Mass distance."""
    mse = jnp.mean((z_curr - z_target) ** 2)
    eps = 1e-8
    total_curr = jnp.sum(z_curr) + eps
    total_target = jnp.sum(z_target) + eps
    n = z_curr.shape[0]
    coords = jnp.arange(n)
    xx, yy = jnp.meshgrid(coords, coords, indexing='ij')
    cx_curr = jnp.sum(xx * z_curr) / total_curr
    cy_curr = jnp.sum(yy * z_curr) / total_curr
    cx_target = jnp.sum(xx * z_target) / total_target
    cy_target = jnp.sum(yy * z_target) / total_target
    com_dist = ((cx_curr - cx_target) ** 2 + (cy_curr - cy_target) ** 2) / (n ** 2)
    return mse + 0.5 * com_dist

# --- MODEL DEFINITION ---
class NS2DControlNet(nn.Module):
    features: Sequence[int]
    u_max: float = 10.0 
    v_max: float = 0.5

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        error = z_curr - z_target
        x = error[..., None]
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        x = x.reshape(-1)
        x = nn.LayerNorm()(x)
        branch = nn.Dense(32)(x)
        branch = nn.tanh(branch)

        freqs = jnp.array([1.0, 2.0, 4.0, 8.0])
        angle = xi_curr[..., None] / (2.0 * jnp.pi) * freqs[None, None, :] * 2.0 * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        encoded = encoded.reshape(xi_curr.shape[0], -1)

        y = encoded
        for feat in [32, 32]:
            y = nn.Dense(feat)(y)
            y = nn.tanh(y)

        branch_rep = jnp.tile(branch, (xi_curr.shape[0], 1))
        h = jnp.concatenate([branch_rep, y], axis=-1)
        h = nn.Dense(64)(h)
        h = nn.tanh(h)

        u_raw = nn.Dense(2)(h)
        v_raw = nn.Dense(2)(h)

        u = self.u_max * jnp.tanh(u_raw)
        v = self.v_max * jnp.tanh(v_raw)
        return u, v

# --- CHECKPOINT UTILS ---
def save_checkpoint(params, opt_state, epoch, metrics_log, path="checkpoint_ns2d.pkl"):
    checkpoint = {'params': params, 'opt_state': opt_state, 'epoch': epoch, 'metrics_log': metrics_log}
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  [Checkpoint saved at epoch {epoch}]")

def load_checkpoint(path="checkpoint_ns2d.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# --- CONFIG ---
n = 64
L = jnp.pi
m_agents = 64
batch_size = 1 
epochs = 200
t_steps = 200
checkpoint_interval = 25

key = jax.random.PRNGKey(0)
key_omega, key_data = jax.random.split(key)

omega_init = sample_initial_vorticity(key_omega, n, V_SCALE_BASE=0.1, V_FALLOFF=0.4).astype(jnp.float64)
xi_init_single = make_actuator_grid(m_agents, L).astype(jnp.float64)

total_samples = 1
data_keys = jax.random.split(key_data, total_samples)
z_init_all, z_target_all = jax.vmap(partial(generate_shape_pair, n=n, L=L))(data_keys)
xi_init_all = jnp.tile(xi_init_single[None, ...], (total_samples, 1, 1))

model = NS2DControlNet(features=(16, 32))
params = model.init(jax.random.PRNGKey(1), jnp.zeros((n, n)), jnp.zeros((n, n)), xi_init_single)

# --- OPTIMIZER ---
total_steps = epochs * (total_samples // batch_size)
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5, peak_value=2e-3, warmup_steps=min(200, total_steps // 20),
    decay_steps=total_steps, end_value=1e-5
)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

metrics_log = {'epochs': [], 'losses': [], 'tracks': [], 'efforts': [], 'final_ious': []}

checkpoint = load_checkpoint()
if checkpoint:
    params, opt_state, start_epoch, metrics_log = checkpoint['params'], checkpoint['opt_state'], checkpoint['epoch'], checkpoint['metrics_log']
else:
    start_epoch = 0

solver_ts = Tesseract.from_image("solver_ns_shape_centralized")

with solver_ts:
    # 1. Initialize Dynamics with the policy function
    dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)

    def loss_fn(p, z_init, xi_init, z_target, dyn):
        # 2. Use the single high-level unroll call
        # Returns: (omega_traj, z_traj, xi_traj, u_traj, v_traj)
        _, z_traj, _, u_traj, v_traj = dyn.unroll_controlled(
            omega_init, z_init, z_target, xi_init, p, t_steps
        )
        
        # 3. Smooth Loss on Terminal Window
        n_loss_steps = max(1, t_steps // 10)
        z_final_window = z_traj[-n_loss_steps:]
        smooth_losses = jax.vmap(lambda z: compute_smooth_loss(z, z_target))(z_final_window)
        
        track = jnp.mean(smooth_losses)
        effort = jnp.mean(u_traj**2) + 0.1 * jnp.mean(v_traj**2)
        final_iou = compute_iou(z_traj[-1], z_target)
        
        return track + 0.0001 * effort, (track, effort, final_iou)

    @partial(jax.jit, static_argnames="dyn")
    def train_step(p, opt_st, z_init_b, xi_init_b, z_target_b, dyn):
        batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))
        def mean_loss_fn(pp):
            losses, auxs = batched_loss_fn(pp, z_init_b, xi_init_b, z_target_b, dyn)
            return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

        (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(p)
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss, aux

    # --- TRAINING LOOP ---
    indices = jnp.arange(total_samples)
    start_time = time.time()

    for epoch in trange(start_epoch, epochs):
        key, subkey = jax.random.split(key)
        shuffled = jax.random.permutation(subkey, indices)
        
        for b in range(total_samples // batch_size):
            idx = shuffled[b*batch_size : (b+1)*batch_size]
            params, opt_state, loss, aux = train_step(
                params, opt_state, z_init_all[idx], xi_init_all[idx], z_target_all[idx], dynamics
            )

        # Log Metrics
        track, effort, iou = aux
        metrics_log['epochs'].append(epoch)
        metrics_log['losses'].append(float(loss))
        metrics_log['tracks'].append(float(track))
        metrics_log['efforts'].append(float(effort))
        metrics_log['final_ious'].append(float(iou))

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.5f} | IoU: {iou:.4f}")
        
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(params, opt_state, epoch + 1, metrics_log)

# --- SAVE RESULTS ---
with open('centralized_params_ns2d.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(params))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.plot(metrics_log['losses']); plt.yscale('log'); plt.title("Total Loss")
plt.subplot(1, 3, 2); plt.plot(metrics_log['tracks'], label="track"); plt.plot(metrics_log['efforts'], label="effort"); plt.legend(); plt.title("Components")
plt.subplot(1, 3, 3); plt.plot(metrics_log['final_ious']); plt.title("Final IoU (1-IoU)")
plt.tight_layout(); plt.savefig("centralized_training_results.png")

print(f"Training finished in {time.time() - start_time:.2f}s.")