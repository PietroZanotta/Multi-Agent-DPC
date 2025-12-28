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
from flax import linen as nn
from tqdm import trange
from tesseract_core import Tesseract

script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

from examples.ns2D.centralized.data_utils import generate_shape_pair, make_actuator_grid
from examples.ns2D.centralized.dynamics import PDEDynamics, sample_initial_vorticity


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


def compute_iou(z_curr, z_target, epsilon=1e-8):
    intersection = jnp.sum(z_curr * z_target)
    union = jnp.sum(z_curr + z_target - z_curr * z_target)
    iou = intersection / (union + epsilon)
    return 1.0 - iou


def compute_mse(z_curr, z_target):
    """MSE loss - always provides gradients."""
    return jnp.mean((z_curr - z_target) ** 2)


def compute_smooth_loss(z_curr, z_target, sigma=5.0):
    """
    Smooth transport-like loss using Gaussian convolution.
    This provides gradients even when shapes don't overlap.
    """
    # Simple approximation: MSE on blurred versions
    from jax.scipy.ndimage import map_coordinates
    
    # Use MSE which always has gradients
    mse = jnp.mean((z_curr - z_target) ** 2)
    
    # Add center-of-mass guidance
    eps = 1e-8
    total_curr = jnp.sum(z_curr) + eps
    total_target = jnp.sum(z_target) + eps
    
    # Compute centers of mass
    n = z_curr.shape[0]
    coords = jnp.arange(n)
    xx, yy = jnp.meshgrid(coords, coords, indexing='ij')
    
    cx_curr = jnp.sum(xx * z_curr) / total_curr
    cy_curr = jnp.sum(yy * z_curr) / total_curr
    cx_target = jnp.sum(xx * z_target) / total_target
    cy_target = jnp.sum(yy * z_target) / total_target
    
    # Center of mass distance (normalized)
    com_dist = ((cx_curr - cx_target) ** 2 + (cy_curr - cy_target) ** 2) / (n ** 2)
    
    return mse + 0.5 * com_dist


# =============================================================================
# Checkpoint utilities
# =============================================================================
def save_checkpoint(params, opt_state, epoch, metrics_log, path="checkpoint.pkl"):
    checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'epoch': epoch,
        'metrics_log': metrics_log,
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  [Checkpoint saved at epoch {epoch}]")


def load_checkpoint(path="checkpoint.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"  [Resuming from epoch {checkpoint['epoch']}]")
        return checkpoint
    return None


# =============================================================================
# Training configuration - OPTIMIZED FOR SPEED
# =============================================================================
n = 64
L = jnp.pi
m_agents = 64

# Training config:
batch_size = 1          # Larger batches = fewer iterations
epochs = 200            # Reduced epochs
t_steps = 200           # Fixed 200 timesteps

# Checkpoint settings
checkpoint_interval = 25
checkpoint_path = "checkpoint_ns2d.pkl"

key = jax.random.PRNGKey(0)
key_omega, key_data = jax.random.split(key)

omega_init = sample_initial_vorticity(key_omega, n, V_SCALE_BASE=0.1, V_FALLOFF=0.4)
xi_init_single = make_actuator_grid(m_agents, L)

total_samples = 1
data_keys = jax.random.split(key_data, 1 * total_samples)
z_init_all, z_target_all = jax.vmap(partial(generate_shape_pair, n=n, L=L))(
    data_keys[:total_samples]
)

plt.imshow(z_init_all[0], origin='lower')
plt.savefig("z_init.png")
plt.close()
plt.imshow(z_target_all[0], origin='lower')
plt.savefig("z_target.png")
plt.close()

xi_init_all = jnp.tile(xi_init_single[None, ...], (total_samples, 1, 1))

model = NS2DControlNet(features=(16, 32))
dummy_z = jnp.zeros((n, n))
dummy_target = jnp.zeros((n, n))
dummy_xi = xi_init_single
params = model.init(jax.random.PRNGKey(1), dummy_z, dummy_target, dummy_xi)

# =============================================================================
# Learning rate schedule
# =============================================================================
num_batches = total_samples // batch_size
total_steps = epochs * num_batches
warmup_steps = min(200, total_steps // 20)

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    peak_value=2e-3,   # Slightly higher peak LR
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
    end_value=1e-5
)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

solver_ts = Tesseract.from_image("solver_ns_shape")

# =============================================================================
# Metrics logging
# =============================================================================
metrics_log = {
    'epochs': [],
    'losses': [],
    'tracks': [],
    'efforts': [],
    'final_ious': [],
    'times': [],
    'grad_norms': [],
    't_steps_used': [],
}

# Try loading checkpoint
start_epoch = 0
checkpoint = load_checkpoint(checkpoint_path)
if checkpoint is not None:
    params = checkpoint['params']
    opt_state = checkpoint['opt_state']
    start_epoch = checkpoint['epoch']
    metrics_log = checkpoint.get('metrics_log', metrics_log)
    if 't_steps_used' not in metrics_log:
        metrics_log['t_steps_used'] = []

with solver_ts:
    dynamics = PDEDynamics(solver_ts, use_tesseract=False)

    # Pre-compile rollout for different lengths (avoid recompilation)
    def make_rollout_fn(t_len):
        """Create JIT-compiled rollout function for a specific length."""
        
        @partial(jax.jit, static_argnames=("dyn",))
        def rollout_and_loss(p, z_init, xi_init, z_target, dyn):
            # Step function WITHOUT checkpoint (faster, uses more memory)
            def step_fn(carry, _):
                omega_curr, z_curr, xi_curr = carry
                u_action, v_action = model.apply(p, z_curr, z_target, xi_curr)
                omega_next, z_next, xi_next = dyn.step(
                    omega_curr, z_curr, xi_curr, u_action, v_action
                )
                return (omega_next, z_next, xi_next), (z_next, u_action, v_action)

            (_, _, _), (z_traj, u_traj, v_traj) = jax.lax.scan(
                step_fn, (omega_init, z_init, xi_init), None, length=t_len
            )
            
            # Compute loss on trajectory - use smooth loss for better gradients
            n_loss_steps = min(10, t_len)
            z_final = z_traj[-n_loss_steps:]
            
            # Primary loss: smooth loss (MSE + center-of-mass) - provides gradients
            smooth_losses = jax.vmap(lambda z: compute_smooth_loss(z, z_target))(z_final)
            track = jnp.mean(smooth_losses)
            
            # Secondary: IoU for monitoring (not primary loss)
            final_iou = compute_iou(z_traj[-1], z_target)
            
            # Control effort - increase weight to encourage more action
            effort = jnp.mean(u_traj**2) + 0.1 * jnp.mean(v_traj**2)
            
            # Total loss: tracking + small effort penalty
            loss = track + 0.0001 * effort  # Reduced effort penalty
            return loss, (track, effort, final_iou)
        
        @partial(jax.jit, static_argnames="dyn")
        def train_step(p, opt_st, z_init_b, xi_init_b, z_target_b, dyn):
            batched_loss = jax.vmap(rollout_and_loss, in_axes=(None, 0, 0, 0, None))

            def mean_loss_fn(pp):
                losses, auxs = batched_loss(pp, z_init_b, xi_init_b, z_target_b, dyn)
                return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

            (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(p)
            grad_norm = optax.global_norm(grads)
            updates, new_opt_st = optimizer.update(grads, opt_st, p)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_opt_st, loss, aux, grad_norm
        
        return train_step

    # Create single train step function for fixed timesteps
    train_step = make_rollout_fn(t_steps)

    indices = jnp.arange(total_samples)
    start_time = time.time()

    print(f"Starting training from epoch {start_epoch}...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Batches/epoch: {num_batches}")
    print(f"Fixed timesteps: {t_steps} (total sim time: {t_steps * 0.02:.1f})")
    print("-" * 70)

    for epoch in trange(start_epoch, epochs, desc="Training"):
        epoch_start = time.time()
        
        # Fixed timesteps - no curriculum
        current_t_steps = t_steps
        
        key, subkey = jax.random.split(key)
        shuffled = jax.random.permutation(subkey, indices)
        
        epoch_losses = []
        epoch_tracks = []
        epoch_efforts = []
        epoch_final_ious = []
        epoch_grad_norms = []
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            current_idx = shuffled[start : start + batch_size]

            z_init_batch = z_init_all[current_idx]
            z_target_batch = z_target_all[current_idx]
            xi_init_batch = xi_init_all[current_idx]

            params, opt_state, loss, aux, grad_norm = train_step(
                params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics
            )
            track, effort, final_iou = aux
            
            epoch_losses.append(loss)
            epoch_tracks.append(track)
            epoch_efforts.append(effort)
            epoch_final_ious.append(final_iou)
            epoch_grad_norms.append(grad_norm)

        # Average metrics
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        avg_track = jnp.mean(jnp.array(epoch_tracks))
        avg_effort = jnp.mean(jnp.array(epoch_efforts))
        avg_final_iou = jnp.mean(jnp.array(epoch_final_ious))
        avg_grad_norm = jnp.mean(jnp.array(epoch_grad_norms))
        epoch_time = time.time() - epoch_start

        # Log metrics
        metrics_log['epochs'].append(epoch)
        metrics_log['losses'].append(float(avg_loss))
        metrics_log['tracks'].append(float(avg_track))
        metrics_log['efforts'].append(float(avg_effort))
        metrics_log['final_ious'].append(float(avg_final_iou))
        metrics_log['times'].append(epoch_time)
        metrics_log['grad_norms'].append(float(avg_grad_norm))
        metrics_log['t_steps_used'].append(current_t_steps)

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            current_lr = lr_schedule(epoch * num_batches)
            print(
                f"\nEpoch {epoch:03d} | t={current_t_steps} | Loss {avg_loss:.5f} | "
                f"IoU {avg_final_iou:.4f} | GradNorm {avg_grad_norm:.3f} | "
                f"LR {current_lr:.1e} | {epoch_time:.1f}s/ep | Total {elapsed:.0f}s"
            )
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(params, opt_state, epoch + 1, metrics_log, checkpoint_path)

# =============================================================================
# Save final results
# =============================================================================
with open("training_metrics.json", "w") as f:
    json.dump(metrics_log, f, indent=2)

epochs_rec = jnp.array(metrics_log['epochs'])
losses = jnp.array(metrics_log['losses'])
tracks = jnp.array(metrics_log['tracks'])
efforts = jnp.array(metrics_log['efforts'])
final_ious = jnp.array(metrics_log['final_ious'])
grad_norms = jnp.array(metrics_log['grad_norms'])

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.plot(epochs_rec, losses)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Total Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 4, 2)
plt.plot(epochs_rec, tracks, label="track")
plt.plot(epochs_rec, efforts, label="effort")
plt.yscale("log")
plt.xlabel("Epoch")
plt.legend()
plt.title("Loss Components")
plt.grid(True, alpha=0.3)

plt.subplot(1, 4, 3)
plt.plot(epochs_rec, final_ious)
plt.xlabel("Epoch")
plt.ylabel("1 - IoU")
plt.title("Final IoU Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 4, 4)
plt.plot(epochs_rec, grad_norms)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Grad Norm")
plt.title("Gradient Norm")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("centralized_training_ns2d.png", dpi=150)

import flax.serialization
with open('centralized_params_ns2d.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(params))

total_time = time.time() - start_time
print(f"\nTraining finished in {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Final loss: {losses[-1]:.6f}, Final IoU: {final_ious[-1]:.6f}")