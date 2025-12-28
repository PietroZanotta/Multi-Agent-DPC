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
    u_max: float = 1.5
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
    """
    Compute Intersection over Union for binary density fields.
    IoU = intersection / union
    Returns 1 - IoU as a loss (0 is perfect, 1 is worst)
    """
    intersection = jnp.sum(z_curr * z_target)
    union = jnp.sum(z_curr + z_target - z_curr * z_target)
    iou = intersection / (union + epsilon)
    return 1.0 - iou  # Convert to loss (lower is better)


n = 64
L = jnp.pi
m_agents = 25
batch_size = 6
epochs = 20
t_steps = 100

key = jax.random.PRNGKey(0)
key_omega, key_data = jax.random.split(key)

omega_init = sample_initial_vorticity(key_omega, n, V_SCALE_BASE=0.5, V_FALLOFF=0.8)

xi_init_single = make_actuator_grid(m_agents, L)

total_samples = 128
data_keys = jax.random.split(key_data, 2 * total_samples)
z_init_all, z_target_all = jax.vmap(partial(generate_shape_pair, n=n, L=L))(
    data_keys[:total_samples]
)
xi_init_all = jnp.tile(xi_init_single[None, ...], (total_samples, 1, 1))

# typecasring to avoid problems
z_init_all = z_init_all.astype(jnp.float64)
z_target_all = z_target_all.astype(jnp.float64)
xi_init_all = xi_init_all.astype(jnp.float64)
omega_init = omega_init.astype(jnp.float64)

model = NS2DControlNet(features=(16, 32))
dummy_z = jnp.zeros((n, n), dtype=jnp.float64)
dummy_target = jnp.zeros((n, n), dtype=jnp.float64)
dummy_xi = xi_init_single.astype(jnp.float64)
params = model.init(jax.random.PRNGKey(1), dummy_z, dummy_target, dummy_xi)

lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

solver_ts = Tesseract.from_image("solver_ns_shape_centralized")

with solver_ts:
    dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False) # using jax to speedup the training

    def loss_fn(p, z_init, xi_init, z_target, dyn):
        omega_init_64 = omega_init.astype(jnp.float64) 

        _, z_traj, _, u_traj, v_traj = dyn.unroll_controlled(
            omega_init_64, z_init, z_target, xi_init, p, t_steps
        )
        
        # Compute IoU loss at each timestep and sum
        def compute_timestep_iou(z_t):
            return compute_iou(z_t, z_target)
        
        iou_losses = jax.vmap(compute_timestep_iou)(z_traj)
        track = jnp.mean(iou_losses)  # Sum over all timesteps
        
        # Control effort
        effort = jnp.mean(u_traj**2) + 0.1 * jnp.mean(v_traj**2)
        
        return track + 0.001 * effort, (track, effort, iou_losses[-1])

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

    indices = jnp.arange(total_samples)
    metrics = []
    start_time = time.time()

    for epoch in trange(epochs):
        key, subkey = jax.random.split(key)
        shuffled = jax.random.permutation(subkey, indices)
        
        # FIX: Properly iterate through all batches
        num_batches = total_samples // batch_size
        epoch_losses = []
        epoch_tracks = []
        epoch_efforts = []
        epoch_final_ious = []
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            current_idx = shuffled[start : start + batch_size]

            z_init_batch = z_init_all[current_idx]
            z_target_batch = z_target_all[current_idx]
            xi_init_batch = xi_init_all[current_idx]

            params, opt_state, loss, aux = train_step(
                params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics
            )
            track, effort, final_iou = aux
            
            epoch_losses.append(loss)
            epoch_tracks.append(track)
            epoch_efforts.append(effort)
            epoch_final_ious.append(final_iou)

        # Average metrics across all batches in the epoch
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        avg_track = jnp.mean(jnp.array(epoch_tracks))
        avg_effort = jnp.mean(jnp.array(epoch_efforts))
        avg_final_iou = jnp.mean(jnp.array(epoch_final_ious))

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:03d} | Loss {avg_loss:.6f} | Track {avg_track:.6f} | "
                f"Effort {avg_effort:.6f} | Final IoU Loss {avg_final_iou:.6f} | Time {elapsed:.1f}s"
            )
            metrics.append((epoch, avg_loss, avg_track, avg_effort, avg_final_iou))

metrics = jnp.array(metrics)
epochs_rec = metrics[:, 0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs_rec, metrics[:, 1])
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Total Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(epochs_rec, metrics[:, 2], label="track (IoU sum)")
plt.plot(epochs_rec, metrics[:, 3], label="effort")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.title("Loss Components")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(epochs_rec, metrics[:, 4])
plt.xlabel("Epoch")
plt.ylabel("1 - IoU")
plt.title("Final Timestep IoU Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("centralized_training_ns2d_fixed.png")

import flax.serialization
with open('centralized_params_ns2d_fixed.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(params))
print(f"Training finished in {time.time() - start_time:.2f}s. Params saved.")