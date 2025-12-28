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

from data_utils import generate_shape_pair, make_actuator_grid
from dynamics import PDEDynamics, sample_initial_vorticity
from models.policy import NS2DControlNet


def compute_iou(z_curr, z_target, epsilon=1e-8):
    intersection = jnp.sum(z_curr * z_target)
    union = jnp.sum(z_curr + z_target - z_curr * z_target)
    iou = intersection / (union + epsilon)
    return 1.0 - iou


def compute_mse(z_curr, z_target):
    """MSE loss - always provides gradients."""
    return jnp.mean((z_curr - z_target) ** 2)


# def compute_smooth_loss(z_curr, z_target, sigma=5.0):
#     """
#     Smooth transport-like loss using Gaussian convolution.
#     This provides gradients even when shapes don't overlap.
#     """
#     # Simple approximation: MSE on blurred versions
#     from jax.scipy.ndimage import map_coordinates
    
#     # Use MSE which always has gradients
#     mse = jnp.mean((z_curr - z_target) ** 2)
    
#     # Add center-of-mass guidance
#     eps = 1e-8
#     total_curr = jnp.sum(z_curr) + eps
#     total_target = jnp.sum(z_target) + eps
    
#     # Compute centers of mass
#     n = z_curr.shape[0]
#     coords = jnp.arange(n)
#     xx, yy = jnp.meshgrid(coords, coords, indexing='ij')
    
#     cx_curr = jnp.sum(xx * z_curr) / total_curr
#     cy_curr = jnp.sum(yy * z_curr) / total_curr
#     cx_target = jnp.sum(xx * z_target) / total_target
#     cy_target = jnp.sum(yy * z_target) / total_target
    
#     # Center of mass distance (normalized)
#     com_dist = ((cx_curr - cx_target) ** 2 + (cy_curr - cy_target) ** 2) / (n ** 2)
    
#     return mse + 0.2 * com_dist


def compute_smooth_loss(z_curr, z_target, com_weight):
    """Annealable smooth loss."""
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
    
    # Center of mass distance (normalized)
    com_dist = ((cx_curr - cx_target) ** 2 + (cy_curr - cy_target) ** 2) / (n ** 2)
    
    return mse + com_weight * com_dist


# =============================================================================
# Training Configuration
# =============================================================================
n = 64
L = jnp.pi
m_agents = 64
batch_size = 5
epochs = 200
t_steps = 200 

# Annealing settings
start_com_weight = 2.
end_com_weight = 0.001
anneal_epochs = 100

# Checkpoint settings
checkpoint_interval = 25
checkpoint_path = "checkpoint_ns2d.pkl"

key = jax.random.PRNGKey(0)
key_omega, key_data = jax.random.split(key)

omega_init = sample_initial_vorticity(key_omega, n, V_SCALE_BASE=0.1, V_FALLOFF=0.4)
xi_init_single = make_actuator_grid(m_agents, L)

total_samples = 10
key = jax.random.PRNGKey(0)
key_omega, key_data = jax.random.split(key)

omega_init_all = jnp.tile(omega_init[None, ...], (total_samples, 1, 1))
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

print("An example of an initial and final profiles saved at z_init.png and z_target.png")

xi_init_all = jnp.tile(xi_init_single[None, ...], (total_samples, 1, 1))

# Initialize Model
model = NS2DControlNet(features=(20, 50))
dummy_z = jnp.zeros((n, n))
dummy_target = jnp.zeros((n, n))
dummy_xi = xi_init_single
params = model.init(jax.random.PRNGKey(1), dummy_z, dummy_target, dummy_xi)

# =============================================================================
# Learning rate schedule
# =============================================================================
num_batches = total_samples // batch_size
total_steps = 200
warmup_steps = 0#min(200, total_steps // 20)

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    peak_value=2e-3,   # Slightly higher peak LR
    warmup_steps=warmup_steps,
    decay_steps=epochs * num_batches,
    end_value=1e-5
)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

solver_ts = Tesseract.from_image("solver_ns_shape_centralized")

# Initialize the PDEDynamics with the policy apply function
dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)

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
    't_steps_used': [],
    'grad_norms': [],
    'sample_xi_traj': None, # To store final xi positions
    'sample_v_traj': None,  # To store velocity control signals
}

# =============================================================================
# Training Logic
# =============================================================================

# @partial(jax.jit, static_argnames=("dyn", "t_len"))
# def rollout_and_loss(p, omega_init, z_init, xi_init, z_target, dyn, t_len, com_weight):
#     omega_traj, z_traj, xi_traj, u_traj, v_traj = dyn.unroll_controlled(
#         omega_init, z_init, z_target, xi_init, p, t_len
#     )
    
#     # Compute loss on the end of the trajectory
#     n_loss_steps = min(10, t_len)
#     z_final = z_traj[-n_loss_steps:]
    
#     smooth_losses = jax.vmap(lambda z: compute_smooth_loss(z, z_target, com_weight))(z_final)
#     track = jnp.mean(smooth_losses)
    
#     final_iou = compute_iou(z_traj[-1], z_target)
#     effort = jnp.mean(u_traj**2) + 0.1 * jnp.mean(v_traj**2)
    
#     # This prevents the erratic "zig-zag" motion by penalizing sharp changes in velocity
#     l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)

#     # Force the integrated positions to stay within [0.05, L-0.05]
#     margin = 0.05
#     # l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 + 
#                     #    jnp.maximum(0, xi_traj - (L - margin))**2)

#     # Combine with appropriate weights
#     total_loss = track + 0.05 * l_accel + 0.0001 * jnp.mean(u_traj**2) #+ 10.0 * l_bound
    
#     return total_loss, (track, effort, final_iou)

@partial(jax.jit, static_argnames=("dyn", "t_len"))
def rollout_and_loss(p, omega_init, z_init, xi_init, z_target, dyn, t_len, com_weight):
    # Differentiable forward pass [cite: 14, 110]
    omega_traj, z_traj, xi_traj, u_traj, v_traj = dyn.unroll_controlled(
        omega_init, z_init, z_target, xi_init, p, t_len
    )
    
    n_loss_steps = min(10, t_len)
    z_final = z_traj[-n_loss_steps:]
    
    # 1. MSE Shape Loss
    # We use MSE as requested, but vmap it over the last few timesteps
    shape_mse = jnp.mean(jax.vmap(lambda z: jnp.mean((z - z_target)**2))(z_final))
    
    # 2. Smooth CoM Loss (Magnet for coarse alignment)
    smooth_losses = jax.vmap(lambda z: compute_smooth_loss(z, z_target, com_weight))(z_final)
    track_com = jnp.mean(smooth_losses)
    
    # 3. Regularization: Force spikes and Acceleration [cite: 81, 104]
    l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)
    effort = jnp.mean(u_traj**2)
    
    # Combined Loss: Multi-scale tracking [cite: 104]
    # We weight MSE and CoM equally to provide both local and global gradients
    total_loss = 10.0 * shape_mse + track_com + 0.01 * l_accel + 0.0001 * effort
    
    final_iou = compute_iou(z_traj[-1], z_target)
    return total_loss, (shape_mse, track_com, final_iou)


@partial(jax.jit, static_argnames=("dyn", "t_len"))
def train_step(p, opt_st, omega_init_b, z_init_b, xi_init_b, z_target_b, dyn, t_len, com_weight):
    # Broadcast com_weight (None) across the vmap
    batched_loss = jax.vmap(rollout_and_loss, in_axes=(None, 0, 0, 0, 0, None, None, None))

    def mean_loss_fn(pp):
        losses, auxs = batched_loss(pp, omega_init_b, z_init_b, xi_init_b, z_target_b, dyn, t_len, com_weight)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(p)
    grad_norm = optax.global_norm(grads)
    updates, new_opt_st = optimizer.update(grads, opt_st, p)
    new_p = optax.apply_updates(p, updates)
    return new_p, new_opt_st, loss, aux, grad_norm

# =============================================================================
# Execution Loop
# =============================================================================
indices = jnp.arange(total_samples)
start_time = time.time()
start_epoch = 0 # Default if no checkpoint loading logic is added back

dt = 0.01 # IMPORTANT: this shoudl match the dt in the solver

print(f"Starting training from epoch {start_epoch}...")
print(f"Epochs: {epochs}, Batch size: {batch_size}, Batches/epoch: {num_batches}")
print(f"Fixed timesteps: {t_steps} (total sim time: {t_steps * dt:.1f})")
print("-" * 70)

for epoch in trange(start_epoch, epochs, desc="Training"):
    epoch_start = time.time()
    
    progress = min(1.0, epoch / anneal_epochs)
    current_com_weight = start_com_weight - progress * (start_com_weight - end_com_weight)
    
    # Shuffle data indices for the epoch
    key, subkey = jax.random.split(key)
    shuffled = jax.random.permutation(subkey, indices)
    
    epoch_losses, epoch_tracks, epoch_efforts, epoch_final_ious, epoch_grad_norms = [], [], [], [], []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        current_idx = shuffled[start : start + batch_size]

        # Extract the unique initial fluid states for this batch
        omega_init_batch = omega_init_all[current_idx]       
        z_init_batch = z_init_all[current_idx]
        z_target_batch = z_target_all[current_idx]
        xi_init_batch = xi_init_all[current_idx]

        # THE CORE TRAIN STEP
        # Sample trajectory for visualization (use the first omega in the batch
        if epoch == epochs - 1 and batch_idx == num_batches - 1:
            # We call the dynamics unroll to capture trajectories for plotting
            omega_traj, z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
                omega_init_batch[0], z_init_batch[0], z_target_batch[0], xi_init_batch[0], params, t_steps
            )
            metrics_log['sample_xi_traj'] = xi_traj
            metrics_log['sample_v_traj'] = v_traj

        params, opt_state, loss, aux, grad_norm = train_step(
            params, opt_state, omega_init_batch, z_init_batch, xi_init_batch, z_target_batch, 
            dynamics, t_steps, current_com_weight
        )
        
        track, effort, final_iou = aux
        epoch_losses.append(loss)
        epoch_tracks.append(track)
        epoch_efforts.append(effort)
        epoch_final_ious.append(final_iou)
        epoch_grad_norms.append(grad_norm)

    # Calculate and Log Average Metrics
    avg_loss = jnp.mean(jnp.array(epoch_losses))
    avg_track = jnp.mean(jnp.array(epoch_tracks))
    avg_effort = jnp.mean(jnp.array(epoch_efforts))
    avg_final_iou = jnp.mean(jnp.array(epoch_final_ious))
    avg_grad_norm = jnp.mean(jnp.array(epoch_grad_norms))
    epoch_time = time.time() - epoch_start

    metrics_log['epochs'].append(epoch)
    metrics_log['losses'].append(float(avg_loss))
    metrics_log['tracks'].append(float(avg_track))
    metrics_log['efforts'].append(float(avg_effort))
    metrics_log['final_ious'].append(float(avg_final_iou))
    metrics_log['times'].append(epoch_time)
    metrics_log['grad_norms'].append(float(avg_grad_norm))
    metrics_log['t_steps_used'].append(t_steps)

    # Periodic Printout
    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        print(
            f"\nEpoch {epoch:03d} | CoM Weight: {current_com_weight:.2f} | Loss {avg_loss:.5f} | "
            f"IoU {avg_final_iou:.4f} | GN {avg_grad_norm:.3f}"
        )

# =============================================================================
# Post-Training: Save Results & Plot Actuator Positions
# =============================================================================
plt.figure(figsize=(18, 10))

# --- Row 1: Standard Training Metrics ---
plt.subplot(2, 4, 1); plt.plot(metrics_log['epochs'], metrics_log['losses']); plt.yscale("log"); plt.title("Total Loss")
plt.subplot(2, 4, 2); plt.plot(metrics_log['epochs'], metrics_log['tracks'], label="track"); plt.yscale("log"); plt.title("Tracking Loss")
plt.subplot(2, 4, 3); plt.plot(metrics_log['epochs'], metrics_log['final_ious']); plt.title("Final IoU (1-IoU)")
plt.subplot(2, 4, 4); plt.plot(metrics_log['epochs'], metrics_log['grad_norms']); plt.yscale("log"); plt.title("Grad Norm")

# --- Row 2: Actuator Analysis ---
if metrics_log['sample_xi_traj'] is not None:
    xi_traj = metrics_log['sample_xi_traj'] # Shape (T, M, 2)
    v_traj = metrics_log['sample_v_traj']   # Shape (T, M, 2)
    
    # 1. Plot Actuator Trajectories in 2D Space
    plt.subplot(2, 4, 5)
    for m in range(min(m_agents, 16)): # Plot first 16 agents for clarity
        plt.plot(xi_traj[:, m, 0], xi_traj[:, m, 1], alpha=0.6)
        plt.scatter(xi_traj[0, m, 0], xi_traj[0, m, 1], marker='o', s=10) # Start
        plt.scatter(xi_traj[-1, m, 0], xi_traj[-1, m, 1], marker='x', s=15) # End
    plt.title("Actuator 2D Paths (Sample)")
    plt.xlabel("X"); plt.ylabel("Y")
    
    # 2. Integrated Position from V (X-component)
    plt.subplot(2, 4, 6)
    # Displacement = Cumulative sum of velocity * dt
    displacement_x = jnp.cumsum(v_traj[:, :, 0], axis=0) * dt
    plt.plot(displacement_x[:, :8]) # Plot first 8 agents
    plt.title("Integrated X-Displacement")
    plt.xlabel("Steps"); plt.ylabel("$\Delta x$")

    # 3. Control Velocity Magnitude
    plt.subplot(2, 4, 7)
    v_mag = jnp.linalg.norm(v_traj, axis=-1)
    plt.plot(jnp.mean(v_mag, axis=1))
    plt.title("Mean Control Vel ($|v|$)")
    plt.xlabel("Steps")

    # 4. Final Actuator Distribution
    plt.subplot(2, 4, 8)
    plt.scatter(xi_traj[-1, :, 0], xi_traj[-1, :, 1], c='red', s=10, label='Final')
    plt.scatter(xi_traj[0, :, 0], xi_traj[0, :, 1], c='blue', s=10, alpha=0.3, label='Initial')
    plt.legend(fontsize='small')
    plt.title("Final Agent Grid")

plt.tight_layout()
plt.savefig("centralized_training_with_actuators.png", dpi=200)
print("Figure saved at centralized_training_with_actuators.png")

import flax.serialization

# Save the parameters
with open('centralized_params_ns2d.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(params))

total_time = time.time() - start_time

# Access the metrics correctly using string keys from the metrics_log dictionary
final_loss = metrics_log['losses'][-1]
final_iou_loss = metrics_log['final_ious'][-1]

print(f"\nTraining finished in {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Final loss: {final_loss:.6f}, Final IoU Loss: {final_iou_loss:.6f}")