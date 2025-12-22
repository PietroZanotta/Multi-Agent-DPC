import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
import sys
import optax
import time
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from tqdm import trange
import flax.serialization

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics # TODO: switch to dynamics.py for Tesseract-only
from data_utils import generate_grf
from models.policy import DecentralizedControlNet 

# --- 1. Initialization ---
print("Initializing environment and Decentralized model...")

solver_ts = Tesseract.from_image("solver_fkpp1d_v1")
n_pde = 100
n_agents = 4
batch_size = 32
epochs = 500
T_steps = 300 
R_safe = 0.05

# Initialize Decentralized Model
model = DecentralizedControlNet(features=(64, 64))
key = jax.random.PRNGKey(0)

# Dummy init
params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))

lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

# --- 2. Loss & Rollout Functions ---
def rollout_fn(params, z_init, xi_init, z_target, dynamics):
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_action, v_action = model.apply(params, z_curr, z_target, xi_curr)
        z_next, xi_next = dynamics.step(z_curr, xi_curr, {'u': u_action, 'v': v_action})
        return (z_next, xi_next), (z_next, xi_next, u_action, v_action)
        
    _, (z_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
        step_fn, (z_init, xi_init), None, length=T_steps
    )
    return z_traj, xi_traj, u_traj, v_traj

def loss_fn(params, z_init, xi_init, z_target, dynamics):
    z_traj, xi_traj, u_traj, v_traj = rollout_fn(params, z_init, xi_init, z_target, dynamics)
    
    # 1. Tracking Loss
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    # 2. Effort Loss
    l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(v_traj ** 2)
    # 3. Boundary Penalty
    margin = 0.02
    l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 + jnp.maximum(0, xi_traj - (1.0 - margin))**2)
    # 4. Collision Avoidance
    dists = jnp.abs(xi_traj[:, :, None] - xi_traj[:, None, :])
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 1.0)) ** 2)
    # 5. Damping
    l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)

    total_loss = 5.0 * l_track + 0.001 * l_effort + 100.0 * l_bound + 1.0 * l_coll + 0.1 * l_accel
    return total_loss, (l_track, l_effort, l_coll, l_bound)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))
    
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, z_init_batch, xi_init_batch, z_target_batch, dynamics)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 3. Main Training Loop ---
with solver_ts:
    print(f"Starting Decentralized DPC Training on {jax.devices()[0]}...")
    dynamics = PDEDynamics(solver_ts, use_tesseract=False)

    metrics = []
    start_time = time.time()
    
    # Pre-generate dataset
    total_samples = 5000 
    all_keys = jax.random.split(key, total_samples)
    _, z_init_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.2))(all_keys)
    _, z_target_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.4))(all_keys)
    
    xi_init_batch = jnp.tile(jnp.linspace(0.2, 0.8, n_agents), (batch_size, 1))
    indices = jnp.arange(total_samples)
    
    for epoch in trange(epochs):
        if epoch % (total_samples // batch_size) == 0:
            key, subkey = jax.random.split(key)
            shuffled_indices = jax.random.permutation(subkey, indices)
                  
        batch_idx = epoch % (total_samples // batch_size)
        current_indices = shuffled_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        params, opt_state, loss, aux = train_step(
            params, opt_state, z_init_all[current_indices], xi_init_batch, z_target_all[current_indices], dynamics
        )
        
        if epoch % 10 == 0:
            print(f"Loss: {loss:.4f} | Track: {aux[0]:.4f}")
            metrics.append((epoch, loss, *aux)) 

# --- 4. Plotting and Saving ---
    metrics = jnp.array(metrics)
    epochs_recorded = metrics[:, 0]
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_recorded, metrics[:, 1], color='black', label='Total Loss')
    plt.yscale('log')
    plt.title('Total Loss (Log Scale)')
    plt.legend()

    # Subplot 2: Performance vs Constraints (Tracking and Boundary)
    plt.subplot(2, 2, 2)
    plt.plot(epochs_recorded, metrics[:, 2], label='Tracking')
    plt.plot(epochs_recorded, metrics[:, 5], label='Boundary', alpha=0.7)
    plt.yscale('log')
    plt.title('Performance vs Constraints')
    plt.legend()

    # Subplot 3: Effort Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs_recorded, metrics[:, 3], color='green', label='Effort')
    plt.title('Effort Loss')
    plt.legend()

    # Subplot 4: Collision Avoidance
    plt.subplot(2, 2, 4)
    plt.plot(epochs_recorded, metrics[:, 4], color='red', label='Collision')
    plt.title('Collision Avoidance')
    plt.legend()

    plt.tight_layout()
    plt.savefig('decentralized_training.png')
    print("Training metrics plotted and saved to decentralized_training.png")
    
    # Save parameters
    with open('decentralized_params.msgpack', 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Training finished in {time.time() - start_time:.2f}s. Params saved.")