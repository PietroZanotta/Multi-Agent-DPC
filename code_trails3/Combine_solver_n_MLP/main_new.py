import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from tesseract_core import Tesseract
import sys
import os
from pathlib import Path
import optax
import time
from functools import partial
import matplotlib.pyplot as plt

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

from dpc_engine.dynamics_dual import PDEDynamics
from models import ControlNet
import data_utils

# --- 1. Initialization ---
print("Initializing environment and model...")

# Load Tesseract Solver
solver_ts = Tesseract.from_image("solver_v1")

# Dimensions
n_pde = 100
n_agents = 4
batch_size = 16  # Moderate batch size for CPU stability

# Hyperparameters
learning_rate = 1e-3 
lr_schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=2000,
    decay_rate=0.5
)
epochs = 1000
T_steps = 300 # Balanced horizon
R_safe = 0.05

# Initialize Model
model = ControlNet(features=(64, 64))
key = jax.random.PRNGKey(0)

dummy_z = jnp.zeros((n_pde,))
dummy_target = jnp.zeros((n_pde,))
dummy_xi = jnp.zeros((n_agents,))
params = model.init(key, dummy_z, dummy_target, dummy_xi)

# Optimizer with Scheduler and Global Gradient Clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr_schedule)
)
opt_state = optimizer.init(params)

print("Model and Optimizer initialized.")

# --- 2. Loss & Rollout Functions ---

def rollout_fn(params, z_init, xi_init, z_target, dynamics):
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_action, v_action = model.apply(params, z_curr, z_target, xi_curr)
        actions = {'u': u_action, 'v': v_action}
        
        with jax.default_device(jax.devices("cpu")[0]):
            z_next, xi_next = dynamics.step(z_curr, xi_curr, actions)
        
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
    left_violation = jnp.maximum(0, margin - xi_traj) 
    right_violation = jnp.maximum(0, xi_traj - (1.0 - margin))
    l_bound = jnp.mean(left_violation**2 + right_violation**2)
    
    # 4. Collision Avoidance
    xi_exp = xi_traj[:, :, None]
    xi_t_exp = xi_traj[:, None, :]
    dists = jnp.abs(xi_exp - xi_t_exp)
    mask = jnp.eye(n_agents)[None, :, :]
    dists = dists + mask * 1.0 
    l_coll = jnp.mean(jnp.maximum(0, R_safe - dists) ** 2)
    
    # 5. Damping (Acceleration Penalty)
    v_diff = jnp.diff(v_traj, axis=0)
    l_accel = jnp.mean(v_diff**2)

    total_loss = 5.0 * l_track + 0.001 * l_effort + 100.0 * l_bound + 1.0 * l_coll + 0.1 * l_accel
    return total_loss, (l_track, l_effort, l_coll, l_bound)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics):
    # Vectorize the loss function over the batch dimension (axis 0)
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
    dynamics = PDEDynamics(solver_ts, use_tesseract=False)
    print(f"Starting Training for {epochs} epochs with Batch Size {batch_size}...")
    
    metrics = []
    start_time = time.time()
    
    for epoch in range(epochs):
        # Sample Batch Environment
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, batch_size)
        
        # Batched data generation
        # We use vmap over the generator to create the batch
        def gen_sample(k):
            k1, k2 = jax.random.split(k)
            _, z_i = data_utils.generate_grf(k1, n_points=n_pde, length_scale=0.2)
            _, z_t = data_utils.generate_grf(k2, n_points=n_pde, length_scale=0.4)
            return z_i, z_t
            
        z_init_batch, z_target_batch = jax.vmap(gen_sample)(subkeys)
        xi_init_batch = jnp.tile(jnp.array([0.2, 0.4, 0.6, 0.8]), (batch_size, 1))

        # Train Step on Batch
        params, opt_state, loss, aux = train_step(
            params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics
        )
        l_track, l_effort, l_coll, l_bound = aux
        
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d} | Loss: {loss:.6f} | Track: {l_track:.6f} | Coll: {l_coll:.6f}")
            metrics.append((epoch, loss, l_track, l_effort, l_coll, l_bound)) 

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    
    # --- 4. Plotting and Saving Loss ---
    metrics = jnp.array(metrics)
    epochs_recorded = metrics[:, 0]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_recorded, metrics[:, 1], color='black', label='Total Loss')
    plt.yscale('log'); plt.title('Total Loss (Log Scale)'); plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs_recorded, metrics[:, 2], label='Tracking')
    plt.plot(epochs_recorded, metrics[:, 5], label='Boundary', alpha=0.7)
    plt.yscale('log'); plt.title('Performance vs Constraints'); plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs_recorded, metrics[:, 3], color='green', label='Effort')
    plt.title('Effort Loss'); plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs_recorded, metrics[:, 4], color='red', label='Collision')
    plt.title('Collision Avoidance'); plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    # Save parameters
    import flax.serialization
    with open('model_params.msgpack', 'wb') as f:
        f.write(flax.serialization.to_bytes(params))