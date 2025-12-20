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
input_dim = n_pde * 2 # PDE state + Target
output_dim = 4 # u(4) only, v(4) is zeroed in split_action

# Hyperparameters
learning_rate = 1e-3 

lr_schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=2000,
    decay_rate=0.5
)
epochs = 1000
T_steps = 500
R_safe = 0.05

# Initialize MLP
model = ControlNet(features=(64, 64))
key = jax.random.PRNGKey(0)

dummy_z = jnp.zeros((n_pde,))
dummy_target = jnp.zeros((n_pde,))
dummy_xi = jnp.zeros((n_agents,))
params = model.init(key, dummy_z, dummy_target, dummy_xi)

# Optimizer
optimizer = optax.adam(lr_schedule)
opt_state = optimizer.init(params)

print("Model and Optimizer initialized.")

# --- 2. Loss & Rollout Functions ---

def rollout_fn(params, z_init, xi_init, z_target, dynamics):
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        
        # Policy now takes xi_curr as an explicit input
        # No more jax.newaxis hacks here; let's keep it clean
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
    
    # 1. Tracking Loss (Global Error)
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    
    # 2. Effort Loss
    l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(v_traj ** 2)
    
    # 3. Improved Boundary Penalty using max(0, dist)^2
    # We define a "safety buffer" (e.g., 0.02) from the edges [0, 1]
    margin = 0.02
    
    # Violation if xi < margin
    left_violation = jnp.maximum(0, margin - xi_traj) 
    # Violation if xi > (1 - margin)
    right_violation = jnp.maximum(0, xi_traj - (1.0 - margin))
    
    # Summing squared violations provides a "wall" that gets steeper the further you go out
    l_bound = jnp.mean(left_violation**2 + right_violation**2)
    
    # 4. Collision Avoidance (Keeping the one-sided logic you already had)
    xi_exp = xi_traj[:, :, None]
    xi_t_exp = xi_traj[:, None, :]
    dists = jnp.abs(xi_exp - xi_t_exp)
    mask = jnp.eye(n_agents)[None, :, :]
    dists = dists + mask * 1.0 
    
    l_coll = jnp.mean(jnp.maximum(0, R_safe - dists) ** 2)
    
    v_diff = jnp.diff(v_traj, axis=0)
    l_accel = jnp.mean(v_diff**2)

    total_loss = 5 * l_track + 0.001 * l_effort + 100.0 * l_bound + 1.0 * l_coll + 0.1 * l_accel
    
    return total_loss, (l_track, l_effort, l_coll, l_bound)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init, xi_init, z_target, dynamics):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, z_init, xi_init, z_target, dynamics
    )
    # Gradient clipping
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 3. Main Training Loop ---
with solver_ts:
    dynamics = PDEDynamics(solver_ts, use_tesseract=False)
    
    print(f"Starting Training for {epochs} epochs...")
    
    metrics = []
    
    start_time = time.time()
    for epoch in range(epochs):
        # Sample Environment
        # Split key
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # Initial condition: GRF with lengthscale 0.2
        _, z_init = data_utils.generate_grf(subkey1, n_points=n_pde, length_scale=0.2)
        if jnp.isnan(z_init).any():
            print("NaN in z_init!")

        # Target: GRF with lengthscale 0.4
        _, z_target = data_utils.generate_grf(subkey2, n_points=n_pde, length_scale=0.4)
        
        # Initial positions: Fixed or random? User says "generated from grf" for initial condition "and target trajectories".
        # Initial agent positions usually fixed or random in domain.
        # Let's keep them fixed for stability or add small noise.
        xi_init = jnp.array([0.2, 0.4, 0.6, 0.8])
        
        # Train Step

        params, opt_state, loss, aux = train_step(
            params, opt_state, z_init, xi_init, z_target, dynamics
        )
        l_track, l_effort, l_coll, l_bound = aux
        
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d} | Time: {elapsed:.2f}s | Loss: {loss:.6f} | Track: {l_track:.6f} | Effort: {l_effort:.6f} | Coll: {l_coll:.6f}")
            metrics.append((epoch, loss, l_track, l_effort, l_coll, l_bound))    

            
    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    
    # --- 4. Plotting and Saving Loss ---
    print("Saving loss plots...")
    metrics = jnp.array(metrics)
    epochs_recorded = metrics[:, 0]
    total_loss = metrics[:, 1]
    tracking_loss = metrics[:, 2]
    effort_loss = metrics[:, 3]
    collision_loss = metrics[:, 4]
    boundary_penalty = metrics[:, 5]
    # print(boundary_penalty)

    plt.figure(figsize=(12, 8))

    # Plot Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_recorded, total_loss, label='Total Loss', color='black')
    plt.yscale('log')
    plt.title('Total Loss (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    # Plot Tracking vs Boundary
    plt.subplot(2, 2, 2)
    plt.plot(epochs_recorded, tracking_loss, label='Tracking (MSE)')
    plt.plot(epochs_recorded, boundary_penalty, label='Boundary Penalty', alpha=0.7)
    plt.yscale('log')
    plt.title('Performance vs Constraints')
    plt.legend()

    # Plot Effort
    plt.subplot(2, 2, 3)
    plt.plot(epochs_recorded, effort_loss, label='Control Effort', color='green')
    plt.title('Effort Loss')
    plt.legend()

    # Plot Collisions
    plt.subplot(2, 2, 4)
    plt.plot(epochs_recorded, collision_loss, label='Collision Penalty', color='red')
    plt.title('Collision Avoidance')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Loss plot saved as training_metrics.png")
    
    # Save parameters
    import flax.serialization
    with open('model_params.msgpack', 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print("Model parameters saved to model_params.msgpack")
