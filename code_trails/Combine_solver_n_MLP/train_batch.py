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

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

from dpc_engine.dynamics import PDEDynamics
from models import MLP, split_action
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
batch_size = 8  # Number of parallel initial conditions
learning_rate = 1e-3
epochs = 100
T_steps = 300
R_safe = 0.05

# Initialize MLP
model = MLP(features=(64, 64, output_dim))
key = jax.random.PRNGKey(0)

dummy_input = jnp.zeros((1, input_dim))
params = model.init(key, dummy_input)

# Optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

print("Model and Optimizer initialized.")

# --- 2. Loss & Rollout Functions ---

def rollout_single(params, z_init, xi_init, z_target, dynamics):
    """
    Simulates a single trajectory forward for T_steps using the policy.
    Args:
        z_init: (N,) - initial PDE state
        xi_init: (n_agents,) - initial agent positions
        z_target: (N,) - target PDE state
    Returns:
        z_traj: (T, N)
        xi_traj: (T, n_agents)
        u_traj: (T, n_agents)
        v_traj: (T, n_agents)
    """
    
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        
        # Policy inference
        policy_input = jnp.concatenate([z_curr, z_target], axis=-1)
        action_flat = model.apply(params, policy_input[jnp.newaxis, :])[0]
        
        u_action, v_action = split_action(action_flat)
        
        actions = {'u': u_action, 'v': v_action}
        
        # Dynamics step
        with jax.default_device(jax.devices("cpu")[0]):
            z_next, xi_next = dynamics.step(z_curr, xi_curr, actions)
        
        return (z_next, xi_next), (z_next, xi_next, u_action, v_action)
        
    _, (z_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
        step_fn, 
        (z_init, xi_init), 
        None, 
        length=T_steps
    )
    
    return z_traj, xi_traj, u_traj, v_traj


def loss_single(params, z_init, xi_init, z_target, dynamics):
    """
    Computes loss for a single trajectory.
    Args:
        z_init: (N,) - initial PDE state
        xi_init: (n_agents,) - initial agent positions
        z_target: (N,) - target PDE state
    Returns:
        total_loss: scalar
        aux: tuple of (l_track, l_effort, l_coll)
    """
    # Run rollout
    z_traj, xi_traj, u_traj, v_traj = rollout_single(params, z_init, xi_init, z_target, dynamics)
    
    # 1. Tracking Loss: MSE between z_traj and z_target
    # z_traj: (T, N), z_target: (N,)
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    
    # 2. Effort Loss: L2 norm of controls
    # u_traj: (T, n_agents)
    l_effort = jnp.mean(u_traj ** 2)
    
    # 3. Collision Avoidance Loss
    # xi_traj: (T, n_agents)
    # Compute pairwise distances
    xi_exp = xi_traj[:, :, None]  # (T, n_agents, 1)
    xi_t_exp = xi_traj[:, None, :]  # (T, 1, n_agents)
    dists = jnp.abs(xi_exp - xi_t_exp)  # (T, n_agents, n_agents)
    
    # Avoid self-collision (diagonal)
    mask = jnp.eye(n_agents)[None, :, :]  # (1, n_agents, n_agents)
    dists = dists + mask * 1.0
    
    # Penalty: max(0, R_safe - dist)^2
    coll_penalty = jnp.sum(jnp.maximum(0, R_safe - dists) ** 2, axis=(1, 2))  # (T,)
    l_coll = jnp.mean(coll_penalty)
    
    # Total Loss
    total_loss = l_track  # + 0.001 * l_effort + 1.0 * l_coll
    
    return total_loss, (l_track, l_effort, l_coll)


# Vectorize rollout and loss over batch dimension using vmap
rollout_fn = jax.vmap(
    rollout_single,
    in_axes=(None, 0, 0, 0, None),  # params and dynamics are shared, rest are batched
    out_axes=0  # All outputs are batched
)

loss_single_vmap = jax.vmap(
    loss_single,
    in_axes=(None, 0, 0, 0, None),  # params and dynamics are shared, rest are batched
    out_axes=(0, 0)  # Both loss and aux are batched
)


def loss_fn(params, z_init_batch, xi_init_batch, z_target_batch, dynamics):
    """
    Computes loss over a batch of trajectories using vmap.
    Args:
        z_init_batch: (B, N) - batch of initial PDE states
        xi_init_batch: (B, n_agents) - batch of initial agent positions
        z_target_batch: (B, N) - batch of target PDE states
    Returns:
        total_loss: scalar (averaged over batch)
        aux: tuple of averaged auxiliary losses
    """
    # Compute per-sample losses using vmap
    losses, aux_batch = loss_single_vmap(params, z_init_batch, xi_init_batch, z_target_batch, dynamics)
    
    # losses: (B,), aux_batch: tuple of (B,) arrays
    # Average over batch
    total_loss = jnp.mean(losses)
    l_track_batch, l_effort_batch, l_coll_batch = aux_batch
    
    # Average auxiliary losses
    l_track = jnp.mean(l_track_batch)
    l_effort = jnp.mean(l_effort_batch)
    l_coll = jnp.mean(l_coll_batch)
    
    return total_loss, (l_track, l_effort, l_coll)


@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, z_init_batch, xi_init_batch, z_target_batch, dynamics
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux


# --- 3. Main Training Loop ---
with solver_ts:
    dynamics = PDEDynamics(solver_ts)
    
    print(f"Starting Training for {epochs} epochs with batch_size={batch_size}...")
    
    metrics = []
    
    start_time = time.time()
    for epoch in range(epochs):
        # Sample Batched Environment
        key, *subkeys = jax.random.split(key, 2 * batch_size + 1)
        
        # Generate batch of initial conditions
        z_init_batch = []
        z_target_batch = []
        
        for i in range(batch_size):
            # Initial condition: GRF with lengthscale 0.2
            _, z_init = data_utils.generate_grf(subkeys[i], n_points=n_pde, length_scale=0.2)
            z_init_batch.append(z_init)
            
            # Target: GRF with lengthscale 0.4
            _, z_target = data_utils.generate_grf(subkeys[batch_size + i], n_points=n_pde, length_scale=0.4)
            z_target_batch.append(z_target)
        
        z_init_batch = jnp.stack(z_init_batch, axis=0)  # (B, N)
        z_target_batch = jnp.stack(z_target_batch, axis=0)  # (B, N)
        
        # Check for NaNs
        if jnp.isnan(z_init_batch).any():
            print("NaN in z_init_batch!")
        if jnp.isnan(z_target_batch).any():
            print("NaN in z_target_batch!")
        
        # Initial agent positions: Same for all samples in batch or randomized
        # Option 1: Same positions for all
        xi_init_single = jnp.array([0.2, 0.4, 0.6, 0.8])
        xi_init_batch = jnp.tile(xi_init_single[None, :], (batch_size, 1))  # (B, n_agents)
        
        # Option 2: Randomize positions for each sample (uncomment if desired)
        # key, subkey = jax.random.split(key)
        # xi_init_batch = jax.random.uniform(subkey, (batch_size, n_agents), minval=0.1, maxval=0.9)
        
        # Train Step
        params, opt_state, loss, aux = train_step(
            params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics
        )
        l_track, l_effort, l_coll = aux
        
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d} | Time: {elapsed:.2f}s | Loss: {loss:.6f} | Track: {l_track:.6f} | Effort: {l_effort:.6f} | Coll: {l_coll:.6f}")
            metrics.append((epoch, loss, l_track, l_effort, l_coll))
                
            
    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    
    # Save parameters
    import flax.serialization
    with open('model_params.msgpack', 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print("Model parameters saved to model_params.msgpack")