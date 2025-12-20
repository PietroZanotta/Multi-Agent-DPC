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
learning_rate = 1e-3 # Reduced LR
epochs = 50
T_steps = 300 # Keep at 10 for now
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

def rollout_fn(params, z_init, xi_init, z_target, dynamics):
    """
    Simulates the system forward for T_steps using the policy.
    """
    
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        
        # Policy inference
        # Input: z_curr (N,) and z_target (N,)
        # MLP expects (Batch, Features) => Add batch dim, then [0]
        policy_input = jnp.concatenate([z_curr, z_target], axis=-1)
        action_flat = model.apply(params, policy_input[jnp.newaxis, :])[0]
        
        u_action, v_action = split_action(action_flat)
        
        actions = {'u': u_action, 'v': v_action}
        
        # Dynamics step
        # Force CPU for Dynamics because it uses Python Callbacks
        with jax.default_device(jax.devices("cpu")[0]):
            z_next, xi_next = dynamics.step(z_curr, xi_curr, actions)
        
        # Carry over next state
        return (z_next, xi_next), (z_next, xi_next, u_action, v_action)
        
    # Scan over T_steps
    # We don't care about inputs per step (second arg of scan), so we pass None or range
    # But lax.scan expects xs to match length. 
    # Actually dynamics.step needs to be pure JAX for this to work inside JIT.
    # Check if PDEDynamics.step is JIT-compatible.
    
    _, (z_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
        step_fn, 
        (z_init, xi_init), 
        None, 
        length=T_steps
    )
    
    return z_traj, xi_traj, u_traj, v_traj

def loss_fn(params, z_init, xi_init, z_target, dynamics):
    z_traj, xi_traj, u_traj, v_traj = rollout_fn(params, z_init, xi_init, z_target, dynamics)
    
    # 1. Tracking Loss: MSE between z_traj and z_target (broadcasted or same shape)
    # z_traj: (T, N), z_target: (N,) -> Broadcast target across time
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    
    # 2. Effort Loss: L2 norm of controls
    l_effort = jnp.mean(u_traj ** 2) # + jnp.mean(v_traj ** 2) (Velocity disabled)
    
    # 3. Collision Avoidance Loss
    # xi_traj: (T, n_agents)
    # Compute pairwise distances
    # Expand dims to (T, N, 1) and (T, 1, N)
    xi_exp = xi_traj[:, :, None]
    xi_t_exp = xi_traj[:, None, :]
    dists = jnp.abs(xi_exp - xi_t_exp)
    
    # Avoid self-collision (diagonal is 0)
    # We can add identity * large_number to dists
    mask = jnp.eye(n_agents)[None, :, :]
    dists = dists + mask * 1.0 # Add 1.0 to diagonal so it doesn't trigger collision
    
    # Penalty: max(0, R_safe - dist)^2
    coll_penalty = jnp.sum(jnp.maximum(0, R_safe - dists) ** 2, axis=(1, 2))
    l_coll = jnp.mean(coll_penalty)
    
    # Total Loss
    # Weights can be tuned
    total_loss = l_track #+ 0.001 * l_effort + 1.0 * l_coll
    
    return total_loss, (l_track, l_effort, l_coll)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init, xi_init, z_target, dynamics):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, z_init, xi_init, z_target, dynamics
    )
    # Gradient clipping
    # grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 3. Main Training Loop ---
with solver_ts:
    dynamics = PDEDynamics(solver_ts)
    
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
