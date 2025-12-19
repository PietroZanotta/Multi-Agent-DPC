
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
import sys
import flax.serialization
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

from dpc_engine.dynamics import PDEDynamics
from models import MLP, split_action
import data_utils

def load_params(model, filepath):
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    # Initialize dummy params to have the correct structure for restoration
    dummy_input = jnp.zeros((1, 100)) # matches main.py n_pde
    key = jax.random.PRNGKey(0)
    init_params = model.init(key, dummy_input)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def visualize_rollout(params, model, z_init, xi_init, z_target, dynamics, T_steps=10):
    
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        
        # Policy inference
        action_flat = model.apply(params, z_curr[jnp.newaxis, :])[0]
        u_action, v_action = split_action(action_flat)
        
        actions = {'u': u_action, 'v': v_action}
        
        # Dynamics step
        z_next, xi_next = dynamics.step(z_curr, xi_curr, actions)
        
        return (z_next, xi_next), (z_next, xi_next, u_action, v_action)
    
    _, (z_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
        step_fn,
        (z_init, xi_init),
        None,
        length=T_steps
    )
    
    return z_traj, xi_traj, u_traj, v_traj

def main():
    # Setup
    n_pde = 100
    n_agents = 4
    output_dim = 8
    T_steps = 10 # Match training for now
    
    # Initialize Environment
    solver_ts = Tesseract.from_image("solver_v1")
    
    with solver_ts:
        dynamics = PDEDynamics(solver_ts)
        model = MLP(features=(64, 64, output_dim))
        
        # Load Params
        try:
            params = load_params(model, 'model_params.msgpack')
            print("Loaded model parameters.")
        except FileNotFoundError:
            print("Error: model_params.msgpack not found. Run main.py first.")
            return

        # Generate Test Cases
        key = jax.random.PRNGKey(42) # Different seed
        
        # Create a figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i in range(2): # 2 examples
            key, subkey1, subkey2 = jax.random.split(key, 3)
            _, z_init = data_utils.generate_grf(subkey1, n_points=n_pde, length_scale=0.2)
            _, z_target = data_utils.generate_grf(subkey2, n_points=n_pde, length_scale=0.4)
            xi_init = jnp.array([0.2, 0.4, 0.6, 0.8])
            
            # Run Rollout
            z_traj, xi_traj, u_traj, v_traj = visualize_rollout(
                params, model, z_init, xi_init, z_target, dynamics, T_steps
            )
            
            # Plotting
            ax = axes[i, 0]
            # Plot Target
            x_grid = jnp.linspace(0, 1, n_pde)
            ax.plot(x_grid, z_target, 'k--', label='Target', linewidth=2)
            
            # Plot Trajectory (Initial and Final)
            ax.plot(x_grid, z_init, 'b:', label='Initial', alpha=0.6)
            ax.plot(x_grid, z_traj[-1], 'r-', label='Final Output', linewidth=2)
            
            # Plot intermediate steps (faint)
            for t in range(0, T_steps, 2):
                ax.plot(x_grid, z_traj[t], 'g-', alpha=0.2)
                
            ax.set_title(f"Example {i+1}: State Evolution")
            ax.legend()
            ax.set_ylim([-2, 2]) # Approximate range
            
            # Plot Controls
            ax2 = axes[i, 1]
            ax2.plot(u_traj, label=[f'u{k}' for k in range(4)])
            ax2.set_title(f"Example {i+1}: Controls (u)")
            ax2.legend()
            
        plt.tight_layout()
        plt.savefig('visualization_results_dpc.png')
        print("Saved visualization to visualization_results_dpc.png")

if __name__ == "__main__":
    main()
