import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
import sys
import flax.serialization
from pathlib import Path
jax.config.update("jax_platform_name", "cpu")

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

from dpc_engine.dynamics_dual import PDEDynamics
from models import ControlNet
import data_utils

def load_params(model, filepath, n_pde=100, n_agents=4):
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    # Match the 3-arg signature
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def visualize_rollout(params, model, z_init, xi_init, z_target, dynamics, T_steps=300):
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

def main():
    n_pde, n_agents, T_steps = 100, 4, 300
    solver_ts = Tesseract.from_image("solver_v1")
    
    with solver_ts:
        dynamics = PDEDynamics(solver_ts, use_tesseract=False)
        model = ControlNet(features=(64, 64))
        
        try:
            params = load_params(model, 'model_params.msgpack', n_pde, n_agents)
            print("Loaded model parameters.")
        except FileNotFoundError:
            print("Error: model_params.msgpack not found.")
            return

        key = jax.random.PRNGKey(42)
        # Added a 4th column for Actuator Positions
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        
        for i in range(2): 
            key, subkey1, subkey2 = jax.random.split(key, 3)
            _, z_init = data_utils.generate_grf(subkey1, n_points=n_pde, length_scale=0.2)
            _, z_target = data_utils.generate_grf(subkey2, n_points=n_pde, length_scale=0.4)
            xi_init = jnp.array([0.2, 0.4, 0.6, 0.8])
            
            z_traj, xi_traj, u_traj, v_traj = visualize_rollout(
                params, model, z_init, xi_init, z_target, dynamics, T_steps
            )
            
            x_grid = jnp.linspace(0, 1, n_pde)
            
            # --- Column 1: State Evolution ---
            ax = axes[i, 0]
            ax.plot(x_grid, z_target, 'k--', label='Target', linewidth=2)
            ax.plot(x_grid, z_init, 'b:', label='Initial', alpha=0.6)
            ax.plot(x_grid, z_traj[-1], 'r-', label='Final Output', linewidth=2)
            # Add dots for final actuator positions
            ax.scatter(xi_traj[-1], z_traj[-1, (xi_traj[-1]*n_pde).astype(int)], color='red', zorder=5, label='Actuators')
            ax.set_title(f"Ex {i+1}: State Evolution")
            ax.legend()
            
            # --- Column 2: Controls (u) ---
            ax2 = axes[i, 1]
            ax2.plot(u_traj)
            ax2.set_title(f"Ex {i+1}: Forcing Intensity (u)")
            
            # --- Column 3: Controls (v) ---
            ax3 = axes[i, 2]
            ax3.plot(v_traj)
            ax3.set_title(f"Ex {i+1}: Velocity (v)")

            # --- Column 4: Actuator Trajectories (xi) ---
            ax4 = axes[i, 3]
            for j in range(n_agents):
                ax4.plot(xi_traj[:, j], label=f'Agent {j}')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax4.axhline(y=1, color='black', linestyle='--', alpha=0.3)
            ax4.set_title(f"Ex {i+1}: Actuator Positions ($\\xi$)")
            ax4.set_ylabel("Position in $\Omega$")
            ax4.set_xlabel("Time Steps")
            ax4.set_ylim([-0.1, 1.1])
            ax4.legend(loc='right')
            
        plt.tight_layout()
        plt.savefig('visualization_results_dpc_new.png')
        print("Saved visualization to visualization_results_dpc_new.png")

if __name__ == "__main__":
    main()
