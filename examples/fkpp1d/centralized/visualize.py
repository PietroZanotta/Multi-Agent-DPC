import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
import sys
import flax.serialization
from pathlib import Path
jax.config.update("jax_platform_name", "cpu")

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics # TODO: switch to dynamics.py for Tesseract-only
from models.policy import ControlNet
import data_utils

def load_params(model, filepath, n_pde=100, n_agents=4):
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    # Match the 3-arg signature of ControlNet
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
    n_pde, n_agents, T_steps = 100, 6, 300
    solver_ts = Tesseract.from_image("solver_v1")
    
    with solver_ts:
        dynamics = PDEDynamics(solver_ts, use_tesseract=False)
        model = ControlNet(features=(64, 64))
        
        try:
            params = load_params(model, 'centralized_params.msgpack', n_pde, n_agents)
            print("Loaded model parameters.")
        except FileNotFoundError:
            print("Error: centralized_params.msgpack not found. Run training first.")
            return

        key = jax.random.PRNGKey(2)
        
        # 2 Examples, 4 Columns (State, U, V, Xi)
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        
        for i in range(2): 
            key, subkey1, subkey2 = jax.random.split(key, 3)
            _, z_init = data_utils.generate_grf(subkey1, n_points=n_pde, length_scale=0.2)
            _, z_target = data_utils.generate_grf(subkey2, n_points=n_pde, length_scale=0.4)
            xi_init = jnp.linspace(0.2, 0.8, n_agents) 
            
            z_traj, xi_traj, u_traj, v_traj = visualize_rollout(
                params, model, z_init, xi_init, z_target, dynamics, T_steps
            )
            
            x_grid = jnp.linspace(0, 1, n_pde)
            
            # --- Column 1: State Evolution (Modified with script 2 features) ---
            ax = axes[i, 0]
            ax.plot(x_grid, z_target, 'k--', label='Target', linewidth=2)
            ax.plot(x_grid, z_init, 'b:', label='Initial', alpha=0.6)
            
            # Feature from script 2: Plot intermediate steps (faint)
            for t in range(0, T_steps, 10): # Plot every 10 steps for clarity
                ax.plot(x_grid, z_traj[t], 'g-', alpha=0.1)
                
            ax.plot(x_grid, z_traj[-1], 'r-', label='Final Output', linewidth=2)
            
            # Highlight final actuator positions (from script 1)
            act_idx = (xi_traj[-1] * n_pde).astype(int)
            ax.scatter(xi_traj[-1], z_traj[-1, act_idx], color='red', zorder=5, label='Actuators')
            
            ax.set_title(f"Ex {i+1}: State Evolution")
            ax.set_ylim([-2, 2]) # Feature from script 2
            ax.legend()
            
            # --- Column 2: Controls (u) (Modified with script 2 labels) ---
            ax2 = axes[i, 1]
            ax2.plot(u_traj, label=[f'u{k}' for k in range(n_agents)])
            ax2.set_title(f"Ex {i+1}: Forcing Intensity (u)")
            ax2.legend()
            
            # --- Column 3: Controls (v) (Modified with script 2 labels) ---
            ax3 = axes[i, 2]
            ax3.plot(v_traj, label=[f'v{k}' for k in range(n_agents)])
            ax3.set_title(f"Ex {i+1}: Velocity (v)")
            ax3.legend()

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
        plt.savefig('centralized_visualization.png')
        print("Saved updated visualization to centralized_visualization.png")

if __name__ == "__main__":
    main()