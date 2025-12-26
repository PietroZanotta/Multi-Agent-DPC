import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
import sys
import flax.serialization
from pathlib import Path
from functools import partial

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import ControlNet
import data_utils

def load_params(model, filepath, n_pde=100, n_agents=8):
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    # The init call ensures we have a PyTree template to load the bytes into
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def main():
    n_pde, n_agents, T_steps = 100, 6, 300 
    solver_ts = Tesseract.from_image("solver_heat_centralized:latest") # Use the image with the flat-vector API
    
    with solver_ts:
        model = ControlNet(features=(64, 64))
        
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=True)
        
        try:
            params = load_params(model, 'centralized_params.msgpack', n_pde, n_agents)
            print(f"Loaded centralized model parameters for {n_agents} agents.")
        except FileNotFoundError:
            print("Error: centralized_params.msgpack not found.")
            return

        key = jax.random.PRNGKey(1234)
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        
        for i in range(2): 
            key, subkey1, subkey2 = jax.random.split(key, 3)
            _, z_init = data_utils.generate_grf(subkey1, n_points=n_pde, length_scale=0.2)
            _, z_target = data_utils.generate_grf(subkey2, n_points=n_pde, length_scale=0.4)
            xi_init = jnp.linspace(0.1, 0.9, n_agents) # Start agents spread out
            
            # The dynamics object handles the flattening/unrolling automatically
            z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
                z_init, xi_init, z_target, params, T_steps
            )
            
            x_grid = jnp.linspace(0, 1, n_pde)
            
            # --- Column 1: State Evolution ---
            ax = axes[i, 0]
            ax.plot(x_grid, z_target, 'k--', label='Target', linewidth=2)
            ax.plot(x_grid, z_init, 'b:', label='Initial', alpha=0.6)
            
            # Ghosting: Plot every 15 steps
            for t in range(0, T_steps, 15):
                ax.plot(x_grid, z_traj[t], 'g-', alpha=0.1)
                
            ax.plot(x_grid, z_traj[-1], 'r-', label='Final Output', linewidth=2)
            
            # Actuator scatter markers
            act_idx = jnp.clip((xi_traj[-1] * n_pde).astype(int), 0, n_pde-1)
            ax.scatter(xi_traj[-1], z_traj[-1, act_idx], color='red', zorder=5)
            
            ax.set_title(f"Ex {i+1}: State Evolution (N={n_agents})")
            ax.set_ylim([-2, 2])
            ax.legend(fontsize='small')
            
            # --- Column 2: Controls (u) ---
            ax2 = axes[i, 1]
            ax2.plot(u_traj)
            ax2.set_title("Forcing Intensity (u)")
            
            # --- Column 3: Controls (v) ---
            ax3 = axes[i, 2]
            ax3.plot(v_traj)
            ax3.set_title("Velocity (v)")

            # --- Column 4: Actuator Trajectories (xi) ---
            ax4 = axes[i, 3]
            for j in range(n_agents):
                ax4.plot(xi_traj[:, j], label=f'A{j}', alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax4.axhline(y=1, color='black', linestyle='--', alpha=0.3)
            ax4.set_title("Actuator Paths ($\\xi$)")
            ax4.set_ylim([-0.1, 1.1])
            if n_agents <= 8: ax4.legend(loc='right', fontsize='xx-small')
            
        plt.tight_layout()
        plt.savefig('centralized_visual_results.png')
        print("Visualization saved to centralized_visual_results.png")

if __name__ == "__main__":
    main()