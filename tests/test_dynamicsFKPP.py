import matplotlib.pyplot as plt
from tesseract_core import Tesseract
import jax.numpy as jnp
import sys
from pathlib import Path

# Adjust pathing to find your dynamics wrapper
script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from examples.fkpp1d.centralized.dynamics_dual import PDEDynamics

def test_and_plot_spacetime_1d():
    # Load the Fisher-KPP specific solver image
    solver_ts = Tesseract.from_image("solver_fkpp1d_v1")
    N = 100
    T_steps = 1000 

    with solver_ts:
        model = PDEDynamics(solver_ts, use_tesseract=False, N=N)

        # 1. Initialize State: 1D Vector (N,)
        z_init = jnp.zeros(N)
        center = N // 2
        r = 6 
        z_init = z_init.at[center-r:center+r].set(1.0)

        # 2. Initialize Agents: (4 agents, scalar positions)
        xi_init = jnp.array([0.4, 0.45, 0.55, 0.6])

        # 3. Create constant control sequences
        u_seq = jnp.tile(jnp.array([0.0, 0.0, 0.0, 0.0]), (T_steps, 1))
        # Move agents outward slowly
        v_seq = jnp.tile(jnp.array([-0.05, -0.02, 0.02, 0.05]), (T_steps, 1))

        # 4. Run the full unroll
        print(f"Unrolling Fisher-KPP 1D trajectory for {T_steps} steps...")
        results = model.unroll(z_init, xi_init, u_seq, v_seq)
        
        z_traj = results["z_trajectory"]   # Shape: (T, N)
        xi_traj = results["xi_trajectory"] # Shape: (T, n_agents)

        # 5. Plotting Spacetime 2D Diagram
        plt.figure(figsize=(10, 8))
        
        # Plot the Density as an image: (Time on Y-axis, Space on X-axis)
        # Using 'magma' to highlight the spreading front
        extent = [0, 1, T_steps, 0] # [x_min, x_max, t_max, t_min]
        im = plt.imshow(z_traj, aspect='auto', extent=extent, cmap='magma', origin='upper')
        
        # Overlay Agent Trajectories
        # xi_traj shape is (T, n_agents). We plot each agent's x position over time.
        time_axis = jnp.arange(T_steps)
        for a in range(xi_traj.shape[1]):
            plt.plot(xi_traj[:, a], time_axis, color='cyan', linestyle='--', alpha=0.8, label=f'Agent {a}' if a==0 else "")

        # Formatting
        plt.colorbar(im, label="Population Density (z)")
        plt.xlabel("Spatial Position (x)")
        plt.ylabel("Time Steps (t)")
        plt.title("Fisher-KPP 1D Spacetime Plot (Hovmöller Diagram)")
        plt.legend()
        
        plt.savefig("fisher1d_spacetime_2d.png", bbox_inches='tight')
        print("Spacetime plot saved as fisher1d_spacetime_2d.png")

if __name__ == "__main__":
    test_and_plot_spacetime_1d()