import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

def main():
    solver_ts = Tesseract.from_image("solver_v1")

    with solver_ts:
        num_spatial_points = 100
        num_time_steps = 5000
        num_agents = 4  # Updated to 4 agents

        # 1. Start with a cold system (all zeros)
        z_init = jnp.zeros(num_spatial_points)
        
        # 2. Position 4 agents evenly across the grid (e.g., 0.2, 0.4, 0.6, 0.8)
        xi_init = jnp.array([0.2, 0.4, 0.6, 0.8]) 
        
        # 3. u_seq: Constant heat intensity for each agent
        # Using 0.5 as a starting point; increase if the "pillars" look too faint
        u_intensity = 0.5
        u_seq = jnp.ones((num_time_steps, num_agents)) * u_intensity
        
        # 4. v_seq: Zero velocity (stationary actuators)
        v_seq = jnp.zeros((num_time_steps, num_agents))

        inputs = {
            "z_init": z_init,
            "xi_init": xi_init,
            "u_seq": u_seq,
            "v_seq": v_seq
        }

        print("--- Running Forward Pass (4 Stationary Heat Sources) ---")
        results = apply_tesseract(solver_ts, inputs)
        
        z_traj = results["z_trajectory"]
        xi_traj = results["xi_trajectory"]

        # 3. Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plotting the heat map of the PDE evolution
        im = ax1.imshow(z_traj, aspect='auto', origin='lower')
        ax1.set_title("PDE State (z) Over Time")
        ax1.set_xlabel("Spatial Grid")
        ax1.set_ylabel("Time Step")
        fig.colorbar(im, ax=ax1)
        
        # Plotting Actuator positions (should look like straight vertical lines)
        ax2.plot(xi_traj)
        ax2.set_title("Actuator Positions (xi) Over Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Position")
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig("fixed_actuator_sim.png")
        print("Visualization saved to fixed_actuator_sim.png")

if __name__ == "__main__":
    main()