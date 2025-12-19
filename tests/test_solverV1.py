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
        num_agents = 4

        # 1. Start cold
        z_init = jnp.zeros(num_spatial_points)
        
        # 2. Starting positions
        xi_init = jnp.array([0.1, 0.3, 0.5, 0.7]) 
        
        # 3. Constant heat intensity
        u_warmup = jnp.ones((2500, num_agents)) * 0.3
        u_cooling = jnp.zeros((2500, num_agents))
        u_seq = jnp.concatenate([u_warmup, u_cooling], axis=0)

        
        # 4. VELOCITY: Move all agents to the right at a constant speed
        # Speed 0.04 means they move 0.2 units of distance over 5000 steps (at dt=0.001)
        velocity = 0.04 
        v_seq = jnp.ones((num_time_steps, num_agents)) * velocity

        inputs = {
            "z_init": z_init,
            "xi_init": xi_init,
            "u_seq": u_seq,
            "v_seq": v_seq
        }

        print("--- Running Forward Pass (Moving Actuators) ---")
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
        plt.savefig("actuator_sim.png")
        print("Visualization saved to actuator_sim.png")

if __name__ == "__main__":
    main()