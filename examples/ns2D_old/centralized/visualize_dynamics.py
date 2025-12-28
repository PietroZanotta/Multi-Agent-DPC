import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from typing import Sequence

# Add project root to path
script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

# Using the requested PDEDynamics and sample_initial_vorticity provided in the prompt
from dynamics import PDEDynamics, sample_initial_vorticity
from data_utils import generate_shape_pair, make_actuator_grid

# Ensure high precision
jax.config.update("jax_enable_x64", True)

def rollout_scene_null(rho_init, xi_init, dynamics, t_steps=100):
    """
    Manually unrolls the physics using the step() method.
    Since null input is required, u and v are fixed to zeros.
    """
    key_omega = jax.random.PRNGKey(42)
    n = rho_init.shape[0]
    
    # 1. Initialize background vorticity (Physical space)
    omega_init = sample_initial_vorticity(key_omega, n).astype(jnp.float64)
    
    # 2. Define zero controls (Null Input)
    m_agents = xi_init.shape[0]
    u_zero = jnp.zeros((m_agents, 2), dtype=jnp.float64)
    v_zero = jnp.zeros((m_agents, 2), dtype=jnp.float64)

    def step_fn(carry, _):
        omega_curr, rho_curr, xi_curr = carry
        
        # Advance physics by one step using the provided dynamics.step method
        omega_next, rho_next, xi_next = dynamics.step(
            omega_curr, rho_curr, xi_curr, u_zero, v_zero
        )
        
        # State to carry, State to record in trajectory
        return (omega_next, rho_next, xi_next), (omega_next, rho_next, xi_next)

    # 3. Perform the manual scan
    _, trajectory = jax.lax.scan(
        step_fn, 
        (omega_init, rho_init.astype(jnp.float64), xi_init.astype(jnp.float64)), 
        None, 
        length=t_steps
    )
    
    # trajectory is (omega_traj, rho_traj, xi_traj)
    return trajectory

def main():
    n, L, m_agents, t_steps = 64, jnp.pi, 25, 200
    solver_ts = Tesseract.from_image("solver_ns_shape")

    with solver_ts:
        # Initializing PDEDynamics as requested (step-based)
        dynamics = PDEDynamics(solver_ts, use_tesseract=False)

        key = jax.random.PRNGKey(64)
        n_scenes, n_cols = 2, 6
        fig, axes = plt.subplots(n_scenes, n_cols, figsize=(4 * n_cols, 4 * n_scenes))
        time_indices = jnp.linspace(0, t_steps - 1, 4, dtype=int)

        for i in range(n_scenes):
            key, subk = jax.random.split(key)
            rho_init, rho_target = generate_shape_pair(subk, n=n, L=L)
            xi_init = make_actuator_grid(m_agents, L)

            # Perform manual rollout with null input
            omega_traj, rho_traj, xi_traj = rollout_scene_null(
                rho_init, xi_init, dynamics, t_steps=t_steps
            )

            ax_row = axes[i] if n_scenes > 1 else axes
            
            # Plot Initial Density
            ax_row[0].contourf(jnp.linspace(0, L, n), jnp.linspace(0, L, n), rho_init, levels=30, cmap="viridis")
            ax_row[0].set_title(f"Scene {i+1}\nInitial Density")
            
            # Plot Target Shape (Static Reference)
            ax_row[1].contourf(jnp.linspace(0, L, n), jnp.linspace(0, L, n), rho_target, levels=30, cmap="magma", alpha=0.3)
            ax_row[1].set_title("Reference Target")

            # Plot Natural Evolution Snapshots
            for j, t_idx in enumerate(time_indices):
                ax = ax_row[2 + j]
                # Background: Vorticity field (the 'motor' of the fluid)
                ax.contour(jnp.linspace(0, L, n), jnp.linspace(0, L, n), omega_traj[t_idx], 
                           levels=10, colors='white', alpha=0.2, linestyles='solid')
                
                # Foreground: Density field (passive scalar being moved by omega)
                ax.contourf(jnp.linspace(0, L, n), jnp.linspace(0, L, n), rho_traj[t_idx], levels=30, cmap="viridis")
                
                # Plot Actuators (should stay static as v=0)
                ax.scatter(xi_traj[t_idx, :, 0], xi_traj[t_idx, :, 1], c='red', s=5, alpha=0.5)
                
                ax.set_title(f"Natural Evolution\nt = {t_idx}")

        plt.tight_layout()
        plt.savefig("ns2d_natural_evolution_check.png", dpi=150)
        print("Natural evolution check saved to ns2d_natural_evolution_check.png")

if __name__ == "__main__":
    main()