import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from flax import linen as nn
from typing import Sequence

# Add project root to path
script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

from dynamics import PDEDynamics, sample_initial_vorticity
from data_utils import generate_shape_pair, make_actuator_grid

# Ensure high precision for NS solver consistency
jax.config.update("jax_enable_x64", True)

# --- 1. Define a Null Policy ---
def null_policy_fn(params, z_curr, z_target, xi_curr):
    """Returns zero forcing and zero actuator velocity regardless of state."""
    n_agents = xi_curr.shape[0]
    u = jnp.zeros((n_agents, 2), dtype=jnp.float64)
    v = jnp.zeros((n_agents, 2), dtype=jnp.float64)
    return u, v

def rollout_scene_null(rho_init, rho_target, xi_init, dynamics, t_steps=100):
    """Unrolls the physics with zero control input."""
    key_omega = jax.random.PRNGKey(42)
    n = rho_init.shape[0]
    
    # Generate the turbulent background flow
    omega_init = sample_initial_vorticity(key_omega, n).astype(jnp.float64)
    
    # params can be None because null_policy_fn ignores them
    trajectories = dynamics.unroll_controlled(
        omega_init, 
        rho_init.astype(jnp.float64), 
        rho_target.astype(jnp.float64), 
        xi_init.astype(jnp.float64), 
        None, # No params needed for null policy
        t_steps
    )
    
    # Returns: omega_traj, rho_traj, xi_traj, u_traj, v_traj
    return trajectories

def main():
    n, L, m_agents, t_steps = 64, jnp.pi, 25, 200
    solver_ts = Tesseract.from_image("solver_ns_shape")

    with solver_ts:
        # IMPORTANT: We pass the null_policy_fn here to check natural evolution
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=null_policy_fn, use_tesseract=False)

        key = jax.random.PRNGKey(64)
        n_scenes, n_cols = 2, 6
        fig, axes = plt.subplots(n_scenes, n_cols, figsize=(4 * n_cols, 4 * n_scenes))
        time_indices = jnp.linspace(0, t_steps - 1, 4, dtype=int)

        for i in range(n_scenes):
            key, subk = jax.random.split(key)
            rho_init, rho_target = generate_shape_pair(subk, n=n, L=L)
            xi_init = make_actuator_grid(m_agents, L)

            # Perform rollout with null input
            trajs = rollout_scene_null(rho_init, rho_target, xi_init, dynamics, t_steps=t_steps)
            omega_traj, rho_traj, xi_traj, u_traj, v_traj = trajs

            ax_row = axes[i] if n_scenes > 1 else axes
            
            # Plot Initial Density
            ax_row[0].contourf(jnp.linspace(0, L, n), jnp.linspace(0, L, n), rho_init, levels=30, cmap="viridis")
            ax_row[0].set_title(f"Scene {i+1}\nInitial Density")
            
            # Plot Target Shape (for reference)
            ax_row[1].contourf(jnp.linspace(0, L, n), jnp.linspace(0, L, n), rho_target, levels=30, cmap="magma", alpha=0.3)
            ax_row[1].set_title("Reference Target")

            # Plot Natural Evolution Snapshots
            for j, t_idx in enumerate(time_indices):
                ax = ax_row[2 + j]
                # Background: Vorticity (shows the 'swirls' driving the evolution)
                ax.contour(jnp.linspace(0, L, n), jnp.linspace(0, L, n), omega_traj[t_idx], 
                           levels=10, colors='white', alpha=0.2, linestyles='solid')
                # Foreground: Density (the shape being advected)
                ax.contourf(jnp.linspace(0, L, n), jnp.linspace(0, L, n), rho_traj[t_idx], levels=30, cmap="viridis")
                ax.set_title(f"Natural Evolution\nt = {t_idx}")

        plt.tight_layout()
        plt.savefig("ns2d_natural_evolution_check.png", dpi=150)
        print("Natural evolution check saved to ns2d_natural_evolution_check.png")

if __name__ == "__main__":
    main()