import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
import sys
from pathlib import Path
import flax.serialization

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
import data_utils

def load_params(model, filepath, n_grid=64, n_agents=16):
    """Load trained parameters from msgpack file."""
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n_grid, n_grid))
    dummy_xi = jnp.zeros((n_agents, 2))
    init_params = model.init(key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def main():
    n_grid = 64
    n_agents = 16
    T_steps = 300

    solver_ts = Tesseract.from_image("solver_heat2d_decentralized:latest")

    with solver_ts:
        model = DecentralizedHeat2DControlNet(features=(16, 32))
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply,
                               use_tesseract=True)

        params = load_params(model, 'decentralized_params_heat2d.msgpack',
                            n_grid, n_agents)

        key = jax.random.PRNGKey(1234)

        # Create visualization grid: 2 scenarios × (initial, target, 5 timepoints)
        n_scenarios = 2
        time_indices = [0, 75, 150, 225, T_steps-1]

        fig, axes = plt.subplots(n_scenarios, len(time_indices) + 2,
                                 figsize=(20, 8))

        for scenario in range(n_scenarios):
            key, k1, k2 = jax.random.split(key, 3)
            xx, yy, z_init = data_utils.generate_grf_2d(k1, n_points=n_grid)
            _, _, z_target = data_utils.generate_grf_2d(k2, n_points=n_grid)

            # Initialize agents in grid pattern
            n_side = int(jnp.sqrt(n_agents))
            spacing = 0.8 / (n_side + 1)
            xi_init = []
            for i in range(n_side):
                for j in range(n_side):
                    if len(xi_init) < n_agents:
                        xi_init.append([0.1 + spacing*(i+1), 0.1 + spacing*(j+1)])
            xi_init = jnp.array(xi_init)

            # Rollout
            print(f"Running scenario {scenario + 1}...")
            z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
                z_init, xi_init, z_target, params, T_steps
            )

            # Determine color scale
            vmin = min(jnp.min(z_init), jnp.min(z_target), jnp.min(z_traj))
            vmax = max(jnp.max(z_init), jnp.max(z_target), jnp.max(z_traj))

            # Plot initial state
            ax = axes[scenario, 0]
            im = ax.imshow(z_init, origin='lower', extent=[0, 1, 0, 1],
                          cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.scatter(xi_init[:, 0], xi_init[:, 1], c='black', s=30, marker='x')
            ax.set_title(f"Scenario {scenario+1}: Initial")
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            # Plot target
            ax = axes[scenario, 1]
            ax.imshow(z_target, origin='lower', extent=[0, 1, 0, 1],
                     cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title("Target")
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            # Plot trajectory snapshots
            for col, t_idx in enumerate(time_indices):
                ax = axes[scenario, col + 2]
                ax.imshow(z_traj[t_idx], origin='lower', extent=[0, 1, 0, 1],
                         cmap='RdBu_r', vmin=vmin, vmax=vmax)

                # Overlay actuator positions
                ax.scatter(xi_traj[t_idx, :, 0], xi_traj[t_idx, :, 1],
                          c='black', s=30, marker='x')

                ax.set_title(f"t={t_idx}")
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            # Compute MSE
            final_mse = jnp.mean((z_traj[-1] - z_target)**2)
            print(f"  Final MSE: {final_mse:.6f}")

        plt.tight_layout()
        # Add colorbar
        fig.colorbar(im, ax=axes, location='right', shrink=0.6, label='Temperature')
        plt.savefig('heat2d_decentralized_visualization.png', dpi=150)
        print("\nSaved: heat2d_decentralized_visualization.png")

if __name__ == "__main__":
    main()
