import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tesseract_core import Tesseract
import sys
from pathlib import Path
import flax.serialization

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import Heat2DControlNet
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

def create_animation(z_traj, xi_traj, z_target, save_path='heat2d_animation.mp4'):
    """Create animated visualization of control trajectory."""
    T, N, _ = z_traj.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Determine color scale
    vmin = min(jnp.min(z_traj), jnp.min(z_target))
    vmax = max(jnp.max(z_traj), jnp.max(z_target))

    # Initialize plots
    im1 = ax1.imshow(z_traj[0], origin='lower', extent=[0, 1, 0, 1],
                     cmap='RdBu_r', vmin=vmin, vmax=vmax)
    scat1 = ax1.scatter(xi_traj[0, :, 0], xi_traj[0, :, 1],
                       c='black', s=50, marker='x', linewidths=2)
    ax1.set_title('Current State (t=0)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    im2 = ax2.imshow(z_target, origin='lower', extent=[0, 1, 0, 1],
                     cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax2.set_title('Target State')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Add colorbar
    fig.colorbar(im1, ax=[ax1, ax2], location='right', shrink=0.8, label='Temperature')

    def update(frame):
        im1.set_array(z_traj[frame])
        scat1.set_offsets(xi_traj[frame])
        ax1.set_title(f'Current State (t={frame})')
        return [im1, scat1]

    anim = animation.FuncAnimation(fig, update, frames=T,
                                   interval=50, blit=True)

    # Save as MP4
    anim.save(save_path, writer='ffmpeg', fps=20, dpi=100)
    print(f"Animation saved: {save_path}")
    plt.close()

    # Also save as GIF
    gif_path = save_path.replace('.mp4', '.gif')
    anim = animation.FuncAnimation(fig, update, frames=T,
                                   interval=50, blit=True)
    anim.save(gif_path, writer='pillow', fps=20)
    print(f"Animation saved: {gif_path}")

def main():
    n_grid = 64
    n_agents = 16
    T_steps = 300

    solver_ts = Tesseract.from_image("solver_heat2d_centralized:latest")

    with solver_ts:
        model = Heat2DControlNet(features=(16, 32))
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply,
                               use_tesseract=True)

        params = load_params(model, 'centralized_params_heat2d.msgpack',
                            n_grid, n_agents)

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        # Generate test scenario
        print("Generating test scenario...")
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
        print("Running controlled trajectory...")
        z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )

        print("Creating animation...")
        create_animation(z_traj, xi_traj, z_target)

        print("\nDone!")

if __name__ == "__main__":
    main()
