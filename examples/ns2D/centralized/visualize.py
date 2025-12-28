
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
from flax import linen as nn
from typing import Sequence

script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

from examples.ns2D.centralized.dynamics import PDEDynamics, sample_initial_vorticity
from examples.ns2D.centralized.data_utils import generate_shape_pair, make_actuator_grid


class NS2DControlNet(nn.Module):
    features: Sequence[int]
    u_max: float = 10.0  # Match train.py
    v_max: float = 0.5

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        error = z_curr - z_target
        x = error[..., None]
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        x = x.reshape(-1)
        x = nn.LayerNorm()(x)
        branch = nn.Dense(32)(x)
        branch = nn.tanh(branch)

        freqs = jnp.array([1.0, 2.0, 4.0, 8.0])
        angle = xi_curr[..., None] / (2.0 * jnp.pi) * freqs[None, None, :] * 2.0 * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        encoded = encoded.reshape(xi_curr.shape[0], -1)

        y = encoded
        for feat in [32, 32]:
            y = nn.Dense(feat)(y)
            y = nn.tanh(y)

        branch_rep = jnp.tile(branch, (xi_curr.shape[0], 1))
        h = jnp.concatenate([branch_rep, y], axis=-1)
        h = nn.Dense(64)(h)
        h = nn.tanh(h)

        u_raw = nn.Dense(2)(h)
        v_raw = nn.Dense(2)(h)

        u = self.u_max * jnp.tanh(u_raw)
        v = self.v_max * jnp.tanh(v_raw)
        return u, v


def load_params(model, filepath, n=128, m_agents=12, L=jnp.pi):
    with open(filepath, "rb") as f:
        serialized = f.read()
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n, n))
    dummy_target = jnp.zeros((n, n))
    dummy_xi = make_actuator_grid(m_agents, L)
    init_params = model.init(key, dummy_z, dummy_target, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized)


def rollout_scene(params, model, rho_init, rho_target, xi_init, dynamics, t_steps=100):
    key = jax.random.PRNGKey(0)
    key_omega, _ = jax.random.split(key)
    omega_init = sample_initial_vorticity(key_omega, rho_init.shape[0], V_SCALE_BASE=0.1, V_FALLOFF=0.4)

    def step_fn(carry, _):
        omega_curr, rho_curr, xi_curr = carry
        u_action, v_action = model.apply(params, rho_curr, rho_target, xi_curr)
        omega_next, rho_next, xi_next = dynamics.step(
            omega_curr, rho_curr, xi_curr, u_action, v_action
        )
        return (omega_next, rho_next, xi_next), (rho_next, xi_next, u_action, v_action)

    (_, _, _), (rho_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
        step_fn, (omega_init, rho_init, xi_init), None, length=t_steps
    )
    return rho_traj, xi_traj, u_traj, v_traj


def main():
    n = 64
    L = jnp.pi
    m_agents = 64
    t_steps = 200  # Match training

    # Use solver directly without Tesseract
    dynamics = PDEDynamics(solver_ts=None, use_tesseract=False)

    model = NS2DControlNet(features=(16, 32))
    try:
        params = load_params(model, "centralized_params_ns2d.msgpack", n, m_agents, L)
        print("Loaded trained parameters.")
    except FileNotFoundError:
        print("centralized_params_ns2d.msgpack not found, using randomly initialized policy.")
        key = jax.random.PRNGKey(1)
        dummy_z = jnp.zeros((n, n))
        dummy_target = jnp.zeros((n, n))
        dummy_xi = make_actuator_grid(m_agents, L)
        params = model.init(key, dummy_z, dummy_target, dummy_xi)

    # Use same key sequence as train.py for identical shapes
    key = jax.random.PRNGKey(0)
    key_omega, key_data = jax.random.split(key)
    
    # Generate same shapes as train.py
    total_samples = 1
    data_keys = jax.random.split(key_data, total_samples)
    z_init_all, z_target_all = jax.vmap(partial(generate_shape_pair, n=n, L=L))(
        data_keys[:total_samples]
    )
    rho_init = z_init_all[0]
    rho_target = z_target_all[0]

    # Debug: save init and target shapes (before creating main figure)
    fig_debug, ax_debug = plt.subplots()
    ax_debug.imshow(rho_init, origin='lower')
    fig_debug.savefig("rho_init.png")
    plt.close(fig_debug)
    
    fig_debug, ax_debug = plt.subplots()
    ax_debug.imshow(rho_target, origin='lower')
    fig_debug.savefig("rho_target.png")
    plt.close(fig_debug)

    n_scenes = 1
    n_cols = 5  # Initial, Target, t=0, t=mid, t=final
    fig, axes = plt.subplots(n_scenes, n_cols, figsize=(3.5 * n_cols, 3.5 * n_scenes))

    # Time indices: start, middle, end
    time_indices = [0, t_steps // 2, t_steps - 1]

    # Collect all data first to determine global color limits
    all_data = []
    scene_data = []

    for i in range(n_scenes):
        xi_init = make_actuator_grid(m_agents, L)

        rho_traj, xi_traj, u_traj, v_traj = rollout_scene(
            params, model, rho_init, rho_target, xi_init, dynamics, t_steps=t_steps
        )
        
        scene_data.append({
            'rho_init': rho_init,
            'rho_target': rho_target,
            'rho_traj': rho_traj,
            'xi_traj': xi_traj,
        })
        
        # Collect for global limits
        all_data.extend([rho_init, rho_target])
        for t_idx in time_indices:
            all_data.append(rho_traj[t_idx])

    # Compute global color limits
    vmin = 0.0
    vmax = max(float(jnp.max(d)) for d in all_data)
    vmax = max(vmax, 1.0)  # Ensure at least 0-1 range

    for i in range(n_scenes):
        data = scene_data[i]
        row_axes = axes if n_scenes == 1 else axes[i]

        # Plot Initial
        im = row_axes[0].imshow(
            data['rho_init'], origin='lower', 
            extent=[0, L, 0, L], cmap='viridis',
            vmin=vmin, vmax=vmax
        )
        row_axes[0].set_title(f"Scene {i+1} - Initial")
        row_axes[0].set_xlabel("x")
        row_axes[0].set_ylabel("y")

        # Plot Target
        row_axes[1].imshow(
            data['rho_target'], origin='lower',
            extent=[0, L, 0, L], cmap='viridis',
            vmin=vmin, vmax=vmax
        )
        row_axes[1].set_title(f"Scene {i+1} - Target")
        row_axes[1].set_xlabel("x")

        # Plot trajectory snapshots
        for j, t_idx in enumerate(time_indices):
            ax = row_axes[2 + j]
            rho_t = data['rho_traj'][t_idx]
            
            ax.imshow(
                rho_t, origin='lower',
                extent=[0, L, 0, L], cmap='viridis',
                vmin=vmin, vmax=vmax
            )
            ax.set_title(f"Scene {i+1} - t = {t_idx}")
            ax.set_xlabel("x")
            
            # Show density stats
            rho_sum = float(jnp.sum(rho_t))
            ax.text(0.02, 0.98, f"sum={rho_sum:.2f}", 
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # Add colorbar with proper positioning
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Density')

    plt.savefig("ns2d_centralized_visualization.png", dpi=150, bbox_inches='tight')
    print("Saved: ns2d_centralized_visualization.png")


if __name__ == "__main__":
    main()
