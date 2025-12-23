import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
import flax.serialization
from flax import linen as nn
from typing import Sequence

script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

from examples.ns2D.centralized.dynamics import PDEDynamics, sample_initial_vorticity
from examples.ns2D.centralized.data_utils import generate_shape_pair, make_actuator_grid
# from examples.ns2D.centralized.train import NS2DControlNet

class NS2DControlNet(nn.Module):
    features: Sequence[int]
    u_max: float = 2.0
    v_max: float = 0.5

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        error = z_curr - z_target
        x = error.reshape(-1)
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        branch = x

        freqs = jnp.array([1.0, 2.0, 4.0, 8.0])
        angle = xi_curr[..., None] / (2.0 * jnp.pi) * freqs[None, None, :] * 2.0 * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        encoded = encoded.reshape(xi_curr.shape[0], -1)

        y = encoded
        for feat in [64, 64]:
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


def load_params(model, filepath, n=128, m_agents=12):
    with open(filepath, "rb") as f:
        serialized = f.read()
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n, n))
    dummy_target = jnp.zeros((n, n))
    dummy_xi = make_actuator_grid(m_agents, 2.0 * jnp.pi)
    init_params = model.init(key, dummy_z, dummy_target, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized)


def rollout_scene(params, model, rho_init, rho_target, xi_init, dynamics, t_steps=100):
    key = jax.random.PRNGKey(42)
    key_omega, _ = jax.random.split(key)
    omega_init = sample_initial_vorticity(key_omega, rho_init.shape[0])

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
    n = 128
    L = 2.0 * jnp.pi
    m_agents = 12
    t_steps = 100

    solver_ts = Tesseract.from_image("solver_ns_shape")

    with solver_ts:
        dynamics = PDEDynamics(solver_ts, use_tesseract=False)

        model = NS2DControlNet(features=(128, 128))
        try:
            params = load_params(model, "centralized_params_ns2d.msgpack", n, m_agents)
        except FileNotFoundError:
            print("centralized_params_ns2d.msgpack not found, using randomly initialized policy.")
            key = jax.random.PRNGKey(1)
            dummy_z = jnp.zeros((n, n))
            dummy_target = jnp.zeros((n, n))
            dummy_xi = make_actuator_grid(m_agents, L)
            params = model.init(key, dummy_z, dummy_target, dummy_xi)

        key = jax.random.PRNGKey(1234)

        n_scenes = 2
        n_cols = 6
        fig, axes = plt.subplots(n_scenes, n_cols, figsize=(4 * n_cols, 4 * n_scenes))

        time_indices = jnp.linspace(0, t_steps - 1, 3, dtype=int)

        for i in range(n_scenes):
            key, subk = jax.random.split(key)
            rho_init, rho_target = generate_shape_pair(subk, n=n, L=L)
            xi_init = make_actuator_grid(m_agents, L)

            rho_traj, xi_traj, u_traj, v_traj = rollout_scene(
                params, model, rho_init, rho_target, xi_init, dynamics, t_steps=t_steps
            )

            row_axes = axes[i] if n_scenes > 1 else axes

            im0 = row_axes[0].contourf(
                jnp.linspace(0, L, n),
                jnp.linspace(0, L, n),
                rho_init,
                levels=30,
                cmap="viridis",
            )
            row_axes[0].set_title(f"Scene {i+1} - Initial")
            row_axes[0].set_xlabel("x")
            row_axes[0].set_ylabel("y")

            im1 = row_axes[1].contourf(
                jnp.linspace(0, L, n),
                jnp.linspace(0, L, n),
                rho_target,
                levels=30,
                cmap="viridis",
            )
            row_axes[1].set_title(f"Scene {i+1} - Target")
            row_axes[1].set_xlabel("x")
            row_axes[1].set_ylabel("y")

            for j, t_idx in enumerate(time_indices):
                ax = row_axes[2 + j]
                im = ax.contourf(
                    jnp.linspace(0, L, n),
                    jnp.linspace(0, L, n),
                    rho_traj[t_idx],
                    levels=30,
                    cmap="viridis",
                )
                ax.set_title(f"Scene {i+1} - t = {int(t_idx)}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            fig.colorbar(im0, ax=row_axes[:2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("ns2d_centralized_visualization.png", dpi=150)


        


if __name__ == "__main__":
    main()

