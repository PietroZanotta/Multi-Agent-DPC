import sys
import time
from functools import partial
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from tqdm import trange
from tesseract_core import Tesseract

script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

from examples.ns2D.centralized.data_utils import generate_shape_pair, make_actuator_grid
from examples.ns2D.centralized.dynamics import PDEDynamics, sample_initial_vorticity


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


n = 128
L = 2.0 * jnp.pi
m_agents = 64
batch_size = 4
epochs = 200
t_steps = 200

key = jax.random.PRNGKey(0)
key_omega, key_data = jax.random.split(key)

omega_init = sample_initial_vorticity(key_omega, n, V_SCALE_BASE=0.5, V_FALLOFF=0.8)

xi_init_single = make_actuator_grid(m_agents, L)

total_samples = 128
data_keys = jax.random.split(key_data, 2 * total_samples)
z_init_all, z_target_all = jax.vmap(partial(generate_shape_pair, n=n, L=L))(
    data_keys[:total_samples]
)
xi_init_all = jnp.tile(xi_init_single[None, ...], (total_samples, 1, 1))

model = NS2DControlNet(features=(128, 128))
dummy_z = jnp.zeros((n, n))
dummy_target = jnp.zeros((n, n))
dummy_xi = xi_init_single
params = model.init(jax.random.PRNGKey(1), dummy_z, dummy_target, dummy_xi)

lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

solver_ts = Tesseract.from_image("solver_ns_shape")

with solver_ts:
    dynamics = PDEDynamics(solver_ts, use_tesseract=False)

    def rollout_fn(p, z_init, xi_init, z_target, dyn):
        def step_fn(carry, _):
            omega_curr, z_curr, xi_curr = carry
            u_action, v_action = model.apply(p, z_curr, z_target, xi_curr)
            omega_next, z_next, xi_next = dyn.step(
                omega_curr, z_curr, xi_curr, u_action, v_action
            )
            return (omega_next, z_next, xi_next), (
                z_next,
                xi_next,
                u_action,
                v_action,
            )

        (_, _, _), (z_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
            step_fn, (omega_init, z_init, xi_init), None, length=t_steps
        )
        return z_traj, xi_traj, u_traj, v_traj

    def loss_fn(p, z_init, xi_init, z_target, dyn):
        z_traj, xi_traj, u_traj, v_traj = rollout_fn(p, z_init, xi_init, z_target, dyn)
        track = jnp.mean((z_traj[-1] - z_target) ** 2)
        effort = jnp.mean(u_traj**2) + 0.1 * jnp.mean(v_traj**2)
        return 10.0 * track + 0.001 * effort, (track, effort)

    @partial(jax.jit, static_argnames="dyn")
    def train_step(p, opt_st, z_init_b, xi_init_b, z_target_b, dyn):
        batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))

        def mean_loss_fn(pp):
            losses, auxs = batched_loss_fn(pp, z_init_b, xi_init_b, z_target_b, dyn)
            return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

        (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(p)
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss, aux

    indices = jnp.arange(total_samples)
    metrics = []
    start_time = time.time()

    for epoch in trange(epochs):
        key, subkey = jax.random.split(key)
        shuffled = jax.random.permutation(subkey, indices)
        start = 0
        current_idx = shuffled[start : start + batch_size]

        z_init_batch = z_init_all[current_idx]
        z_target_batch = z_target_all[current_idx]
        xi_init_batch = xi_init_all[current_idx]

        params, opt_state, loss, aux = train_step(
            params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics
        )
        track, effort = aux

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:03d} | Loss {loss:.6f} | Track {track:.6f} | Effort {effort:.6f} | Time {elapsed:.1f}s"
            )
            metrics.append((epoch, loss, track, effort))

metrics = jnp.array(metrics)
epochs_rec = metrics[:, 0]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_rec, metrics[:, 1])
plt.yscale("log")
plt.title("Total loss")

plt.subplot(1, 2, 2)
plt.plot(epochs_rec, metrics[:, 2], label="track")
plt.plot(epochs_rec, metrics[:, 3], label="effort")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("centralized_training_ns2d.png")

import flax.serialization
with open('centralized_params_ns2d.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(params))
print(f"Training finished in {time.time() - start_time:.2f}s. Params saved.")
