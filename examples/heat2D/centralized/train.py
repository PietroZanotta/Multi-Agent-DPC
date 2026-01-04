import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
import sys
from pathlib import Path
import optax
import time
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange
import flax.serialization

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import Heat2DControlNet
from data_utils import generate_grf_2d

# --- Configuration ---
solver_ts = Tesseract.from_image("solver_heat2d_centralized:latest")
n_grid = 64
n_agents = 16  # 4×4 grid of agents
batch_size = 16  # Smaller batch due to 2D memory
T_steps = 300
R_safe = 0.08  # Collision radius in 2D
epochs = 500

model = Heat2DControlNet(features=(16, 32))
key = jax.random.PRNGKey(0)

# Initialize with dummy 2D inputs
dummy_z = jnp.zeros((n_grid, n_grid))
dummy_xi = jnp.zeros((n_agents, 2))
params = model.init(key, dummy_z, dummy_z, dummy_xi)

lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

# --- Loss Function ---
def loss_fn(params, z_init, xi_init, z_target, dynamics):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )

    # 1. Tracking loss
    l_track = jnp.mean((z_traj - z_target[None, :, :]) ** 2)

    # 2. Effort loss
    l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(jnp.sum(v_traj ** 2, axis=-1))

    # 3. Boundary penalty (2D)
    margin = 0.02
    x_penalty = jnp.maximum(0, margin - xi_traj[:, :, 0])**2 + \
                jnp.maximum(0, xi_traj[:, :, 0] - (1.0 - margin))**2
    y_penalty = jnp.maximum(0, margin - xi_traj[:, :, 1])**2 + \
                jnp.maximum(0, xi_traj[:, :, 1] - (1.0 - margin))**2
    l_bound = jnp.mean(x_penalty + y_penalty)

    # 4. Collision avoidance (2D Euclidean distance)
    # Compute pairwise distances: (T, M, M)
    diff = xi_traj[:, :, None, :] - xi_traj[:, None, :, :]  # (T, M, M, 2)
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)  # (T, M, M)

    # Mask diagonal
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 10.0)) ** 2)

    # 5. Acceleration penalty
    l_accel = jnp.mean(jnp.sum(jnp.diff(v_traj, axis=0)**2, axis=-1))

    total_loss = 5.0 * l_track + 0.001 * l_effort + 100.0 * l_bound + \
                 1.0 * l_coll + 0.1 * l_accel

    return total_loss, (l_track, l_effort, l_coll, l_bound)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))

    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, z_init_batch, xi_init_batch,
                                        z_target_batch, dynamics)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- Training Loop ---
with solver_ts:
    dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply,
                           use_tesseract=False)

    print("Generating 2D dataset...")
    print("This may take several minutes due to eigendecomposition...")
    all_keys = jax.random.split(key, 5000)

    # Generate data (this will be slow - consider pre-generating)
    z_init_list, z_target_list = [], []
    for i, k in enumerate(all_keys):
        if i % 500 == 0:
            print(f"  Generated {i}/5000...")
        k1, k2 = jax.random.split(k)
        _, _, z_i = generate_grf_2d(k1, n_points=n_grid, length_scale=0.25)
        _, _, z_t = generate_grf_2d(k2, n_points=n_grid, length_scale=0.4)
        z_init_list.append(z_i)
        z_target_list.append(z_t)

    z_init_all = jnp.stack(z_init_list)
    z_target_all = jnp.stack(z_target_list)

    print("Dataset generation complete!")

    # Initialize agents in grid pattern
    n_side = int(jnp.sqrt(n_agents))
    spacing = 0.8 / (n_side + 1)
    xi_template = []
    for i in range(n_side):
        for j in range(n_side):
            if len(xi_template) < n_agents:
                xi_template.append([0.1 + spacing * (i+1),
                                   0.1 + spacing * (j+1)])
    xi_init_single = jnp.array(xi_template)
    xi_init_batch = jnp.tile(xi_init_single, (batch_size, 1, 1))

    metrics = []
    start_time = time.time()

    print("\nStarting training...")
    for epoch in trange(epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, 5000)
        z_init_b, z_target_b = z_init_all[idx], z_target_all[idx]

        params, opt_state, loss, aux = train_step(
            params, opt_state, z_init_b, xi_init_batch, z_target_b, dynamics
        )

        if epoch % 10 == 0:
            metrics.append((epoch, loss, *aux))
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Track: {aux[0]:.4f} | " +
                  f"Effort: {aux[1]:.4f} | Coll: {aux[2]:.4f} | Bound: {aux[3]:.4f}")

    print(f"\nTraining finished in {time.time() - start_time:.2f}s.")

    # Save parameters
    with open('centralized_params_heat2d.msgpack', 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print("Parameters saved to centralized_params_heat2d.msgpack")

    # Plot metrics
    if metrics:
        metrics = jnp.array(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(metrics[:, 0], metrics[:, 1])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_yscale('log')

        axes[0, 1].plot(metrics[:, 0], metrics[:, 2])
        axes[0, 1].set_title('Tracking Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_yscale('log')

        axes[1, 0].plot(metrics[:, 0], metrics[:, 3])
        axes[1, 0].set_title('Effort Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_yscale('log')

        axes[1, 1].plot(metrics[:, 0], metrics[:, 4])
        axes[1, 1].set_title('Collision Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig('training_metrics_heat2d.png')
        print("Training metrics saved to training_metrics_heat2d.png")
