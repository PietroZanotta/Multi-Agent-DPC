import sys
import time
from pathlib import Path
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
from flax import linen as nn
from tesseract_core import Tesseract

script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

from dynamics import PDEDynamics, sample_initial_vorticity
from data_utils import generate_shape_pair_v2, make_actuator_grid

# --- MODEL DEFINITION ---
class NS2DControlNet(nn.Module):
    features: Sequence[int]
    u_max: float = 10.0
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

# --- UTILITIES ---
def load_params(model, filepath, n, m_agents, L):
    with open(filepath, "rb") as f:
        serialized = f.read()
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n, n), dtype=jnp.float64)
    dummy_target = jnp.zeros((n, n), dtype=jnp.float64)
    dummy_xi = make_actuator_grid(m_agents, L).astype(jnp.float64)
    init_params = model.init(key, dummy_z, dummy_target, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized)

def rollout_scene(params, rho_init, rho_target, xi_init, dynamics, t_steps):
    """Uses the unroll_controlled method for a single sequence."""
    # Match the key split used in training
    key = jax.random.PRNGKey(0)
    key_omega, _ = jax.random.split(key)
    n = rho_init.shape[0]
    omega_init = sample_initial_vorticity(key_omega, n, V_SCALE_BASE=0.8, V_FALLOFF=0.7)

    # All inputs to float64 for the solver
    res = dynamics.unroll_controlled(
        omega_init.astype(jnp.float64), 
        rho_init.astype(jnp.float64), 
        rho_target.astype(jnp.float64), 
        xi_init.astype(jnp.float64), 
        params, 
        t_steps
    )
    # returns rho_traj, xi_traj, u_traj, v_traj (skipping omega_traj)
    return res[1], res[2], res[3], res[4], omega_init


def rollout_uncontrolled(rho_init, rho_target, xi_init, dynamics, t_steps, omega_init):
    """
    Rollout with ZERO control (u=0, v=0) to show uncontrolled dynamics.
    Uses the same omega_init as controlled for fair comparison.
    """
    from tesseracts.solverNS_shape_centralized import solver as ns_solver
    
    n = rho_init.shape[0]
    m_agents = xi_init.shape[0]
    
    # Convert omega to spectral space
    omega_hat = jnp.fft.rfft2(omega_init.astype(jnp.float64))
    rho = rho_init.astype(jnp.float64)
    xi = xi_init.astype(jnp.float64)
    
    # Zero control
    u_zero = jnp.zeros((m_agents, 2), dtype=jnp.float64)
    v_zero = jnp.zeros((m_agents, 2), dtype=jnp.float64)
    
    # Step function for uncontrolled rollout
    def step_fn(carry, _):
        omega_hat, rho, xi = carry
        omega_hat_new, rho_new, xi_new = ns_solver.full_step(omega_hat, rho, xi, u_zero, v_zero)
        return (omega_hat_new, rho_new, xi_new), rho_new
    
    # Run uncontrolled simulation
    _, rho_traj = jax.lax.scan(step_fn, (omega_hat, rho, xi), None, length=t_steps)
    
    return rho_traj


# --- MAIN ---
def main():
    n = 64
    L = jnp.pi
    m_agents = 64
    t_steps = 200
    n_scenes = 3 # Set as desired
    
    # Initialize Tesseract or Local Solver
    solver_ts = Tesseract.from_image("solver_ns_shape_centralized") # or use None
    model = NS2DControlNet(features=(16, 32))
    
    # Dynamics setup
    dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)

    try:
        params = load_params(model, "centralized_params_ns2d.msgpack", n, m_agents, L)
        print("Loaded trained parameters.")
    except FileNotFoundError:
        print("Parameters not found, using random init.")
        params = model.init(jax.random.PRNGKey(1), jnp.zeros((n,n)), jnp.zeros((n,n)), make_actuator_grid(m_agents, L))

    # Data generation
    key = jax.random.PRNGKey(0)
    _, key_data = jax.random.split(key)
    data_keys = jax.random.split(key_data, n_scenes)
    
    time_indices = [0, t_steps // 2, t_steps - 1]
    n_cols = 5  # Init, Target, t=0, t=mid, t=final
    n_rows = 2 * n_scenes  # 2 rows per scene: uncontrolled (top) + controlled (bottom)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

    scene_results = []
    all_rho_values = []

    # 1. Collect results (both controlled and uncontrolled)
    for i in range(n_scenes):
        rho_init, rho_target = generate_shape_pair_v2(data_keys[i], n=n, L=L)
        xi_init = make_actuator_grid(m_agents, L)

        # Controlled rollout
        rho_traj_ctrl, xi_traj, _, _, omega_init = rollout_scene(
            params, rho_init, rho_target, xi_init, dynamics, t_steps
        )
        
        # Uncontrolled rollout (same initial conditions, zero control)
        rho_traj_unctrl = rollout_uncontrolled(
            rho_init, rho_target, xi_init, dynamics, t_steps, omega_init
        )
        
        scene_results.append({
            'init': rho_init, 
            'target': rho_target, 
            'traj_controlled': rho_traj_ctrl,
            'traj_uncontrolled': rho_traj_unctrl
        })
        
        # Aggregate for global color scale
        all_rho_values.extend([rho_init, rho_target])
        for t in time_indices:
            all_rho_values.append(rho_traj_ctrl[t])
            all_rho_values.append(rho_traj_unctrl[t])

    # 2. Compute Global Vmin/Vmax
    vmin = 0.0
    vmax = float(jnp.max(jnp.array(all_rho_values)))
    vmax = max(vmax, 1.0)

    # 3. Plotting with labels and stats
    for i in range(n_scenes):
        data = scene_results[i]
        
        # Row indices: uncontrolled on top, controlled on bottom
        row_unctrl = 2 * i
        row_ctrl = 2 * i + 1
        
        if n_rows == 2:
            axes_unctrl = axes[0]
            axes_ctrl = axes[1]
        else:
            axes_unctrl = axes[row_unctrl]
            axes_ctrl = axes[row_ctrl]
        
        imshow_kwargs = dict(origin='lower', extent=[0, L, 0, L], cmap='viridis', vmin=vmin, vmax=vmax)

        # ===== UNCONTROLLED ROW (Top) =====
        axes_unctrl[0].imshow(data['init'], **imshow_kwargs)
        axes_unctrl[0].set_title(f"Scene {i+1} - Initial")
        axes_unctrl[0].set_ylabel("Uncontrolled", fontsize=12, fontweight='bold')
        
        axes_unctrl[1].imshow(data['target'], **imshow_kwargs)
        axes_unctrl[1].set_title("Target Shape")

        for j, t_idx in enumerate(time_indices):
            ax = axes_unctrl[2 + j]
            rho_t = data['traj_uncontrolled'][t_idx]
            im = ax.imshow(rho_t, **imshow_kwargs)
            ax.set_title(f"t = {t_idx}")
            
            rho_sum = float(jnp.sum(rho_t))
            ax.text(0.05, 0.95, f"sum={rho_sum:.2f}", transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # ===== CONTROLLED ROW (Bottom) =====
        axes_ctrl[0].imshow(data['init'], **imshow_kwargs)
        axes_ctrl[0].set_title(f"Scene {i+1} - Initial")
        axes_ctrl[0].set_ylabel("Controlled", fontsize=12, fontweight='bold')
        
        axes_ctrl[1].imshow(data['target'], **imshow_kwargs)
        axes_ctrl[1].set_title("Target Shape")

        for j, t_idx in enumerate(time_indices):
            ax = axes_ctrl[2 + j]
            rho_t = data['traj_controlled'][t_idx]
            im = ax.imshow(rho_t, **imshow_kwargs)
            ax.set_title(f"t = {t_idx}")
            
            rho_sum = float(jnp.sum(rho_t))
            ax.text(0.05, 0.95, f"sum={rho_sum:.2f}", transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # Colorbar
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Density')

    plt.savefig("ns2d_centralized_visualization.png", dpi=150, bbox_inches='tight')
    print("Saved: ns2d_centralized_visualization.png")

if __name__ == "__main__":
    main()