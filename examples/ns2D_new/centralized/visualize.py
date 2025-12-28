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
from data_utils import generate_shape_pair, make_actuator_grid
from models.policy import NS2DControlNet

# --- UTILITIES ---
def load_params(model, filepath, n, m_agents, L):
    with open(filepath, "rb") as f:
        serialized = f.read()
    key = jax.random.PRNGKey(0)
    # The model internally handles the gradient stacking now
    dummy_z = jnp.zeros((n, n), dtype=jnp.float64)
    dummy_target = jnp.zeros((n, n), dtype=jnp.float64)
    dummy_xi = make_actuator_grid(m_agents, L).astype(jnp.float64)
    init_params = model.init(key, dummy_z, dummy_target, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized)

def rollout_scene(params, rho_init, rho_target, xi_init, dynamics, t_steps):
    """Uses the unroll_controlled method for a single sequence."""
    key_omega = jax.random.PRNGKey(42)
    n = rho_init.shape[0]
    omega_init = sample_initial_vorticity(key_omega, n, V_SCALE_BASE=0.1, V_FALLOFF=0.4)

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
    return res[1], res[2], res[3], res[4]

# --- MAIN ---
def main():
    n = 64
    L = jnp.pi
    m_agents = 64
    t_steps = 200
    n_scenes = 2
    
    # Initialize Tesseract or Local Solver
    solver_ts = Tesseract.from_image("solver_ns_shape") 
    model = NS2DControlNet(features=(20, 50))
    
    # Dynamics setup
    dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)

    try:
        params = load_params(model, "centralized_params_ns2d.msgpack", n, m_agents, L)
        print("Loaded trained parameters.")
    except FileNotFoundError:
        print("Parameters not found, using random init.")
        params = model.init(jax.random.PRNGKey(1), jnp.zeros((n,n)), jnp.zeros((n,n)), make_actuator_grid(m_agents, L))

    # Data generation - Splitting keys ensures different I.C.s for each scene
    key = jax.random.PRNGKey(0)
    _, key_data = jax.random.split(key)
    data_keys = jax.random.split(key_data, n_scenes)
    
    time_indices = [0, t_steps // 2, t_steps - 1]
    n_cols = 5  # Init, Target, t=0, t=mid, t=final
    
    # Create the stacked plot (n_scenes rows)
    fig, axes = plt.subplots(n_scenes, n_cols, figsize=(3.5 * n_cols, 3.5 * n_scenes))
    
    # --- MODIFIED: Ensure axes is always 2D for consistent indexing ---
    if n_scenes == 1:
        axes = jnp.expand_dims(axes, axis=0)

    scene_results = []
    all_rho_values = []

    # 1. Collect results
    for i in range(n_scenes):
        # Unique key per iteration = unique shape pair
        rho_init, rho_target = generate_shape_pair(data_keys[i], n=n, L=L)
        xi_init = make_actuator_grid(m_agents, L)

        rho_traj, xi_traj, _, _ = rollout_scene(
            params, rho_init, rho_target, xi_init, dynamics, t_steps
        )
        
        scene_results.append({
            'init': rho_init, 'target': rho_target, 'traj': rho_traj
        })
        
        # Aggregate for global color scale
        all_rho_values.extend([rho_init, rho_target])
        for t in time_indices:
            all_rho_values.append(rho_traj[t])

    # 2. Compute Global Vmin/Vmax
    vmin = 0.0
    vmax = float(jnp.max(jnp.array(all_rho_values)))
    vmax = max(vmax, 1.0)

    # 3. Plotting
    for i in range(n_scenes):
        data = scene_results[i]
        row_axes = axes[i] # Now safe because of the expansion above
        
        imshow_kwargs = dict(origin='lower', extent=[0, L, 0, L], cmap='viridis', vmin=vmin, vmax=vmax)

        # Plot Init/Target
        row_axes[0].imshow(data['init'], **imshow_kwargs)
        row_axes[0].set_title(f"Scene {i+1} - Initial")
        row_axes[1].imshow(data['target'], **imshow_kwargs)
        row_axes[1].set_title("Target Shape")

        # Plot Snapshots
        for j, t_idx in enumerate(time_indices):
            ax = row_axes[2 + j]
            rho_t = data['traj'][t_idx]
            im = ax.imshow(rho_t, **imshow_kwargs)
            ax.set_title(f"t = {t_idx}")
            
            # Density Stats
            rho_sum = float(jnp.sum(rho_t))
            ax.text(0.05, 0.95, f"sum={rho_sum:.2f}", transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # Adjust layout and colorbar
    fig.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Density')

    plt.savefig("ns2d_centralized_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved: ns2d_centralized_visualization.png with {n_scenes} scenes.")

if __name__ == "__main__":
    main()