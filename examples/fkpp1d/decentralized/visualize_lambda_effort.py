import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import flax.serialization
import optax
from pathlib import Path
from functools import partial
from tqdm import trange
from tesseract_core import Tesseract
# NEW IMPORT for formatting
from matplotlib.ticker import ScalarFormatter

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
import data_utils

# --- 1. Setup Directories ---
MODELS_DIR = Path("models/analysis")
FIGURES_DIR = Path("figures/conjecture")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Training Logic ---
def loss_fn(params, z_init, xi_init, z_target, dynamics, l_effort_weight, n_agents, T_steps, R_safe=0.05):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    # Tracking Error [cite: 29]
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    # Control Effort Penalty [cite: 31, 125]
    l_effort = jnp.mean(u_traj ** 2) 
    
    # Boundary and Collision Penalties [cite: 34, 43]
    margin = 0.02
    l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 + 
                       jnp.maximum(0, xi_traj - (1.0 - margin))**2)
    dists = jnp.abs(xi_traj[:, :, None] - xi_traj[:, None, :])
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 1.0)) ** 2)
    l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)

    return 5.0 * l_track + l_effort_weight * l_effort + 100.0 * l_bound + 1.0 * l_coll + 0.1 * l_accel, l_track

@partial(jax.jit, static_argnames=('dynamics', 'l_effort_weight', 'n_agents', 'T_steps', 'optimizer'))
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics, l_effort_weight, n_agents, T_steps, optimizer):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None, None, None, None))
    def mean_loss_fn(p):
        losses, track_losses = batched_loss_fn(p, z_init_batch, xi_init_batch, z_target_batch, dynamics, l_effort_weight, n_agents, T_steps)
        return jnp.mean(losses), jnp.mean(track_losses)
    (loss, track_l), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, track_l

def train_model(l_weight, n_pde, n_agents, epochs, dynamics, model, optimizer):
    param_path = MODELS_DIR / f"params_lambda_{l_weight}.msgpack"
    if param_path.exists():
        print(f"Skipping training: Model for lambda={l_weight} already exists.")
        return
    
    print(f"Training model for lambda_effort = {l_weight}...")
    key = jax.random.PRNGKey(int(l_weight * 1000) if l_weight > 0 else 42)
    params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    opt_state = optimizer.init(params)
    
    all_keys = jax.random.split(key, 5000)
    _, z_init_all = jax.vmap(partial(data_utils.generate_grf, n_points=n_pde, length_scale=0.2))(all_keys)
    _, z_target_all = jax.vmap(partial(data_utils.generate_grf, n_points=n_pde, length_scale=0.4))(all_keys)
    xi_init_batch = jnp.tile(jnp.linspace(0.2, 0.8, n_agents), (32, 1))

    for epoch in trange(epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (32,), 0, 5000)
        params, opt_state, _, _ = train_step(params, opt_state, z_init_all[idx], xi_init_batch, z_target_all[idx], dynamics, l_weight, n_agents, 300, optimizer)

    with open(param_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))

# --- 3. Evaluation with Temporal Windowing ---
def run_comparison(solver_ts, n_agents_list, lambda_list, n_pde, T_steps, z_init, z_target, window_ratio=0.7):
    all_results = []
    start_idx = int(T_steps * (1.0 - window_ratio))
    print(f"Analyzing effort from step {start_idx} to {T_steps} (Window: {window_ratio*100:.0f}%)")

    for l_weight in lambda_list:
        param_path = MODELS_DIR / f"params_lambda_{l_weight}.msgpack"
        model = DecentralizedControlNet(features=(64, 64))
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=True)
        
        with open(param_path, 'rb') as f:
            bytes_data = f.read()
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((20,)))
        params = flax.serialization.from_bytes(params, bytes_data)

        for n in n_agents_list:
            print(f"Testing Lambda={l_weight}, N={n}...")
            xi_init = jnp.linspace(0.2, 0.8, n)
            z_traj, _, u_traj, _ = dynamics.unroll_controlled(z_init, xi_init, z_target, params, T_steps)
            
            mse = float(jnp.mean((z_traj[-1] - z_target)**2))
            u_window = u_traj[start_idx:] 
            
            # Total squared effort in window [cite: 120]
            window_steps = T_steps - start_idx
            total_effort_sq = float(jnp.sum(u_window**2) / window_steps) 
            
            all_results.append({
                "lambda": l_weight, 
                "n_agents": n, 
                "mse": mse, 
                "total_effort_sq": total_effort_sq,
                "window": f"Last {window_ratio*100:.0f}%"
            })
    return pd.DataFrame(all_results)

# --- 4. Plotting ---
def plot_conjecture_results(df, window_label):
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22']
    for i, l in enumerate(df['lambda'].unique()):
        sub = df[df['lambda'] == l]
        ax1.semilogy(sub['n_agents'], sub['mse'], marker='o', markersize=6, 
                     label=f'$\lambda_u={l}$', color=colors[i], linewidth=2)
        ax2.loglog(sub['n_agents'], sub['total_effort_sq'], marker='s', markersize=6,
                   label=f'$\lambda_u={l}$', color=colors[i], linewidth=2)

    # Subplot 1: MSE
    ax1.set_title("Zero-Shot Scalability: Tracking MSE", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Number of Agents ($N$)", fontsize=11)
    ax1.set_ylabel("Final $L^2$ Error", fontsize=11)
    ax1.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Training $N$')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Subplot 2: Effort Decay
    ax2.set_title(f"Steady-State Effort ({window_label}): $\sum u_i^2 = O(1/N)$", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Number of Agents ($N$)", fontsize=11)
    ax2.set_ylabel("Mean Window Effort ($\sum u_i^2$)", fontsize=11)
    
    # --- Y-AXIS FORMATTING: Show 10, 20, etc. instead of scientific notation ---
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.yaxis.get_major_formatter().set_scientific(False)
    ax2.yaxis.get_major_formatter().set_useOffset(False)
    # Ensure minor ticks also use plain formatting if they appear
    ax2.yaxis.set_minor_formatter(ScalarFormatter())
    ax2.yaxis.get_minor_formatter().set_scientific(False)

    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"conjecture_scaling_{window_label.replace(' ', '_')}.pdf", dpi=300)
    print(f"Plots saved to {FIGURES_DIR}")

def main():
    n_pde, T_steps = 100, 300
    lambda_list = [1e-2, 1e-1, 0.5, 1]
    n_agents_list = [15, 20, 30, 40, 50, 60]
    WINDOW_RATIO = 0.7 

    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    model = DecentralizedControlNet(features=(64, 64))
    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))

    with solver_ts:
        dynamics_local = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)
        for l in lambda_list:
            train_model(l, n_pde, 20, 500, dynamics_local, model, optimizer)

        key = jax.random.PRNGKey(42)
        _, z_init = data_utils.generate_grf(key, n_points=n_pde, length_scale=0.2)
        _, z_target = data_utils.generate_grf(jax.random.PRNGKey(43), n_points=n_pde, length_scale=0.4)
        
        results_df = run_comparison(solver_ts, n_agents_list, lambda_list, n_pde, T_steps, 
                                    z_init, z_target, window_ratio=WINDOW_RATIO)
        
        plot_conjecture_results(results_df, f"Last {int(WINDOW_RATIO*100)}%")
        results_df.to_csv(FIGURES_DIR / "conjecture_data_windowed.csv", index=False)

if __name__ == "__main__":
    main()