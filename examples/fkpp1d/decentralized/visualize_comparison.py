import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import sys
import flax.serialization
from pathlib import Path
from tesseract_core import Tesseract

# Import decentralized components
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
import data_utils

def load_params(model, filepath, n_pde=100, n_agents=8):
    """Loads the weights from a msgpack file."""
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    return flax.serialization.from_bytes(init_params, serialized_bytes)

# def run_experiment(solver_ts, n_agents_list, params_file, experiment_name, output_dir, z_init, z_target, n_pde, T_steps):
#     """Runs a single simulation suite and saves/plots results."""
#     print(f"\n--- Starting Experiment: {experiment_name} ---")
#     results = {"n_agents": [], "mse": [], "u_effort": []}

#     for n_agents in n_agents_list:
#         print(f"Simulating Case: N={n_agents}...")
#         model = DecentralizedControlNet(features=(64, 64))
#         dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=True)
        
#         try:
#             params = load_params(model, params_file, n_pde, n_agents)
#         except Exception as e:
#             print(f"Skipping N={n_agents}: {e}")
#             continue

#         xi_init = jnp.linspace(0.2, 0.8, n_agents)
#         z_traj, _, u_traj, _ = dynamics.unroll_controlled(
#             z_init, xi_init, z_target, params, T_steps
#         )

#         # Compute metrics
#         final_mse = float(jnp.mean((z_traj[-1] - z_target)**2))
#         forcing_effort = float(jnp.sum(u_traj)) 

#         results["n_agents"].append(int(n_agents))
#         results["mse"].append(final_mse)
#         results["u_effort"].append(forcing_effort)

#     # --- Save Data ---
#     df = pd.DataFrame(results)
#     csv_path = output_dir / f"metrics_{experiment_name}.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"Data saved to {csv_path}")

#     # --- Plotting ---
#     fig, (ax_mse, ax_u) = plt.subplots(1, 2, figsize=(14, 6))
#     n_vals = np.array(results["n_agents"])
#     mse_vals = np.array(results["mse"])
#     effort_vals = np.array(results["u_effort"])

#     # MSE Plot
#     ax_mse.bar(n_vals, mse_vals, color='skyblue', edgecolor='navy', width=0.6 if len(n_vals) < 15 else 1.2)
#     ax_mse.set_title(f"Final MSE ({experiment_name})", fontsize=14)
#     ax_mse.set_xlabel("Number of Actuators ($n$)", fontsize=12)
#     ax_mse.set_ylabel("Log MSE", fontsize=12)
#     ax_mse.set_yscale('log')
#     ax_mse.set_xticks(n_vals)
#     ax_mse.tick_params(axis='x', rotation=45)

#     # Effort Plot
#     if len(n_vals) > 2:
#         n_smooth = np.linspace(n_vals.min(), n_vals.max(), 300)
#         spline = make_interp_spline(n_vals, effort_vals, k=2)
#         effort_smooth = spline(n_smooth)
#         ax_u.plot(n_smooth, effort_smooth, 'orange', alpha=0.6, linestyle='--')
#         ax_u.scatter(n_vals, effort_vals, color='darkorange', s=80, edgecolors='black', zorder=5)
#     else:
#         ax_u.plot(n_vals, effort_vals, 'o-', color='orange')

#     ax_u.set_title(f"Control Effort ({experiment_name})", fontsize=14)
#     ax_u.set_xlabel("Number of Actuators ($n$)", fontsize=12)
#     ax_u.set_ylabel("Energy", fontsize=12)
#     ax_u.set_xticks(n_vals)
#     ax_u.tick_params(axis='x', rotation=45)
#     ax_u.grid(True, linestyle=':', alpha=0.6)

#     plt.tight_layout()
#     plot_path = output_dir / f"plot_{experiment_name}.png"
#     plt.savefig(plot_path)
#     plt.close() # Close figure to free memory
#     print(f"Plot saved to {plot_path}")

def run_experiment(solver_ts, n_agents_list, params_file, experiment_name, output_dir, z_init, z_target, n_pde, T_steps):
    print(f"\n--- Starting Experiment: {experiment_name} ---")
    results = {"n_agents": [], "mse": [], "u_effort": []}

    for n_agents in n_agents_list:
        print(f"Simulating Case: N={n_agents}...")
        model = DecentralizedControlNet(features=(64, 64))
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=True)
        
        try:
            params = load_params(model, params_file, n_pde, n_agents)
        except Exception as e:
            print(f"Skipping N={n_agents}: {e}")
            continue

        xi_init = jnp.linspace(0.2, 0.8, n_agents)
        z_traj, _, u_traj, _ = dynamics.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )

        final_mse = float(jnp.mean((z_traj[-1] - z_target)**2))
        # Total forcing effort as sum of intensities
        forcing_effort = float(jnp.sum(u_traj)) 

        results["n_agents"].append(int(n_agents))
        results["mse"].append(final_mse)
        results["u_effort"].append(forcing_effort)

    # --- Data Processing ---
    df = pd.DataFrame(results)
    n_vals = df["n_agents"].values
    mse_vals = df["mse"].values
    effort_vals = df["u_effort"].values

    # --- Dual-Axis Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis: MSE (Log Scale)
    color_mse = 'tab:blue'
    ax1.set_xlabel('Number of Actuators ($n$)', fontsize=12)
    ax1.set_ylabel('Final Tracking MSE', color=color_mse, fontsize=12)
    ax1.semilogy(n_vals, mse_vals, marker='o', linestyle='-', color=color_mse, 
                 linewidth=2, label='Tracking MSE')
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.grid(True, which="both", ls="-", alpha=0.15)

    # Secondary Axis: Control Effort (Linear Scale)
    ax2 = ax1.twinx()  
    color_effort = 'tab:orange'
    ax2.set_ylabel('Total Control Effort ($\sum u_i$)', color=color_effort, fontsize=12)
    ax2.plot(n_vals, effort_vals, marker='s', linestyle='--', color=color_effort, 
             alpha=0.8, linewidth=2, label='Control Effort')
    ax2.tick_params(axis='y', labelcolor=color_effort)

    # Highlight Training Point (N=20)
    training_n = 20
    if training_n in n_vals:
        ax1.axvline(x=training_n, color='red', linestyle=':', alpha=0.5)
        plt.text(training_n + 1, ax2.get_ylim()[1]*0.9, 'Training Size ($N=20$)', 
                 color='red', fontweight='bold')

    plt.title(f"Zero-Shot Scalability Analysis", fontsize=14, pad=15)
    
    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plot_path = output_dir / f"scalability_dual_axis_{experiment_name}.pdf" # PDF for papers
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"plot saved to {plot_path}")

def main():
    # --- Configuration ---
    n_pde = 100
    T_steps = 300
    jax.config.update("jax_platform_name", "cpu")
    output_dir = Path("figures/fkpp_decentralized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Shared Initial Conditions ---
    key = jax.random.PRNGKey(42)
    key, sub1, sub2 = jax.random.split(key, 3)
    _, z_init = data_utils.generate_grf(sub1, n_points=n_pde, length_scale=0.2)
    _, z_target = data_utils.generate_grf(sub2, n_points=n_pde, length_scale=0.4)

    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    
    with solver_ts:
        # --- Experiment 1: Standard ---
        run_experiment(
            solver_ts, 
            n_agents_list=list(np.arange(10, 65, 5)), 
            params_file='decentralized_params_larger.msgpack',
            experiment_name="standard",
            output_dir=output_dir,
            z_init=z_init,
            z_target=z_target,
            n_pde=n_pde,
            T_steps=T_steps
        )

        # # --- Experiment 2: Larger Parameters ---
        # run_experiment(
        #     solver_ts, 
        #     n_agents_list=list(np.arange(10, 21, 1)), 
        #     params_file='decentralized_params_larger.msgpack',
        #     experiment_name="larger_params",
        #     output_dir=output_dir,
        #     z_init=z_init,
        #     z_target=z_target,
        #     n_pde=n_pde,
        #     T_steps=T_steps
        # )

if __name__ == "__main__":
    main()