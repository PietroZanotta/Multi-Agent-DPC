import sys
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

# --- Path Configuration ---
SOLVER_PATH = "/Users/dibakarroysarkar/Desktop/PhDThesis/Codes/tesseract_hackathon/group_git/Tesseract-Hackathon/tesseracts/solverNS_shape"
sys.path.append(SOLVER_PATH)

# --- Import Domain Physics ---
from solver import (
    multi_scale_velocity_field,  # Needed for Vorticity
    multi_scale_density_field,
    uniform_actuator_positions,
    V_SCALE_BASE, V_FALLOFF, BUOYANCY_STRENGTH,
    fixed_dt, N, L
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def generate_hybrid_init(key, grid_n, domain_l, num_blobs=15):
    """
    Hybrid Initialization:
    - Vorticity (Omega): Multi-scale turbulent noise.
    - Density (Rho): Gaussian blobs (clouds).
    """
    key_vel, key_scale, key_blobs = jax.random.split(key, 3)
    
    # ---------------------------------------------------------
    # PART A: Multi-Scale Vorticity Generation
    # ---------------------------------------------------------
    
    # 1. Generate multi-scale velocity scaling
    v_scale = V_SCALE_BASE * (1.0 + (jax.random.uniform(key_scale) - 0.5) * 2.0)
    
    # 2. Generate velocity field (using physics constants from solver)
    velocity = multi_scale_velocity_field(key_vel, grid_n, v_scale, V_FALLOFF)
    vx = velocity[:, :, 0]
    vy = velocity[:, :, 1]
    
    # 3. Convert Velocity to Vorticity (Omega) via FFT
    vx_hat = jnp.fft.rfft2(vx)
    vy_hat = jnp.fft.rfft2(vy)
    
    kx = jnp.fft.fftfreq(grid_n, d=domain_l/grid_n)
    ky = jnp.fft.rfftfreq(grid_n, d=domain_l/grid_n)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    
    two_pi_i = 2.0 * jnp.pi * 1j
    omega_hat = two_pi_i * (KX * vy_hat - KY * vx_hat)
    
    # Return to real space
    omega = jnp.fft.irfft2(omega_hat)

    # ---------------------------------------------------------
    # PART B: Gaussian Blob Density Generation
    # ---------------------------------------------------------
    
    key_pos, key_sigma, key_amp = jax.random.split(key_blobs, 3)
    
    # Grid setup
    x = jnp.linspace(0, domain_l, grid_n)
    y = jnp.linspace(0, domain_l, grid_n)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # Randomize Blob Parameters
    centers = jax.random.uniform(key_pos, (num_blobs, 2), minval=0.1*domain_l, maxval=0.9*domain_l)
    # sigmas = jax.random.uniform(key_sigma, (num_blobs,), minval=domain_l/20.0, maxval=domain_l/10.0)
    sigmas = jnp.full(num_blobs, 0.04)
    amplitudes = jax.random.uniform(key_amp, (num_blobs,), minval=0.5, maxval=1.0)

    # Vectorized Gaussian calculation
    def gaussian(cx, cy, sig, amp):
        return amp * jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sig**2))

    blobs = jax.vmap(gaussian)(centers[:, 0], centers[:, 1], sigmas, amplitudes)
    rho = jnp.sum(blobs, axis=0)
    rho = jnp.clip(rho, 0.0, 1.0)
    
    return omega, rho

def main():
    # 1. Setup Solver
    solver_ts = Tesseract.from_image("solver_ns_shape")

    with solver_ts:
        # Simulation Parameters
        dt = fixed_dt
        T_sim = 64
        M = 4 
        
        key_init = jax.random.PRNGKey(101)

        # 2. Generate Initial State (Hybrid)
        print(f"Initializing Hybrid Fields (N={N}, L={L:.2f})...")
        print(" -> Vorticity: Multiscale Noise")
        print(" -> Density:   Gaussian Blobs")
        
        # --- CALL HYBRID FUNCTION ---
        omega_init, rho_init = generate_hybrid_init(key_init, grid_n=N, domain_l=L)

        # 3. Actuator Placement
        xi_init = uniform_actuator_positions(M)
        
        # 4. Control Inputs
        u_seq = jnp.zeros((T_sim, M, 2))
        v_seq = jnp.zeros((T_sim, M, 2))

        # Ensure correct precision
        omega_init = omega_init.astype(jnp.float64)
        rho_init = rho_init.astype(jnp.float64)
        xi_init = xi_init.astype(jnp.float64)
        u_seq = u_seq.astype(jnp.float64)
        v_seq = v_seq.astype(jnp.float64)

        inputs = {
            "omega_init": omega_init,
            "rho_init": rho_init,
            "xi_init": xi_init,
            "u_seq": u_seq,
            "v_seq": v_seq,
        }

        # 5. Run Simulation
        print("="*70)
        print(f"Hybrid Simulation Run")
        print("="*70)
        
        results = apply_tesseract(solver_ts, inputs)
        
        rho_traj = results["rho_trajectory"]
        omega_traj = results["omega_trajectory"]
        xi_traj = results["xi_trajectory"]

        # Diagnostics
        print(f"\n✓ Simulation completed!")
        print(f"Trajectory Shape: {rho_traj.shape}")

        if jnp.isnan(rho_traj).any():
            print("❌ WARNING: NaNs detected in output!")
        else:
            print("✓ No NaNs - simulation is stable!")

        # 6. Visualization
        print("\nPlotting results...")
        num_plots = 6
        timesteps = jnp.linspace(0, T_sim - 1, num_plots, dtype=int)
        
        fig, axes = plt.subplots(1, num_plots, figsize=(18, 3.5), constrained_layout=True)
        fig.suptitle('Smoke Evolution (Hybrid: Multiscale Omega + Blob Rho)', fontsize=16)

        vmin = 0.0
        vmax = 1.0

        for i, t_idx in enumerate(timesteps):
            ax = axes[i]
            # Plot Density
            im = ax.imshow(rho_traj[t_idx].T, cmap="viridis", origin="lower", 
                           extent=[0, L, 0, L], vmin=vmin, vmax=vmax)
            
            # Plot Actuators
            current_actuators = xi_traj[t_idx]
            ax.scatter(current_actuators[:, 0], current_actuators[:, 1], 
                       c="cyan", marker="x", s=40, linewidths=1.5, alpha=0.7)

            ax.set_title(f"t = {int(t_idx) * dt:.2f}")
            ax.set_xlabel("x")
            if i > 0: ax.set_yticks([])

        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(r'$\rho$')

        output_file = "ns_hybrid_evolution.png"
        plt.savefig(output_file, dpi=150)
        print(f"✓ Visualization saved to {output_file}")

if __name__ == "__main__":
    main()