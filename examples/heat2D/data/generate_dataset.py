"""
Offline dataset generator for 2D Heat Equation Control.

Generates and saves GRF datasets for reuse, avoiding repeated expensive eigendecompositions.
Supports both 32×32 and 64×64 grid resolutions.

Usage:
    python generate_dataset.py --grid-size 32 --samples 5000
    python generate_dataset.py --grid-size 64 --samples 1000
    
NOTE: This might take a while to run, especially for 64×64 grids.
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from centralized.data_utils import generate_grf_2d


def main():
    parser = argparse.ArgumentParser(description='Generate 2D Heat Equation Dataset')
    parser.add_argument('--grid-size', type=int, default=32, choices=[32, 64],
                        help='Grid resolution (32 or 64)')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    n_grid = args.grid_size
    n_samples = args.samples
    output_file = f'heat2d_dataset_{n_grid}x{n_grid}.npz'

    print("=" * 60)
    print("2D Heat Dataset Generator")
    print("=" * 60)
    print(f"Grid size: {n_grid}×{n_grid}")
    print(f"Samples: {n_samples}")
    print(f"Output: {output_file}")
    print(f"Seed: {args.seed}")
    print()

    # Estimate time
    if n_grid == 32:
        est_time_per_sample = 1.5  # seconds
    else:  # 64
        est_time_per_sample = 15.0  # seconds (rough estimate)

    est_total_min = (est_time_per_sample * n_samples * 2) / 60  # *2 for init + target
    print(f"Estimated time: ~{est_total_min:.1f} minutes")
    print()

    key = jax.random.PRNGKey(args.seed)
    all_keys = jax.random.split(key, n_samples * 2)

    z_init_list = []
    z_target_list = []

    start_time = time.time()
    last_print_time = start_time

    print("Generating initial conditions...")
    for i in range(n_samples):
        # Progress reporting
        current_time = time.time()
        if current_time - last_print_time > 5.0 or i % 50 == 0:  # Every 5 seconds or 50 samples
            elapsed = current_time - start_time
            if i > 0:
                time_per_sample = elapsed / i
                remaining = (n_samples - i) * time_per_sample
                print(f"  Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")
            else:
                print(f"  Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%)")
            last_print_time = current_time

        _, _, z_i = generate_grf_2d(all_keys[i], n_points=n_grid, length_scale=0.25)
        z_init_list.append(z_i)

    print(f"  Completed: {n_samples}/{n_samples} (100.0%)")
    print()

    init_time = time.time() - start_time
    print(f"Initial conditions generated in {init_time/60:.1f} minutes")
    print()

    print("Generating target conditions...")
    target_start = time.time()
    last_print_time = target_start

    for i in range(n_samples):
        # Progress reporting
        current_time = time.time()
        if current_time - last_print_time > 5.0 or i % 50 == 0:
            elapsed = current_time - target_start
            if i > 0:
                time_per_sample = elapsed / i
                remaining = (n_samples - i) * time_per_sample
                print(f"  Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")
            else:
                print(f"  Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%)")
            last_print_time = current_time

        _, _, z_t = generate_grf_2d(all_keys[n_samples + i], n_points=n_grid,
                                     length_scale=0.4)
        z_target_list.append(z_t)

    print(f"  Completed: {n_samples}/{n_samples} (100.0%)")
    print()

    target_time = time.time() - target_start
    print(f"Target conditions generated in {target_time/60:.1f} minutes")
    print()

    # Convert to arrays
    print("Converting to arrays...")
    z_init_all = np.array(z_init_list)
    z_target_all = np.array(z_target_list)

    # Verify shapes
    print(f"Initial conditions shape: {z_init_all.shape}")
    print(f"Target conditions shape: {z_target_all.shape}")
    print()

    # Verify boundary conditions
    print("Verifying boundary conditions...")
    max_bc_error = max(
        np.max(np.abs(z_init_all[:, 0, :])),   # Left edge
        np.max(np.abs(z_init_all[:, -1, :])),  # Right edge
        np.max(np.abs(z_init_all[:, :, 0])),   # Bottom edge
        np.max(np.abs(z_init_all[:, :, -1])),  # Top edge
        np.max(np.abs(z_target_all[:, 0, :])),
        np.max(np.abs(z_target_all[:, -1, :])),
        np.max(np.abs(z_target_all[:, :, 0])),
        np.max(np.abs(z_target_all[:, :, -1]))
    )
    print(f"  Max boundary error: {max_bc_error:.2e}")
    if max_bc_error > 1e-6:
        print("  WARNING: Boundary conditions not satisfied!")
    else:
        print("  ✓ Boundary conditions satisfied")
    print()

    # Save to compressed archive
    print(f"Saving to {output_file}...")
    np.savez_compressed(
        output_file,
        z_init=z_init_all,
        z_target=z_target_all,
        grid_size=n_grid,
        n_samples=n_samples
    )

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print()

    total_time = time.time() - start_time
    print("=" * 60)
    print("Dataset generation complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Output: {output_file}")
    print("=" * 60)
    print()
    print("To use this dataset in training, it will be loaded automatically")
    print("if found in examples/heat2D/data/")


if __name__ == "__main__":
    main()
