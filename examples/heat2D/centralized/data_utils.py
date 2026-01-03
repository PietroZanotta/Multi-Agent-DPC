import jax
import jax.numpy as jnp

def rbf_kernel_2d(x1, x2, length_scale=0.3, sigma=1.0):
    """
    2D RBF kernel for GRF generation.

    Args:
        x1: Grid points (N1, 2)
        x2: Grid points (N2, 2)
        length_scale: Correlation length
        sigma: Signal variance

    Returns:
        Covariance matrix (N1, N2)
    """
    # Pairwise distances: (N1, 1, 2) - (1, N2, 2) -> (N1, N2, 2)
    diff = x1[:, None, :] - x2[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)  # (N1, N2)
    return sigma**2 * jnp.exp(-0.5 * dist_sq / length_scale**2)

def generate_grf_2d(key, n_points=64, length_scale=0.4, sigma=1.0):
    """
    Generate 2D Gaussian Random Field with zero Dirichlet BCs.

    Strategy:
    1. Generate GRF on full grid using eigendecomposition
    2. Apply 2D bridge correction to enforce z=0 on all edges

    Args:
        key: JAX random key
        n_points: Number of grid points per dimension
        length_scale: Correlation length for smoothness
        sigma: Signal variance

    Returns:
        meshgrid (xx, yy), field (N, N)
    """
    with jax.default_device(jax.devices("cpu")[0]):  # CPU for eigen-decomp
        # Grid
        x_grid = jnp.linspace(0, 1, n_points)
        y_grid = jnp.linspace(0, 1, n_points)
        xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')

        # Flatten grid to (N², 2)
        grid_flat = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

        # Covariance matrix (N², N²) - this is large!
        K = rbf_kernel_2d(grid_flat, grid_flat, length_scale, sigma)

        # Eigendecomposition
        w, V = jnp.linalg.eigh(K)
        w_sqrt = jnp.sqrt(jnp.maximum(w, 1e-12))

        # Sample
        z = jax.random.normal(key, shape=(n_points**2,))
        f_flat = V @ (w_sqrt * z)

    # Reshape to 2D
    f = f_flat.reshape((n_points, n_points))

    # --- 2D Bridge Correction ---
    # We need to enforce f=0 on edges while minimizing distortion
    # Use bilinear interpolation from edges

    # Extract edge values
    f_left = f[0, :]      # x=0 edge
    f_right = f[-1, :]    # x=1 edge
    f_bottom = f[:, 0]    # y=0 edge
    f_top = f[:, -1]      # y=1 edge

    # Extract corner values
    f_00 = f[0, 0]
    f_10 = f[-1, 0]
    f_01 = f[0, -1]
    f_11 = f[-1, -1]

    # Build trend surface using bilinear interpolation
    x_norm = xx
    y_norm = yy

    # Corner contribution (bilinear interpolation of corners)
    corner_trend = ((1 - x_norm) * (1 - y_norm) * f_00 +
                    x_norm * (1 - y_norm) * f_10 +
                    (1 - x_norm) * y_norm * f_01 +
                    x_norm * y_norm * f_11)

    # Edge contributions (linear interpolation along each edge, minus corner overlap)
    # Left edge: varies with y, weighted by (1-x)
    left_edge_interp = (1 - y_norm) * f_00 + y_norm * f_01
    left_contrib = (1 - x_norm) * (f_left[None, :] - left_edge_interp)

    # Right edge: varies with y, weighted by x
    right_edge_interp = (1 - y_norm) * f_10 + y_norm * f_11
    right_contrib = x_norm * (f_right[None, :] - right_edge_interp)

    # Bottom edge: varies with x, weighted by (1-y)
    bottom_edge_interp = (1 - x_norm) * f_00 + x_norm * f_10
    bottom_contrib = (1 - y_norm) * (f_bottom[:, None] - bottom_edge_interp)

    # Top edge: varies with x, weighted by y
    top_edge_interp = (1 - x_norm) * f_01 + x_norm * f_11
    top_contrib = y_norm * (f_top[:, None] - top_edge_interp)

    # Total trend surface
    trend = corner_trend + left_contrib + right_contrib + bottom_contrib + top_contrib

    # Subtract trend to get zero BCs
    field = f - trend

    return xx, yy, field


if __name__ == "__main__":
    # Test block to visualize if run directly
    import matplotlib.pyplot as plt

    print("Generating 2D GRF samples...")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, k in enumerate(keys):
        xx, yy, field = generate_grf_2d(k, n_points=64, length_scale=0.4)

        # Verify boundary conditions
        max_boundary_error = max(
            jnp.max(jnp.abs(field[0, :])),   # Left edge
            jnp.max(jnp.abs(field[-1, :])),  # Right edge
            jnp.max(jnp.abs(field[:, 0])),   # Bottom edge
            jnp.max(jnp.abs(field[:, -1]))   # Top edge
        )
        print(f"Sample {i+1}: Max boundary error = {max_boundary_error:.2e}")

        im = axes[i].imshow(field, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("grf_2d_samples.png")
    print("Saved grf_2d_samples.png")
