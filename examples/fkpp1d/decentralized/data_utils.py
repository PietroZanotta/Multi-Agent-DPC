import jax
import jax.numpy as jnp

def rbf_kernel(x1, x2, length_scale=0.2, sigma=1.0):
    """
    Computes the RBF (Squared Exponential) kernel covariance matrix.
    Args:
        x1: array of shape (N, D) or (N,)
        x2: array of shape (M, D) or (M,)
        length_scale: scale parameter controlling smoothness
        sigma: signal variance
    """
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
        
    dist_sq = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return sigma**2 * jnp.exp(-0.5 * dist_sq / length_scale**2)

def generate_grf_unbounded(key, n_points=100, length_scale=0.2, sigma=1.0):
    """
    Generates a smooth Gaussian Random Field on [0, 1] with zero boundary conditions.
    Running on CPU to avoid Metal/GPU issues with Cholesky.
    """
    # with jax.default_device(jax.devices("cpu")[0]):
    x_grid = jnp.linspace(0, 1, n_points)
    
    # 1. Compute Covariance Matrix
    K = rbf_kernel(x_grid, x_grid, length_scale, sigma) + jnp.eye(n_points) * 1e-4
    
    # 2. Sample from Multivariate Normal samples ~ N(0, K)
    L = jnp.linalg.cholesky(K)
    z = jax.random.normal(key, shape=(n_points,))
    f_sample = L @ z
    
    # 3. Apply Bridge Correction to enforce f(0) = 0 and f(1) = 0
    f_0 = f_sample[0]
    f_1 = f_sample[-1]
    
    linear_trend = f_0 + x_grid * (f_1 - f_0)
    field = f_sample - linear_trend
    
    return x_grid, field

# def generate_grf(key, n_points=100, length_scale=0.2, sigma=1.0):
#     x_grid, field = generate_grf_unbounded(key, n_points, length_scale, sigma)
    
#     # Apply a sigmoid to squash values to (0, 1)
#     # Note: Since the GRF bridge forces boundaries to 0, 
#     # sigmoid(0) = 0.5. We need to shift it back to 0 at the boundaries.
    
#     bounded_field = jax.nn.sigmoid(field) 
    
#     # Re-enforce zero boundaries if needed for your specific control task
#     # (e.g., by multiplying by a window function like sin(pi * x))
#     window = jnp.sin(jnp.pi * x_grid)
#     return x_grid, bounded_field * window

def generate_grf(key, n_points=100, length_scale=0.1, sigma=2.0):
    x_grid, field = generate_grf_unbounded(key, n_points, length_scale, sigma)
    
    # Method A: Softplus (Allows for very sparse "islands" of population)
    # Raising the field or lowering sigma creates more 'empty' space
    bounded_field = jax.nn.softplus(field - 1.0) 
    
    # Normalize to [0, 1]
    bounded_field = bounded_field / (jnp.max(bounded_field) + 1e-6)
    
    return x_grid, bounded_field

if __name__ == "__main__":
    # Test block to visualize if run directly
    import matplotlib.pyplot as plt
    
    print("Generating GRF samples...")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    plt.figure(figsize=(10, 6))
    
    for k in keys:
        x, y = generate_grf(k, n_points=100, length_scale=0.15)
        plt.plot(x, y)
        
    plt.title("Smooth GRF Samples with Zero Boundaries")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig("grf_samples.png")
    print("Saved grf_samples.png")
