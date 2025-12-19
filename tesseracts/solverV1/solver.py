# solver.py
import jax
import jax.numpy as jnp
from jax import jit

# Configuration
N = 100
dx = 1.0 / N
x_grid = jnp.linspace(0, 1, N)
nu = 0.01    # Diffusion coefficient
sigma = 0.05  # Width of the actuator influence
# REDUCED DT: 0.001 is well within the 0.005 stability limit
fixed_dt = 0.001 

def laplacian(z):
    """Central difference for the Laplacian."""
    z_interior = z[1:-1]
    z_left = z[:-2]
    z_right = z[2:]
    lapl = (z_left - 2 * z_interior + z_right) / (dx**2)
    return jnp.pad(lapl, (1, 1), mode='constant', constant_values=0.0)

def forcing_fn(x_grid, xi, u):
    sq_dist = (x_grid[None, :] - xi[:, None])**2
    basis = jnp.exp(-sq_dist / (2 * sigma**2))
    return jnp.sum(u[:, None] * basis, axis=0)

def f_combined(z, xi, u, dxi_dt):
    """
    The full system RHS.
    Returns (dz/dt, dxi/dt)
    """
    dz_dt = nu * laplacian(z) + forcing_fn(x_grid, xi, u)
    # The actuator velocity is provided directly as an input
    return dz_dt, dxi_dt

@jit
def rk4_step(z, xi, u, v):
    """
    RK4 step for both the PDE state (z) and actuator positions (xi).
    v: the velocity of actuators (dxi_dt).
    """
    dt = fixed_dt
    
    # k1
    k1_z, k1_xi = f_combined(z, xi, u, v)
    
    # k2
    k2_z, k2_xi = f_combined(z + 0.5 * dt * k1_z, xi + 0.5 * dt * k1_xi, u, v)
    
    # k3
    k3_z, k3_xi = f_combined(z + 0.5 * dt * k2_z, xi + 0.5 * dt * k2_xi, u, v)
    
    # k4
    k4_z, k4_xi = f_combined(z + dt * k3_z, xi + dt * k3_xi, u, v)
    
    # Final Combine
    z_next = z + (dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z)
    xi_next = xi + (dt / 6.0) * (k1_xi + 2*k2_xi + 2*k3_xi + k4_xi)
    
    return z_next, xi_next

@jit
def solve_pde_trajectory(z_init, xi_init, controls_u, controls_v):
    def step_fn(carry, t_idx):
        z, xi = carry
        u = controls_u[t_idx]
        v = controls_v[t_idx]
        
        z_next, xi_next = rk4_step(z, xi, u, v)
        return (z_next, xi_next), (z_next, xi_next)

    _, trajectory = jax.lax.scan(step_fn, (z_init, xi_init), jnp.arange(len(controls_u)))
    return trajectory # This is a tuple (z_traj, xi_traj)