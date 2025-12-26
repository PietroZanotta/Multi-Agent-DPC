import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

# --- Configuration ---
N_grid = 100 
L = 1.0
dx = L / N_grid
nu = 0.005   # Diffusion
rho = 3.0    # Growth rate
dt = 0.001   # Time step
sigma = 0.05 # Actuator width

def forcing_fn_1d(xi, u, N):
    """
    Calculates the 1D Gaussian influence of actuators.
    xi: (M,) positions
    u:  (M,) intensities
    """
    x_coords = jnp.linspace(0, 1, N)
    def single_actuator(pos, intensity):
        dist_sq = (x_coords - pos)**2
        return intensity * jnp.exp(-dist_sq / (2 * sigma**2))
    
    forcings = jax.vmap(single_actuator)(xi, u)
    return jnp.sum(forcings, axis=0)

def solve_tridiagonal_diffusion(z_explicit, r, N):
    # Dirichlet (z=0 at boundaries)
    d = jnp.ones(N) * (1 + 2 * r)
    d = d.at[0].set(1.0)
    d = d.at[-1].set(1.0)

    ld = jnp.ones(N) * (-r)
    ld = ld.at[0].set(0.0)

    ud = jnp.ones(N) * (-r)
    ud = ud.at[-1].set(0.0)
    
    rhs_values = z_explicit.at[0].set(0.0).at[-1].set(0.0)
    rhs = rhs_values[:, jnp.newaxis]
    
    out = jax.lax.linalg.tridiagonal_solve(ld, d, ud, rhs)
    return out.ravel()

@jit
def fkpp_step_1d(z, xi, u, v):
    """Refactored to match heat equation step signature."""
    N = z.shape[0]
    # 1. Reaction + Forcing (Explicit)
    f_t = forcing_fn_1d(xi, u, N) 
    reaction = rho * z * (1.0 - z)
    z_explicit = z + dt * (reaction + f_t)

    # 2. Diffusion (Implicit)
    r = nu * dt / (dx**2)
    z_next = solve_tridiagonal_diffusion(z_explicit, r, N)
    
    # 3. Updates & Clipping
    z_next = jnp.clip(z_next, 0.0, 1.0)
    xi_next = jnp.clip(xi + dt * v, 0.0, 1.0)

    return z_next, xi_next

@partial(jax.jit, static_argnums=(4, 5))
def solve_with_policy(z_init, xi_init, z_target, params, policy_apply_fn, t_steps):
    """
    FKPP Loop: Policy determines intensity (u) and velocity (v) 
    at every step based on current state vs target.
    """
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        
        # 1. Policy Inference
        # Returns u (intensities) and v (velocities)
        u, v = policy_apply_fn(params, z_curr, z_target, xi_curr)
        
        # 2. FKPP Physics Step
        z_next, xi_next = fkpp_step_1d(z_curr, xi_curr, u, v)
        
        return (z_next, xi_next), (z_next, xi_next, u, v)

    _, trajectory = jax.lax.scan(
        step_fn, 
        (z_init, xi_init), 
        None, 
        length=t_steps
    )
    
    return trajectory # (z_traj, xi_traj, u_traj, v_traj)