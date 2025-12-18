# solver.py
import jax
import jax.numpy as jnp
from jax import jit

# Configuration (Global constants for the solver)
N = 100
dx = 1.0 / N
x = jnp.linspace(0, 1, N, endpoint=False)
nu = 0.1  # diffusion coefficient
sigma = 0.1
centers = jnp.array([0.2, 0.4, 0.6, 0.8])
fixed_dt = 0.001

def build_matrices(N, r):
    """Builds the dense matrices for the implicit scheme."""
    main_diag = jnp.ones(N) * (1 + 2 * r)
    off_diag = jnp.ones(N - 1) * (-r)
    
    A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    
    # Enforce Dirichlet BCs
    A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
    A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)
    
    return A

# Pre-compute control spatial profiles (Gaussian blobs)
# Shape: (4, N)
profiles = jnp.exp(-0.5 * ((x[None, :] - centers[:, None]) / sigma) ** 2)

@jit
def step_heat_equation(u_current, control):
    """
    Single-step forward simulation for open-loop control.
    
    Args:
        u_current: Current state (N,)
        control: Single control input (4,)
    
    Returns:
        u_next: Next state (N,)
    """
    r = nu * fixed_dt / (2 * dx**2)
    A = build_matrices(N, r)
    
    # Calculate forcing from control
    f_t = jnp.dot(control, profiles)
    
    # Explicit step (RHS)
    u_interior = u_current[1:-1]
    u_left = u_current[:-2]
    u_right = u_current[2:]
    
    diff = r * (u_left - 2*u_interior + u_right)
    rhs_interior = u_interior + diff + fixed_dt * f_t[1:-1]
    
    rhs = jnp.zeros_like(u_current)
    rhs = rhs.at[1:-1].set(rhs_interior)
    
    # Implicit solve
    u_next = jnp.linalg.solve(A, rhs)
    
    return u_next

@jit
def solve_heat_equation(u_init, controls):
    """
    Multi-step forward solver for MPC planning.
    
    Args:
        u_init: Initial state (N,)
        controls: Control signals over time (T, 4)
    
    Returns:
        trajectory: The full history of states (T, N)
    """
    r = nu * fixed_dt / (2 * dx**2)
    A = build_matrices(N, r)
    
    def step_fn(u_prev, control_t):
        # Calculate forcing
        f_t = jnp.dot(control_t, profiles)
        
        # Explicit step
        u_interior = u_prev[1:-1]
        u_left = u_prev[:-2]
        u_right = u_prev[2:]
        
        diff = r * (u_left - 2*u_interior + u_right)
        rhs_interior = u_interior + diff + fixed_dt * f_t[1:-1]
        
        rhs = jnp.zeros_like(u_prev)
        rhs = rhs.at[1:-1].set(rhs_interior)
        
        # Implicit solve
        u_next = jnp.linalg.solve(A, rhs)
        
        return u_next, u_next

    final_u, trajectory = jax.lax.scan(step_fn, u_init, controls)
    
    return trajectory