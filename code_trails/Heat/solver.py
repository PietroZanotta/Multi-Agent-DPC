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
    # Main diagonal and off-diagonals
    main_diag = jnp.ones(N) * (1 + 2 * r)
    off_diag = jnp.ones(N - 1) * (-r)
    
    # Construct dense matrix A (LHS)
    A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    
    # Enforce Dirichlet BCs: first and last rows
    # We want u[0] = 0, u[-1] = 0.
    # Modify A to be identity at boundaries
    A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
    A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)
    
    return A

@jit
def solve_heat_equation(u_init, controls):
    """
    Differentiable forward solver.
    
    Args:
        u_init: Initial state (N,)
        controls: Control signals over time (T, 4)
    
    Returns:
        final_u: The state at the final time step.
        trajectory: The full history of states (T, N)
    """
    r = nu * fixed_dt / (2 * dx**2)
    A = build_matrices(N, r)
    
    # Pre-compute control spatial profiles (Gaussian blobs)
    # Shape: (4, N)
    profiles = jnp.exp(-0.5 * ((x[None, :] - centers[:, None]) / sigma) ** 2)
    
    def step_fn(u_prev, control_t):
        # 1. Calculate Forcing
        # control_t is (4,) -> f_t is (N,)
        f_t = jnp.dot(control_t, profiles)
        
        # 2. Explicit Step (RHS of Crank-Nicolson)
        # Note: We implement the matrix multiplication manually or via convolution for speed,
        # but for clarity/differentiability, we can just use the indices logic or a stored RHS matrix.
        # Here we do the standard stencil operation:
        u_interior = u_prev[1:-1]
        u_left = u_prev[:-2]
        u_right = u_prev[2:]
        
        # Laplacian part
        diff = r * (u_left - 2*u_interior + u_right)
        
        rhs_interior = u_interior + diff + fixed_dt * f_t[1:-1]
        
        # Reconstruct full RHS vector with BCs (0 at boundaries)
        rhs = jnp.zeros_like(u_prev)
        rhs = rhs.at[1:-1].set(rhs_interior)
        
        # 3. Implicit Solve (LHS)
        u_next = jnp.linalg.solve(A, rhs)
        
        return u_next, u_next

    # Run the simulation loop efficiently
    # scan returns: (final_carry, stacked_outputs)
    final_u, trajectory = jax.lax.scan(step_fn, u_init, controls)
    
    return trajectory