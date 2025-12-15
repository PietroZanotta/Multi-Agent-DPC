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
# centers = jnp.array([0.2, 0.4, 0.6, 0.8]) # REMOVED: No longer hardcoded
fixed_dt = 0.001

def build_matrices(N, r):
    """Builds the dense matrices for the implicit scheme."""
    main_diag = jnp.ones(N) * (1 + 2 * r)
    off_diag = jnp.ones(N - 1) * (-r)
    
    A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    
    # Dirichlet BCs
    A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
    A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)
    
    return A

@jit
def solve_heat_equation(u_init, controls, actuator_centers):
    """
    Differentiable forward solver.
    
    Args:
        u_init: Initial state (N,)
        controls: Control signals over time (T, 4)
        actuator_centers: Locations of actuators (4,) <-- NEW ARGUMENT
    
    Returns:
        trajectory: The full history of states (T, N)
    """
    r = nu * fixed_dt / (2 * dx**2)
    A = build_matrices(N, r)
    
    # Pre-compute control spatial profiles (Gaussian blobs)
    # This uses the DYNAMIC actuator_centers passed in.
    # Shape: (4, N)
    # Since this is a smooth Gaussian, gradients flow through 'actuator_centers' perfectly.
    profiles = jnp.exp(-0.5 * ((x[None, :] - actuator_centers[:, None]) / sigma) ** 2)
    
    def step_fn(u_prev, control_t):
        # 1. Calculate Forcing
        f_t = jnp.dot(control_t, profiles)
        
        # 2. Explicit Step (RHS)
        u_interior = u_prev[1:-1]
        u_left = u_prev[:-2]
        u_right = u_prev[2:]
        
        diff = r * (u_left - 2*u_interior + u_right)
        rhs_interior = u_interior + diff + fixed_dt * f_t[1:-1]
        
        rhs = jnp.zeros_like(u_prev)
        rhs = rhs.at[1:-1].set(rhs_interior)
        
        # 3. Implicit Solve (LHS)
        u_next = jnp.linalg.solve(A, rhs)
        
        return u_next, u_next

    # Run the simulation loop
    final_u, trajectory = jax.lax.scan(step_fn, u_init, controls)
    
    return trajectory