import jax
import jax.numpy as jnp
from jax import jit, lax

# --- Configuration (Physical Parameters) ---
N_grid = 100 
L = 1.0
dx = L / N_grid
nu = 0.005    # Diffusion
rho = 3.     # Growth rate
dt = 0.001   # Time step
sigma = 0.05  # Actuator width

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
    
    # Vectorize the actuator calculation over all agents
    forcings = jax.vmap(single_actuator)(xi, u)
    return jnp.sum(forcings, axis=0)

def solve_tridiagonal_diffusion(z_explicit, r, N):
    """
    Solves (I - r*L)z = z_explicit using 1D tridiagonal solver.
    """
    # 1. Diagonals (N,)
    d = jnp.ones(N) * (1 + 2 * r)
    d = d.at[0].set(1 + r)
    d = d.at[-1].set(1 + r)
    
    ld = jnp.ones(N) * (-r)
    ud = jnp.ones(N) * (-r)
    
    # 2. Fix the RHS shape
    # z_explicit is (N,) -> reshape to (N, 1)
    rhs = z_explicit[:, jnp.newaxis]
    
    # 3. Solve
    # out will have shape (N, 1)
    out = jax.lax.linalg.tridiagonal_solve(ld, d, ud, rhs)
    
    # Flatten back to (N,)
    return out.ravel()

@jit
def fkpp_step_1d(carry, actions):
    """
    Single time-step for 1D Fisher-KPP.
    """
    z, xi = carry
    u, v = actions['u'], actions['v']
    N = z.shape[0]

    # 1. Reaction + Forcing (Explicit)
    # Using +u for growth stimulation or -u for suppression
    f_t = forcing_fn_1d(xi, u, N) 
    reaction = rho * z * (1.0 - z)
    z_explicit = z + dt * (reaction + f_t)

    # 2. Diffusion (Implicit)
    r = nu * dt / (dx**2)
    z_next = solve_tridiagonal_diffusion(z_explicit, r, N)
    
    # 3. Updates
    z_next = jnp.clip(z_next, 0.0, 1.0)
    xi_next = jnp.clip(xi + dt * v, 0.0, 1.0)

    return (z_next, xi_next), (z_next, xi_next)

@jit
def solve_pde_trajectory(z_init, xi_init, u_seq, v_seq):
    """
    Unrolls the trajectory over the given control sequences.
    """
    def scan_body(carry, inputs):
        u, v = inputs
        return fkpp_step_1d(carry, {'u': u, 'v': v})

    _, (z_traj, xi_traj) = lax.scan(scan_body, (z_init, xi_init), (u_seq, v_seq))
    return z_traj, xi_traj