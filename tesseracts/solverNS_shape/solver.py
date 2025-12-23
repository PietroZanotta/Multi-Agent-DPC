"""
2D Incompressible Navier-Stokes Solver with Gaussian Control Forcing

Spectral method using vorticity-streamfunction formulation.
Structure mirrors the 1D diffusion solver pattern.

Equations:
    ∂ω/∂t = -(v·∇)ω + ν∇²ω + (∇×f)_z
    ∇·v = 0  (automatic via streamfunction)
    ∂ρ/∂t = -(v·∇)ρ  (passive scalar)
"""

import jax
import jax.numpy as jnp
from jax import jit
import jax.numpy.fft as jfft

jax.config.update("jax_enable_x64", True)
# =============================================================================
# Configuration
# =============================================================================
N = 128
L = 2.0 * jnp.pi
dx = L / N
nu = 0.001
sigma = 0.3
fixed_dt = 0.01
drag = 0.0

# Grid coordinates
x = jnp.linspace(0.0, L, N, endpoint=False)
y = jnp.linspace(0.0, L, N, endpoint=False)
xx, yy = jnp.meshgrid(x, y, indexing='ij')

# Wavenumbers for spectral method (rfft2)
kx = jnp.fft.fftfreq(N, d=dx)
ky = jnp.fft.rfftfreq(N, d=dx)
KX, KY = jnp.meshgrid(kx, ky, indexing='ij')

# Precomputed spectral operators
two_pi = 2.0 * jnp.pi
K_SQ = (two_pi**2) * (KX**2 + KY**2)
K_SQ_safe = jnp.where(K_SQ == 0.0, 1.0, K_SQ)

# Linear operator: ν∇² - drag
LINEAR_TERM = -nu * K_SQ - drag

kappa = 0.0

# 2/3 dealiasing filter
kx_max = 1.0 / (2.0 * dx)
ky_max = 1.0 / (2.0 * dx)
DEALIAS = ((jnp.abs(KX) < (2.0/3.0) * kx_max) & 
           (jnp.abs(KY) < (2.0/3.0) * ky_max)).astype(jnp.float64)


# =============================================================================
# Spectral Helper Functions
# =============================================================================

def vorticity_to_velocity(omega_hat):
    """
    Convert vorticity to velocity via streamfunction.
    ω = -∇²ψ  =>  ψ_hat = ω_hat / |k|²
    v_x = ∂ψ/∂y, v_y = -∂ψ/∂x
    
    Returns velocity in physical space.
    """
    psi_hat = omega_hat / K_SQ_safe
    psi_hat = jnp.where(K_SQ == 0, 0.0, psi_hat)
    
    two_pi_i = 2.0 * jnp.pi * 1j
    vx_hat = two_pi_i * KY * psi_hat
    vy_hat = -two_pi_i * KX * psi_hat
    
    vx = jfft.irfft2(vx_hat)
    vy = jfft.irfft2(vy_hat)
    return vx, vy


def spectral_gradient(f_hat):
    """Compute gradient in Fourier space, return in physical space."""
    two_pi_i = 2.0 * jnp.pi * 1j
    df_dx = jfft.irfft2(two_pi_i * KX * f_hat)
    df_dy = jfft.irfft2(two_pi_i * KY * f_hat)
    return df_dx, df_dy


def spectral_curl_z(fx, fy):
    """Compute z-component of curl: (∇×f)_z = ∂f_y/∂x - ∂f_x/∂y"""
    fx_hat = jfft.rfft2(fx)
    fy_hat = jfft.rfft2(fy)
    two_pi_i = 2.0 * jnp.pi * 1j
    curl_z_hat = two_pi_i * (KX * fy_hat - KY * fx_hat)
    return curl_z_hat


# =============================================================================
# Forcing Function (Gaussian Actuators)
# =============================================================================

def forcing_fn(xi, u):
    """
    Gaussian actuator forcing field.
    
    Args:
        xi: Actuator positions (M, 2) - each row is [x, y]
        u: Force amplitudes (M, 2) - each row is [fx, fy]
        
    Returns:
        fx, fy: Force field components (N, N) each
    """
    # Compute Gaussian basis for each actuator
    # xi[:, 0] are x-positions, xi[:, 1] are y-positions
    # Broadcast: (M, 1, 1) with (N, N) grid
    
    dx_sq = (xx[None, :, :] - xi[:, 0, None, None])**2
    dy_sq = (yy[None, :, :] - xi[:, 1, None, None])**2
    sq_dist = dx_sq + dy_sq  # (M, N, N)
    
    basis = jnp.exp(-sq_dist / (2.0 * sigma**2))  # (M, N, N)
    
    # Sum contributions from all actuators
    fx = jnp.sum(u[:, 0, None, None] * basis, axis=0)  # (N, N)
    fy = jnp.sum(u[:, 1, None, None] * basis, axis=0)  # (N, N)
    
    return fx, fy


# =============================================================================
# Time Stepping (Semi-Implicit: CN for diffusion, explicit for advection)
# =============================================================================

@jit
def navier_stokes_step(omega_hat, xi, u, v):
    """
    Single time step for Navier-Stokes + actuator dynamics.
    
    Uses Crank-Nicolson for diffusion (implicit) and 
    explicit Euler for advection and forcing.
    
    Args:
        omega_hat: Vorticity in Fourier space (N, N//2+1) complex
        xi: Actuator positions (M, 2)
        u: Force amplitudes (M, 2)
        v: Actuator velocities (M, 2)
        
    Returns:
        omega_hat_new: Updated vorticity
        xi_new: Updated actuator positions
    """
    dt = fixed_dt
    
    # 1. Get velocity from vorticity
    vx, vy = vorticity_to_velocity(omega_hat)
    
    # 2. Compute advection: -(v·∇)ω
    grad_omega_x, grad_omega_y = spectral_gradient(omega_hat)
    advection = -(vx * grad_omega_x + vy * grad_omega_y)
    advection_hat = jfft.rfft2(advection)
    advection_hat = DEALIAS * advection_hat
    
    # 3. Compute forcing curl: (∇×f)_z
    fx, fy = forcing_fn(xi, u)
    curl_f_hat = spectral_curl_z(fx, fy)
    curl_f_hat = DEALIAS * curl_f_hat
    
    # 4. Explicit terms (advection + forcing)
    explicit_terms = advection_hat + curl_f_hat
    
    # 5. Semi-implicit update (Crank-Nicolson for diffusion)
    # ω_new = ω + dt * [0.5 * L * ω + 0.5 * L * ω_new + explicit]
    # => (1 - 0.5*dt*L) * ω_new = (1 + 0.5*dt*L) * ω + dt * explicit
    # => ω_new = [(1 + 0.5*dt*L) * ω + dt * explicit] / (1 - 0.5*dt*L)
    
    lhs_coeff = 1.0 - 0.5 * dt * LINEAR_TERM
    rhs = (1.0 + 0.5 * dt * LINEAR_TERM) * omega_hat + dt * explicit_terms
    omega_hat_new = rhs / lhs_coeff
    
    # 6. Update actuator positions
    xi_new = xi + dt * v
    xi_new = xi_new % L  # Periodic wrapping
    
    return omega_hat_new, xi_new


@jit
def advect_density_step(rho, omega_hat):
    """
    Advect passive scalar by one time step with small diffusion.
    ∂ρ/∂t = -(v·∇)ρ + κ∇²ρ
    """
    dt = fixed_dt
    vx, vy = vorticity_to_velocity(omega_hat)

    def advection_hat_from_rho(r):
        r_hat = jfft.rfft2(r)
        grad_r_x, grad_r_y = spectral_gradient(r_hat)
        adv = -(vx * grad_r_x + vy * grad_r_y)
        adv_hat = jfft.rfft2(adv)
        return DEALIAS * adv_hat

    rho_hat = jfft.rfft2(rho)
    adv0_hat = advection_hat_from_rho(rho)
    rho_tilde = jfft.irfft2(rho_hat + dt * adv0_hat).astype(rho.dtype)
    adv1_hat = advection_hat_from_rho(rho_tilde)
    adv_hat = 0.5 * (adv0_hat + adv1_hat)

    diffusion_lhs = jnp.asarray(1.0 + dt * kappa * K_SQ, dtype=rho_hat.dtype)
    rho_hat_new = (rho_hat + dt * adv_hat) / diffusion_lhs
    return jfft.irfft2(rho_hat_new).astype(rho.dtype)


@jit
def full_step(omega_hat, rho, xi, u, v):
    """
    Full time step: vorticity + density + actuators.
    """
    omega_hat_new, xi_new = navier_stokes_step(omega_hat, xi, u, v)
    rho_new = advect_density_step(rho, omega_hat_new)
    return omega_hat_new, rho_new, xi_new


# =============================================================================
# Trajectory Solver
# =============================================================================

@jit
def solve_trajectory(omega_hat_init, rho_init, xi_init, controls_u, controls_v):
    """
    Solve the PDE trajectory given initial conditions and control sequence.
    
    Args:
        omega_hat_init: Initial vorticity in Fourier space (N, N//2+1)
        rho_init: Initial passive scalar (N, N)
        xi_init: Initial actuator positions (M, 2)
        controls_u: Force amplitudes over time (T, M, 2)
        controls_v: Actuator velocities over time (T, M, 2)
        
    Returns:
        trajectory: Tuple of (omega_hat, rho, xi) arrays over time
    """
    def step_fn(carry, controls):
        omega_hat, rho, xi = carry
        u, v = controls
        
        omega_hat_new, rho_new, xi_new = full_step(omega_hat, rho, xi, u, v)
        
        return (omega_hat_new, rho_new, xi_new), (omega_hat_new, rho_new, xi_new)
    
    _, trajectory = jax.lax.scan(
        step_fn, 
        (omega_hat_init, rho_init, xi_init), 
        (controls_u, controls_v)
    )
    
    return trajectory

@jit
def solve_pde_trajectory(z_init, xi_init, controls_u, controls_v):
    key = jax.random.PRNGKey(42)
    omega_hat_init = random_vorticity(key, k_peak=6.0, amplitude=80.0)
    _, rho_traj, xi_traj = solve_trajectory(omega_hat_init, z_init, xi_init, controls_u, controls_v)
    return rho_traj, xi_traj


# =============================================================================
# Initial Conditions
# =============================================================================

def taylor_green_vorticity(k=2, amplitude=1.0):
    """Taylor-Green vortex initial condition."""
    omega = (2.0 * k * amplitude) * jnp.cos(k * xx) * jnp.cos(k * yy)
    return jfft.rfft2(omega)


# def random_vorticity(key, k_peak=4.0, amplitude=1.0):
#     """Random vorticity with energy peaked at k_peak."""
#     k_mag = jnp.sqrt(KX**2 + KY**2)
#     energy_spectrum = jnp.exp(jnp.float32(-0.5) * ((k_mag - jnp.float32(k_peak)) / (jnp.float32(k_peak) / jnp.float32(2.0)))**2).astype(jnp.float32)
    
#     key1, key2 = jax.random.split(key)
#     phases = jax.random.uniform(key1, shape=KX.shape, minval=0, maxval=2*jnp.pi).astype(jnp.float32)
#     amplitudes = jax.random.normal(key2, shape=KX.shape).astype(jnp.float32)
    
#     omega_hat = jnp.float32(amplitude) * energy_spectrum * amplitudes * jnp.exp(1j * phases).astype(jnp.complex64)
#     return omega_hat.astype(jnp.complex64)

def random_vorticity(key, k_peak=4.0, amplitude=2.0):
    """Random turbulent vorticity field."""
    kx_grid = jnp.fft.fftfreq(N, d=dx)
    ky_grid = jnp.fft.rfftfreq(N, d=dx)
    KX_local, KY_local = jnp.meshgrid(kx_grid, ky_grid, indexing='ij')
    k_mag = jnp.sqrt(KX_local**2 + KY_local**2)
    
    # Energy spectrum peaked at k_peak
    energy_spectrum = (k_mag ** 2) * jnp.exp(-0.5 * ((k_mag - k_peak) / (k_peak / 3))**2)
    energy_spectrum = energy_spectrum / (jnp.max(energy_spectrum) + 1e-10)
    
    key1, key2 = jax.random.split(key)
    phases = jax.random.uniform(key1, shape=KX_local.shape, minval=0, maxval=2*jnp.pi)
    rand_amp = jax.random.normal(key2, shape=KX_local.shape)
    
    omega_hat = amplitude * energy_spectrum * rand_amp * jnp.exp(1j * phases)
    omega_hat = omega_hat.at[0, 0].set(0.0)  # Zero mean
    return omega_hat
    

def gaussian_density(center, sigma_rho=0.5, amplitude=1.0):
    """Gaussian blob for passive scalar."""
    sq_dist = (xx - center[0])**2 + (yy - center[1])**2
    return amplitude * jnp.exp(-sq_dist / (2.0 * sigma_rho**2))


def uniform_actuator_positions(M):
    """Place M actuators uniformly in domain."""
    import math
    n_side = int(math.ceil(math.sqrt(M)))
    spacing = L / (n_side + 1)
    
    positions = []
    for i in range(n_side):
        for j in range(n_side):
            if len(positions) < M:
                positions.append([spacing * (i + 1), spacing * (j + 1)])
    
    return jnp.array(positions[:M])

# =============================================================================
# Diagnostics
# =============================================================================

def kinetic_energy(omega_hat):
    """Compute total kinetic energy."""
    vx, vy = vorticity_to_velocity(omega_hat)
    return 0.5 * jnp.mean(vx**2 + vy**2) * L**2


def enstrophy(omega_hat):
    """Compute total enstrophy."""
    omega = jfft.irfft2(omega_hat)
    return 0.5 * jnp.mean(omega**2) * L**2


def vorticity_field(omega_hat):
    """Convert to physical space."""
    return jfft.irfft2(omega_hat)


# # =============================================================================
# # Example Usage
# # =============================================================================

# if __name__ == "__main__":
#     import time
    
#     print("=" * 60)
#     print("2D Incompressible Navier-Stokes with Gaussian Control")
#     print("=" * 60)
#     print(f"Grid: {N}x{N}, Domain: [0, {L:.4f}]²")
#     print(f"Viscosity: {nu}, dt: {fixed_dt}")
    
#     # Number of actuators
#     M = 4
    
#     # Initial conditions
#     omega_hat_init = taylor_green_vorticity(k=2, amplitude=1.0)
#     rho_init = gaussian_density(center=(L/2, L/2), sigma_rho=L/6)
#     xi_init = uniform_actuator_positions(M)
    
#     print(f"\nInitial KE: {kinetic_energy(omega_hat_init):.6f}")
#     print(f"Initial Enstrophy: {enstrophy(omega_hat_init):.6f}")
    
#     # Control sequence (random forcing, stationary actuators)
#     T = 100  # Number of time steps
#     key = jax.random.PRNGKey(42)
#     key1, key2 = jax.random.split(key)
    
#     controls_u = 0.5 * jax.random.normal(key1, (T, M, 2))  # Force amplitudes
#     controls_v = jnp.zeros((T, M, 2))  # Actuators don't move
    
#     # Run simulation
#     print(f"\nRunning {T} time steps...")
    
#     start = time.time()
#     trajectory = solve_trajectory(omega_hat_init, rho_init, xi_init, 
#                                    controls_u, controls_v)
#     omega_hat_traj, rho_traj, xi_traj = trajectory
#     jax.block_until_ready(omega_hat_traj)
#     compile_time = time.time() - start
    
#     # Run again (after JIT compile)
#     start = time.time()
#     trajectory = solve_trajectory(omega_hat_init, rho_init, xi_init,
#                                    controls_u, controls_v)
#     omega_hat_traj, rho_traj, xi_traj = trajectory
#     jax.block_until_ready(omega_hat_traj)
#     run_time = time.time() - start
    
#     print(f"Compile + run time: {compile_time:.3f}s")
#     print(f"Run time (after JIT): {run_time:.3f}s")
#     print(f"Time per step: {run_time/T*1000:.3f}ms")
    
#     # Final state
#     omega_hat_final = omega_hat_traj[-1]
#     print(f"\nFinal KE: {kinetic_energy(omega_hat_final):.6f}")
#     print(f"Final Enstrophy: {enstrophy(omega_hat_final):.6f}")
    
#     # Test gradient computation
#     print("\n" + "=" * 60)
#     print("Testing Differentiability")
#     print("=" * 60)
    
#     def loss_fn(controls_u):
#         """Loss: final enstrophy."""
#         traj = solve_trajectory(omega_hat_init, rho_init, xi_init,
#                                 controls_u, controls_v)
#         return enstrophy(traj[0][-1])
    
#     grad_fn = jax.grad(loss_fn)
    
#     start = time.time()
#     gradient = grad_fn(controls_u)
#     jax.block_until_ready(gradient)
#     grad_time = time.time() - start
    
#     print(f"Gradient computation time: {grad_time:.3f}s")
#     print(f"Gradient shape: {gradient.shape}")
#     print(f"Gradient norm: {jnp.linalg.norm(gradient):.6f}")
#     print("\n✓ Solver is fully differentiable!")

