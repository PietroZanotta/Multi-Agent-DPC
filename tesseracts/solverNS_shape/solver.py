"""
2D Incompressible Navier-Stokes Solver with Gaussian Control Forcing

Spectral method using vorticity-streamfunction formulation.
PHIFLOW MULTI-SCALE VERSION - Reproduces dramatic turbulent mixing

Equations:
    ∂ω/∂t = -(v·∇)ω + ν∇²ω + (∇×f)_z
    ∇·v = 0  (automatic via streamfunction)
    ∂ρ/∂t = -(v·∇)ρ + buoyancy  (passive scalar with optional buoyancy)
"""

import jax
import jax.numpy as jnp
from jax import jit
import jax.numpy.fft as jfft

jax.config.update("jax_enable_x64", True)
# =============================================================================
# Configuration - PHIFLOW PARAMETERS
# =============================================================================
N = 128  # Grid resolution (confirmed from PhiFlow code)
L = 2.0 * jnp.pi  # Domain size
dx = L / N
nu = 0.0  # NO viscosity (only numerical diffusion from advection)
sigma = 0.3  # Gaussian actuator width
fixed_dt = 0.01  # Large timestep from PhiFlow
drag = 0.0

# Multi-scale initialization parameters (from PhiFlow RandomSmoke)
V_SCALE_BASE = 1.0  # Base velocity scale
V_FALLOFF = 0.9  # Power spectrum falloff
BUOYANCY_STRENGTH = 0.5  # Upward force from density
use_buoyancy = True  # Enable buoyancy-driven flow

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

kappa = 0  # No density diffusion (pure advection for sharp filaments)

# 2/3 dealiasing filter
kx_max = 1.0 / (2.0 * dx)
ky_max = 1.0 / (2.0 * dx)
DEALIAS = ((jnp.abs(KX) < (2.0/3.0) * kx_max) & 
           (jnp.abs(KY) < (2.0/3.0) * ky_max)).astype(jnp.float64)


# =============================================================================
# Multi-Scale Initialization (PhiFlow Method)
# =============================================================================

def upsample_2x(field):
    """Upsample field by factor of 2 using linear interpolation"""
    h, w = field.shape[0], field.shape[1]
    
    # Create 2x larger array
    upsampled = jnp.zeros((h*2, w*2) + field.shape[2:], dtype=field.dtype)
    
    # Fill in original values at even indices
    upsampled = upsampled.at[::2, ::2].set(field)
    
    # Linear interpolation in x direction
    upsampled = upsampled.at[1::2, ::2].set((field + jnp.roll(field, -1, axis=0)) / 2)
    
    # Linear interpolation in y direction  
    upsampled = upsampled.at[::2, 1::2].set((upsampled[::2, ::2] + jnp.roll(upsampled[::2, ::2], -1, axis=1)) / 2)
    
    # Diagonal interpolation
    upsampled = upsampled.at[1::2, 1::2].set((upsampled[1::2, ::2] + jnp.roll(upsampled[1::2, ::2], -1, axis=1)) / 2)
    
    return upsampled


def multi_scale_velocity_field(key, resolution=N, v_scale=V_SCALE_BASE, v_falloff=V_FALLOFF):
    """
    Multi-scale random velocity field initialization.
    
    This is the ACTUAL method from PhiFlow RandomSmoke class!
    Builds velocity field from coarse to fine scales with decreasing weights.
    """
    # Start at 1×1 resolution
    size = 1
    velocity = jnp.zeros((1, 1, 2), dtype=jnp.float64)
    
    i = 0
    while size < resolution:
        # Upsample current field by 2x
        velocity = upsample_2x(velocity)
        size = size * 2
        
        # Add weighted random noise at this scale
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (size, size, 2), dtype=jnp.float64)
        weight = v_scale * (v_falloff ** i)
        velocity = velocity + noise * weight
        
        i += 1
    
    return velocity


def multi_scale_density_field(key, resolution=N):
    """
    Multi-scale random density field initialization.
    
    From PhiFlow: combines 3 scales (1/4, 1/8, 1/16 resolution).
    """
    # Scale 1: 1/4 resolution, upsampled 2x
    key, k1 = jax.random.split(key)
    d1 = jax.random.uniform(k1, (resolution//4, resolution//4), dtype=jnp.float64)
    d1 = upsample_2x(upsample_2x(d1))
    
    # Scale 2: 1/8 resolution, upsampled 3x
    key, k2 = jax.random.split(key)
    d2 = jax.random.uniform(k2, (resolution//8, resolution//8), dtype=jnp.float64)
    d2 = upsample_2x(upsample_2x(upsample_2x(d2)))
    
    # Scale 3: 1/16 resolution, upsampled 4x
    key, k3 = jax.random.split(key)
    d3 = jax.random.uniform(k3, (resolution//16, resolution//16), dtype=jnp.float64)
    d3 = upsample_2x(upsample_2x(upsample_2x(upsample_2x(d3))))
    
    # Combine scales
    density = d1 + d2 + d3
    
    # Scale and clamp: clip(density * 0.66 - 1, 0, 1)
    density = jnp.clip(density * 0.66 - 1.0, 0.0, 1.0)
    
    # Apply margin (set edges to zero)
    margin = 16
    density = density.at[:margin, :].set(0.0)
    density = density.at[-margin:, :].set(0.0)
    density = density.at[:, :margin].set(0.0)
    density = density.at[:, -margin:].set(0.0)
    
    return density


# =============================================================================
# Spectral Helper Functions
# =============================================================================

def vorticity_to_velocity(omega_hat):
    """
    Convert vorticity to velocity via streamfunction.
    ω = -∇²ψ  =>  ψ_hat = -ω_hat / |k|²
    v_x = ∂ψ/∂y, v_y = -∂ψ/∂x
    
    Returns velocity in physical space.
    """
    psi_hat = -omega_hat / K_SQ_safe
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
    Advect passive scalar by one time step with optional buoyancy.
    ∂ρ/∂t = -(v·∇)ρ + κ∇²ρ
    
    PhiFlow uses pure advection (κ=0) for sharp filaments.
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
    
    PhiFlow order:
    1. Update vorticity/velocity (with actuators)
    2. Add buoyancy to velocity before recomputing vorticity
    3. Advect density
    """
    dt = fixed_dt
    
    # Get current velocity
    vx, vy = vorticity_to_velocity(omega_hat)
    
    # Compute advection
    grad_omega_x, grad_omega_y = spectral_gradient(omega_hat)
    advection = -(vx * grad_omega_x + vy * grad_omega_y)
    advection_hat = jfft.rfft2(advection)
    advection_hat = DEALIAS * advection_hat
    
    # Compute forcing curl
    fx, fy = forcing_fn(xi, u)
    curl_f_hat = spectral_curl_z(fx, fy)
    curl_f_hat = DEALIAS * curl_f_hat
    
    # Add buoyancy to velocity (PhiFlow method)
    if use_buoyancy:
        vy_buoyant = vy + BUOYANCY_STRENGTH * rho * dt
        # Recompute vorticity from buoyant velocity
        vx_hat = jfft.rfft2(vx)
        vy_buoyant_hat = jfft.rfft2(vy_buoyant)
        two_pi_i = 2.0 * jnp.pi * 1j
        buoyancy_vorticity_hat = two_pi_i * (KX * vy_buoyant_hat - KY * vx_hat)
        buoyancy_contribution = buoyancy_vorticity_hat - omega_hat
    else:
        buoyancy_contribution = 0.0
    
    # Explicit terms
    explicit_terms = advection_hat + curl_f_hat + buoyancy_contribution
    
    # Semi-implicit update
    lhs_coeff = 1.0 - 0.5 * dt * LINEAR_TERM
    rhs = (1.0 + 0.5 * dt * LINEAR_TERM) * omega_hat + dt * explicit_terms
    omega_hat_new = rhs / lhs_coeff
    
    # Update actuator positions
    xi_new = xi + dt * v
    xi_new = xi_new % L
    
    # Advect density
    rho_new = advect_density_step(rho, omega_hat_new)
    
    return omega_hat_new, rho_new, xi_new


# =============================================================================
# Trajectory Solver
# =============================================================================

@jit
def solve_trajectory(omega_hat_init, rho_init, xi_init, controls_u, controls_v):
    """
    Solve the PDE trajectory given initial conditions and control sequence.
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
    """
    Solve PDE with multi-scale initialization.
    
    Uses PhiFlow method: multi-scale random velocity field.
    """
    key = jax.random.PRNGKey(42)
    
    # Randomize v_scale per scene (PhiFlow does this)
    key, k_vel = jax.random.split(key)
    v_scale = V_SCALE_BASE * (1.0 + (jax.random.uniform(k_vel) - 0.5) * 2.0)
    
    # Generate multi-scale velocity
    key, k_vel2 = jax.random.split(key)
    velocity_field = multi_scale_velocity_field(k_vel2, N, v_scale, V_FALLOFF)
    
    # Convert velocity to vorticity
    vx = velocity_field[:, :, 0]
    vy = velocity_field[:, :, 1]
    vx_hat = jfft.rfft2(vx)
    vy_hat = jfft.rfft2(vy)
    two_pi_i = 2.0 * jnp.pi * 1j
    omega_hat_init = two_pi_i * (KX * vy_hat - KY * vx_hat)
    
    _, rho_traj, xi_traj = solve_trajectory(omega_hat_init, z_init, xi_init, controls_u, controls_v)
    return rho_traj, xi_traj


# =============================================================================
# Initial Conditions
# =============================================================================

def vortex_dipoles_initial(key, num_dipoles=5, strength=12.0, separation=0.4):
    """
    Create coherent vortex dipoles for strong advection.
    
    NOTE: PhiFlow uses multi_scale_velocity_field instead!
    This is kept for compatibility.
    """
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Random positions for dipole centers
    centers = jax.random.uniform(key1, shape=(num_dipoles, 2), minval=L*0.2, maxval=L*0.8)
    
    # Random orientations
    angles = jax.random.uniform(key2, shape=(num_dipoles,), minval=0, maxval=2*jnp.pi)
    
    # Random strengths (with variation)
    strengths = jax.random.uniform(key3, shape=(num_dipoles,), minval=0.7, maxval=1.3) * strength
    
    omega = jnp.zeros_like(xx)
    sigma_vortex = 0.18
    
    for i in range(num_dipoles):
        cx, cy = centers[i, 0], centers[i, 1]
        angle = angles[i]
        s = strengths[i]
        
        # Positive vortex
        dx_pos = separation * jnp.cos(angle)
        dy_pos = separation * jnp.sin(angle)
        dist_sq_pos = (xx - (cx + dx_pos))**2 + (yy - (cy + dy_pos))**2
        omega_pos = s * jnp.exp(-dist_sq_pos / (2.0 * sigma_vortex**2))
        
        # Negative vortex
        dx_neg = -separation * jnp.cos(angle)
        dy_neg = -separation * jnp.sin(angle)
        dist_sq_neg = (xx - (cx + dx_neg))**2 + (yy - (cy + dy_neg))**2
        omega_neg = -s * jnp.exp(-dist_sq_neg / (2.0 * sigma_vortex**2))
        
        omega = omega + omega_pos + omega_neg
    
    return jfft.rfft2(omega)


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
    
    omega_hat_seed = energy_spectrum * rand_amp * jnp.exp(1j * phases)
    omega_hat_seed = omega_hat_seed.at[0, 0].set(0.0)

    omega = jfft.irfft2(omega_hat_seed)
    omega = omega - jnp.mean(omega)
    omega = omega / (jnp.std(omega) + 1e-12)
    omega = amplitude * omega
    return jfft.rfft2(omega)


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