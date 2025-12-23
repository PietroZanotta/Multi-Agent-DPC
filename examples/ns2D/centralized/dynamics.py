import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from tesseract_jax import apply_tesseract
from tesseracts.solverNS_shape import solver as ns_solver


class PDEDynamics:
    def __init__(self, solver_ts, use_tesseract=True):
        self.solver_ts = solver_ts
        self.use_tesseract = use_tesseract
        self.dt = ns_solver.fixed_dt

    def step(self, omega, rho, xi, u, v):
        if self.use_tesseract:
            omega64 = omega.astype(jnp.float64)
            rho64 = rho.astype(jnp.float64)
            xi64 = xi.astype(jnp.float64)
            u64 = u.astype(jnp.float64)[jnp.newaxis, ...]
            v64 = v.astype(jnp.float64)[jnp.newaxis, ...]

            inputs = {
                "omega_init": omega64,
                "rho_init": rho64,
                "xi_init": xi64,
                "u_seq": u64,
                "v_seq": v64,
            }
            results = apply_tesseract(self.solver_ts, inputs)
            omega_next = results["omega_trajectory"][0].astype(omega.dtype)
            rho_next = results["rho_trajectory"][0].astype(rho.dtype)
            xi_next = results["xi_trajectory"][0].astype(xi.dtype)
            return omega_next, rho_next, xi_next
        else:
            omega_hat = jfft.rfft2(omega)
            omega_hat_new, rho_new, xi_new = ns_solver.full_step(
                omega_hat, rho, xi, u, v
            )
            omega_new = jfft.irfft2(omega_hat_new)
            return omega_new, rho_new, xi_new


def sample_initial_vorticity(key, N, V_SCALE_BASE=0.5, V_FALLOFF=0.8):
    velocity = ns_solver.multi_scale_velocity_field(key, resolution=N, v_scale=V_SCALE_BASE, v_falloff=V_FALLOFF)
    vx = velocity[:, :, 0]
    vy = velocity[:, :, 1]
    
    # 3. Convert Velocity to Vorticity (Omega) via FFT
    vx_hat = jnp.fft.rfft2(vx)
    vy_hat = jnp.fft.rfft2(vy)
    
    kx = jnp.fft.fftfreq(N, d=ns_solver.L/N)
    ky = jnp.fft.rfftfreq(N, d=ns_solver.L/N)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    
    two_pi_i = 2.0 * jnp.pi * 1j
    omega_hat = two_pi_i * (KX * vy_hat - KY * vx_hat)
    
    # Return to real space
    omega = jnp.fft.irfft2(omega_hat)
    return omega
