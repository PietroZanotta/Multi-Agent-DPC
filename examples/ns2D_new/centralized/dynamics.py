import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from tesseract_jax import apply_tesseract
from tesseracts.solverNS_shape_centralized import solver as ns_solver
from jax.flatten_util import ravel_pytree

class PDEDynamics:
    def __init__(self, solver_ts, policy_apply_fn, use_tesseract=True):
        self.solver_ts = solver_ts
        self.policy_apply_fn = policy_apply_fn
        self.use_tesseract = use_tesseract
        self.dt = ns_solver.fixed_dt

    def unroll_controlled(self, omega_init, rho_init, rho_target, xi_init, params, t_steps):
        """Performs a FULL controlled simulation (Closed-Loop) in ONE call."""
        if self.use_tesseract:
            # 1. Flatten the PyTree into a 1D vector for Tesseract serialization
            flat_params, _ = ravel_pytree(params)
            
            # 2. Prepare inputs matching the new InputSchema in tesseract_api.py
            inputs = {
                "omega_init": omega_init.astype(jnp.float64),
                "rho_init": rho_init.astype(jnp.float64),
                "rho_target": rho_target.astype(jnp.float64),
                "xi_init": xi_init.astype(jnp.float64),
                "flat_params": flat_params.astype(jnp.float64),
                "t_steps": t_steps
            }
            
            # 3. Call remote/accelerated solver
            results = apply_tesseract(self.solver_ts, inputs)
            
            return (
                results["omega_trajectory"], 
                results["rho_trajectory"], 
                results["xi_trajectory"],
                results["u_trajectory"], 
                results["v_trajectory"]
            )
        else:
            # Native JAX implementation
            # Convert physical vorticity to spectral for the native solver
            omega_hat_init = jfft.rfft2(omega_init)
            
            # solve_with_policy returns (o_hat_traj, r_traj, xi_traj, u_traj, v_traj)
            res = ns_solver.solve_with_policy(
                omega_hat_init, 
                rho_init, 
                rho_target, 
                xi_init, 
                params, 
                self.policy_apply_fn, 
                t_steps
            )
            
            # Convert spectral vorticity trajectory back to physical space
            omega_trajectory = jfft.irfft2(res[0])
            
            return (
                omega_trajectory, 
                res[1], # rho_trajectory
                res[2], # xi_trajectory
                res[3], # u_trajectory
                res[4]  # v_trajectory
            )

def sample_initial_vorticity(key, N, V_SCALE_BASE=0.5, V_FALLOFF=0.8):
    """Generates initial turbulent vorticity field using multi-scale velocity."""
    velocity = ns_solver.multi_scale_velocity_field(
        key, resolution=N, v_scale=V_SCALE_BASE, v_falloff=V_FALLOFF
    )
    vx = velocity[:, :, 0]
    vy = velocity[:, :, 1]
    
    vx_hat = jnp.fft.rfft2(vx)
    vy_hat = jnp.fft.rfft2(vy)
    
    kx = jnp.fft.fftfreq(N, d=ns_solver.L/N)
    ky = jnp.fft.rfftfreq(N, d=ns_solver.L/N)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    
    two_pi_i = 2.0 * jnp.pi * 1j
    omega_hat = two_pi_i * (KX * vy_hat - KY * vx_hat)
    
    return jnp.fft.irfft2(omega_hat)