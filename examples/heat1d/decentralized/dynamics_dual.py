"""
Wrapper for Decentralized 1D Heat Equation Dynamics using Tesseract-JAX
Enables controlled simulations via a ControlNet policy, either through Tesseract runtime or native JAX (for fast prototyping).
""" 
import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract
from tesseracts.solverHeat_decentralized import solver 
from jax.flatten_util import ravel_pytree

class PDEDynamics:
    def __init__(self, solver_ts, policy_apply_fn, use_tesseract=True):
        self.solver_ts = solver_ts
        self.policy_apply_fn = policy_apply_fn
        self.use_tesseract = use_tesseract

    def unroll_controlled(self, z_init, xi_init, z_target, params, t_steps):
        """Performs a FULL controlled simulation in ONE call."""
        if self.use_tesseract:
            # Flatten the PyTree into a 1D vector before sending
            flat_params, _ = ravel_pytree(params)
            
            inputs = {
                "z_init": z_init,
                "xi_init": xi_init,
                "z_target": z_target,
                "flat_params": flat_params, 
                "t_steps": t_steps
            }
            
            results = apply_tesseract(self.solver_ts, inputs)
            return (
                results["z_trajectory"], 
                results["xi_trajectory"], 
                results["u_trajectory"], 
                results["v_trajectory"]
            )
        else:
            # Native JAX handles the dict 'params' directly
            return solver.solve_with_policy(
                z_init, xi_init, z_target, params, 
                self.policy_apply_fn, t_steps
            )