import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract
from jax.flatten_util import ravel_pytree

import tesseracts.solverFKPP_centralized.solver as solver 

class PDEDynamics:
    def __init__(self, solver_ts, policy_apply_fn, use_tesseract=True):
        """
        Initializes the dynamics wrapper for Centralized 1D Fisher-KPP.
        
        Args:
            solver_ts: The loaded Tesseract solver object.
            policy_apply_fn: The .apply method of your ControlNet.
            use_tesseract: If False, runs direct JAX code (solver.py).
        """
        self.solver_ts = solver_ts
        self.policy_apply_fn = policy_apply_fn
        self.use_tesseract = use_tesseract

    def unroll_controlled(self, z_init, xi_init, z_target, params, t_steps):
        """
        Performs a FULL controlled FKPP simulation in ONE call.
        The policy dictates agent movement and forcing intensities at each step.
        """
        if self.use_tesseract:
            # 1. Flatten the PyTree weights into a 1D vector for Tesseract serialization
            flat_params, _ = ravel_pytree(params)
            
            # 2. Prepare inputs matching the new InputSchema (api.py)
            inputs = {
                "z_init": z_init,
                "xi_init": xi_init,
                "z_target": z_target,
                "flat_params": flat_params, 
                "t_steps": t_steps
            }
            
            # 3. Execute through the Tesseract runtime
            results = apply_tesseract(self.solver_ts, inputs)
            
            # Return all four trajectories as expected by the centralized pattern
            return (
                results["z_trajectory"], 
                results["xi_trajectory"], 
                results["u_trajectory"], 
                results["v_trajectory"]
            )
        else:
            # 4. Native JAX fallback using the solver's internal scan loop
            # Note: Native JAX handles the 'params' dict (PyTree) directly
            return solver.solve_with_policy(
                z_init, 
                xi_init, 
                z_target, 
                params, 
                self.policy_apply_fn, 
                t_steps
            )