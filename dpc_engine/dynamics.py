# dynamics.py
import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract

class PDEDynamics:
    def __init__(self, solver_ts):
        """
        Initializes the dynamics wrapper with a loaded Tesseract solver.
        """
        self.solver_ts = solver_ts

    def step(self, z_curr, xi_curr, actions):
        """
        Applies a single RK4 step using the policy's actions.
        
        Args:
            z_curr: Current PDE field (N,)
            xi_curr: Current actuator positions (Num_Agents,)
            actions: Dictionary or array containing:
                     - 'u': Intensities (Num_Agents,)
                     - 'v': Velocities (Num_Agents,)
        
        Returns:
            z_next, xi_next
        """
        # We need to format the inputs for the Tesseract solver.
        # Even for a single step, many solvers expect a sequence dimension (T=1).
        # We wrap them in [None, ...] to add that dimension.
        
        inputs = {
            "z_init": z_curr,
            "xi_init": xi_curr,
            "u_seq": actions['u'][jnp.newaxis, :], 
            "v_seq": actions['v'][jnp.newaxis, :]
        }
        
        # Apply the differentiable solver
        results = apply_tesseract(self.solver_ts, inputs)
        
        # Extract the first (and only) step from the trajectory output
        z_next = results["z_trajectory"][0]
        xi_next = results["xi_trajectory"][0]
        
        return z_next, xi_next

    def unroll(self, z_init, xi_init, u_seq, v_seq):
        """
        Utility to run the full sequence. 
        This is what your Loss function will call during training.
        """
        inputs = {
            "z_init": z_init,
            "xi_init": xi_init,
            "u_seq": u_seq,
            "v_seq": v_seq
        }
        return apply_tesseract(self.solver_ts, inputs)