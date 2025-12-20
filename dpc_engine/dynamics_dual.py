import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract
from tesseracts.solverV1.solver import solve_pde_trajectory, implicit_step

class PDEDynamics:
    def __init__(self, solver_ts, use_tesseract=True):
        """
        Initializes the dynamics wrapper.
        
        Args:
            solver_ts: The loaded Tesseract solver object.
            use_tesseract: Boolean flag. If False, runs direct JAX code on GPU.
        """
        self.solver_ts = solver_ts
        self.use_tesseract = use_tesseract

    def step(self, z_curr, xi_curr, actions):
        """
        Applies a single step using either Tesseract or direct JAX.
        """
        if self.use_tesseract:
            inputs = {
                "z_init": z_curr,
                "xi_init": xi_curr,
                "u_seq": actions['u'][jnp.newaxis, :], 
                "v_seq": actions['v'][jnp.newaxis, :]
            }
            results = apply_tesseract(self.solver_ts, inputs)
            return results["z_trajectory"][0], results["xi_trajectory"][0]
        else:
            # Call the JAX step function directly
            # This will run on GPU automatically if JAX is configured for it
            return implicit_step(z_curr, xi_curr, actions['u'], actions['v'])

    def unroll(self, z_init, xi_init, u_seq, v_seq):
        """
        Runs the full sequence.
        """
        if self.use_tesseract:
            inputs = {
                "z_init": z_init,
                "xi_init": xi_init,
                "u_seq": u_seq,
                "v_seq": v_seq
            }
            return apply_tesseract(self.solver_ts, inputs)
        else:
            # Direct JAX unroll using lax.scan (as defined in your solver)
            z_traj, xi_traj = solve_pde_trajectory(z_init, xi_init, u_seq, v_seq)
            return {
                "z_trajectory": z_traj,
                "xi_trajectory": xi_traj
            }