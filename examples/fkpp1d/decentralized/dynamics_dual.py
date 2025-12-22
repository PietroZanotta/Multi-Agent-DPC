import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract
from tesseracts.solverFKPP.solver import solve_pde_trajectory, fkpp_step_1d as fkpp_step

class PDEDynamics:
    def __init__(self, solver_ts, use_tesseract=True, N=100):
        """
        Initializes the dynamics wrapper for 1D Fisher-KPP.
        
        Args:
            solver_ts: The loaded Tesseract solver object.
            use_tesseract: If False, runs direct JAX code (solver.py).
            N: Grid resolution (N).
        """
        self.solver_ts = solver_ts
        self.use_tesseract = use_tesseract
        self.N = N

    def step(self, z_curr, xi_curr, actions):
        """
        Applies a single step. 
        z_curr: (N,) vector
        xi_curr: (M,) scalar positions
        actions: {'u': (M,), 'v': (M,)}
        """
        if self.use_tesseract:
            # Tesseract expects sequences for u and v (Time, ...)
            # Add a time dimension of size 1
            inputs = {
                "z_init": z_curr,
                "xi_init": xi_curr,
                "u_seq": actions['u'][jnp.newaxis, :], 
                "v_seq": actions['v'][jnp.newaxis, :]
            }
            results = apply_tesseract(self.solver_ts, inputs)
            
            # In 1D, z_trajectory is (T, N), so no extra reshaping is needed
            # beyond grabbing the first time step.
            z_next = results["z_trajectory"][0]
            xi_next = results["xi_trajectory"][0]
            
            return z_next, xi_next
        else:
            # Direct JAX call to the single step function
            (z_next, xi_next), _ = fkpp_step((z_curr, xi_curr), actions)
            return z_next, xi_next

    def unroll(self, z_init, xi_init, u_seq, v_seq):
        """
        Runs the full sequence over T steps.
        """
        if self.use_tesseract:
            inputs = {
                "z_init": z_init,
                "xi_init": xi_init,
                "u_seq": u_seq,
                "v_seq": v_seq
            }
            results = apply_tesseract(self.solver_ts, inputs)
            
            return {
                "z_trajectory": results["z_trajectory"],
                "xi_trajectory": results["xi_trajectory"]
            }
        else:
            # Direct JAX unroll using solve_pde_trajectory
            z_traj, xi_traj = solve_pde_trajectory(z_init, xi_init, u_seq, v_seq)
            return {
                "z_trajectory": z_traj,
                "xi_trajectory": xi_traj
            }