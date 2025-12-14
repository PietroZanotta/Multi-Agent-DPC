# tesseract_api.py
import jax
import jax.numpy as jnp
import solver # Import your solver code

# Initialize JAX (optional, but good for setup)
jax.config.update("jax_enable_x64", False)

def apply(inputs):
    """
    The main entry point.
    Expects inputs dict with:
      - 'u_init': array of shape (N,)
      - 'controls': array of shape (T, 4)
    """
    u_init = jnp.array(inputs['u_init'])
    controls = jnp.array(inputs['controls'])
    
    trajectory = solver.solve_heat_equation(u_init, controls)
    
    return {"trajectory": trajectory}

# --- Differentiability Endpoints ---

def vector_jacobian_product(inputs, cotangents):
    """
    Computes v^T * J. This enables reverse-mode differentiation (backprop).
    Useful if you have a scalar loss function at the end.
    """
    u_init = jnp.array(inputs['u_init'])
    controls = jnp.array(inputs['controls'])
    
    # We define a function that maps inputs -> outputs (trajectory)
    def forward_fn(u, c):
        return solver.solve_heat_equation(u, c)

    # Use JAX to compute the VJP
    # primal_out is the trajectory, vjp_fn is the backward function
    primal_out, vjp_fn = jax.vjp(forward_fn, u_init, controls)
    
    # cotangents['trajectory'] contains the gradients flowing back from the loss
    cotan_traj = jnp.array(cotangents['trajectory'])
    
    # Compute gradients w.r.t inputs
    grad_u_init, grad_controls = vjp_fn(cotan_traj)
    
    return {
        "u_init": grad_u_init,
        "controls": grad_controls
    }

def jacobian(inputs):
    """
    Computes the full Jacobian matrix.
    Warning: This can be huge (Output_Size x Input_Size).
    Use only for debugging or small T.
    """
    u_init = jnp.array(inputs['u_init'])
    controls = jnp.array(inputs['controls'])
    
    def forward_fn(u, c):
        return solver.solve_heat_equation(u, c)
        
    jac_u, jac_c = jax.jacfwd(forward_fn, argnums=(0, 1))(u_init, controls)
    
    return {
        "u_init": jac_u,
        "controls": jac_c
    }