# tesseract_api.py
import jax
import jax.numpy as jnp
import solver 

jax.config.update("jax_enable_x64", False)

# Default centers in case the user doesn't provide them (Backward Compatibility)
DEFAULT_CENTERS = jnp.array([0.2, 0.4, 0.6, 0.8])

def apply(inputs):
    """
    The main entry point.
    Expects inputs dict with:
      - 'u_init': array of shape (N,)
      - 'controls': array of shape (T, 4)
      - 'actuator_centers': array of shape (4,) [OPTIONAL]
    """
    u_init = jnp.array(inputs['u_init'])
    controls = jnp.array(inputs['controls'])
    
    # Get centers from input, or use default if missing
    centers = inputs.get('actuator_centers', DEFAULT_CENTERS)
    centers = jnp.array(centers)
    
    trajectory = solver.solve_heat_equation(u_init, controls, centers)
    
    return {"trajectory": trajectory}

# --- Differentiability Endpoints ---

def vector_jacobian_product(inputs, cotangents):
    """
    Computes v^T * J. Enables backprop for controls AND positions.
    """
    u_init = jnp.array(inputs['u_init'])
    controls = jnp.array(inputs['controls'])
    
    # Handle centers
    centers = inputs.get('actuator_centers', DEFAULT_CENTERS)
    centers = jnp.array(centers)
    
    # We define a function that maps (u, c, p) -> outputs
    def forward_fn(u, c, p):
        return solver.solve_heat_equation(u, c, p)

    # Use JAX to compute the VJP
    # We now differentiate w.r.t three arguments
    primal_out, vjp_fn = jax.vjp(forward_fn, u_init, controls, centers)
    
    cotan_traj = jnp.array(cotangents['trajectory'])
    
    # Compute gradients
    grad_u_init, grad_controls, grad_centers = vjp_fn(cotan_traj)
    
    return {
        "u_init": grad_u_init,
        "controls": grad_controls,
        "actuator_centers": grad_centers  # <--- Now returns position gradients!
    }

def jacobian(inputs):
    """
    Computes the full Jacobian matrix.
    """
    u_init = jnp.array(inputs['u_init'])
    controls = jnp.array(inputs['controls'])
    centers = inputs.get('actuator_centers', DEFAULT_CENTERS)
    centers = jnp.array(centers)
    
    def forward_fn(u, c, p):
        return solver.solve_heat_equation(u, c, p)
        
    # argnums=(0, 1, 2) asks for Jacobians of u_init, controls, AND centers
    jac_u, jac_c, jac_p = jax.jacfwd(forward_fn, argnums=(0, 1, 2))(u_init, controls, centers)
    
    return {
        "u_init": jac_u,
        "controls": jac_c,
        "actuator_centers": jac_p
    }