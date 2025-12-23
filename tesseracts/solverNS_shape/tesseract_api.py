import sys
import os
import jax
from typing import Any
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import solver

# 1. Define the Input Schema
class InputSchema(BaseModel):
    omega_init: Differentiable[Array[(None, None), Float64]] = Field(description="Initial 2D vorticity field")
    rho_init: Differentiable[Array[(None, None), Float64]] = Field(description="Initial 2D density field")
    xi_init: Differentiable[Array[(None, 2), Float64]] = Field(description="Initial 2D actuator positions")
    u_seq: Differentiable[Array[(None, None, 2), Float64]] = Field(description="Control forces over time")
    v_seq: Differentiable[Array[(None, None, 2), Float64]] = Field(description="Actuator velocities over time")

# 2. Define the Output Schema
class OutputSchema(BaseModel):
    omega_trajectory: Differentiable[Array[(None, None, None), Float64]] = Field(description="Vorticity trajectory over time")
    rho_trajectory: Differentiable[Array[(None, None, None), Float64]] = Field(description="Density trajectory over time")
    xi_trajectory: Differentiable[Array[(None, None, 2), Float64]] = Field(description="Actuator trajectory over time")

# 3. Updated Apply Function
def apply(inputs: InputSchema) -> OutputSchema:
    # Convert physical vorticity to spectral vorticity if needed
    # The solver expects omega_hat_init
    omega_hat_init = jax.numpy.fft.rfft2(inputs.omega_init)
    
    omega_hat_traj, rho_traj, xi_traj = solver.solve_trajectory(
        omega_hat_init, 
        inputs.rho_init, 
        inputs.xi_init, 
        inputs.u_seq, 
        inputs.v_seq
    )
    
    # Convert spectral vorticity trajectory back to physical space for output
    omega_traj = jax.numpy.fft.irfft2(omega_hat_traj)
    
    return OutputSchema(
        omega_trajectory=omega_traj,
        rho_trajectory=rho_traj,
        xi_trajectory=xi_traj
    )

# 4. Define Abstract Evaluation (Shape Inference)
def abstract_eval(abstract_inputs: InputSchema):
    omega_meta = abstract_inputs.omega_init
    rho_meta = abstract_inputs.rho_init
    u_meta = abstract_inputs.u_seq
    xi_meta = abstract_inputs.xi_init

    t_steps = u_meta.shape[0]
    nx = rho_meta.shape[0]
    ny = rho_meta.shape[1]
    m_agents = xi_meta.shape[0]

    return {
        "omega_trajectory": ShapeDType(
            shape=(t_steps, nx, ny),
            dtype="float64"
        ),
        "rho_trajectory": ShapeDType(
            shape=(t_steps, nx, ny),
            dtype="float64"
        ),
        "xi_trajectory": ShapeDType(
            shape=(t_steps, m_agents, 2),
            dtype="float64"
        )
    }


# 5. Updated VJP (Vector-Jacobian Product)
def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    def forward(omega_init, rho_init, xi_init, u_seq, v_seq):
        omega_hat_init = jax.numpy.fft.rfft2(omega_init)
        omega_hat_traj, rho_traj, xi_traj = solver.solve_trajectory(
            omega_hat_init, rho_init, xi_init, u_seq, v_seq
        )
        omega_traj = jax.numpy.fft.irfft2(omega_hat_traj)
        return omega_traj, rho_traj, xi_traj

    # Use JAX VJP on the core solver
    primal_out, vjp_fn = jax.vjp(
        forward, 
        inputs.omega_init, 
        inputs.rho_init, 
        inputs.xi_init, 
        inputs.u_seq, 
        inputs.v_seq
    )
    
    # Map cotangent_vector (dict) to the JAX vjp_fn expectations
    cotan_tuple = (
        cotangent_vector.get("omega_trajectory"),
        cotangent_vector.get("rho_trajectory"), 
        cotangent_vector.get("xi_trajectory")
    )
    
    # Ensure cotangents are zeroed if missing
    cotan_tuple = tuple(c if c is not None else jnp.zeros_like(p) for c, p in zip(cotan_tuple, primal_out))
    
    grads = vjp_fn(cotan_tuple)
    
    # Return a dictionary mapping input names to their gradients
    full_grads = {
        "omega_init": grads[0],
        "rho_init": grads[1],
        "xi_init": grads[2],
        "u_seq": grads[3],
        "v_seq": grads[4]
    }
    
    # Filter only the inputs requested in vjp_inputs
    return {k: v for k, v in full_grads.items() if k in vjp_inputs}
