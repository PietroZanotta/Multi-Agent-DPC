import sys
import os
import jax
import jax.numpy as jnp
from typing import Any
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import solver

# 1. Define the Input Schema
class InputSchema(BaseModel):
    z_init: Differentiable[Array[(None,), Float32]] = Field(description="Initial state z")
    xi_init: Differentiable[Array[(None,), Float32]] = Field(description="Initial agent states xi")
    u_seq: Differentiable[Array[(None, None), Float32]] = Field(description="Control sequence u")
    v_seq: Differentiable[Array[(None, None), Float32]] = Field(description="Control sequence v")

# 2. Define the Output Schema
class OutputSchema(BaseModel):
    z_trajectory: Differentiable[Array[(None, None), Float32]] = Field(description="Solved z trajectory")
    xi_trajectory: Differentiable[Array[(None, None), Float32]] = Field(description="Solved xi trajectory")

# 3. Updated Apply Function
def apply(inputs: InputSchema) -> OutputSchema:
    z_traj, xi_traj = solver.solve_pde_trajectory(
        inputs.z_init, 
        inputs.xi_init, 
        inputs.u_seq, 
        inputs.v_seq
    )
    return OutputSchema(z_trajectory=z_traj, xi_trajectory=xi_traj)

# 4. Define Abstract Evaluation (Shape Inference)
# def abstract_eval(abstract_inputs: InputSchema):
#     # Determine shapes based on input dimensions
#     # For example, if solve_pde_trajectory returns (Time, State)
#     t_steps = abstract_inputs.u_seq.shape[0]
#     z_dim = abstract_inputs.z_init.shape[0]
#     xi_dim = abstract_inputs.xi_init.shape[0]

#     return {
#         "z_trajectory": ShapeDType(shape=(t_steps, z_dim), dtype=jnp.float32),
#         "xi_trajectory": ShapeDType(shape=(t_steps, xi_dim), dtype=jnp.float32)
#     }
def abstract_eval(abstract_inputs: InputSchema):
    """
    Calculate output shape dynamically using the metadata 
    passed in via abstract_inputs.
    """
    # These are ShapeDType objects provided by the runtime
    z_meta = abstract_inputs.z_init
    u_meta = abstract_inputs.u_seq
    xi_meta = abstract_inputs.xi_init

    # Extract dimensions from the shapes
    t_steps = u_meta.shape[0]
    z_dim = z_meta.shape[0]
    xi_dim = xi_meta.shape[0]

    return {
        "z_trajectory": ShapeDType(
            shape=(t_steps, z_dim), 
            # Use the string dtype already present in the input metadata
            dtype=z_meta.dtype 
        ),
        "xi_trajectory": ShapeDType(
            shape=(t_steps, xi_dim), 
            dtype=xi_meta.dtype
        )
    }


# 5. Updated VJP (Vector-Jacobian Product)
def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    def forward(z_init, xi_init, u_seq, v_seq):
        return solver.solve_pde_trajectory(z_init, xi_init, u_seq, v_seq)

    # Use JAX VJP on the core solver
    primal_out, vjp_fn = jax.vjp(
        forward, 
        inputs.z_init, 
        inputs.xi_init, 
        inputs.u_seq, 
        inputs.v_seq
    )
    
    # Map cotangent_vector (dict) to the JAX vjp_fn expectations
    # We assume cotangents are provided for 'z_trajectory' and 'xi_trajectory'
    cotan_tuple = (
        cotangent_vector.get("z_trajectory"), 
        cotangent_vector.get("xi_trajectory")
    )
    
    grads = vjp_fn(cotan_tuple)
    
    # Return a dictionary mapping input names to their gradients
    full_grads = {
        "z_init": grads[0],
        "xi_init": grads[1],
        "u_seq": grads[2],
        "v_seq": grads[3]
    }
    
    # Filter only the inputs requested in vjp_inputs
    return {k: v for k, v in full_grads.items() if k in vjp_inputs}