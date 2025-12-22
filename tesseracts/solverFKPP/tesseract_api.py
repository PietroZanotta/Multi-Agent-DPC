import sys
import os
import jax
import jax.numpy as jnp
from typing import Any
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

# Assuming the 1D solver is in solver.py
import solver 

# 1. Define the Input Schema
class InputSchema(BaseModel):
    # z_init is now 1D: (N,)
    z_init: Differentiable[Array[(None,), Float32]] = Field(description="Initial state 1D grid")
    # xi_init: (M,) where M is number of agents (scalar positions)
    xi_init: Differentiable[Array[(None,), Float32]] = Field(description="Initial agent scalar positions")
    # u_seq: (Time, M) control sequence (forcing intensity)
    u_seq: Differentiable[Array[(None, None), Float32]] = Field(description="Control sequence (forcing intensity)")
    # v_seq: (Time, M) scalar velocity vectors for agents
    v_seq: Differentiable[Array[(None, None), Float32]] = Field(description="Scalar velocity vectors for agents")

# 2. Define the Output Schema
class OutputSchema(BaseModel):
    # In 1D, spatial dims (N) and time (T) result in (T, N)
    z_trajectory: Differentiable[Array[(None, None), Float32]] = Field(description="Solved z trajectory (Time, N)")
    xi_trajectory: Differentiable[Array[(None, None), Float32]] = Field(description="Solved xi trajectory (Time, M)")

# 3. Apply Function
def apply(inputs: InputSchema) -> OutputSchema:
    # The 1D solver returns (Time, N) and (Time, M)
    z_traj, xi_traj = solver.solve_pde_trajectory(
        inputs.z_init, 
        inputs.xi_init, 
        inputs.u_seq, 
        inputs.v_seq
    )
    
    return OutputSchema(z_trajectory=z_traj, xi_trajectory=xi_traj)

# 4. Abstract Evaluation (Shape Inference)
def abstract_eval(abstract_inputs: InputSchema):
    z_meta = abstract_inputs.z_init
    u_meta = abstract_inputs.u_seq
    xi_meta = abstract_inputs.xi_init

    t_steps = u_meta.shape[0]
    z_dim = z_meta.shape[0]
    num_agents = xi_meta.shape[0]

    return {
        "z_trajectory": ShapeDType(
            shape=(t_steps, z_dim), 
            dtype=z_meta.dtype 
        ),
        "xi_trajectory": ShapeDType(
            shape=(t_steps, num_agents), 
            dtype=xi_meta.dtype
        )
    }

# 5. VJP (Vector-Jacobian Product)
def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    def forward(z_init, xi_init, u_seq, v_seq):
        # returns (Time, N) and (Time, M)
        return solver.solve_pde_trajectory(z_init, xi_init, u_seq, v_seq)

    primal_out, vjp_fn = jax.vjp(
        forward, 
        inputs.z_init, 
        inputs.xi_init, 
        inputs.u_seq, 
        inputs.v_seq
    )
    
    cotan_tuple = (
        cotangent_vector.get("z_trajectory"), 
        cotangent_vector.get("xi_trajectory")
    )
    
    grads = vjp_fn(cotan_tuple)
    
    full_grads = {
        "z_init": grads[0],
        "xi_init": grads[1],
        "u_seq": grads[2],
        "v_seq": grads[3]
    }
    
    return {k: v for k, v in full_grads.items() if k in vjp_inputs}