import sys
import os
import jax
import jax.numpy as jnp
from typing import Any
from pydantic import BaseModel, Field, ConfigDict
from jax.flatten_util import ravel_pytree
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import solver
from models.policy import ControlNet 

# --- 0. Model Initialization & Flattening ---
POLICY_MODEL = ControlNet(features=(64, 64))

_DUMMY_PARAMS = POLICY_MODEL.init(
    jax.random.PRNGKey(0), 
    jnp.zeros((100,)), # State (N_grid)
    jnp.zeros((100,)), # Target (N_grid)
    jnp.zeros((1,))    # 1 Agent template
)
_INITIAL_FLAT, _UNFLATTEN_FN = ravel_pytree(_DUMMY_PARAMS)
_PARAM_SIZE = _INITIAL_FLAT.size

# --- 1. Define the Input Schema ---
class InputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    z_init: Differentiable[Array[(None,), Float32]] = Field(description="Initial PDE state")
    xi_init: Differentiable[Array[(None,), Float32]] = Field(description="Initial agent positions")
    z_target: Differentiable[Array[(None,), Float32]] = Field(description="Target PDE state")
    
    flat_params: Differentiable[Array[(_PARAM_SIZE,), Float32]] = Field(
        description="Flattened NN weight vector"
    )
    
    t_steps: int = Field(default=300)

# --- 2. Define the Output Schema ---
class OutputSchema(BaseModel):
    z_trajectory: Differentiable[Array[(None, None), Float32]]
    xi_trajectory: Differentiable[Array[(None, None), Float32]]
    u_trajectory: Differentiable[Array[(None, None), Float32]]
    v_trajectory: Differentiable[Array[(None, None), Float32]]

# --- 3. Apply Function ---
def apply(inputs: InputSchema) -> OutputSchema:
    params = _UNFLATTEN_FN(inputs.flat_params)
    
    # Solve using the policy-integrated FKPP solver
    z_traj, xi_traj, u_traj, v_traj = solver.solve_with_policy(
        inputs.z_init, 
        inputs.xi_init, 
        inputs.z_target,
        params, 
        POLICY_MODEL.apply, 
        inputs.t_steps
    )
    
    return OutputSchema(
        z_trajectory=z_traj, 
        xi_trajectory=xi_traj,
        u_trajectory=u_traj,
        v_trajectory=v_traj
    )

# --- 4. Abstract Evaluation ---
def abstract_eval(abstract_inputs: InputSchema):
    t_steps = abstract_inputs.t_steps
    z_dim = abstract_inputs.z_init.shape[0]
    n_agents = abstract_inputs.xi_init.shape[0]

    return {
        "z_trajectory": ShapeDType(shape=(t_steps, z_dim), dtype="float32"),
        "xi_trajectory": ShapeDType(shape=(t_steps, n_agents), dtype="float32"),
        "u_trajectory": ShapeDType(shape=(t_steps, n_agents), dtype="float32"),
        "v_trajectory": ShapeDType(shape=(t_steps, n_agents), dtype="float32"),
    }

# --- 5. VJP Function ---
def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    def forward(z_i, xi_i, z_t, p_flat):
        p_tree = _UNFLATTEN_FN(p_flat)
        return solver.solve_with_policy(
            z_i, xi_i, z_t, p_tree, POLICY_MODEL.apply, inputs.t_steps
        )

    primal_out, vjp_fn = jax.vjp(
        forward, 
        inputs.z_init, 
        inputs.xi_init, 
        inputs.z_target, 
        inputs.flat_params
    )
    
    cotan_tuple = (
        cotangent_vector.get("z_trajectory"), 
        cotangent_vector.get("xi_trajectory"),
        cotangent_vector.get("u_trajectory"),
        cotangent_vector.get("v_trajectory")
    )
    
    grads = vjp_fn(cotan_tuple)
    
    return {
        "z_init": grads[0],
        "xi_init": grads[1],
        "z_target": grads[2],
        "flat_params": grads[3]
    }