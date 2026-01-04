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
from models.policy import Heat2DControlNet

# --- 0. Setup Model & Flattening Logic ---
POLICY_MODEL = Heat2DControlNet(features=(16, 32))

# Initialize once to capture the parameter structure
# The weights structure is independent of the number of agents
# NOTE: Grid size must match training (32×32 for default, can be changed to 64×64 if needed)
_DUMMY_PARAMS = POLICY_MODEL.init(
    jax.random.PRNGKey(0),
    jnp.zeros((32, 32)),   # 2D State (changed from 64 to match training default)
    jnp.zeros((32, 32)),   # 2D Target (changed from 64 to match training default)
    jnp.zeros((1, 2))      # 1 Agent template with 2D position
)
_INITIAL_FLAT, _UNFLATTEN_FN = ravel_pytree(_DUMMY_PARAMS)
_PARAM_SIZE = _INITIAL_FLAT.size

# --- 1. Define the Input Schema ---
class InputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    z_init: Differentiable[Array[(None, None), Float32]]      # 2D initial state
    xi_init: Differentiable[Array[(None, 2), Float32]]        # 2D actuator positions
    z_target: Differentiable[Array[(None, None), Float32]]    # 2D target state

    # 1D Flattened weight vector for clean serialization
    flat_params: Differentiable[Array[(_PARAM_SIZE,), Float32]] = Field(
        description="Flattened Heat2D Centralized NN weight vector"
    )

    t_steps: int = Field(default=300)

# --- 2. Define the Output Schema ---
class OutputSchema(BaseModel):
    z_trajectory: Differentiable[Array[(None, None, None), Float32]]   # (T, N, N)
    xi_trajectory: Differentiable[Array[(None, None, 2), Float32]]     # (T, M, 2)
    u_trajectory: Differentiable[Array[(None, None), Float32]]         # (T, M)
    v_trajectory: Differentiable[Array[(None, None, 2), Float32]]      # (T, M, 2)

# --- 3. Apply Function ---
def apply(inputs: InputSchema) -> OutputSchema:
    # 1. Reconstruct PyTree from flattened params
    params = _UNFLATTEN_FN(inputs.flat_params)

    # 2. Run 2D Heat Solver Logic
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

# --- 4. Abstract Evaluation (Dynamic Agent Support) ---
def abstract_eval(abstract_inputs: InputSchema):
    t_steps = abstract_inputs.t_steps
    n_x, n_y = abstract_inputs.z_init.shape
    n_agents = abstract_inputs.xi_init.shape[0]  # Detect agents from input

    return {
        "z_trajectory": ShapeDType(shape=(t_steps, n_x, n_y), dtype="float32"),
        "xi_trajectory": ShapeDType(shape=(t_steps, n_agents, 2), dtype="float32"),
        "u_trajectory": ShapeDType(shape=(t_steps, n_agents), dtype="float32"),
        "v_trajectory": ShapeDType(shape=(t_steps, n_agents, 2), dtype="float32"),
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