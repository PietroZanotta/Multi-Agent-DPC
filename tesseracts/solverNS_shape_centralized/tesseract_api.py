import sys
import os
import jax
import jax.numpy as jnp
from typing import Any, Sequence
import flax.linen as nn
from pydantic import BaseModel, Field, ConfigDict
from jax.flatten_util import ravel_pytree
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import solver

# --- 0. Policy Model Definition ---
class NS2DControlNet(nn.Module):
    features: Sequence[int]
    u_max: float = 1.5
    v_max: float = 0.5

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        # 1. Field Processing (Global context)
        error = z_curr - z_target
        x = error[..., None]
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        x = x.reshape(-1)
        x = nn.LayerNorm()(x)
        branch = nn.Dense(32)(x)
        branch = nn.tanh(branch)

        # 2. Agent Position Encoding (Local context)
        freqs = jnp.array([1.0, 2.0, 4.0, 8.0])
        angle = xi_curr[..., None] / (2.0 * jnp.pi) * freqs[None, None, :] * 2.0 * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        encoded = encoded.reshape(xi_curr.shape[0], -1)

        y = encoded
        for feat in [32, 32]:
            y = nn.Dense(feat)(y)
            y = nn.tanh(y)

        # 3. Fusion & Multi-Agent Action Output
        branch_rep = jnp.tile(branch, (xi_curr.shape[0], 1))
        h = jnp.concatenate([branch_rep, y], axis=-1)
        h = nn.Dense(64)(h)
        h = nn.tanh(h)

        u_raw = nn.Dense(2)(h)
        v_raw = nn.Dense(2)(h)

        u = self.u_max * jnp.tanh(u_raw)
        v = self.v_max * jnp.tanh(v_raw)
        return u, v

# --- 1. Model Initialization & Flattening ---
# Grid N=64 is assumed from solver.py
POLICY_MODEL = NS2DControlNet(features=(16, 32)) 
_DUMMY_PARAMS = POLICY_MODEL.init(
    jax.random.PRNGKey(0), 
    jnp.zeros((64, 64)), # z_curr
    jnp.zeros((64, 64)), # z_target
    jnp.zeros((1, 2))    # 1 Agent template [x, y]
)
_INITIAL_FLAT, _UNFLATTEN_FN = ravel_pytree(_DUMMY_PARAMS)
_PARAM_SIZE = _INITIAL_FLAT.size

# --- 2. Define the Input Schema ---
class InputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    omega_init: Differentiable[Array[(None, None), Float64]] = Field(description="Initial 2D vorticity")
    rho_init: Differentiable[Array[(None, None), Float64]] = Field(description="Initial 2D density")
    rho_target: Differentiable[Array[(None, None), Float64]] = Field(description="Target density field")
    xi_init: Differentiable[Array[(None, 2), Float64]] = Field(description="Initial actuator positions")
    
    flat_params: Differentiable[Array[(_PARAM_SIZE,), Float64]] = Field(
        description="Flattened NS2DControlNet weight vector"
    )
    
    t_steps: int = Field(default=100)

# --- 3. Define the Output Schema ---
class OutputSchema(BaseModel):
    omega_trajectory: Differentiable[Array[(None, None, None), Float64]]
    rho_trajectory: Differentiable[Array[(None, None, None), Float64]]
    xi_trajectory: Differentiable[Array[(None, None, 2), Float64]]
    u_trajectory: Differentiable[Array[(None, None, 2), Float64]]
    v_trajectory: Differentiable[Array[(None, None, 2), Float64]]

# --- 4. Apply Function ---
def apply(inputs: InputSchema) -> OutputSchema:
    params = _UNFLATTEN_FN(inputs.flat_params)
    
    # Pre-transform omega to spectral space for the solver
    omega_hat_init = jnp.fft.rfft2(inputs.omega_init)
    
    o_traj_hat, r_traj, xi_traj, u_traj, v_traj = solver.solve_with_policy(
        omega_hat_init,
        inputs.rho_init,
        inputs.rho_target,
        inputs.xi_init,
        params,
        POLICY_MODEL.apply,
        inputs.t_steps
    )
    
    return OutputSchema(
        omega_trajectory=jnp.fft.irfft2(o_traj_hat),
        rho_trajectory=r_traj,
        xi_trajectory=xi_traj,
        u_trajectory=u_traj,
        v_trajectory=v_traj
    )

# --- 5. Abstract Evaluation ---
def abstract_eval(abstract_inputs: InputSchema):
    t_steps = abstract_inputs.t_steps
    nx, ny = abstract_inputs.rho_init.shape
    n_agents = abstract_inputs.xi_init.shape[0]

    return {
        "omega_trajectory": ShapeDType(shape=(t_steps, nx, ny), dtype="float64"),
        "rho_trajectory": ShapeDType(shape=(t_steps, nx, ny), dtype="float64"),
        "xi_trajectory": ShapeDType(shape=(t_steps, n_agents, 2), dtype="float64"),
        "u_trajectory": ShapeDType(shape=(t_steps, n_agents, 2), dtype="float64"),
        "v_trajectory": ShapeDType(shape=(t_steps, n_agents, 2), dtype="float64"),
    }

# --- 6. VJP Function ---
def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    def forward(o_init, r_init, r_target, xi_i, p_flat):
        p_tree = _UNFLATTEN_FN(p_flat)
        o_hat_i = jnp.fft.rfft2(o_init)
        res = solver.solve_with_policy(
            o_hat_i, r_init, r_target, xi_i, p_tree, POLICY_MODEL.apply, inputs.t_steps
        )
        # Convert omega back to physical space for the VJP
        return (jnp.fft.irfft2(res[0]), res[1], res[2], res[3], res[4])

    primal_out, vjp_fn = jax.vjp(
        forward, 
        inputs.omega_init, 
        inputs.rho_init, 
        inputs.rho_target,
        inputs.xi_init, 
        inputs.flat_params
    )
    
    cotan_tuple = (
        cotangent_vector.get("omega_trajectory"), 
        cotangent_vector.get("rho_trajectory"),
        cotangent_vector.get("xi_trajectory"),
        cotangent_vector.get("u_trajectory"),
        cotangent_vector.get("v_trajectory")
    )
    
    # Fill Nones with zeros
    cotan_tuple = tuple(c if c is not None else jnp.zeros_like(p) for c, p in zip(cotan_tuple, primal_out))
    grads = vjp_fn(cotan_tuple)
    
    full_grads = {
        "omega_init": grads[0],
        "rho_init": grads[1],
        "rho_target": grads[2],
        "xi_init": grads[3],
        "flat_params": grads[4]
    }
    
    return {k: v for k, v in full_grads.items() if k in vjp_inputs}