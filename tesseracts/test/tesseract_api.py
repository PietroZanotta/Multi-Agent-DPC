# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any  # Added the missing import

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel

from tesseract_core.runtime import Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

#
# Schemata
#

class InputSchema(BaseModel):
    dummy_in: Differentiable[Float32] = 0.0

class OutputSchema(BaseModel):
    dummy_out: Differentiable[Float32]
    compute_time_ms: float
    device_info: str

#
# Logic
#

@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """
    Batched Matrix Multiplication: 
    Multiplies 128 pairs of 2048x2048 matrices in parallel.
    """
    batch_size = 128
    dim = 2048
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # Generate batched tensors: (128, 2048, 2048)
    mats_a = jax.random.normal(k1, (batch_size, dim, dim))
    mats_b = jax.random.normal(k2, (batch_size, dim, dim))
    
    # Use vmap to push all 128 multiplications to the GPU cores at once
    # This is much more "parallel-friendly" than a single large matmul
    results = jax.vmap(jnp.matmul)(mats_a, mats_b)
    
    return {"dummy_out": jnp.sum(results)}

def apply(inputs: InputSchema) -> OutputSchema:
    # 1. Compilation/Warmup 
    # (JAX compiles the HLO graph here; we don't want to time this)
    _ = apply_jit(inputs.model_dump())
    
    # 2. Timed Execution
    start = time.perf_counter()
    out_dict = apply_jit(inputs.model_dump())
    
    # Crucial: JAX is asynchronous. block_until_ready() ensures 
    # the GPU has actually finished before we stop the clock.
    jax.block_until_ready(out_dict)
    end = time.perf_counter()
    
    duration_ms = (end - start) * 1000
    
    current_device = jax.devices()[0]
    device_str = f"{current_device.platform.upper()}: {current_device.device_kind}"

    # Print to console for immediate feedback
    print(f"Device: {device_str}")
    print(f"Compute Time: {duration_ms:.2f} ms")

    return OutputSchema(
        dummy_out=out_dict["dummy_out"],
        compute_time_ms=duration_ms,
        device_info=device_str
    )

#
# Jax-handled AD endpoints
#

def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))

def jacobian_vector_product(inputs: InputSchema, jvp_inputs: set[str], jvp_outputs: set[str], tangent_vector: dict[str, Any]):
    return jvp_jit(inputs.model_dump(), tuple(jvp_inputs), tuple(jvp_outputs), tangent_vector)

def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict[str, Any]):
    return vjp_jit(inputs.model_dump(), tuple(vjp_inputs), tuple(vjp_outputs), cotangent_vector)

def abstract_eval(abstract_inputs):
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(jaxified_inputs, filter_spec=is_shapedtype_struct)

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    
    # Metadata for abstract evaluation
    jax_shapes["compute_time_ms"] = 0.0
    jax_shapes["device_info"] = "string"

    return jax.tree.map(
        lambda x: ({"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )

#
# Helper functions
#

@eqx.filter_jit
def jac_jit(inputs: dict, jac_inputs: tuple[str], jac_outputs: tuple[str]):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(flatten_with_paths(inputs, include_paths=jac_inputs))

@eqx.filter_jit
def jvp_jit(inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(filtered_apply, [flatten_with_paths(inputs, include_paths=jvp_inputs)], [tangent_vector])[1]

@eqx.filter_jit
def vjp_jit(inputs: dict, vjp_inputs: tuple[str], vjp_outputs: tuple[str], cotangent_vector: dict):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs))
    return vjp_func(cotangent_vector)[0]