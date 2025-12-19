# test_dynamics.py
from tesseract_core import Tesseract
import jax.numpy as jnp
import sys
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))
print(script_dir)

from dpc_engine.dynamics import PDEDynamics

def test():
    solver_ts = Tesseract.from_image("solver_v1")
    
    # Wrap everything that uses apply_tesseract in the context manager
    with solver_ts:
        model = PDEDynamics(solver_ts)

        # Initial state
        z = jnp.zeros(100)
        xi = jnp.array([0.2, 0.4, 0.6, 0.8])
        
        # Fake actions
        actions = {
            'u': jnp.array([0.5, 0.5, 0.5, 0.5]),
            'v': jnp.array([0.01, 0.01, 0.01, 0.01])
        }

        # Take one step
        z_next, xi_next = model.step(z, xi, actions)
        
        print(f"Position updated from {xi[0]:.6f} to {xi_next[0]:.6f}")
        print(f"Max state value: {jnp.max(z_next):.5f}")


if __name__ == "__main__":
    test()