# test_dynamics.py
import jax
import jax.numpy as jnp
import sys
from pathlib import Path
from tesseract_core import Tesseract

# Setup paths
script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from dpc_engine.dynamics import PDEDynamics
# Assuming you saved the mock policy code above
from models.policy_mock import init_mock_policy_params, mock_policy_forward

def test():
    solver_ts = Tesseract.from_image("solver_v1")
    
    # 1. Setup dimensions
    num_spatial_points = 100
    num_agents = 4
    key = jax.random.PRNGKey(42)

    # 2. Initialize Policy and Dynamics
    params = init_mock_policy_params(num_spatial_points, num_agents, key)
    
    with solver_ts:
        model = PDEDynamics(solver_ts)

        # 3. Initial State
        z = jnp.zeros(num_spatial_points)
        xi = jnp.array([0.2, 0.4, 0.6, 0.8])
        
        print("--- Initializing Policy Test ---")
        
        # 4. Policy Inference (The Brain thinks...)
        actions = mock_policy_forward(params, z, xi)
        
        print(f"Policy suggests intensities (u): {actions['u']}")
        print(f"Policy suggests velocities  (v): {actions['v']}")

        # 5. Physics Step (The World reacts...)
        z_next, xi_next = model.step(z, xi, actions)
        
        print("\n--- Results after 1 RK4 Step ---")
        print(f"Position 0: {xi[0]:.6f} -> {xi_next[0]:.6f} (delta: {xi_next[0]-xi[0]:.6f})")
        print(f"Max field intensity: {jnp.max(z_next):.6f}")
        
        # 6. Verify Differentiability (Crucial for DPC!)
        def simple_loss(p):
            # A dummy loss: minimize the sum of the field
            acts = mock_policy_forward(p, z, xi)
            zn, _ = model.step(z, xi, acts)
            return jnp.sum(zn**2)

        grad_fn = jax.grad(simple_loss)
        grads = grad_fn(params)
        print("\n--- Differentiation Check ---")
        print(f"Gradient of loss w.r.t policy weights (mean): {jnp.mean(grads['w']):.8f}")

if __name__ == "__main__":
    test()