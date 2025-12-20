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
from models.policy_mock import init_mock_policy_params, mock_policy_forward

def test_modes():
    # Load the Tesseract solver (used for both modes if initialized)
    solver_ts = Tesseract.from_image("solver_v1")
    
    # 1. Setup dimensions
    num_spatial_points = 100
    num_agents = 4
    key = jax.random.PRNGKey(42)

    # 2. Initialize Policy Params
    params = init_mock_policy_params(num_spatial_points, num_agents, key)
    
    # Initial State
    z_init = jnp.zeros(num_spatial_points)
    xi_init = jnp.array([0.2, 0.4, 0.6, 0.8])

    # We iterate through both modes to compare outputs and gradients
    modes = [True, False]
    results = {}

    with solver_ts:
        for use_ts in modes:
            mode_name = "TESSERACT" if use_ts else "NATIVE JAX (GPU)"
            print(f"\n{'='*20} Testing Mode: {mode_name} {'='*20}")
            
            # Initialize Dynamics with the specific flag
            model = PDEDynamics(solver_ts, use_tesseract=use_ts)

            # 3. Policy Inference
            actions = mock_policy_forward(params, z_init, xi_init)
            
            # 4. Physics Step
            z_next, xi_next = model.step(z_init, xi_init, actions)
            
            print(f"Position 0 change: {xi_init[0]:.6f} -> {xi_next[0]:.6f}")
            print(f"Max field intensity: {jnp.max(z_next):.6e}")

           # 5. Verify Differentiability
            def simple_loss(p):
                acts = mock_policy_forward(p, z_init, xi_init)
                zn, _ = model.step(z_init, xi_init, acts)
                return jnp.sum(zn**2)

            grad_fn = jax.grad(simple_loss)
            grads = grad_fn(params)
            
            # Flatten the gradient tree to calculate a global mean
            flat_grads, _ = jax.tree_util.tree_flatten(grads)
            mean_grad = jnp.mean(jnp.array([jnp.mean(g) for g in flat_grads]))
            
            print(f"Mean gradient across all params: {mean_grad:.8e}")
            
            # Store results for a quick comparison at the end
            results[mode_name] = {
                "z_next": z_next,
                "xi_next": xi_next,
                "grad": mean_grad
            }

    # 6. Final Comparison
    print(f"\n{'='*20} Final Comparison {'='*20}")
    diff_z = jnp.abs(results["TESSERACT"]["z_next"] - results["NATIVE JAX (GPU)"]["z_next"]).max()
    diff_grad = abs(results["TESSERACT"]["grad"] - results["NATIVE JAX (GPU)"]["grad"])
    
    print(f"Max Field Difference: {diff_z:.8e}")
    print(f"Gradient Difference:  {diff_grad:.8e}")
    
    if diff_z < 1e-5:
        print("\nSUCCESS: Both modes produce consistent physics results.")
    else:
        print("\nWARNING: Numerical mismatch detected between modes.")

if __name__ == "__main__":
    test_modes()