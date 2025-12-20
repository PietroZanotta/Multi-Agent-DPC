# test_grad.py
import jax
import jax.numpy as jnp
import tesseract_api

# Create dummy data
u_init = jnp.zeros(100)
controls = jnp.zeros((400, 4)) # T=400, 4 centers

# Define a scalar loss function on the output
def loss_fn(u, c):
    traj = tesseract_api.solver.solve_heat_equation(u, c)
    # Goal: Maximize temperature at center at final time
    return -jnp.sum(traj[-1, 45:55]) 

# Compute gradient w.r.t controls
grads = jax.grad(loss_fn, argnums=1)(u_init, controls)

print("Gradient shape:", grads.shape)
print("Gradient norm:", jnp.linalg.norm(grads))
print("If norm > 0, backprop is working!")