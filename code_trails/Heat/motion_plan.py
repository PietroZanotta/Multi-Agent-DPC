import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import time

import tesseract_api

# --- 2. Configuration ---
N = 100
T_total = 300
Horizon = 30        # INCREASED: Longer lookahead
dt = 0.001

x_grid = jnp.linspace(0, 1, N, endpoint=False)

# MODIFIED: Make target easier to reach (less narrow)
# target_state = 0.8 * jnp.exp(-50 * (x_grid - 0.5)**2)  # Changed from -100 to -50
# target_state = 0.8 * jnp.sin(jnp.pi * x_grid) # works
# target_state = 0.5 * jnp.exp(-20 * (x_grid - 0.5)**2)

def get_smooth_grf_target(x_grid, key):
    """
    Generates a random smooth curve that is 0 at both boundaries.
    It sums the first few sine modes with random weights.
    """
    # 1. Random weights for the first 4 sine modes (Low frequency = Smooth)
    # We decay the power (1/k) so higher frequencies are weaker
    k1, k2, k3, k4 = jax.random.normal(key, (4,))
    
    w1 = k1 * 1.0   # Main arch
    w2 = k2 * 0.5   # Secondary bump
    w3 = k3 * 0.3   # Small details
    w4 = k4 * 0.1   # Tiny details

    # 2. Combine modes
    # sin(n * pi * x) ensures it is exactly 0 at x=0 and x=1
    target = (w1 * jnp.sin(1 * jnp.pi * x_grid) + 
              w2 * jnp.sin(2 * jnp.pi * x_grid) + 
              w3 * jnp.sin(3 * jnp.pi * x_grid) + 
              w4 * jnp.sin(4 * jnp.pi * x_grid))
    
    # 3. Scale to reasonable temperature (e.g., max amplitude ~0.5 to 0.8)
    target = 0.6 * target / (jnp.max(jnp.abs(target)) + 1e-6)
    
    return target

# --- Usage in your main function ---
key = jax.random.PRNGKey(2) # 42 Change seed for different shapes
target_state = get_smooth_grf_target(x_grid, key)

# --- 3. JAX Objective Function ---
def mpc_loss_fn(plan_controls, current_u):
    """
    Evaluates a candidate plan (Horizon, 4) starting from current_u.
    """
    # 1. Predict Future using Tesseract API
    inputs = {
        'u_init': current_u,
        'controls': plan_controls
    }
    results = tesseract_api.apply(inputs)
    trajectory = results['trajectory']  # Shape: (Horizon, N)
    final_pred_u = trajectory[-1]
    
    # 2. Cost Terms
    # a. Terminal Cost (Hit the target at the end)
    terminal_cost = jnp.mean((final_pred_u - target_state) ** 2)
    
    # b. ENABLED: Energy Cost (Prevent excessive control)
    # Scale this appropriately - too large and it won't reach target
   # energy_cost = 0.001 * jnp.sum(plan_controls ** 2)
    
    # c. ENABLED: Smoothness (Prevent control flickering)
   # smoothness_cost = 0.01 * jnp.sum((plan_controls[1:] - plan_controls[:-1])**2)
    
    # d. ADDED: Tracking cost (encourage progress throughout trajectory)
    # This helps the optimizer find better intermediate states
    #tracking_cost = 0.0001 * jnp.mean((trajectory - target_state[None, :])**2)
    
    total_cost = terminal_cost #+ energy_cost + smoothness_cost + tracking_cost
    
    return total_cost

# Compile the value_and_grad function once
jit_loss_and_grad = jit(value_and_grad(mpc_loss_fn))

# --- 4. SciPy Bridge ---
def scipy_objective(flat_plan, current_u, shape):
    """
    Wraps JAX for SciPy.
    """
    # Reshape Flat Numpy -> Shaped JAX
    plan = jnp.array(flat_plan.reshape(shape))
    
    # Compute Loss & Gradients
    loss_val, grad_val = jit_loss_and_grad(plan, current_u)
    
    # Convert JAX -> Flat Numpy
    return float(loss_val), np.array(grad_val).astype(np.float64).flatten()

# --- 5. Main MPC Loop ---
def main():
    print(f"Starting SciPy MPC (Horizon={Horizon})...")
    
    u_current = jnp.zeros(N)
    
    # Initialize Plan (Warm Start Buffer)
    current_plan = np.zeros((Horizon, 4))
    
    history_u = [np.array(u_current)]
    history_c = []
    history_loss = []
    
    start_time = time.time()

    for t in range(T_total):
        
        # --- A. PLANNING (Inner Loop) ---
        obj_fun = lambda x: scipy_objective(x, u_current, (Horizon, 4))
        
        # IMPROVED: More iterations early on, fewer later (adaptive)
        # Early timesteps need more optimization since warm start is poor
        if t < 50:
            max_iter = 30
        elif t < 150:
            max_iter = 20
        else:
            max_iter = 15
        
        # Run L-BFGS-B
        res = scipy.optimize.minimize(
            fun=obj_fun,
            x0=current_plan.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': 1e-6,  # ADDED: Convergence tolerance
                'gtol': 1e-5,  # ADDED: Gradient tolerance
                'disp': False
            }
        )
        
        # Check if optimization succeeded
        if not res.success and t % 50 == 0:
            print(f"  Warning at t={t}: {res.message}")
        
        # Extract optimal plan
        optimized_plan = res.x.reshape(Horizon, 4)
        
        # --- B. EXECUTION ---
        # Take the first action
        action_t = optimized_plan[0]
        
        # Apply to Physics (Real Step)
        step_inputs = {'u_init': u_current, 'controls': jnp.array(action_t)[None, :]}
        step_res = tesseract_api.apply(step_inputs)
        u_next = step_res['trajectory'][0]  # Get first (and only) timestep
        
        # --- C. UPDATE ---
        history_u.append(np.array(u_next))
        history_c.append(action_t)
        history_loss.append(res.fun)
        u_current = u_next
        
        # --- D. WARM START ---
        # Shift the plan left
        next_plan = np.zeros_like(optimized_plan)
        next_plan[:-1] = optimized_plan[1:]
        next_plan[-1] = optimized_plan[-1]  # Repeat last action
        current_plan = next_plan

        if t % 50 == 0:
            current_error = np.mean((np.array(u_current) - np.array(target_state))**2)
            print(f"Time {t:03d} | Loss: {res.fun:.6f} | Current Error: {current_error:.6f} | Iters: {res.nit}")

    print(f"MPC Finished in {time.time() - start_time:.2f}s")

    # --- 6. Visualization ---
    history_u = np.array(history_u)
    history_c = np.array(history_c)
    history_loss = np.array(history_loss)
    
    # Final error
    final_error = np.mean((history_u[-1] - np.array(target_state))**2)
    print(f"\nFinal MSE: {final_error:.6f}")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: State evolution over time (heatmap)
    plt.subplot(2, 3, 1)
    plt.imshow(history_u.T, aspect='auto', origin='lower', extent=[0, T_total, 0, 1])
    plt.colorbar(label='State value')
    plt.xlabel('Time step')
    plt.ylabel('Spatial position')
    plt.title('State Evolution (Heatmap)')
    
    # Plot 2: Initial vs Final vs Target
    plt.subplot(2, 3, 2)
    plt.plot(x_grid, history_u[0], 'k--', label="Start", linewidth=2)
    plt.plot(x_grid, target_state, 'r-', linewidth=3, label="Target")
    plt.plot(x_grid, history_u[-1], 'b-', linewidth=2, label="Final Result")
    plt.plot(x_grid, history_u[T_total//2], 'g:', linewidth=2, label=f"Mid (t={T_total//2})")
    plt.title("State Comparison")
    plt.xlabel("Spatial position")
    plt.ylabel("State value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Control Actions over time
    plt.subplot(2, 3, 3)
    for i in range(4):
        plt.plot(history_c[:, i], label=f"Actuator {i+1}", alpha=0.7)
    plt.title("Control Actions")
    plt.xlabel("Time step")
    plt.ylabel("Control value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss over time
    plt.subplot(2, 3, 4)
    plt.semilogy(history_loss)
    plt.title("Optimization Loss vs Time")
    plt.xlabel("Time step")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Error over time
    plt.subplot(2, 3, 5)
    errors = [np.mean((u - np.array(target_state))**2) for u in history_u]
    plt.semilogy(errors)
    plt.title("Tracking Error vs Time")
    plt.xlabel("Time step")
    plt.ylabel("MSE (log scale)")
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Control effort
    plt.subplot(2, 3, 6)
    effort = np.sum(history_c**2, axis=1)
    plt.plot(effort)
    plt.title("Control Effort")
    plt.xlabel("Time step")
    plt.ylabel("Sum of squared controls")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mpc_results.png', dpi=150)
    plt.show()
    
    return history_u, history_c, history_loss

if __name__ == "__main__":
    main()