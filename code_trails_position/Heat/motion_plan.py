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
Horizon = 30        # Lookahead horizon
dt = 0.001
NUM_ACTUATORS = 4   # Define clearly for reshaping

x_grid = jnp.linspace(0, 1, N, endpoint=False)

def get_smooth_grf_target(x_grid, key):
    """
    Generates a random smooth curve that is 0 at both boundaries.
    """
    k1, k2, k3, k4 = jax.random.normal(key, (4,))
    
    w1 = k1 * 1.0   
    w2 = k2 * 0.5   
    w3 = k3 * 0.3   
    w4 = k4 * 0.1   

    target = (w1 * jnp.sin(1 * jnp.pi * x_grid) + 
              w2 * jnp.sin(2 * jnp.pi * x_grid) + 
              w3 * jnp.sin(3 * jnp.pi * x_grid) + 
              w4 * jnp.sin(4 * jnp.pi * x_grid))
    
    target = 0.6 * target / (jnp.max(jnp.abs(target)) + 1e-6)
    return target

# --- Usage in your main function ---
key = jax.random.PRNGKey(42) # Changed seed for variety
target_state = get_smooth_grf_target(x_grid, key)

# --- 3. JAX Objective Function ---
def mpc_loss_fn(plan_controls, plan_centers, current_u):
    """
    Args:
        plan_controls: (Horizon, NUM_ACTUATORS)
        plan_centers: (NUM_ACTUATORS,)  <-- Optimizing this now!
        current_u: (N,)
    """
    inputs = {
        'u_init': current_u,
        'controls': plan_controls,
        'actuator_centers': plan_centers # Pass dynamic centers
    }
    
    results = tesseract_api.apply(inputs)
    trajectory = results['trajectory'] # Shape (Horizon, N)
    
    # Cost 1: Terminal Cost (Hit the target at the end of horizon)
    final_pred_u = trajectory[-1]
    terminal_cost = 5 * np.mean((final_pred_u - target_state) ** 2)
    
    # Cost 2: Running Cost (Track target throughout horizon)
    # This helps gradients flow better for the centers
    running_cost = 1 * jnp.mean((trajectory - target_state[None, :]) ** 2)

    # Cost 3: Control Effort (Regularization)
    energy_cost = 5e-5 * jnp.sum(plan_controls ** 2)
    
    return terminal_cost + running_cost + energy_cost

# Differentiate w.r.t argnums 0 (controls) and 1 (centers)
jit_loss_and_grad = jit(value_and_grad(mpc_loss_fn, argnums=(0, 1)))

# --- 4. SciPy Bridge ---
def scipy_objective(flat_vars, current_u, horizon, num_actuators):
    """
    flat_vars contains: [controls (Horizon*num_actuators) ... | centers (num_actuators) ...]
    """
    # 1. Unpack
    split_idx = horizon * num_actuators
    
    flat_controls = flat_vars[:split_idx]
    flat_centers = flat_vars[split_idx:]
    
    plan_controls = jnp.array(flat_controls.reshape(horizon, num_actuators))
    plan_centers = jnp.array(flat_centers)
    
    # 2. Compute Loss & Gradients
    # grads is a tuple: (grad_controls, grad_centers)
    loss_val, (grad_c, grad_p) = jit_loss_and_grad(plan_controls, plan_centers, current_u)
    
    # 3. Repack gradients into a single flat vector
    flat_grad = np.concatenate([
        np.array(grad_c).flatten(),
        np.array(grad_p).flatten()
    ]).astype(np.float64)
    
    return float(loss_val), flat_grad

def main():
    print(f"Starting Co-Design MPC (Horizon={Horizon})...")
    
    u_current = jnp.zeros(N)
    
    # --- Initialize Variables ---
    # 1. Controls Plan (Warm Start Buffer)
    # MODIFICATION 1: Initialize with noise instead of zeros to ensure initial gradients
    current_plan = np.random.normal(0, 0.1, size=(Horizon, NUM_ACTUATORS))
    
    # 2. Actuator Centers (Initial Guess)
    # Start them evenly spaced. They will migrate automatically.
    current_centers = np.array([0.2, 0.4, 0.6, 0.8])
    
    history_u = [np.array(u_current)]
    history_c = []
    history_centers = [current_centers.copy()] # Track center movement
    history_loss = []
    
    start_time = time.time()
    
    for t in range(T_total):
        
        # --- A. PACKING ---
        # Combine controls and centers into one vector for SciPy
        x0 = np.concatenate([current_plan.flatten(), current_centers.flatten()])
        
        # --- DEFINE BOUNDS ---
        # Controls: Unbounded (None, None)
        # Centers: [0.01, 0.99] (Stay strictly inside grid to avoid index errors)
        bounds = [(None, None)] * (Horizon * NUM_ACTUATORS) + \
                 [(0.01, 0.99)] * NUM_ACTUATORS
        
        # --- B. OPTIMIZATION ---
        obj_fun = lambda x: scipy_objective(x, u_current, Horizon, NUM_ACTUATORS)
        
        # Adaptive iterations
        if t < 20: max_iter = 50     # Hard work early on
        elif t < 100: max_iter = 20
        else: max_iter = 10
        
        res = scipy.optimize.minimize(
            fun=obj_fun,
            x0=x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-5}
        )
        
        if not res.success and t % 50 == 0:
            print(f"  Warning at t={t}: {res.message}")
        
        # --- C. UNPACKING RESULTS ---
        optimized_x = res.x
        split_idx = Horizon * NUM_ACTUATORS
        
        optimized_plan = optimized_x[:split_idx].reshape(Horizon, NUM_ACTUATORS)
        optimized_centers = optimized_x[split_idx:]
        
        # Update persistent variables
        current_centers = optimized_centers # Centers stay updated!
        
        # --- D. EXECUTION ---
        # Take the first action from the plan
        action_t = optimized_plan[0]
        
        # Apply to Physics (Real Step) using the OPTIMIZED centers
        step_inputs = {
            'u_init': u_current, 
            'controls': jnp.array(action_t)[None, :],
            'actuator_centers': jnp.array(current_centers) 
        }
        step_res = tesseract_api.apply(step_inputs)
        u_next = step_res['trajectory'][0]
        
        # --- E. RECORDING ---
        history_u.append(np.array(u_next))
        history_c.append(action_t)
        history_centers.append(current_centers.copy())
        history_loss.append(res.fun)
        u_current = u_next
        
        # --- F. WARM START & DEFIBRILLATOR ---
        # Shift the plan left
        next_plan = np.zeros_like(optimized_plan)
        next_plan[:-1] = optimized_plan[1:]
        next_plan[-1] = optimized_plan[-1]
        
        # MODIFICATION 2: The Defibrillator
        # If plan is too close to zero (due to energy penalty), inject noise to revive gradients.
        plan_magnitude = np.mean(np.abs(next_plan))
        if plan_magnitude < 0.01:
            noise = np.random.normal(0, 0.05, size=next_plan.shape)
            next_plan += noise
            
        current_plan = next_plan

        if t % 50 == 0:
            current_error = np.mean((np.array(u_current) - np.array(target_state))**2)
            # Print centers to verify they are moving
            print(f"Time {t:03d} | Loss: {res.fun:.5f} | MSE: {current_error:.5f} | Centers: {np.round(current_centers, 2)}")

    print(f"MPC Finished in {time.time() - start_time:.2f}s")

    # --- 6. Visualization ---
    history_u = np.array(history_u)
    history_c = np.array(history_c)
    history_centers = np.array(history_centers)
    history_loss = np.array(history_loss)
    
    # Final error
    final_error = np.mean((history_u[-1] - np.array(target_state))**2)
    print(f"\nFinal MSE: {final_error:.6f}")
    
    plt.figure(figsize=(18, 12))
    
    # Plot 1: State Heatmap
    plt.subplot(3, 3, 1)
    plt.imshow(history_u.T, aspect='auto', origin='lower', extent=[0, T_total, 0, 1])
    plt.colorbar(label='State value')
    plt.title('State Evolution (Heatmap)')
    
    # Plot 2: Target Comparison
    plt.subplot(3, 3, 2)
    plt.plot(x_grid, history_u[0], 'k--', label="Start")
    plt.plot(x_grid, target_state, 'r-', linewidth=3, label="Target")
    plt.plot(x_grid, history_u[-1], 'b-', linewidth=2, label="Final")
    plt.title("State Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Control Actions
    plt.subplot(3, 3, 3)
    for i in range(NUM_ACTUATORS):
        plt.plot(history_c[:, i], label=f"Act {i+1}", alpha=0.6)
    plt.title("Control Actions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss
    plt.subplot(3, 3, 4)
    plt.semilogy(history_loss)
    plt.title("Optimization Loss")
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Tracking Error
    plt.subplot(3, 3, 5)
    errors = [np.mean((u - np.array(target_state))**2) for u in history_u]
    plt.semilogy(errors)
    plt.title("Tracking MSE")
    plt.grid(True, alpha=0.3)

    # Plot 6: Control Effort
    plt.subplot(3, 3, 6)
    effort = np.sum(history_c**2, axis=1)
    plt.plot(effort)
    plt.title("Total Control Effort")
    plt.grid(True, alpha=0.3)

    # --- Plot 7: Actuator Migration (NEW) ---
    plt.subplot(3, 1, 3) # Full width at bottom
    for i in range(NUM_ACTUATORS):
        plt.plot(history_centers[:, i], linewidth=2, label=f"Center {i+1}")
    plt.ylabel("Position (0 to 1)")
    plt.xlabel("Time Step")
    plt.title("Actuator Migration (Self-Optimization)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('codesign_results.png', dpi=150)
    plt.show()
    
    return history_u, history_c, history_loss

if __name__ == "__main__":
    main()