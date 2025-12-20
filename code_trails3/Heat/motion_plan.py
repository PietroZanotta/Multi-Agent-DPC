import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import time

import tesseract_api

# --- Configuration ---
N = 100
T_total = 300
Horizon = 30
dt = 0.001

x_grid = jnp.linspace(0, 1, N, endpoint=False)

def get_smooth_grf_target(x_grid, key):
    """Generates a random smooth curve using smooth Fourier basis that is 0 at both boundaries."""
    n = len(x_grid)
    
    # Number of low-frequency modes to use (fewer = smoother)
    n_modes = 2  # Use only 5 lowest frequency modes for very smooth curves
    
    # Generate random coefficients with decreasing variance for higher frequencies
    keys = jax.random.split(key, n_modes)
    coefficients = []
    
    for i, k in enumerate(keys):
        # Variance decreases with frequency for smoothness
        freq = i + 1
        variance = 1.0 / (freq ** 2)  # Quadratic decay for very smooth curves
        coeff = jax.random.normal(k) * jnp.sqrt(variance)
        coefficients.append(coeff)
    
    coefficients = jnp.array(coefficients)
    
    # Build the function using sine basis
    target = jnp.zeros_like(x_grid)
    for i, coeff in enumerate(coefficients):
        freq = i + 1
        # Adjust for endpoint=False: scale x to [0, 1] including the implicit endpoint
        x_adjusted = x_grid * n / (n - 1)  # This maps [0, (n-1)/n] -> [0, 1]
        target += coeff * jnp.sin(freq * jnp.pi * x_adjusted)
    
    # Normalize to desired scale
    max_val = jnp.maximum(jnp.max(jnp.abs(target)), 1e-8)
    target = 0.6 * target / max_val
    
    return target

# Usage
key = jax.random.PRNGKey(4)
target_state = get_smooth_grf_target(x_grid, key)

# --- JAX Loss Function with Single-Step Rollout via Tesseract API ---
def mpc_loss_fn(plan_controls, current_u):
    """
    Evaluates plan using single-step rollout with explicit feedback via Tesseract API.
    Accumulates tracking error over entire prediction horizon.
    """
    def rollout_step(u_current, control_t):
        """Apply one step via Tesseract API with feedback."""
        step_result = tesseract_api.step({
            'u_current': u_current,
            'control': control_t
        })
        u_next = step_result['u_next']
        return u_next, u_next
    
    # Rollout trajectory with explicit state feedback at each step
    _, trajectory = jax.lax.scan(rollout_step, current_u, plan_controls)
    
    # Accumulated tracking error over entire horizon
    tracking_error = jnp.mean((trajectory - target_state[None, :])**2)
    
    # Optional: Control regularization (uncomment if needed)
    # control_penalty = 0.001 * jnp.mean(plan_controls ** 2)
    # smoothness_penalty = 0.01 * jnp.mean((plan_controls[1:] - plan_controls[:-1])**2)
    # total_cost = tracking_error + control_penalty + smoothness_penalty
    
    return tracking_error

# Compile loss and gradient
jit_loss_and_grad = jit(value_and_grad(mpc_loss_fn))

# --- SciPy Bridge ---
def scipy_objective(flat_plan, current_u, shape):
    """Wraps JAX for SciPy optimizer."""
    plan = jnp.array(flat_plan.reshape(shape))
    loss_val, grad_val = jit_loss_and_grad(plan, current_u)
    return float(loss_val), np.array(grad_val).astype(np.float64).flatten()

# --- Main MPC Loop ---
def main():
    print(f"Starting MPC with Tesseract API (Horizon={Horizon})...")
    
    u_current = jnp.zeros(N)
    current_plan = np.zeros((Horizon, 4))
    
    history_u = [np.array(u_current)]
    history_c = []
    history_loss = []
    
    start_time = time.time()

    for t in range(T_total):
        # --- PLANNING PHASE ---
        obj_fun = lambda x: scipy_objective(x, u_current, (Horizon, 4))
        
        # Adaptive iteration budget
        if t < 50:
            max_iter = 30
        elif t < 150:
            max_iter = 20
        else:
            max_iter = 15
        
        res = scipy.optimize.minimize(
            fun=obj_fun,
            x0=current_plan.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': 1e-6,
                'gtol': 1e-5,
                'disp': False
            }
        )
        
        if not res.success and t % 50 == 0:
            print(f"  Warning at t={t}: {res.message}")
        
        optimized_plan = res.x.reshape(Horizon, 4)
        
        # --- EXECUTION PHASE ---
        # Apply first control action via Tesseract API
        action_t = optimized_plan[0]
        step_result = tesseract_api.step({
            'u_current': u_current,
            'control': jnp.array(action_t)
        })
        u_next = step_result['u_next']
        
        # --- UPDATE ---
        history_u.append(np.array(u_next))
        history_c.append(action_t)
        history_loss.append(res.fun)
        u_current = u_next
        
        # --- WARM START ---
        # Shift plan left and repeat last action
        current_plan[:-1] = optimized_plan[1:]
        current_plan[-1] = optimized_plan[-1]

        if t % 50 == 0:
            current_error = np.mean((np.array(u_current) - np.array(target_state))**2)
            print(f"Time {t:03d} | Loss: {res.fun:.6f} | Error: {current_error:.6f} | Iters: {res.nit}")

    elapsed = time.time() - start_time
    print(f"\nMPC Finished in {elapsed:.2f}s")

    # --- Analysis ---
    history_u = np.array(history_u)
    history_c = np.array(history_c)
    history_loss = np.array(history_loss)
    
    final_error = np.mean((history_u[-1] - np.array(target_state))**2)
    print(f"Final MSE: {final_error:.6f}")
    
    # --- Visualization ---
    plt.figure(figsize=(15, 10))
    
    # 1. State evolution heatmap
    plt.subplot(2, 3, 1)
    plt.imshow(history_u.T, aspect='auto', origin='lower', extent=[0, T_total, 0, 1])
    plt.colorbar(label='State value')
    plt.xlabel('Time step')
    plt.ylabel('Spatial position')
    plt.title('State Evolution')
    
    # 2. Initial vs Final vs Target
    plt.subplot(2, 3, 2)
    plt.plot(x_grid, history_u[0], 'k--', label="Start", linewidth=2)
    plt.plot(x_grid, target_state, 'r-', linewidth=3, label="Target")
    plt.plot(x_grid, history_u[-1], 'b-', linewidth=2, label="Final")
    plt.plot(x_grid, history_u[T_total//2], 'g:', linewidth=2, label=f"Mid (t={T_total//2})")
    plt.title("State Comparison")
    plt.xlabel("Spatial position")
    plt.ylabel("State value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Control actions
    plt.subplot(2, 3, 3)
    for i in range(4):
        plt.plot(history_c[:, i], label=f"Actuator {i+1}", alpha=0.7)
    plt.title("Control Actions")
    plt.xlabel("Time step")
    plt.ylabel("Control value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Optimization loss
    plt.subplot(2, 3, 4)
    plt.semilogy(history_loss)
    plt.title("Optimization Loss")
    plt.xlabel("Time step")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, alpha=0.3)
    
    # 5. Tracking error
    plt.subplot(2, 3, 5)
    errors = [np.mean((u - np.array(target_state))**2) for u in history_u]
    plt.semilogy(errors)
    plt.title("Tracking Error")
    plt.xlabel("Time step")
    plt.ylabel("MSE (log scale)")
    plt.grid(True, alpha=0.3)
    
    # 6. Control effort
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