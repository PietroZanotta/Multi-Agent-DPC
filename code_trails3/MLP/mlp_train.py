import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
import time
from functools import partial

# --- 1. Data Generation ---
def generate_data(n_samples=100):
    """Generates training data for y = sin(x)."""
    x = np.linspace(-np.pi, 2*np.pi, n_samples).reshape(-1, 1)
    y = np.sin(x)
    return jnp.array(x), jnp.array(y)

# --- 2. Model Definition (MLP) ---
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # Hidden layers
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)  # Using tanh as in your reference
        
        # Output layer (linear activation)
        x = nn.Dense(self.features[-1])(x)
        return x

# --- 3. Loss & Update Steps ---
def mse_loss(params, x_batch, y_batch, model):
    preds = model.apply(params, x_batch)
    return jnp.mean((preds - y_batch) ** 2)

@partial(jax.jit, static_argnums=(4, 5))
def train_step(params, opt_state, x_batch, y_batch, model, optimizer):
    """Performs one training step."""
    loss, grads = jax.value_and_grad(mse_loss)(params, x_batch, y_batch, model)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# --- 4. Main Training Loop ---
def main():
    # Hyperparameters
    learning_rate = 1e-3
    epochs = 6000
    hidden_layers = [64, 64, 64]
    output_dim = 1
    seed = 0

    print(f"Training MLP on sin(x) with layers {hidden_layers}...")

    # Data
    x_train, y_train = generate_data(n_samples=200)
    
    # Model Init
    key = jax.random.PRNGKey(seed)
    # Convert to tuple to make the model hashable for static_argnums
    model = MLP(features=tuple(hidden_layers + [output_dim]))
    
    # Initialize parameters with dummy input
    dummy_input = jnp.ones((1, 1))
    params = model.init(key, dummy_input)
    
    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Training
    train_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, x_train, y_train, model, optimizer)
        
        if epoch % 100 == 0:
            train_losses.append(loss)
            print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s. Final Loss: {loss:.6f}")
    
    # --- 5. Visualization ---
    # Prediction
    y_pred = model.apply(params, x_train)
    
    plt.figure(figsize=(10, 5))
    
    # Plot Function
    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, label='True sin(x)', alpha=0.5, color='blue')
    plt.plot(x_train, y_pred, label='MLP Prediction', color='red', linewidth=2)
    plt.title('MLP Regression: sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Loss using standard matplotlib (as per request for simple code)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.title('Training Loss (every 100 epochs)')
    plt.xlabel('Log Step')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_sinx_result.png')
    print("Results saved to mlp_sinx_result.png")
    # plt.show() # Uncomment to show interactivly if running locally

if __name__ == "__main__":
    main()
