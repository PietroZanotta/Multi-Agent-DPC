import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class MLP(nn.Module):
    """
    Simple MLP for regression/control.
    Takes state as input and outputs control actions (u, v).
    """
    features: Sequence[int]
    u_max: float = 40.0  # High bound for forcing
    # v_max: float = 5.0     # Moderate bound for velocity
    
    @nn.compact
    def __call__(self, x):
        # Hidden layers
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        
        # Output layer
        # Output dimension is now only 4 for u
        x = nn.Dense(self.features[-1])(x)
        
        # Apply bounds with smooth activations
        # u is the forcing intensity
        u = self.u_max * jnp.tanh(x)  # u ∈ [-u_max, u_max]
        
        return u

def split_action(action_vec):
    """
    Handles the 4-dim output for u and provides zero for v.
    Args:
        action_vec: (4,) or (Batch, 4) array (u only)
    Returns:
        u, v (zeros)
    """
    u = action_vec
    if action_vec.ndim == 1:
        v = jnp.zeros((4,))
    else:
        v = jnp.zeros((action_vec.shape[0], 4))
    return u, v
