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
    
    @nn.compact
    def __call__(self, x):
        # Hidden layers
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        
        # Output layer
        # Output dimension should be 8: 4 for u, 4 for v
        x = nn.Dense(self.features[-1])(x)
        return x

def split_action(action_vec):
    """
    Splits the 8-dim output into u (4) and v (4).
    Args:
        action_vec: (8,) or (Batch, 8) array
    Returns:
        u, v
    """
    if action_vec.ndim == 1:
        u = action_vec[:4]
        v = action_vec[4:]
    else:
        u = action_vec[:, :4]
        v = action_vec[:, 4:]
    return u, v
