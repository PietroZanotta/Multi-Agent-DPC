import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class MLP(nn.Module):
    features: Sequence[int]
    u_max: float = 40.0
    v_max: float = 2.0  # Velocity bound for moving xi
    
    def setup(self):
        # Fourier feature frequencies for coordinate encoding
        # This helps the MLP "understand" the [0, 1] spatial domain better
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def encode_coords(self, xi):
        # xi is (N_agents,)
        # Returns (N_agents, len(freqs)*2)
        angle = xi[:, None] * self.frequencies * jnp.pi
        return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1).flatten()

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        # 1. Feature Engineering
        # Concatenate field state and the encoded positions of agents
        xi_encoded = self.encode_coords(xi_curr)
        x = jnp.concatenate([z_curr, z_target, xi_encoded], axis=-1)

        # 2. Shared Hidden Layers
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        
        # 3. Separate Heads for u (Intensity) and v (Velocity)
        # This allows the network to learn different scales for movement vs force
        u_raw = nn.Dense(4, name="u_head")(x)
        v_raw = nn.Dense(4, name="v_head")(x)
        
        # Apply physical bounds
        u = self.u_max * jnp.tanh(u_raw)
        v = self.v_max * jnp.tanh(v_raw)
        
        return u, v




class ControlNet(nn.Module):
    features: Sequence[int]
    u_max: float = 40.0
    v_max: float = 2.0
    
    def setup(self):
        # Fourier frequencies for the Trunk
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def trunk_net(self, xi):
        """Processes actuator coordinates."""
        angle = xi[:, None] * self.frequencies * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        # Process each agent's coordinate independently through a small MLP
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded # Shape: (n_agents, 32)

    def branch_net(self, error_field, error_grad):
        combined = jnp.concatenate([error_field, error_grad], axis=-1)
        x = combined
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.LayerNorm()(x) # Added Normalization
            x = nn.tanh(x)
        return x # Shape: (branch_feature_dim,)

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        # 1. Calculate Error Field and its Spatial Gradient
        error = z_curr - z_target
        # Central difference for gradient
        error_grad = jnp.gradient(error) 

        # 2. Branch: Global Error Context
        branch_out = self.branch_net(error, error_grad)

        # 3. Trunk: Spatial Agent Context
        trunk_out = self.trunk_net(xi_curr)

        # 4. Fusion (DeepONet style)
        # We broadcast the global branch context to all agents and concatenate with trunk
        branch_repeated = jnp.tile(branch_out, (xi_curr.shape[0], 1))
        combined = jnp.concatenate([branch_repeated, trunk_out], axis=-1)

        # 5. Control Heads (Per Agent)
        # We process each agent's combined feature to get u and v
        x = combined
        for feat in [32]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        
        u_raw = nn.Dense(1)(x).squeeze(-1) # Output 1 per agent
        v_raw = nn.Dense(1)(x).squeeze(-1) # Output 1 per agent
        
        u = self.u_max * jnp.tanh(u_raw)
        v = self.v_max * jnp.tanh(v_raw)
        
        return u, v
    