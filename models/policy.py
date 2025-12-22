import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class ControlNet(nn.Module):
    """
    DeepONet-based Controller for spatial systems.
    
    This module implements a feedback controller that maps global state errors 
    to local actuator commands. It uses a dual-pathway architecture:
    
    1. Branch Net: Encodes the global error 'field' and its gradient, 
       capturing the high-level state of the system.
    2. Trunk Net: Encodes spatial coordinates (xi) using Fourier features,
       capturing 'where' the control is being applied. xi is a function of time as
       the actuator is moving.
    
    The two representations are fused to produce per-agent control signals (u, v)
    scaled by physical saturation limits.
    """
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
            x = nn.LayerNorm()(x)
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


class DecentralizedControlNet(nn.Module):
    features: Sequence[int]
    u_max: float = 40.0
    v_max: float = 1.0  # Max velocity in units/step
    sensor_range: float = 0.08 
    
    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    def branch_net(self, local_patch):
        x = local_patch
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.GroupNorm(num_groups=1)(x) # Local-only normalization
            x = nn.tanh(x)
        return x

    def trunk_net(self, xi):
        angle = xi[:, None] * self.frequencies * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded 

    @nn.compact
    def __call__(self, z_curr, z_target, xi_curr):
        error = z_curr - z_target
        n_pde = z_curr.shape[0]
        window_size = int(self.sensor_range * n_pde)
        half_window = window_size // 2

        # --- Boundary Padding (Prevents Boundary Bounce) ---
        # 'edge' padding mimics the physical boundary of the PDE to avoid info sharing among agents
        padded_error = jnp.pad(error, (half_window, half_window), mode='edge')

        def get_local_obs(xi):
            # Map GPS coordinate to padded array index
            center_idx = (xi * (n_pde - 1)).astype(int) + half_window
            start = center_idx - half_window
            
            # --- Pure Local Slicing (Prevents Global Leaks) ---
            p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
            
            # Local gradient calculation on the slice
            p_grad = jnp.gradient(p_err) 

            # Standardize input for the Branch Net
            p_err = jax.image.resize(p_err, (20,), method='bilinear')
            p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
            
            return jnp.concatenate([p_err, p_grad])

        local_patches = jax.vmap(get_local_obs)(xi_curr)
        branch_outs = jax.vmap(self.branch_net)(local_patches)
        trunk_outs = self.trunk_net(xi_curr)

        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        x = nn.Dense(32)(combined)
        x = nn.tanh(x)
        
        # Output Heads
        u_raw = nn.Dense(1)(x).squeeze(-1)
        v_raw = nn.Dense(1)(x).squeeze(-1)

        # --- Bi-directional Velocity ---
        # Using tanh allows agents to move both left and right
        return self.u_max * jnp.tanh(u_raw), self.v_max * jnp.tanh(v_raw)
    


