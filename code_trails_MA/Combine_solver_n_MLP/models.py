import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

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

        # --- FIX 1: Boundary Padding (Prevents Boundary Bounce) ---
        # 'edge' padding mimics the physical boundary of the PDE
        padded_error = jnp.pad(error, (half_window, half_window), mode='edge')

        def get_local_obs(xi):
            # Map GPS coordinate to padded array index
            center_idx = (xi * (n_pde - 1)).astype(int) + half_window
            start = center_idx - half_window
            
            # --- FIX 2: Pure Local Slicing (Prevents Global Leaks) ---
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

        # --- FIX 3: Bi-directional Velocity ---
        # Using tanh allows agents to move both left and right
        return self.u_max * jnp.tanh(u_raw), self.v_max * jnp.tanh(v_raw)
    


