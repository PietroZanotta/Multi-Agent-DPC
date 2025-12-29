import math

import jax
import jax.numpy as jnp


def generate_field2d(key, n=128, L=2.0 * jnp.pi, length_scale=0.4):
    x = jnp.linspace(0.0, L, n, endpoint=False)
    y = jnp.linspace(0.0, L, n, endpoint=False)
    noise = jax.random.normal(key, (n, n))

    dx = L / n
    kx = jnp.fft.fftfreq(n, d=dx)
    ky = jnp.fft.rfftfreq(n, d=dx)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    k_mag = jnp.sqrt(KX**2 + KY**2)

    k0 = 1.0 / (length_scale + 1e-6)
    filt = jnp.exp(-((k_mag / k0) ** 2))

    noise_hat = jnp.fft.rfft2(noise)
    field = jnp.fft.irfft2(noise_hat * filt)

    field = field - jnp.min(field)
    field = field / (jnp.max(field) + 1e-12)
    return field


def _polygon_mask(vertices, n):
    k = vertices.shape[0]
    xs = jnp.arange(n)
    ys = jnp.arange(n)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    points = jnp.stack([xx, yy], axis=-1)

    def body(mask, i):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % k]
        edge = v1 - v0
        rel = points - v0
        cross = edge[0] * rel[..., 1] - edge[1] * rel[..., 0]
        inside_edge = cross >= 0.0
        mask = mask & inside_edge
        return mask, None

    init_mask = jnp.ones((n, n), dtype=bool)
    mask, _ = jax.lax.scan(body, init_mask, jnp.arange(k))
    return mask


def _sample_convex_polygon(key, n, center_region="left"):
    key_angle, key_r, key_center = jax.random.split(key, 3)
    K = 6
    angles = jax.random.uniform(key_angle, (K,), minval=0.0, maxval=2.0 * jnp.pi)
    angles = jnp.sort(angles)

    base = 0.08 * n
    radii = base * (0.7 + 0.6 * jax.random.uniform(key_r, (K,)))

    if center_region == "left":
        cx = jax.random.uniform(key_center, (), minval=0.15 * n, maxval=0.35 * n)
        cy = jax.random.uniform(key_center, (), minval=0.25 * n, maxval=0.75 * n)
    elif center_region == "right_top":
        cx = jax.random.uniform(key_center, (), minval=0.65 * n, maxval=0.85 * n)
        cy = jax.random.uniform(key_center, (), minval=0.55 * n, maxval=0.85 * n)
    else:
        cx = jax.random.uniform(key_center, (), minval=0.25 * n, maxval=0.75 * n)
        cy = jax.random.uniform(key_center, (), minval=0.25 * n, maxval=0.75 * n)

    xs = cx + radii * jnp.cos(angles)
    ys = cy + radii * jnp.sin(angles)

    xs = jnp.clip(xs, 1.0, n - 2.0)
    ys = jnp.clip(ys, 1.0, n - 2.0)

    verts = jnp.stack([xs, ys], axis=-1)
    verts = jnp.round(verts).astype(jnp.int32)
    return verts


def _random_bar_mask(key, n, orientation="horizontal", region="right_top"):
    key_pos, key_size = jax.random.split(key)

    min_len = 0.25 * n
    max_len = 0.6 * n
    thickness = jax.random.uniform(key_size, (), minval=0.04 * n, maxval=0.09 * n)

    if orientation == "horizontal":
        length = jax.random.uniform(key_size, (), minval=min_len, maxval=max_len)
        if region == "right_top":
            y_center = jax.random.uniform(key_pos, (), minval=0.7 * n, maxval=0.9 * n)
            x_center = jax.random.uniform(key_pos, (), minval=0.55 * n, maxval=0.85 * n)
        else:
            y_center = jax.random.uniform(key_pos, (), minval=0.3 * n, maxval=0.7 * n)
            x_center = jax.random.uniform(key_pos, (), minval=0.3 * n, maxval=0.7 * n)

        x_min = jnp.clip(x_center - 0.5 * length, 0.0, n - 1.0)
        x_max = jnp.clip(x_center + 0.5 * length, 0.0, n - 1.0)
        y_min = jnp.clip(y_center - 0.5 * thickness, 0.0, n - 1.0)
        y_max = jnp.clip(y_center + 0.5 * thickness, 0.0, n - 1.0)
    else:
        length = jax.random.uniform(key_size, (), minval=min_len, maxval=max_len)
        if region == "right_top":
            x_center = jax.random.uniform(key_pos, (), minval=0.8 * n, maxval=0.95 * n)
            y_center = jax.random.uniform(key_pos, (), minval=0.3 * n, maxval=0.7 * n)
        else:
            x_center = jax.random.uniform(key_pos, (), minval=0.3 * n, maxval=0.7 * n)
            y_center = jax.random.uniform(key_pos, (), minval=0.3 * n, maxval=0.7 * n)

        x_min = jnp.clip(x_center - 0.5 * thickness, 0.0, n - 1.0)
        x_max = jnp.clip(x_center + 0.5 * thickness, 0.0, n - 1.0)
        y_min = jnp.clip(y_center - 0.5 * length, 0.0, n - 1.0)
        y_max = jnp.clip(y_center + 0.5 * length, 0.0, n - 1.0)

    xs = jnp.arange(n)
    ys = jnp.arange(n)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")

    mask = (
        (xx >= x_min)
        & (xx <= x_max)
        & (yy >= y_min)
        & (yy <= y_max)
    )
    return mask


def generate_shape_pair(key, n=128, L=2.0 * jnp.pi):
    key_init, key_target_shape, key_target_type, key_orient = jax.random.split(key, 4)

    verts_init = _sample_convex_polygon(key_init, n, center_region="left")
    mask_init = _polygon_mask(verts_init, n)

    is_bar = jax.random.bernoulli(key_target_type)

    def make_bar(_):
        orient_flag = jax.random.bernoulli(key_orient)

        def bar_horizontal(_):
            return _random_bar_mask(
                key_target_shape, n, orientation="horizontal", region="right_top"
            )

        def bar_vertical(_):
            return _random_bar_mask(
                key_target_shape, n, orientation="vertical", region="right_top"
            )

        return jax.lax.cond(orient_flag, bar_horizontal, bar_vertical, operand=None)

    def make_polygon(_):
        verts_target = _sample_convex_polygon(
            key_target_shape, n, center_region="right_top"
        )
        return _polygon_mask(verts_target, n)

    mask_target = jax.lax.cond(is_bar, make_bar, make_polygon, operand=None)

    rho_init = mask_init.astype(jnp.float32)
    rho_target = mask_target.astype(jnp.float32)

    mass_init = jnp.sum(rho_init)
    mass_target = jnp.sum(rho_target)

    eps = jnp.array(1e-12, dtype=mass_init.dtype)
    one = jnp.array(1.0, dtype=mass_init.dtype)

    cond = mass_init <= mass_target

    scale_init_if_init_smaller = one
    scale_target_if_init_smaller = mass_init / (mass_target + eps)

    scale_init_if_target_smaller = mass_target / (mass_init + eps)
    scale_target_if_target_smaller = one

    scale_init = jnp.where(
        cond, scale_init_if_init_smaller, scale_init_if_target_smaller
    )
    scale_target = jnp.where(
        cond, scale_target_if_init_smaller, scale_target_if_target_smaller
    )
    rho_init = rho_init * scale_init
    rho_target = rho_target * scale_target

    return rho_init, rho_target


def make_actuator_grid(m, L):
    n_side = int(math.ceil(math.sqrt(m)))
    spacing = L / (n_side + 1)
    coords = []
    for i in range(n_side):
        for j in range(n_side):
            if len(coords) < m:
                coords.append([spacing * (i + 1), spacing * (j + 1)])
    return jnp.array(coords)


# =============================================================================
# NEW: Diverse Shape Generation (v2)
# =============================================================================

def _create_grid(n):
    """Create normalized coordinate grid [-1, 1] x [-1, 1]."""
    coords = jnp.linspace(-1, 1, n)
    xx, yy = jnp.meshgrid(coords, coords, indexing='ij')
    return xx, yy


def _circle_mask(n, center, scale, aspect_ratio=1.0):
    """Generate ellipse/circle mask."""
    xx, yy = _create_grid(n)
    cx, cy = center[0], center[1]
    # Normalize center to [-1, 1]
    cx_norm = 2.0 * cx / n - 1.0
    cy_norm = 2.0 * cy / n - 1.0
    # Scale to grid units
    r = 2.0 * scale / n
    
    dist_sq = ((xx - cx_norm) / aspect_ratio)**2 + (yy - cy_norm)**2
    mask = dist_sq <= r**2
    return mask.astype(jnp.float32)


def _regular_polygon_mask(n, center, scale, num_sides, rotation=0.0):
    """Generate regular polygon mask (triangle to hexagon)."""
    xx, yy = _create_grid(n)
    cx_norm = 2.0 * center[0] / n - 1.0
    cy_norm = 2.0 * center[1] / n - 1.0
    r = 2.0 * scale / n
    
    # Translate to center
    dx = xx - cx_norm
    dy = yy - cy_norm
    
    # Convert to polar
    angle = jnp.arctan2(dy, dx) - rotation
    dist = jnp.sqrt(dx**2 + dy**2)
    
    # Polygon boundary in polar: r_boundary = r / cos(theta_local)
    # where theta_local is angle within each sector
    sector_angle = 2.0 * jnp.pi / num_sides
    angle_in_sector = jnp.mod(angle + sector_angle / 2, sector_angle) - sector_angle / 2
    
    # Distance to polygon edge at this angle
    r_boundary = r * jnp.cos(jnp.pi / num_sides) / (jnp.cos(angle_in_sector) + 1e-8)
    
    mask = dist <= r_boundary
    return mask.astype(jnp.float32)


def _star_mask(n, center, scale, num_points=5, inner_ratio=0.4):
    """Generate star mask."""
    xx, yy = _create_grid(n)
    cx_norm = 2.0 * center[0] / n - 1.0
    cy_norm = 2.0 * center[1] / n - 1.0
    r_outer = 2.0 * scale / n
    r_inner = r_outer * inner_ratio
    
    dx = xx - cx_norm
    dy = yy - cy_norm
    angle = jnp.arctan2(dy, dx)
    dist = jnp.sqrt(dx**2 + dy**2)
    
    # Star has 2*num_points vertices alternating between inner and outer radii
    sector_angle = jnp.pi / num_points
    angle_in_sector = jnp.mod(angle, sector_angle) / sector_angle
    
    # Interpolate between inner and outer radius
    # At sector boundaries (0 and 1): outer radius
    # At sector midpoint (0.5): inner radius
    interp = jnp.abs(2.0 * angle_in_sector - 1.0)
    r_boundary = r_inner + (r_outer - r_inner) * interp
    
    mask = dist <= r_boundary
    return mask.astype(jnp.float32)


def _gaussian_blob_mask(key, n, center, scale, num_blobs=3):
    """Generate smooth blob from overlapping Gaussians."""
    keys = jax.random.split(key, num_blobs * 2)
    
    xx, yy = _create_grid(n)
    cx_norm = 2.0 * center[0] / n - 1.0
    cy_norm = 2.0 * center[1] / n - 1.0
    sigma = 2.0 * scale / n * 0.4  # Base sigma
    
    density = jnp.zeros((n, n))
    
    for i in range(num_blobs):
        # Random offset for each sub-blob
        offset_x = 0.5 * sigma * jax.random.uniform(keys[2*i], (), minval=-1, maxval=1)
        offset_y = 0.5 * sigma * jax.random.uniform(keys[2*i+1], (), minval=-1, maxval=1)
        
        blob_cx = cx_norm + offset_x
        blob_cy = cy_norm + offset_y
        
        dist_sq = (xx - blob_cx)**2 + (yy - blob_cy)**2
        density = density + jnp.exp(-dist_sq / (2 * sigma**2))
    
    # Threshold to create binary mask, then normalize
    threshold = 0.3 * num_blobs
    mask = (density >= threshold).astype(jnp.float32)
    return mask


def _sample_separated_centers(key, n, min_dist_frac=0.3):
    """Sample two centers with minimum separation."""
    key1, key2 = jax.random.split(key)
    margin = 0.2 * n
    
    # First center
    cx1 = jax.random.uniform(key1, (), minval=margin, maxval=n - margin)
    cy1 = jax.random.uniform(key1, (), minval=margin, maxval=n - margin)
    
    # Second center - ensure separation
    min_dist = min_dist_frac * n
    
    # Simple approach: sample angle and distance
    key_ang, key_dist = jax.random.split(key2)
    angle = jax.random.uniform(key_ang, (), minval=0, maxval=2*jnp.pi)
    dist = jax.random.uniform(key_dist, (), minval=min_dist, maxval=0.5*n)
    
    cx2 = jnp.clip(cx1 + dist * jnp.cos(angle), margin, n - margin)
    cy2 = jnp.clip(cy1 + dist * jnp.sin(angle), margin, n - margin)
    
    return jnp.array([cx1, cy1]), jnp.array([cx2, cy2])


def _generate_single_shape(key, shape_type, n, center, scale):
    """Generate a single shape based on type (0-3)."""
    key1, key2 = jax.random.split(key)
    
    # Type 0: Circle/Ellipse
    aspect = 0.7 + 0.6 * jax.random.uniform(key1)  # 0.7-1.3
    circle = _circle_mask(n, center, scale, aspect)
    
    # Type 1: Polygon (4-6 sides)
    num_sides = 4 + jax.random.randint(key1, (), 0, 3)  # 4, 5, or 6
    rotation = jax.random.uniform(key2, (), minval=0.0, maxval=2*jnp.pi)
    polygon = _regular_polygon_mask(n, center, scale, num_sides, rotation)
    
    # Type 2: Star
    star = _star_mask(n, center, scale, num_points=5, inner_ratio=0.4)
    
    # Type 3: Gaussian blob
    blob = _gaussian_blob_mask(key1, n, center, scale, num_blobs=3)
    
    # Select based on type
    shapes = jnp.stack([circle, polygon, star, blob], axis=0)
    return shapes[shape_type]


def generate_shape_pair_v2(key, n=64, L=2.0 * jnp.pi):
    """
    Generate diverse initial and target shape pairs.
    
    Shape types:
    0: Circle/Ellipse
    1: Regular polygon (4-6 sides)
    2: Star (5-pointed)
    3: Gaussian blob
    
    Returns rho_init, rho_target with matched mass.
    """
    keys = jax.random.split(key, 8)
    
    # Random shape types (0-3), ensure different types
    init_type = jax.random.randint(keys[0], (), 0, 4)
    offset = 1 + jax.random.randint(keys[1], (), 0, 3)  # 1, 2, or 3
    target_type = (init_type + offset) % 4
    
    # Random separated centers
    center_init, center_target = _sample_separated_centers(keys[2], n, min_dist_frac=0.25)
    
    # Random scales (0.08-0.13 of n)
    scale_init = n * (0.08 + 0.05 * jax.random.uniform(keys[3]))
    scale_target = n * (0.08 + 0.05 * jax.random.uniform(keys[4]))
    
    # Generate shapes
    mask_init = _generate_single_shape(keys[5], init_type, n, center_init, scale_init)
    mask_target = _generate_single_shape(keys[6], target_type, n, center_target, scale_target)
    
    # Ensure shapes are not empty (fallback to circle if empty)
    init_empty = jnp.sum(mask_init) < 1.0
    target_empty = jnp.sum(mask_target) < 1.0
    
    # Fallback circles if empty
    fallback_init = _circle_mask(n, center_init, scale_init * 1.5, 1.0)
    fallback_target = _circle_mask(n, center_target, scale_target * 1.5, 1.0)
    
    mask_init = jnp.where(init_empty, fallback_init, mask_init)
    mask_target = jnp.where(target_empty, fallback_target, mask_target)
    
    # Match masses: scale the LARGER shape down to match the smaller one
    mass_init = jnp.sum(mask_init) + 1e-8
    mass_target = jnp.sum(mask_target) + 1e-8
    
    # Scale factor: min(1, smaller_mass / larger_mass)
    # This ensures we never scale UP (which keeps max ≤ 1)
    scale_init = jnp.where(mass_init > mass_target, mass_target / mass_init, 1.0)
    scale_target = jnp.where(mass_target > mass_init, mass_init / mass_target, 1.0)
    
    # Apply scaling but ensure min density is at least 0.5 for numerical stability
    scale_init = jnp.maximum(scale_init, 0.5)
    scale_target = jnp.maximum(scale_target, 0.5)
    
    rho_init = (mask_init * scale_init).astype(jnp.float32)
    rho_target = (mask_target * scale_target).astype(jnp.float32)
    
    return rho_init, rho_target

