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
