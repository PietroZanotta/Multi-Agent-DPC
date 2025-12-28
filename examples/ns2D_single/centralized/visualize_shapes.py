import sys
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(script_dir))

from examples.ns2D.centralized.data_utils import generate_shape_pair


def main():
    n = 128
    L = 2.0 * jnp.pi
    seed = int(time.time()) & 0xFFFFFFFF
    key = jax.random.PRNGKey(seed)
    rho_init, rho_target = generate_shape_pair(key, n=n, L=L)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    im0 = axes[0].imshow(rho_init, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    axes[0].set_title("Initial rho")

    im1 = axes[1].imshow(rho_target, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    axes[1].set_title("Target rho")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.04)
    plt.savefig("ns2d_shapes_preview.png", dpi=150)


if __name__ == "__main__":
    main()
