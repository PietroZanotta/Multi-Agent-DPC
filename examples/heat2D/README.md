# 2D Heat Equation Control Examples

This directory contains examples for controlling the 2D heat equation using differentiable predictive control (DPC) with neural operators.

## Overview

The 2D heat equation control system extends the 1D heat example to two spatial dimensions:

- **PDE**: ∂z/∂t = ν∇²z + B(x,y,t) on [0,1]×[0,1]
- **Boundary Conditions**: Dirichlet (zero at all edges)
- **Solver**: Alternating Direction Implicit (ADI) method (Crank-Nicolson)
- **Grid**: 64×64 spatial points
- **Actuators**: Mobile agents that apply Gaussian-filtered forcing and can move in 2D
- **Goal**: Drive the temperature field from initial state to target state

## Directory Structure

```
heat2D/
├── centralized/
│   ├── data_utils.py          # 2D GRF generation with Dirichlet BCs
│   ├── dynamics_dual.py        # Wrapper for solver integration
│   ├── train.py                # Training with Heat2DControlNet
│   ├── visualize.py            # 2D heatmap visualization
│   └── animate.py              # Animation generation
└── decentralized/
    ├── data_utils.py          # Same as centralized
    ├── dynamics_dual.py       # Same as centralized
    ├── train.py               # Uses DecentralizedHeat2DControlNet
    ├── visualize.py
    └── animate.py
```

## Key Differences from 1D Heat

### Spatial Domain
- **1D**: Line segment [0,1] with 100 grid points
- **2D**: Square domain [0,1]×[0,1] with 64×64 grid points

### Solver
- **1D**: Crank-Nicolson with tridiagonal solve
- **2D**: ADI method (splits 2D problem into two 1D problems)

### Actuator Dynamics
- **1D**: Positions are scalars, velocities are scalars
- **2D**: Positions are (x,y) pairs, velocities are 2D vectors

### Policy Architecture
- **Centralized**:
  - Branch: CNN for 2D spatial error processing
  - Input: (64, 64, 3) channels [error, ∂error/∂x, ∂error/∂y]
  - Output: u (M,) scalar forcing + v (M, 2) 2D velocity
- **Decentralized**:
  - Each agent sees 12×12 local patch
  - Same CNN structure but on local patches
  - Zero communication between agents

## Usage

### 1. Build Tesseracts

```bash
# Build all tesseracts (including heat2D)
./buildall.sh

# Or build specific ones
tesseract build tesseracts/solverHeat2D_centralized/
tesseract build tesseracts/solverHeat2D_decentralized/
```

### 2. Train Controllers

**Centralized:**
```bash
cd examples/heat2D/centralized/
python train.py
# Outputs: centralized_params_heat2d.msgpack
```

**Decentralized:**
```bash
cd examples/heat2D/decentralized/
python train.py
# Outputs: decentralized_params_heat2d.msgpack
```

**Note**: Data generation may take several minutes due to eigendecomposition of large covariance matrices.

### 3. Visualize Results

```bash
# Centralized
cd examples/heat2D/centralized/
python visualize.py  # Creates heat2d_centralized_visualization.png
python animate.py    # Creates heat2d_animation.mp4 and .gif

# Decentralized
cd examples/heat2D/decentralized/
python visualize.py  # Creates heat2d_decentralized_visualization.png
python animate.py    # Creates heat2d_animation.mp4 and .gif
```

## Configuration

### Hyperparameters (train.py)

```python
n_grid = 64          # Grid resolution (64×64)
n_agents = 16        # Number of actuators (4×4 grid)
batch_size = 16      # Training batch size
T_steps = 300        # Control horizon
epochs = 500         # Training epochs
R_safe = 0.08        # Collision radius
```

### Physics Parameters (solver.py)

```python
N = 64               # Grid points per dimension
nu = 0.2             # Diffusion coefficient
sigma = 0.15         # Actuator width (Gaussian basis)
dt = 0.001           # Time step
```

### Loss Weights

- Tracking: 5.0 (primary objective)
- Effort: 0.001 (minimize actuation)
- Boundary: 100.0 (hard constraint)
- Collision: 1.0 (avoid agent overlap)
- Acceleration: 0.1 (smooth motion)

## Data Generation

The system generates training data using 2D Gaussian Random Fields (GRF) with zero Dirichlet boundary conditions:

1. **RBF Kernel**: K(x₁, x₂) = σ² exp(-||x₁ - x₂||² / (2ℓ²))
2. **Eigendecomposition**: For numerical stability
3. **Bridge Correction**: Enforces zero BCs on all four edges via bilinear interpolation

**Length scales**:
- Initial states: 0.25 (sharper features)
- Target states: 0.4 (smoother)

## Architecture and Performance Comparison

The following table summarizes the key differences between centralized and decentralized controllers:

| **Metric** | **Centralized** | **Decentralized** |
|------------|-----------------|-------------------|
| **Branch Input Dim** | 32×32×3 (3,072) | 12×12×3 (432) |
| **CNN Features** | [16, 32] (3×3 conv) | [16, 32] (3×3 conv) |
| **Fusion Hidden** | 64 (tanh) | 64 (tanh) |
| **Output Heads** | u∈ℝ, v∈ℝ² | u∈ℝ, v∈ℝ² |
| **Total Parameters** | 2,116,003 | 158,531 |
| **Checkpoint Size** | 8.1 MB | 620 KB |
| **Final MSE (Controlled)** | 1.50×10⁻⁴ | 8.60×10⁻⁵ |
| **Final MSE (Uncontrolled)** | 3.45×10⁻² | 3.45×10⁻² |
| **Improvement** | 99.6% | 99.8% |
| **Scalability** | Zero-shot (global) | Zero-shot (local) |
| **Communication** | Global | None |

### Key Observations

1. **Model Efficiency**: The decentralized controller is 13× smaller (158K vs 2.1M parameters) while achieving comparable or better performance
2. **Performance**: Both controllers achieve >99.5% reduction in tracking error compared to uncontrolled evolution
3. **Architecture Trade-offs**:
   - **Centralized**: Processes full 32×32 grid, larger trunk network (64-dim), global coordination
   - **Decentralized**: Each agent sees only 12×12 local patch, smaller trunk (32-dim), stigmergic coordination
4. **Surprising Result**: The decentralized controller achieves slightly lower final MSE (8.6×10⁻⁵ vs 1.5×10⁻⁴), demonstrating that local sensing can be sufficient for effective coordination

### Training Time
- ~30-60 minutes for 500 epochs (depending on hardware)
- Data generation: ~10-20 minutes

## Technical Details

### ADI Method

The Alternating Direction Implicit (ADI) method splits the 2D Crank-Nicolson scheme:

**Step 1 (Implicit in x, Explicit in y)**:
```
(I - r/2 * ∂²/∂x²) z* = (I + r/2 * ∂²/∂y²) z^n + dt/2 * B
```

**Step 2 (Implicit in y, Explicit in x)**:
```
(I - r/2 * ∂²/∂y²) z^(n+1) = (I + r/2 * ∂²/∂x²) z* + dt/2 * B
```

Each step requires solving N tridiagonal systems (64 systems per direction).

### 2D Gaussian Forcing

```python
basis = exp(-((x-ξ_x)² + (y-ξ_y)²) / (2σ²))
B(x,y) = Σᵢ uᵢ * basis_i(x,y)
```

## Troubleshooting

### Data Generation is Slow
- Pre-generate dataset once and save to disk
- Reduce dataset size (5000 → 1000 samples)
- Use smaller grid for prototyping (64 → 32)

### Training Diverges
- Reduce learning rate (1e-3 → 5e-4)
- Increase gradient clipping
- Verify boundary conditions in GRF data
- Start with smaller T_steps (100)

### Memory Issues
- Reduce batch size (16 → 8)
- Reduce T_steps (300 → 200)
- Use gradient accumulation

### Actuators Cluster
- Increase collision penalty (1.0 → 5.0)
- Reduce R_safe (0.08 → 0.05)
- Initialize agents more spread out

## References

See the main repository README and `main.tex` for theoretical background on:
- Differentiable Predictive Control (DPC)
- DeepONet operator learning
- Tesseract architecture for autodiff pipelines
