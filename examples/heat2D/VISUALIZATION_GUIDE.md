# 2D Heat Equation Visualization Guide

This guide explains how to use and customize the visualization and animation scripts for the 2D heat equation control examples.

## Quick Start

### Generate Visualizations
```bash
cd examples/heat2D/centralized
python visualize.py          # Creates PDF and PNG figures

cd ../decentralized
python visualize.py          # Creates PDF and PNG figures
```

### Generate Animations
```bash
cd examples/heat2D/centralized
python animate.py            # Creates MP4 and GIF animations

cd ../decentralized
python animate.py            # Creates MP4 and GIF animations
```

## Output Files

### Visualization (visualize.py)
- **PDF**: `heat2d_centralized_visualization.pdf` (vector graphics, publication-ready)
- **PNG**: `heat2d_centralized_visualization.png` (high-res raster, 300 DPI)

### Animation (animate.py)
- **MP4**: `heat2d_animation.mp4` (high quality, 200 DPI, requires ffmpeg)
- **GIF**: `heat2d_animation.gif` (moderate quality, 150 DPI)

## Visualization Layout

### Static Figure (visualize.py)

The figure contains:
- **3 rows of 2D field plots** (6 timesteps each):
  1. **Uncontrolled Evolution**: Natural diffusion without control
  2. **DPC Controlled Evolution**: With actuators (colored by control intensity)
  3. **Tracking Error**: Absolute error |z - z_target|

- **3 colorbars**:
  1. **Temperature** (RdBu_r): For field values
  2. **Control u** (YlOrRd): Actuator forcing intensity
  3. **|Error|** (hot): Tracking error magnitude

- **3 time-series plots** (bottom row):
  1. **MSE Tracking Error**: Controlled vs Uncontrolled (log scale)
  2. **Agent Speed**: Average actuator velocity
  3. **Control Intensity**: Average |u| over time

### Animation (animate.py)

2×2 layout:
```
[Uncontrolled Evolution]  [DPC Controlled Evolution]
[Tracking Error]          [MSE Tracking Error Plot]
```

- Fields animated over 300 timesteps
- Actuators move dynamically
- MSE curves build up in real-time
- 10 seconds @ 30 fps (300 frames total)

## Customization Parameters

### Location in Code

All parameters are in the `main()` function of each script.

### Key Parameters

#### Grid and Agents
```python
n_grid = 32       # Grid resolution (32×32 = 1024 points)
n_agents = 16     # Number of actuators (4×4 grid)
T_steps = 300     # Simulation timesteps
```

**Note**: `n_grid` and `n_agents` must match the trained model. If you train with different values, update these accordingly.

#### Random Seed (Scenario Selection)
```python
key = jax.random.PRNGKey(1234)  # Change to generate different scenarios
```

Try different seeds to get different initial/target conditions:
- `1234` - Default (scenario 1 from original)
- `42`, `999`, `2024`, etc. - Different scenarios

#### Timesteps for Field Plots (visualize.py only)
```python
timesteps = get_log_timesteps(T_steps, n_points=6)
```

Modify `n_points` to change number of columns:
- `n_points=6` - Default (6 timesteps, logarithmically spaced)
- `n_points=8` - More detailed evolution
- `n_points=4` - Fewer snapshots

The `get_log_timesteps()` function emphasizes early dynamics (first 80 steps are more densely sampled).

#### Animation Settings (animate.py only)
```python
fps = 30          # Frames per second
duration = 10     # Animation duration in seconds
```

Quality settings:
```python
# GIF
anim.save(gif_path, writer='pillow', fps=fps, dpi=150)

# MP4
anim.save(mp4_path, writer='ffmpeg', fps=fps, dpi=200, ...)
```

Higher DPI = larger file size but better quality for presentations.

### GRF Generation Parameters (Advanced)

To control the smoothness/complexity of scenarios, modify `data_utils.generate_grf_2d()` calls:

```python
xx, yy, z_init = data_utils.generate_grf_2d(
    k1,
    n_points=n_grid,
    length_scale=0.4,  # Default: 0.4 (increase for smoother fields)
    sigma=1.0          # Default: 1.0 (increase for larger amplitude)
)
```

**Length scale effects**:
- `0.2-0.3`: More detailed, complex patterns
- `0.4-0.5`: Moderate smoothness (default)
- `0.6-0.8`: Very smooth, gradual variations

**Sigma effects**:
- `0.5`: Gentle temperature variations
- `1.0`: Moderate variations (default)
- `2.0`: Dramatic temperature differences

## Actuator Visualization

### Representation Methods

**In visualize.py**:
- **DPC Controlled row**: Dots colored by control intensity `u` (YlOrRd colormap)
  - Yellow = low control, Red = high control
  - Size: 25 points (small, uniform)

- **Tracking Error row**: Cyan dots with black edges
  - Shows actuator positions on error field
  - Size: 20 points

**In animate.py**:
- **DPC Controlled panel**: Animated dots with dynamic color
  - Color changes with control intensity
  - Size: 30 points

- **Tracking Error panel**: Cyan animated dots
  - Size: 25 points

All dots use `edgecolors='black'` with `linewidths=0.5-0.6` for visibility.

### Field Interpolation

**Setting**: `interpolation='nearest'`

This ensures **no interpolation** between grid points (as requested). The field is displayed with sharp pixel boundaries matching the 32×32 grid.

Other options:
- `'none'`: No interpolation (same as 'nearest' for our case)
- `'bilinear'`: Smooth interpolation (NOT recommended per your requirements)

## Output for Publications

### Recommended Settings

**For papers/conference proceedings**:
- Use **PDF** output from `visualize.py` (vector graphics scale infinitely)
- DPI doesn't matter for PDF (vectors), but keep at 300 for embedded rasters

**For presentations/posters**:
- Use **PNG** from `visualize.py` at 300 DPI
- Use **MP4** from `animate.py` at 200 DPI or higher

**For web/supplementary materials**:
- Use **PNG** at 150-200 DPI (smaller file size)
- Use **GIF** from `animate.py` (universally compatible)

### Font and Style

Both scripts use publication-quality styling:
- Font family: Times New Roman (serif)
- Consistent font sizes (10-14pt for figures, 12-16pt for animations)
- Grid lines with 0.3 alpha for readability
- No unnecessary decorations

## Troubleshooting

### Error: "centralized_params_heat2d.msgpack not found"
**Solution**: Run training first:
```bash
cd examples/heat2D/centralized
python train.py
```

### Error: "ffmpeg not available" (animate.py)
**Solution**: MP4 save will fail, but GIF will still be created. To enable MP4:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Grid size mismatch
**Error**: Shape errors when loading parameters

**Solution**: Ensure `n_grid` matches the training configuration. Check `train.py` for the value used during training.

### Agent count mismatch (centralized only)
**Error**: Parameter shape errors

**Solution**: For centralized models, `n_agents` must match training. Decentralized models support zero-shot scaling (can use any number of agents at inference).

### Slow GRF generation
**Issue**: `generate_grf_2d()` takes long

**Solution**: This is expected for high-res grids. The eigendecomposition is O(N^6). For n_grid=32, it takes ~1-2 seconds per sample. For faster testing, pre-generate data using `data/generate_dataset.py`.

## Comparing Centralized vs Decentralized

To generate both and compare:
```bash
cd examples/heat2D

# Centralized
cd centralized
python visualize.py
python animate.py

# Decentralized
cd ../decentralized
python visualize.py
python animate.py
```

The decentralized controller typically shows:
- Similar or better tracking performance (due to local sensing)
- More distributed control effort
- Scalability to different agent counts

## File Structure

```
examples/heat2D/
├── centralized/
│   ├── train.py                    # Training script
│   ├── visualize.py                # Static figure generation
│   ├── animate.py                  # Animation generation
│   ├── data_utils.py               # GRF data generation
│   ├── dynamics_dual.py            # PDE dynamics
│   └── centralized_params_heat2d.msgpack  # Trained parameters
├── decentralized/
│   ├── train.py
│   ├── visualize.py
│   ├── animate.py
│   ├── data_utils.py
│   ├── dynamics_dual.py
│   └── decentralized_params_heat2d.msgpack
└── data/
    ├── generate_dataset.py         # Offline dataset generator
    └── heat2d_dataset_32x32.npz    # Pre-generated data
```

## Advanced: Batch Generation

To generate multiple scenarios programmatically:

```python
# In visualize.py or animate.py, modify:
for seed in [42, 123, 456, 789, 1234]:
    key = jax.random.PRNGKey(seed)
    # ... generate and save with unique filenames
    plt.savefig(f'heat2d_visualization_seed{seed}.pdf')
```

This creates a gallery of different test cases.

---

For questions or issues, refer to the main project README or check the 1D examples in `examples/heat1/` for reference.
