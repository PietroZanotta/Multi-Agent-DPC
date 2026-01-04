# 2D Heat Equation Control - Quick Start Guide

## Quick Pipeline Test (Recommended First Step)

Before running the full training (which takes ~1 hour), verify the pipeline works with a quick test:

```bash
# From repository root
./run_heat2d_test.sh
```

**OR** run the main script with test flag:

```bash
./run_heat2d.sh --test
```

**OR** run individual components:

```bash
# Build tesseracts
tesseract build tesseracts/solverHeat2D_centralized/
tesseract build tesseracts/solverHeat2D_decentralized/

# Test centralized training (1 sample, 10 epochs)
cd examples/heat2D/centralized/
python train.py --test

# Test decentralized training
cd ../decentralized/
python train.py --test

# Test visualization
cd ../centralized/
python visualize.py
```

### Quick Test Parameters

- **Dataset**: 1 initial condition + 1 target
- **Epochs**: 10 (instead of 500)
- **Timesteps**: 100 (instead of 300)
- **Expected time**: ~2-5 minutes total

### What the Quick Test Verifies

✅ Tesseracts build correctly
✅ 2D ADI solver works
✅ 2D GRF data generation works
✅ Policy networks initialize correctly
✅ Training loop runs without errors
✅ Gradient flow works end-to-end
✅ Visualization works with Tesseracts

---

## Full Training

Once the quick test passes, run full training:

```bash
./run_heat2d.sh
```

### Full Training Parameters

- **Dataset**: 5000 samples (5000 init + 5000 targets)
- **Epochs**: 500
- **Timesteps**: 300
- **Expected time**: ~30-60 minutes per controller

### What Full Training Produces

**Centralized**:
- `centralized_params_heat2d.msgpack` (~200KB)
- `training_metrics_heat2d.png` (loss curves)
- `heat2d_centralized_visualization.png` (2 test scenarios)
- `heat2d_animation.mp4` and `.gif`

**Decentralized**:
- `decentralized_params_heat2d.msgpack` (~100KB)
- `training_metrics_heat2d_decentralized.png`
- `heat2d_decentralized_visualization.png`
- `heat2d_animation.mp4` and `.gif`

---

## Manual Training

For more control, run steps individually:

### 1. Build Tesseracts

```bash
tesseract build tesseracts/solverHeat2D_centralized/
tesseract build tesseracts/solverHeat2D_decentralized/
```

### 2. Train Controllers

**Centralized** (full training):
```bash
cd examples/heat2D/centralized/
python train.py
```

**Centralized** (quick test):
```bash
python train.py --test
```

**Decentralized** (full training):
```bash
cd examples/heat2D/decentralized/
python train.py
```

**Decentralized** (quick test):
```bash
python train.py --test
```

### 3. Visualize Results

**Static visualization**:
```bash
cd examples/heat2D/centralized/
python visualize.py  # Creates heat2d_centralized_visualization.png
```

**Animation** (requires ffmpeg):
```bash
python animate.py  # Creates .mp4 and .gif
```

---

## Troubleshooting

### Issue: Data generation is slow

The eigendecomposition of a 4096×4096 matrix (64² grid points) takes time.

**Solutions**:
- Use `--test` mode first (only 1 sample)
- Pre-generate dataset once and save to disk
- Reduce grid size (modify `n_grid` in solver.py)

### Issue: Training diverges (NaN losses)

**Solutions**:
- Reduce learning rate in train.py: `1e-3` → `5e-4`
- Increase gradient clipping: `1.0` → `0.5`
- Check boundary conditions in generated data
- Use `--test` mode to debug with smaller problem

### Issue: Out of memory

**Solutions**:
- Reduce batch size in train.py: `16` → `8`
- Reduce T_steps: `300` → `200`
- Use `--test` mode (batch_size=1, T_steps=100)

### Issue: Tesseract build fails

**Solutions**:
- Ensure tesseract-core is installed: `pip install tesseract-core`
- Check that JAX is installed: `pip install jax flax`
- Verify models/policy.py was copied to tesseract directories

---

## Understanding the Results

### Training Metrics

The training metrics plot shows 4 subplots:
1. **Total Loss**: Should decrease over time (log scale)
2. **Tracking Loss**: Main objective (MSE to target)
3. **Effort Loss**: Actuation cost
4. **Collision Loss**: Agent overlap penalty

**Good training**:
- Total loss decreases smoothly
- Tracking loss reaches <0.1 (centralized) or <0.2 (decentralized)
- Collision loss near zero (agents don't overlap)

### Visualization

The visualization shows:
- **Column 1**: Initial state with agent positions
- **Column 2**: Target state
- **Columns 3-7**: Trajectory at t=0, 75, 150, 225, 299

**Good control**:
- Temperature field evolves from initial → target
- Agents move strategically to guide the field
- Final state closely matches target (similar colors)

### Animation

The animation shows:
- **Left panel**: Current state evolving over time
- **Right panel**: Fixed target state
- **Black X markers**: Mobile actuator positions

**Good control**:
- Agents coordinate to shape the temperature field
- Actuators move to regions that need heating/cooling
- Field converges to target by end of trajectory

---

## Performance Benchmarks

**Expected training times** (on modern GPU):

| Mode | Centralized | Decentralized |
|------|-------------|---------------|
| Test (1 sample, 10 epochs) | ~1-2 min | ~1-2 min |
| Full (5000 samples, 500 epochs) | ~30-45 min | ~25-40 min |

**Expected final losses** (after 500 epochs):

| Metric | Centralized | Decentralized |
|--------|-------------|---------------|
| Tracking Loss | <0.1 | <0.2 |
| Total Loss | <1.0 | <2.0 |

**Model sizes**:
- Centralized: ~200KB (global sensing)
- Decentralized: ~100KB (local sensing)

---

## Next Steps

After successful training:

1. **Experiment with different targets**: Modify data_utils.py to generate different patterns
2. **Adjust agent count**: Change `n_agents` in train.py (16 → 25 for 5×5 grid)
3. **Modify physics**: Change `nu`, `sigma`, or `dt` in solver.py
4. **Compare centralized vs decentralized**: Visualizations side-by-side
5. **Test zero-shot scalability** (decentralized): Train with N=16, deploy with N=25

---

## Command Reference

```bash
# Quick test (recommended first)
./run_heat2d_test.sh

# Full workflow
./run_heat2d.sh

# Full workflow in test mode
./run_heat2d.sh --test

# Individual training (full)
cd examples/heat2D/centralized/
python train.py

# Individual training (test)
python train.py --test

# Visualization only (requires trained params)
python visualize.py
python animate.py
```
