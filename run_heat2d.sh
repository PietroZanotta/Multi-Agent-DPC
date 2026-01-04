#!/bin/bash
# Complete workflow for 2D Heat Equation Control Examples
# This script builds tesseracts, trains models, and generates visualizations
#
# Usage:
#   ./run_heat2d.sh          # Full training (5000 samples, 500 epochs)
#   ./run_heat2d.sh --test   # Quick test (1 sample, 10 epochs)

set -e  # Exit on error

# Parse arguments
TEST_MODE=""
if [[ "$1" == "--test" ]]; then
    TEST_MODE="--test"
    echo "========================================"
    echo "2D Heat Equation Control - TEST MODE"
    echo "========================================"
    echo ""
    echo "Running quick test (1 sample, 10 epochs)..."
else
    echo "========================================"
    echo "2D Heat Equation Control - Full Workflow"
    echo "========================================"
    echo ""
    echo "Running full training (5000 samples, 500 epochs)..."
fi
echo ""

# ============================================
# STEP 1: Build Tesseracts
# ============================================
echo "STEP 1: Building Tesseracts..."
echo "--------------------------------------"

echo "Building centralized 2D heat solver..."
tesseract build tesseracts/solverHeat2D_centralized/
echo "✓ Centralized tesseract built successfully"
echo ""

echo "Building decentralized 2D heat solver..."
tesseract build tesseracts/solverHeat2D_decentralized/
echo "✓ Decentralized tesseract built successfully"
echo ""

# ============================================
# STEP 2: Train Centralized Controller
# ============================================
echo "STEP 2: Training Centralized Controller..."
echo "--------------------------------------"
cd examples/heat2D/centralized/

if [[ -z "$TEST_MODE" ]]; then
    echo "Starting centralized training (this may take 30-60 minutes)..."
else
    echo "Starting centralized test training (~1-2 minutes)..."
fi
python train.py $TEST_MODE

echo "✓ Centralized training complete!"
echo "  Output: centralized_params_heat2d.msgpack"
echo "  Metrics: training_metrics_heat2d.png"
echo ""

# ============================================
# STEP 3: Train Decentralized Controller
# ============================================
echo "STEP 3: Training Decentralized Controller..."
echo "--------------------------------------"
cd ../decentralized/

if [[ -z "$TEST_MODE" ]]; then
    echo "Starting decentralized training (this may take 30-60 minutes)..."
else
    echo "Starting decentralized test training (~1-2 minutes)..."
fi
python train.py $TEST_MODE

echo "✓ Decentralized training complete!"
echo "  Output: decentralized_params_heat2d.msgpack"
echo "  Metrics: training_metrics_heat2d_decentralized.png"
echo ""

# ============================================
# STEP 4: Inference & Visualization (Centralized)
# ============================================
echo "STEP 4: Running Inference & Visualization (Centralized)..."
echo "--------------------------------------"
cd ../centralized/

echo "Generating centralized visualizations (using Tesseract)..."
python visualize.py

echo "Creating centralized animation (using Tesseract)..."
python animate.py

echo "✓ Centralized visualization complete!"
echo "  Output: heat2d_centralized_visualization.png"
echo "  Output: heat2d_animation.mp4"
echo "  Output: heat2d_animation.gif"
echo ""

# ============================================
# STEP 5: Inference & Visualization (Decentralized)
# ============================================
echo "STEP 5: Running Inference & Visualization (Decentralized)..."
echo "--------------------------------------"
cd ../decentralized/

echo "Generating decentralized visualizations (using Tesseract)..."
python visualize.py

echo "Creating decentralized animation (using Tesseract)..."
python animate.py

echo "✓ Decentralized visualization complete!"
echo "  Output: heat2d_decentralized_visualization.png"
echo "  Output: heat2d_animation.mp4"
echo "  Output: heat2d_animation.gif"
echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo "✓ ALL STEPS COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Summary of outputs:"
echo "  Centralized:"
echo "    - examples/heat2D/centralized/centralized_params_heat2d.msgpack"
echo "    - examples/heat2D/centralized/training_metrics_heat2d.png"
echo "    - examples/heat2D/centralized/heat2d_centralized_visualization.png"
echo "    - examples/heat2D/centralized/heat2d_animation.mp4"
echo "    - examples/heat2D/centralized/heat2d_animation.gif"
echo ""
echo "  Decentralized:"
echo "    - examples/heat2D/decentralized/decentralized_params_heat2d.msgpack"
echo "    - examples/heat2D/decentralized/training_metrics_heat2d_decentralized.png"
echo "    - examples/heat2D/decentralized/heat2d_decentralized_visualization.png"
echo "    - examples/heat2D/decentralized/heat2d_animation.mp4"
echo "    - examples/heat2D/decentralized/heat2d_animation.gif"
echo ""
