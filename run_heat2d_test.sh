#!/bin/bash
# Quick test workflow for 2D Heat Equation Control
# Uses 1 sample and 10 epochs for fast pipeline verification

set -e  # Exit on error

echo "========================================"
echo "2D Heat Equation Control - QUICK TEST"
echo "========================================"
echo ""
echo "This script runs a quick test with:"
echo "  - 1 training sample (1 init + 1 target)"
echo "  - 10 epochs"
echo "  - T=100 timesteps"
echo "  - Should complete in ~2-5 minutes"
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
# STEP 2: Test Centralized Controller
# ============================================
echo "STEP 2: Testing Centralized Controller..."
echo "--------------------------------------"
cd examples/heat2D/centralized/

echo "Starting centralized test training..."
python train.py --test

echo "✓ Centralized test complete!"
echo "  Output: centralized_params_heat2d.msgpack"
echo ""

# ============================================
# STEP 3: Test Decentralized Controller
# ============================================
echo "STEP 3: Testing Decentralized Controller..."
echo "--------------------------------------"
cd ../decentralized/

echo "Starting decentralized test training..."
python train.py --test

echo "✓ Decentralized test complete!"
echo "  Output: decentralized_params_heat2d.msgpack"
echo ""

# ============================================
# STEP 4: Test Visualization (Centralized)
# ============================================
echo "STEP 4: Testing Visualization (Centralized)..."
echo "--------------------------------------"
cd ../centralized/

echo "Generating centralized visualizations (using Tesseract)..."
python visualize.py

echo "✓ Centralized visualization complete!"
echo "  Output: heat2d_centralized_visualization.png"
echo ""

# ============================================
# STEP 5: Test Visualization (Decentralized)
# ============================================
echo "STEP 5: Testing Visualization (Decentralized)..."
echo "--------------------------------------"
cd ../decentralized/

echo "Generating decentralized visualizations (using Tesseract)..."
python visualize.py

echo "✓ Decentralized visualization complete!"
echo "  Output: heat2d_decentralized_visualization.png"
echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo "✓ QUICK TEST COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Pipeline verification successful!"
echo ""
echo "Summary of test outputs:"
echo "  Centralized:"
echo "    - examples/heat2D/centralized/centralized_params_heat2d.msgpack"
echo "    - examples/heat2D/centralized/training_metrics_heat2d.png"
echo "    - examples/heat2D/centralized/heat2d_centralized_visualization.png"
echo ""
echo "  Decentralized:"
echo "    - examples/heat2D/decentralized/decentralized_params_heat2d.msgpack"
echo "    - examples/heat2D/decentralized/training_metrics_heat2d_decentralized.png"
echo "    - examples/heat2D/decentralized/heat2d_decentralized_visualization.png"
echo ""
echo "To run full training (5000 samples, 500 epochs):"
echo "  ./run_heat2d.sh"
echo ""
