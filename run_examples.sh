#!/bin/bash

# Define the list of directories
repos=(
    "./examples/heat1d/centralized/"
    "./examples/heat1d/decentralized/"
    "./examples/fkpp1d/centralized/"
    "./examples/fkpp1d/decentralized/"
)

# --- STAGE 1: DATA PREPARATION ---
echo "================================================"
echo "STAGE 1: Preparing data for all experiments"
echo "================================================"

for repo in "${repos[@]}"; do
    if [ -d "$repo" ]; then
        echo "Preparing data in: $repo"
        cd "$repo" && python3 data_utils.py
        cd - > /dev/null
    else
        echo "Directory $repo not found. Skipping..."
    fi
done

# --- STAGE 2: TRAINING AND VISUALIZATION ---
echo ""
echo "================================================"
echo "STAGE 2: Running Training and Visualizations"
echo "================================================"

for repo in "${repos[@]}"; do
    if [ -d "$repo" ]; then
        echo "------------------------------------------------"
        echo "Processing: $repo"
        cd "$repo"
        
        # Run training, then visualization if training succeeds
        python3 train.py && python3 visualize.py
        
        if [ $? -eq 0 ]; then
            echo "Success: $repo completed."
        else
            echo "Error: Training or Visualization failed in $repo."
        fi
        
        cd - > /dev/null
    fi
done

echo "================================================"
echo "All stages finished."