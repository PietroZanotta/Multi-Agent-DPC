#!/bin/bash

# Define the list of directories
repos=(
    "./examples/heat1d/centralized/"
    "./examples/heat1d/decentralized/"
    "./examples/fkpp1d/centralized/"
    "./examples/fkpp1d/decentralized/"
)

echo "================================================"
echo "Running Training and Visualizations"
echo "================================================"

for repo in "${repos[@]}"; do
    if [ -d "$repo" ]; then
        echo "------------------------------------------------"
        echo "Processing: $repo"
        
        # Move into the directory
        cd "$repo" || { echo "Could not enter $repo"; continue; }
        
        # Execute scripts in sequence. 
        # Using '&&' ensures that if one fails, the rest are skipped for this repo.
        python3 train.py && \
        python3 visualize.py && \
        python3 visualize_conference.py && \
        python3 visualize_comparison.py && \
        python3 animate.py
        
        # Capturing the exit status of the chain
        if [ $? -eq 0 ]; then
            echo ">>> Success: All scripts completed for $repo."
        else
            echo ">>> Error: A script failed in $repo. Skipping to next repository..."
        fi
        
        # Return to original directory
        cd - > /dev/null
    else
        echo "Warning: Directory $repo not found. Skipping..."
    fi
done

echo ""
echo "================================================"
echo "All tasks finished."