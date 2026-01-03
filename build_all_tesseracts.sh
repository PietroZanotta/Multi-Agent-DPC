#!/bin/bash

# Define the list of repositories based on the image
repos=(
    "tesseracts/solverFKPP_centralized"
    "tesseracts/solverFKPP_decentralized"
    "tesseracts/solverHeat_centralized"
    "tesseracts/solverHeat_decentralized"
    "tesseracts/solverHeat2D_centralized"
    "tesseracts/solverHeat2D_decentralized"
    "tesseracts/solverNS_shape"
    "tesseracts/solverNS_shape_centralized"
)

echo "================================================"
echo "Starting Tesseract Build Process"
echo "================================================"

for repo in "${repos[@]}"; do
    if [ -d "$repo" ]; then
        echo "------------------------------------------------"
        echo "Building: $repo"
        
        # Enter the directory
        cd "$repo" || { echo "Failed to enter $repo"; continue; }
        
        # Run the build command
        tesseract build .
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo ">>> Success: $repo built successfully."
        else
            echo ">>> Error: 'tesseract build' failed in $repo. Skipping..."
        fi
        
        # Return to the parent directory
        cd ../..
    else
        echo "Warning: Directory $repo not found. Skipping..."
    fi
done

echo ""
echo "================================================"
echo "All builds finished."