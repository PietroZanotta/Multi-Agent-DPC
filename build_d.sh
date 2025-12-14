#!/bin/bash
# Build all tesseracts in the project using the Pasteur Tesseract CLI

set -e

echo "========================================="
echo "Building Tesseract Hackathon Template"
echo "========================================="
echo ""

# Path to the Tesseract Core CLI inside your conda env
TESS_CLI="/opt/anaconda3/envs/tess/bin/tesseract"

# Check that the binary exists
if [ ! -x "$TESS_CLI" ]; then
    echo "Error: Pasteur Tesseract CLI not found at $TESS_CLI"
    echo "Make sure tesseract-core is installed in the tess environment."
    exit 1
fi

# Check that this CLI looks like the Pasteur one
if ! "$TESS_CLI" --help | grep -q "autodiff"; then
    echo "Error: CLI at $TESS_CLI does not look like Pasteur Tesseract Core."
    echo "You may still be hitting the OCR tesseract or a wrong install."
    exit 1
fi

for tess_dir in tesseracts/*/
do
    echo "Building ${tess_dir}"
    "$TESS_CLI" build "${tess_dir}"
    echo "✓ ${tess_dir} built successfully"
    echo ""
done

echo "========================================="
echo "✓ All tesseracts built successfully!"
echo "========================================="