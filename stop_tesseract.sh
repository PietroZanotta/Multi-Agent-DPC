#!/bin/bash

# Usage: ./stop_tesseract.sh vlasov:latest
IMAGE_NAME=$1

if [ -z "$IMAGE_NAME" ]; then
    echo "Usage: ./stop_tesseract.sh <image_name>"
    exit 1
fi

SAFE_NAME=$(echo $IMAGE_NAME | tr ':.' '_')

echo "🛑 Shutting down Tesseract container: $SAFE_NAME..."

# Find the container by the name we gave it
ID=$(docker ps -aq -f name=$SAFE_NAME)

if [ ! -z "$ID" ]; then
    docker rm -f "$ID"
    # Clean up the local port just in case
    fuser -k 8080/tcp > /dev/null 2>&1
    echo "✨ GPU memory released and container removed."
else
    echo "❓ No running container found for $IMAGE_NAME."
fi