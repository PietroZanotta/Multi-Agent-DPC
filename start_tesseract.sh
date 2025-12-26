#!/bin/bash

# Usage: ./start_tesseract.sh vlasov:latest
IMAGE_NAME=$1
PORT=8080

if [ -z "$IMAGE_NAME" ]; then
    echo "Usage: ./start_tesseract.sh <image_name>"
    exit 1
fi

# Create a name for the container
SAFE_NAME=$(echo $IMAGE_NAME | tr ':.' '_')

echo "🚀 Starting Tesseract for image: $IMAGE_NAME..."

# 1. Run tesseract in the background
# We pipe output to a log file so it doesn't clutter your terminal
nohup tesseract serve --gpus all "$IMAGE_NAME" --port $PORT > tesseract_server.log 2>&1 &

echo "⏳ Waiting for container to spin up..."
sleep 5 # Give Docker a moment to actually create the container

# 2. Find the container ID created from this image and rename it
CONTAINER_ID=$(docker ps -q --filter "ancestor=$IMAGE_NAME" | head -n 1)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ Error: Could not find the started container. Check tesseract_server.log"
    exit 1
fi

docker rename "$CONTAINER_ID" "$SAFE_NAME"
echo "✅ Container identified and tracked as: $SAFE_NAME"

# 3. Health Check
echo "⏳ Performing GPU/JAX health check..."
until $(curl --output /dev/null --silent --head --fail http://localhost:$PORT/docs); do
    printf '.'
    sleep 2
done

echo -e "\n✅ Tesseract is UP and ready on port $PORT!"