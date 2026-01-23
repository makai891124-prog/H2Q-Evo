#!/bin/bash

# Script to build the Docker image

set -e

# Define image name and tag
IMAGE_NAME="h2q-app"
IMAGE_TAG="latest"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Optionally, push the image to a registry (uncomment and configure)
# docker push ${IMAGE_NAME}:${IMAGE_TAG}

echo "Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
