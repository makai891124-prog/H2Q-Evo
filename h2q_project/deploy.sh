#!/bin/bash

# Script to deploy the application using Docker

set -e

# Navigate to the docker directory
cd h2q_project/docker

# Build the docker image
./build_image.sh

# Optionally, tag the image for a specific registry
# docker tag h2q-app:latest your-registry/h2q-app:latest

# Navigate back to the project root
cd ../..

# Run the Docker container
docker run -d -p 80:80 h2q-app:latest

echo "Application deployed successfully."
