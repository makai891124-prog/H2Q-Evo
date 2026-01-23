#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Update package lists
sudo apt-get update

# Install Python 3 and pip
sudo apt-get install -y python3 python3-pip

# Upgrade pip
pip3 install --upgrade pip

# Install dependencies
pip3 install numpy

# Install test dependencies
pip3 install unittest

# Run tests
python3 -m unittest h2q_project/test_quaternion.py