#!/bin/bash

# Activate the virtual environment (if applicable)
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

# Install coverage if not already installed
pip install coverage

# Run tests with coverage
coverage run -m unittest discover -s h2q_project/tests

# Generate coverage report
coverage report -m

# Optionally, generate HTML report:
# coverage html