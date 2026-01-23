import pytest
import numpy as np

# This file can contain pytest configuration and fixtures
# For example:

@pytest.fixture
def sample_quaternion():
    return np.array([1.0, 0.0, 0.0, 0.0])
