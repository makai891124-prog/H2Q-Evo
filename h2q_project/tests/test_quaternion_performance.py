import pytest
import numpy as np
from h2q_project.quaternion import Quaternion
from h2q_project.fractal_geometry import julia_set

@pytest.fixture(params=[100, 1000, 10000])
def size(request):
    return request.param

@pytest.fixture()
def quaternion_values():
    return (Quaternion(1, 2, 3, 4), Quaternion(5, 6, 7, 8))


def test_quaternion_multiplication(benchmark, quaternion_values):
    q1, q2 = quaternion_values
    benchmark(lambda: q1 * q2)


def test_quaternion_addition(benchmark, quaternion_values):
    q1, q2 = quaternion_values
    benchmark(lambda: q1 + q2)


def test_quaternion_conjugate(benchmark, quaternion_values):
    q1, _ = quaternion_values
    benchmark(lambda: q1.conjugate())


def test_quaternion_norm(benchmark, quaternion_values):
    q1, _ = quaternion_values
    benchmark(lambda: q1.norm())


def test_julia_set_generation(benchmark, size):
    width, height = size, size
    c = complex(-0.4, 0.6)
    benchmark(julia_set, width, height, c)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
