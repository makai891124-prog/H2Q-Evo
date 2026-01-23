import pytest
import numpy as np
from h2q_project.geometry_utils import create_rotation_matrix_from_quaternion, transform_points


def test_create_rotation_matrix_from_quaternion():
    # Test with identity quaternion
    q = np.array([1, 0, 0, 0])
    rotation_matrix = create_rotation_matrix_from_quaternion(q)
    expected_matrix = np.eye(3)
    assert np.allclose(rotation_matrix, expected_matrix)

    # Test with a rotation quaternion (e.g., 90 degrees around x-axis)
    q = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
    rotation_matrix = create_rotation_matrix_from_quaternion(q)
    expected_matrix = np.array([[1, 0, 0],
                                [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                                [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
    assert np.allclose(rotation_matrix, expected_matrix)


def test_transform_points():
    # Test with identity matrix
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotation_matrix = np.eye(3)
    transformed_points = transform_points(rotation_matrix, points)
    assert np.allclose(transformed_points, points)

    # Test with a rotation matrix (e.g., 90 degrees around x-axis)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
    transformed_points = transform_points(rotation_matrix, points)
    expected_points = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(transformed_points, expected_points)
