import numpy as np


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculates the Euclidean distance between two 3D points.

    Args:
        point1 (np.ndarray): The first point.
        point2 (np.ndarray): The second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(point1 - point2)


def calculate_midpoint(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Calculates the midpoint between two 3D points.

    Args:
        point1 (np.ndarray): The first point.
        point2 (np.ndarray): The second point.

    Returns:
        np.ndarray: The midpoint between the two points.
    """
    return (point1 + point2) / 2


def rotate_vector(vector: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotates a vector using a rotation matrix.

    Args:
        vector (np.ndarray): The vector to rotate.
        rotation_matrix (np.ndarray): The rotation matrix.

    Returns:
        np.ndarray: The rotated vector.
    """
    return np.dot(rotation_matrix, vector)