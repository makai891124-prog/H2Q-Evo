import numpy as np


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Calculates the conjugate of a quaternion.

    Args:
        q (np.ndarray): The quaternion to conjugate (w, x, y, z).

    Returns:
        np.ndarray: The conjugate of the quaternion (w, -x, -y, -z).
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])



def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions.

    Args:
        q1 (np.ndarray): The first quaternion (w, x, y, z).
        q2 (np.ndarray): The second quaternion (w, x, y, z).

    Returns:
        np.ndarray: The product of the two quaternions.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])



def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a quaternion to a rotation matrix.

    Args:
        q (np.ndarray): The quaternion (w, x, y, z).

    Returns:
        np.ndarray: The rotation matrix.
    """
    w, x, y, z = q

    # Normalize the quaternion
    norm = np.linalg.norm(q)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Create the rotation matrix
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

    return rotation_matrix

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion.

    Args:
        q (np.ndarray): The quaternion (w, x, y, z).

    Returns:
        np.ndarray: The normalized quaternion.
    """
    norm = np.linalg.norm(q)
    return q / norm