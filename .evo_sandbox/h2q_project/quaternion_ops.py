import numpy as np


def quaternion_multiply(q1, q2):
    """ 
    Multiply two quaternions. 
    Quaternions are represented as numpy arrays in the form [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    """ 
    Compute the conjugate of a quaternion. 
    Quaternions are represented as numpy arrays in the form [w, x, y, z].
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quaternion_magnitude(q):
    """ 
    Compute the magnitude of a quaternion.
    Quaternions are represented as numpy arrays in the form [w, x, y, z].
    """
    w, x, y, z = q
    return np.sqrt(w*w + x*x + y*y + z*z)


def quaternion_normalize(q):
    """ 
    Normalize a quaternion.
    Quaternions are represented as numpy arrays in the form [w, x, y, z].
    """
    magnitude = quaternion_magnitude(q)
    if magnitude == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Return identity quaternion if magnitude is zero to avoid division by zero
    return q / magnitude


def rotate_vector_by_quaternion(vector, quaternion):
    """ 
    Rotate a 3D vector by a quaternion.
    The vector is represented as a numpy array [x, y, z].
    The quaternion is represented as a numpy array [w, x, y, z].
    """
    vector = np.array(vector)
    quaternion = np.array(quaternion)
    q_vector = np.array([0.0, vector[0], vector[1], vector[2]])
    q_conjugate = quaternion_conjugate(quaternion)
    rotated_q = quaternion_multiply(quaternion_multiply(quaternion, q_vector), q_conjugate)
    return rotated_q[1:]
