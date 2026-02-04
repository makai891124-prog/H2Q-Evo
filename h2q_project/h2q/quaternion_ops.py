"""
Quaternion Operations Module (NumPy)
Input format: [w, x, y, z]
"""

from __future__ import annotations

import numpy as np


def quaternion_multiply(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_conjugate(q):
    """Return conjugate [w, -x, -y, -z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_norm(q):
    """Return quaternion magnitude."""
    w, x, y, z = q
    return float(np.sqrt(w * w + x * x + y * y + z * z))


def quaternion_normalize(q):
    """Return unit quaternion; fall back to identity if norm is zero."""
    n = quaternion_norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array(q, dtype=np.float64) / n


def quaternion_slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.

    Args:
        q1, q2: quaternions [w, x, y, z]
        t: interpolation factor in [0, 1]

    Returns:
        Interpolated unit quaternion.
    """
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)

    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return quaternion_normalize(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q3 = q2 - q1 * dot
    q3 = quaternion_normalize(q3)

    return q1 * np.cos(theta) + q3 * np.sin(theta)