import numpy as np

def trefoil_knot(num_points):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = np.sin(t) + 2 * np.sin(2*t)
    y = np.cos(t) - 2 * np.cos(2*t)
    z = -np.sin(3*t)
    return np.column_stack((x, y, z))