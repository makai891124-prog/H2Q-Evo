import numpy as np

# Using generators for potentially large point clouds to save memory.
def generate_cube_points(center=(0, 0, 0), size=1):
    """Generates points for a cube centered at the given center.

    Args:
        center (tuple): The center of the cube (x, y, z).
        size (float): The side length of the cube.

    Yields:
        tuple: Coordinates of a point on the cube.
    """
    x0, y0, z0 = center
    half_size = size / 2.0

    # Define the vertices of the cube
    vertices = [
        (x0 - half_size, y0 - half_size, z0 - half_size),
        (x0 + half_size, y0 - half_size, z0 - half_size),
        (x0 + half_size, y0 + half_size, z0 - half_size),
        (x0 - half_size, y0 + half_size, z0 - half_size),
        (x0 - half_size, y0 - half_size, z0 + half_size),
        (x0 + half_size, y0 - half_size, z0 + half_size),
        (x0 + half_size, y0 + half_size, z0 + half_size),
        (x0 - half_size, y0 + half_size, z0 + half_size),
    ]

    # Iterate through the vertices and yield them
    for vertex in vertices:
        yield vertex


def calculate_bounding_box(points):
    """Calculates the bounding box of a set of points.

    Args:
        points (iterable): An iterable of 3D points (e.g., a list of tuples).

    Returns:
        tuple: A tuple containing the minimum and maximum coordinates of the bounding box.
               (min_x, min_y, min_z), (max_x, max_y, max_z)
    """
    # Convert the iterable to a list to allow for iteration if needed.
    points = list(points)

    if not points:
        return None, None

    # Initialize min/max with the first point.
    min_x = max_x = points[0][0]
    min_y = max_y = points[0][1]
    min_z = max_z = points[0][2]

    for point in points:
        x, y, z = point
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)

        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)
