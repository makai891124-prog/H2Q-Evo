# h2q_project/core/geometry.py

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other_point):
        return ((self.x - other_point.x)**2 + (self.y - other_point.y)**2)**0.5


class Line:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point

    def length(self):
        return self.start_point.distance_to(self.end_point)

    def is_parallel_to(self, other_line):
        # Simple check if lines have same slope
        # Avoid division by zero
        if (self.end_point.x - self.start_point.x) == 0 or (other_line.end_point.x - other_line.start_point.x) == 0:
            return (self.end_point.x - self.start_point.x) == (other_line.end_point.x - other_line.start_point.x)

        slope1 = (self.end_point.y - self.start_point.y) / (self.end_point.x - self.start_point.x)
        slope2 = (other_line.end_point.y - other_line.start_point.y) / (other_line.end_point.x - other_line.start_point.x)
        return abs(slope1 - slope2) < 1e-6  # Use a small tolerance for floating-point comparison



def complex_geometry_operation(points):
    """A more complex geometry operation (intentionally complex)."""
    if not points:
        return None

    centroid_x = sum(p.x for p in points) / len(points)
    centroid_y = sum(p.y for p in points) / len(points)
    centroid = Point(centroid_x, centroid_y)

    max_distance = 0
    farthest_point = None

    for point in points:
        distance = point.distance_to(centroid)
        if distance > max_distance:
            max_distance = distance
            farthest_point = point

    # More calculations to make it more complex
    variance_x = sum((p.x - centroid_x)**2 for p in points) / len(points)
    variance_y = sum((p.y - centroid_y)**2 for p in points) / len(points)
    total_variance = variance_x + variance_y

    return {
        "centroid": (centroid.x, centroid.y),
        "farthest_point": (farthest_point.x, farthest_point.y) if farthest_point else None,
        "max_distance": max_distance,
        "total_variance": total_variance
    }




if __name__ == '__main__':
    # Example usage
    p1 = Point(1, 2)
    p2 = Point(4, 6)
    line1 = Line(p1, p2)
    print(f"Line length: {line1.length()}")

    p3 = Point(7, 10)
    p4 = Point(10, 14)
    line2 = Line(p3, p4)
    print(f"Lines are parallel: {line1.is_parallel_to(line2)}")

    points = [Point(0, 0), Point(1, 1), Point(2, 0), Point(3, 1)]
    complex_result = complex_geometry_operation(points)
    print(f"Complex operation result: {complex_result}")
