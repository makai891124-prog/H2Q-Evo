from h2q_project.core.geometry import Point, distance

class Kernel:
    def __init__(self, points):
        self.points = points

    def nearest_neighbor(self, query_point: Point) -> Point:
        """Finds the nearest neighbor to the query point.
        """
        if not self.points:
            return None

        nearest = self.points[0]
        min_distance = distance(query_point, nearest)

        for point in self.points[1:]:
            dist = distance(query_point, point)
            if dist < min_distance:
                min_distance = dist
                nearest = point

        return nearest

    def performance_analysis(self, query_point: Point) -> dict:
        """Performs a lightweight performance analysis based on the
        geometric kernel.  This is a self-reflection module.
        """
        nearest_neighbor = self.nearest_neighbor(query_point)
        nearest_neighbor_distance = distance(query_point, nearest_neighbor)

        analysis = {
            "nearest_neighbor_distance": nearest_neighbor_distance,
            "num_points": len(self.points),
            "average_distance": self._average_distance(query_point)
        }
        return analysis

    def _average_distance(self, query_point: Point) -> float:
        """Calculates the average distance to all points.
        """
        if not self.points:
            return 0.0

        total_distance = sum(distance(query_point, point) for point in self.points)
        return total_distance / len(self.points)
