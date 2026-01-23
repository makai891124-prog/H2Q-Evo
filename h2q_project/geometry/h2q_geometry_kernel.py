class H2QGeometryKernel:
    def __init__(self, points):
        self.points = points
        self.num_points = len(points)

    def calculate_pairwise_distances(self):
        """Calculates pairwise distances between points.

        Optimized for memory usage by storing only the upper triangle
        of the distance matrix. This assumes the distance function
        is symmetric.
        """

        distances = []
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                dist = self.distance(self.points[i], self.points[j])
                distances.append((i, j, dist))
        return distances

    def distance(self, point1, point2):
        """Calculates the Euclidean distance between two points.
        Override this method for different distance metrics.
        """
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def find_neighbors_within_radius(self, query_point, radius):
        """Finds all points within a given radius of a query point.

        Uses the pre-computed pairwise distances for efficiency.  This approach trades
        memory for speed, but the memory optimization in `calculate_pairwise_distances`
        mitigates the cost.
        """
        neighbors = []
        pairwise_distances = self.calculate_pairwise_distances()
        for i, j, dist in pairwise_distances:
            if i == query_point or j == query_point:
                other_point_index = j if i == query_point else i
                if dist <= radius:
                    neighbors.append((other_point_index, dist))

        # Now consider the distances from the query point to itself (distance 0).
        # This handles the case where the query point is one of the original points.
        # But avoids adding it multiple times in case of duplicate points
        if query_point < self.num_points:
            neighbors.append((query_point, 0.0))
        return neighbors

if __name__ == '__main__':
    # Example usage
    points = [(1, 2), (3, 4), (5, 6), (1,2)]
    kernel = H2QGeometryKernel(points)

    # Calculate pairwise distances
    pairwise_distances = kernel.calculate_pairwise_distances()
    print("Pairwise Distances:", pairwise_distances)

    # Find neighbors within a radius
    query_point = 0
    radius = 3
    neighbors = kernel.find_neighbors_within_radius(query_point, radius)
    print(f"Neighbors of point {query_point} within radius {radius}: {neighbors}")
