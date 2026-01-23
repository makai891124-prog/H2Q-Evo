import unittest
from h2q_project.core.kernel import Kernel
from h2q_project.core.geometry import Point

class TestKernel(unittest.TestCase):

    def test_nearest_neighbor(self):
        points = [Point(1, 1), Point(2, 2), Point(3, 3)]
        kernel = Kernel(points)
        query_point = Point(1.5, 1.5)
        nearest = kernel.nearest_neighbor(query_point)
        self.assertEqual(nearest, Point(1, 1))

    def test_nearest_neighbor_empty(self):
        points = []
        kernel = Kernel(points)
        query_point = Point(1.5, 1.5)
        nearest = kernel.nearest_neighbor(query_point)
        self.assertIsNone(nearest)

    def test_performance_analysis(self):
        points = [Point(1, 1), Point(2, 2), Point(3, 3)]
        kernel = Kernel(points)
        query_point = Point(1.5, 1.5)
        analysis = kernel.performance_analysis(query_point)
        self.assertIn("nearest_neighbor_distance", analysis)
        self.assertIn("num_points", analysis)
        self.assertIn("average_distance", analysis)
        self.assertEqual(analysis["num_points"], 3)

    def test_performance_analysis_empty(self):
        points = []
        kernel = Kernel(points)
        query_point = Point(1.5, 1.5)
        analysis = kernel.performance_analysis(query_point)
        self.assertEqual(analysis["nearest_neighbor_distance"], 0.0)
        self.assertEqual(analysis["num_points"], 0)
        self.assertEqual(analysis["average_distance"], 0.0)

if __name__ == '__main__':
    unittest.main()
