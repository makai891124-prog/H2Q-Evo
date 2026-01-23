import unittest
import numpy as np
from h2q_project.geometry import Point, Sphere


class TestPoint(unittest.TestCase):

    def test_point_creation(self):
        p = Point(1, 2, 3)
        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)
        self.assertEqual(p.z, 3)

    def test_point_distance(self):
        p1 = Point(0, 0, 0)
        p2 = Point(3, 4, 0)
        self.assertEqual(p1.distance(p2), 5)


class TestSphere(unittest.TestCase):

    def test_sphere_creation(self):
        center = Point(0, 0, 0)
        sphere = Sphere(center, 5)
        self.assertEqual(sphere.center.x, 0)
        self.assertEqual(sphere.radius, 5)

    def test_sphere_contains(self):
        center = Point(0, 0, 0)
        sphere = Sphere(center, 5)
        p1 = Point(1, 1, 1)
        p2 = Point(6, 0, 0)
        self.assertTrue(sphere.contains(p1))
        self.assertFalse(sphere.contains(p2))


if __name__ == '__main__':
    unittest.main()