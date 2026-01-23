import unittest
import numpy as np
from h2q_project.core.h2q_kernel import H2QKernel, SelfReflectionModule

class TestH2QKernel(unittest.TestCase):

    def setUp(self):
        self.kernel = H2QKernel()

    def test_distance(self):
        point1 = np.array([1, 2, 3])
        point2 = np.array([4, 5, 6])
        expected_distance = np.sqrt(27)
        self.assertAlmostEqual(self.kernel.distance(point1, point2), expected_distance)

    def test_is_within_tolerance(self):
        value = 1.0001
        expected_value = 1.0
        self.assertTrue(self.kernel.is_within_tolerance(value, expected_value))

        value = 1.001
        expected_value = 1.0
        self.assertFalse(self.kernel.is_within_tolerance(value, expected_value))

    def test_reflect_point(self):
        point = np.array([1, 1, 1])
        normal = np.array([0, 1, 0])
        origin = np.array([0, 0, 0])
        expected_reflected_point = np.array([1, -1, 1])
        reflected_point = self.kernel.reflect_point(point, normal,origin)
        np.testing.assert_array_equal(reflected_point, expected_reflected_point)

class TestSelfReflectionModule(unittest.TestCase):

    def setUp(self):
        self.kernel = H2QKernel()
        self.reflection_module = SelfReflectionModule(self.kernel)

    def test_assess_reflection(self):
        point = np.array([1, 1, 1])
        normal = np.array([0, 1, 0])
        origin = np.array([0, 0, 0])
        reflected_point_calculated = np.array([1, -1, 1])
        self.assertTrue(self.reflection_module.assess_reflection(point, normal, reflected_point_calculated, origin))

        reflected_point_calculated = np.array([1.1, -1.1, 1.1]) # Slightly incorrect reflection
        self.assertFalse(self.reflection_module.assess_reflection(point, normal, reflected_point_calculated, origin))
