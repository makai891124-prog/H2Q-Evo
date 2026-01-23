import unittest
import numpy as np
from h2q_project.knot import trefoil_knot

class TestKnot(unittest.TestCase):

    def test_trefoil_knot_shape(self):
        # Basic test to ensure the trefoil knot function returns the correct shape
        num_points = 100
        knot = trefoil_knot(num_points)
        self.assertEqual(knot.shape, (num_points, 3))

    def test_trefoil_knot_values(self):
        # Check a few specific points to ensure they are within a reasonable range
        num_points = 50
        knot = trefoil_knot(num_points)
        self.assertTrue(np.all(knot >= -2) and np.all(knot <= 2))


if __name__ == '__main__':
    unittest.main()