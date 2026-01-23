import unittest
import numpy as np
from h2q_project.quaternion import Quaternion

class TestQuaternion(unittest.TestCase):

    def test_magnitude(self):
        q = Quaternion(1, 2, 3, 4)
        self.assertAlmostEqual(q.magnitude(), np.sqrt(30))

    def test_normalize(self):
        q = Quaternion(1, 2, 3, 4)
        q_normalized = q.normalize()
        magnitude = q_normalized.magnitude()
        self.assertAlmostEqual(magnitude, 1.0)

        # Test normalization of a zero quaternion
        q_zero = Quaternion(0, 0, 0, 0)
        q_zero_normalized = q_zero.normalize()
        self.assertEqual(q_zero_normalized.w, 0)
        self.assertEqual(q_zero_normalized.x, 0)
        self.assertEqual(q_zero_normalized.y, 0)
        self.assertEqual(q_zero_normalized.z, 0)

if __name__ == '__main__':
    unittest.main()
