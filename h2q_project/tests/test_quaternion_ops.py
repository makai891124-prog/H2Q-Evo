import unittest
import numpy as np
from h2q_project import quaternion_ops

class TestQuaternionOps(unittest.TestCase):

    def test_quaternion_multiply(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        expected = np.array([0, 1, 0, 0])
        result = quaternion_ops.quaternion_multiply(q1, q2)
        np.testing.assert_allclose(result, expected)

        q1 = np.array([0.707, 0.707, 0, 0])
        q2 = np.array([0.707, 0, 0.707, 0])
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternion_ops.quaternion_multiply(q1, q2)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_quaternion_conjugate(self):
        q = np.array([1, 2, 3, 4])
        expected = np.array([1, -2, -3, -4])
        result = quaternion_ops.quaternion_conjugate(q)
        np.testing.assert_allclose(result, expected)

    def test_quaternion_magnitude(self):
        q = np.array([1, 2, 3, 4])
        expected = np.sqrt(1 + 4 + 9 + 16)
        result = quaternion_ops.quaternion_magnitude(q)
        self.assertAlmostEqual(result, expected)

    def test_quaternion_normalize(self):
        q = np.array([1, 2, 3, 4])
        magnitude = np.sqrt(1 + 4 + 9 + 16)
        expected = q / magnitude
        result = quaternion_ops.quaternion_normalize(q)
        np.testing.assert_allclose(result, expected)

        # Test for zero magnitude quaternion
        q_zero = np.array([0, 0, 0, 0])
        expected_zero = np.array([1.0, 0.0, 0.0, 0.0]) # Identity quaternion
        result_zero = quaternion_ops.quaternion_normalize(q_zero)
        np.testing.assert_allclose(result_zero, expected_zero)

    def test_rotate_vector_by_quaternion(self):
        vector = np.array([1, 0, 0])
        quaternion = np.array([0.707, 0, 0, 0.707])
        expected = np.array([0, 1, 0])
        result = quaternion_ops.rotate_vector_by_quaternion(vector, quaternion)
        np.testing.assert_allclose(result, expected, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
