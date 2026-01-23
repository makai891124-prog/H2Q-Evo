import unittest
import numpy as np
from h2q_project.quaternion import quaternion

class TestQuaternion(unittest.TestCase):

    def test_creation(self):
        q = quaternion.Quaternion(1, 2, 3, 4)
        self.assertEqual(q.w, 1)
        self.assertEqual(q.x, 2)
        self.assertEqual(q.y, 3)
        self.assertEqual(q.z, 4)

    def test_norm(self):
        q = quaternion.Quaternion(1, 2, 2, 3)
        self.assertAlmostEqual(q.norm(), 4.0)

    def test_normalize(self):
        q = quaternion.Quaternion(1, 2, 2, 3)
        q_normalized = q.normalize()
        self.assertAlmostEqual(q_normalized.norm(), 1.0)

    def test_conjugate(self):
        q = quaternion.Quaternion(1, 2, 3, 4)
        q_conjugate = q.conjugate()
        self.assertEqual(q_conjugate.w, 1)
        self.assertEqual(q_conjugate.x, -2)
        self.assertEqual(q_conjugate.y, -3)
        self.assertEqual(q_conjugate.z, -4)

    def test_quaternion_multiplication(self):
        q1 = quaternion.Quaternion(1, 2, 3, 4)
        q2 = quaternion.Quaternion(5, 6, 7, 8)
        q_product = q1 * q2
        self.assertEqual(q_product.w, -60.0)
        self.assertEqual(q_product.x, 12.0)
        self.assertEqual(q_product.y, 30.0)
        self.assertEqual(q_product.z, 24.0)

    def test_rotation_matrix_conversion(self):        # Create a quaternion representing a rotation of 90 degrees around the z-axis
        angle = np.pi / 2
        q = quaternion.Quaternion(np.cos(angle/2), 0, 0, np.sin(angle/2))

        # Convert the quaternion to a rotation matrix
        rotation_matrix = q.to_rotation_matrix()

        # Define the expected rotation matrix
        expected_rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Assert that the rotation matrix is close to the expected rotation matrix
        np.testing.assert_allclose(rotation_matrix, expected_rotation_matrix, atol=1e-7)


if __name__ == '__main__':
    unittest.main()