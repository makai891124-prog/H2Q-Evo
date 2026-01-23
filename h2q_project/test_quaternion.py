import unittest
import numpy as np
from h2q_project.quaternion import Quaternion

class TestQuaternion(unittest.TestCase):

    def test_conjugate(self):
        q = Quaternion(1, 2, 3, 4)
        q_conj = q.conjugate()
        self.assertEqual(q_conj.w, 1)
        self.assertEqual(q_conj.x, -2)
        self.assertEqual(q_conj.y, -3)
        self.assertEqual(q_conj.z, -4)

    def test_magnitude(self):
        q = Quaternion(1, 2, 3, 4)
        mag = q.magnitude()
        self.assertAlmostEqual(mag, np.sqrt(30))

    def test_normalize(self):
        q = Quaternion(1, 2, 3, 4)
        q_norm = q.normalize()
        mag = q_norm.magnitude()
        self.assertAlmostEqual(mag, 1.0)

    def test_multiplication(self):
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        q3 = q1 * q2
        self.assertEqual(q3.w, -60.0)
        self.assertEqual(q3.x, 12.0)
        self.assertEqual(q3.y, 30.0)
        self.assertEqual(q3.z, 24.0)

    def test_to_rotation_matrix(self):
        # Test identity rotation (no rotation)
        q_identity = Quaternion(1, 0, 0, 0)
        rotation_matrix_identity = q_identity.to_rotation_matrix()
        expected_identity = np.eye(3)
        np.testing.assert_allclose(rotation_matrix_identity, expected_identity, atol=1e-7)

        # Test a 90-degree rotation around the Z-axis (approximately)
        # Quaternion representing 90-degree rotation around Z-axis: cos(45), 0, 0, sin(45)
        q_z_rotation = Quaternion(np.cos(np.pi/4), 0, 0, np.sin(np.pi/4))
        rotation_matrix_z = q_z_rotation.to_rotation_matrix()
        expected_z = np.array([
            [np.cos(np.pi/2), -np.sin(np.pi/2), 0],
            [np.sin(np.pi/2), np.cos(np.pi/2), 0],
            [0, 0, 1]
        ])
        np.testing.assert_allclose(rotation_matrix_z, expected_z, atol=1e-7)


if __name__ == '__main__':
    unittest.main()