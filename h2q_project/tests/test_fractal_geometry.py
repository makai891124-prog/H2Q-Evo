import unittest
import numpy as np
from h2q_project.h2q.fractal_geometry import quaternion_to_rotation_matrix

class TestFractalGeometry(unittest.TestCase):

    def test_quaternion_to_rotation_matrix(self):
        # Test case 1: Identity quaternion
        q = np.array([1, 0, 0, 0])
        expected_result = np.eye(3)
        result = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(result, expected_result)

        # Test case 2: Rotation of 90 degrees around the x-axis
        q = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
        expected_result = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # Approximate rotation matrix
        result = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(result, expected_result, atol=1e-7)

        # Test case 3: Non-unit quaternion (should normalize internally)
        q = np.array([2, 0, 0, 0])  # Same rotation as identity, but scaled quaternion
        expected_result = np.eye(3)
        result = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(result, expected_result)


if __name__ == '__main__':
    unittest.main()