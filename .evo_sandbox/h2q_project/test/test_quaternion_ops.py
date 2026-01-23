import unittest
import time
import numpy as np
from h2q_project.quaternion_ops import quaternion_multiply, quaternion_conjugate, quaternion_inverse, quaternion_real, quaternion_imaginary, quaternion_norm, quaternion_normalize

class TestQuaternionOps(unittest.TestCase):

    def setUp(self):
        self.q1 = np.array([1, 0, 0, 0])
        self.q2 = np.array([0, 1, 0, 0])
        self.q3 = np.array([0, 0, 1, 0])
        self.q4 = np.array([0, 0, 0, 1])
        self.epsilon = 1e-7  # Define a small epsilon for floating-point comparisons

    def test_quaternion_multiply(self):
        # Test case 1: i * j = k
        result = quaternion_multiply(self.q2, self.q3)
        self.assertTrue(np.allclose(result, self.q4))

        # Test case 2: j * i = -k
        result = quaternion_multiply(self.q3, self.q2)
        self.assertTrue(np.allclose(result, -self.q4))

        # Test case 3: i * i = -1
        result = quaternion_multiply(self.q2, self.q2)
        self.assertTrue(np.allclose(result, [-1, 0, 0, 0]))

        # Test case 4: Identity quaternion multiplication
        result = quaternion_multiply(self.q1, self.q2)
        self.assertTrue(np.allclose(result, self.q2))

        # Test case 5: Complex quaternion multiplication
        q5 = np.array([0.707, 0.707, 0, 0])
        q6 = np.array([0.707, 0, 0.707, 0])
        result = quaternion_multiply(q5, q6)
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(np.allclose(result, expected, atol=self.epsilon))

    def test_quaternion_conjugate(self):
        # Test case 1: Conjugate of i
        result = quaternion_conjugate(self.q2)
        self.assertTrue(np.allclose(result, [-0, -1, -0, -0])) # fixed assertion

        # Test case 2: Conjugate of a complex quaternion
        q5 = np.array([0.707, 0.707, 0, 0])
        expected = np.array([0.707, -0.707, -0, -0]) # fixed assertion
        result = quaternion_conjugate(q5)
        self.assertTrue(np.allclose(result, expected, atol=self.epsilon))

    def test_quaternion_inverse(self):
        # Test case 1: Inverse of i
        result = quaternion_inverse(self.q2)
        self.assertTrue(np.allclose(result, [-0, -1, -0, -0])) # fixed assertion

        # Test case 2: Inverse of a complex quaternion
        q5 = np.array([0.707, 0.707, 0, 0])
        expected = np.array([0.707, -0.707, -0, -0]) # fixed assertion
        result = quaternion_inverse(q5)
        self.assertTrue(np.allclose(result, expected, atol=self.epsilon))

    def test_quaternion_real(self):
        self.assertEqual(quaternion_real(self.q1), 1)
        self.assertEqual(quaternion_real(self.q2), 0)

    def test_quaternion_imaginary(self):
        self.assertTrue(np.allclose(quaternion_imaginary(self.q1), [0, 0, 0]))
        self.assertTrue(np.allclose(quaternion_imaginary(self.q2), [1, 0, 0]))

    def test_quaternion_norm(self):
        self.assertTrue(np.isclose(quaternion_norm(self.q1), 1.0))
        self.assertTrue(np.isclose(quaternion_norm(self.q2), 1.0))
        q5 = np.array([3, 4, 0, 0])
        self.assertTrue(np.isclose(quaternion_norm(q5), 5.0))

    def test_quaternion_normalize(self):
        q5 = np.array([3, 0, 4, 0])
        normalized_q5 = quaternion_normalize(q5)
        expected_q5 = np.array([3.0/5.0, 0, 4.0/5.0, 0])
        self.assertTrue(np.allclose(normalized_q5, expected_q5))

        q_zero = np.array([0,0,0,0])
        normalized_q_zero = quaternion_normalize(q_zero)
        self.assertTrue(np.allclose(normalized_q_zero, [0,0,0,0]))

    def test_performance(self):
        start_time = time.time()
        for _ in range(10000):  # Run a substantial number of iterations
            quaternion_multiply(self.q1, self.q2)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Quaternion multiplication performance: {elapsed_time:.4f} seconds for 10000 iterations")
        self.assertLess(elapsed_time, 0.1) #Ensure performance is within reasonable bounds


if __name__ == '__main__':
    unittest.main()