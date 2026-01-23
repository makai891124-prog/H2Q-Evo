import unittest
import numpy as np
from h2q_project.geometric_algebra import commutator, anti_commutator

class TestGeometricAlgebra(unittest.TestCase):

    def test_commutator(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = commutator(a, b)
        expected = np.cross(a, b) * 2  # Commutator of vectors is 2 * (a x b)
        np.testing.assert_allclose(result, expected)

    def test_anti_commutator(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = anti_commutator(a, b)
        expected = 2 * np.dot(a, b) # Anti-commutator of vectors is 2 * (a . b)
        self.assertAlmostEqual(result, expected)

if __name__ == '__main__':
    unittest.main()