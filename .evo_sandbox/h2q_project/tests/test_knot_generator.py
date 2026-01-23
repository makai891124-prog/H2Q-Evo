import unittest
import numpy as np
from h2q_project.geometry.knot_generator import generate_knot

class TestKnotGenerator(unittest.TestCase):

    def test_generate_knot_default_parameters(self):
        # Test with default parameters.
        curve = generate_knot()
        self.assertIsInstance(curve, np.ndarray)
        self.assertEqual(len(curve.shape), 2)
        self.assertEqual(curve.shape[1], 3)  # Expect 3D points.

    def test_generate_knot_varying_turns(self):
        # Test with varying number of turns.
        turns = 5
        curve = generate_knot(turns=turns)
        self.assertIsInstance(curve, np.ndarray)
        self.assertEqual(len(curve.shape), 2)
        self.assertEqual(curve.shape[1], 3)

    def test_generate_knot_varying_segments(self):
        # Test with varying number of segments.
        segments = 200
        curve = generate_knot(segments=segments)
        self.assertIsInstance(curve, np.ndarray)
        self.assertEqual(len(curve), segments)
        self.assertEqual(curve.shape[1], 3)

    def test_generate_knot_ensure_finite(self):
        # Test that the generated curve does not contain NaN or infinite values
        curve = generate_knot()
        self.assertFalse(np.any(np.isnan(curve)))
        self.assertFalse(np.any(np.isinf(curve)))

    def test_generate_knot_radius_control(self):
        # Test the impact of the radius parameter
        radius = 2.0
        curve = generate_knot(radius=radius)
        # Basic validation: ensure no NaNs/Infs and right dimensionality
        self.assertFalse(np.any(np.isnan(curve)))
        self.assertFalse(np.any(np.isinf(curve)))
        self.assertEqual(curve.shape[1], 3)

if __name__ == '__main__':
    unittest.main()
