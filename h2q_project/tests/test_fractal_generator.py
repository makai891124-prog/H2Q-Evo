import unittest
import numpy as np
from h2q_project import fractal_generator

class TestFractalGenerator(unittest.TestCase):

    def test_generate_cube_fractal(self):
        generator = fractal_generator.FractalGenerator()
        fractal = generator.generate_cube_fractal(iterations=1)
        self.assertIsInstance(fractal, np.ndarray)
        self.assertEqual(fractal.shape[1], 3) # check if 3D points
        self.assertTrue(fractal.shape[0] > 0) # check if points were actually generated

    def test_rotate_fractal(self):
        generator = fractal_generator.FractalGenerator()
        fractal = generator.generate_cube_fractal(iterations=1)
        rotated_fractal = generator.rotate_fractal(fractal, axis=[0, 1, 0], angle=np.pi/2)
        self.assertIsInstance(rotated_fractal, np.ndarray)
        self.assertEqual(rotated_fractal.shape, fractal.shape)

    def test_scale_fractal(self):
        generator = fractal_generator.FractalGenerator()
        fractal = generator.generate_cube_fractal(iterations=1)
        scaled_fractal = generator.scale_fractal(fractal, scale=2.0)
        self.assertIsInstance(scaled_fractal, np.ndarray)
        self.assertEqual(scaled_fractal.shape, fractal.shape)

    def test_translate_fractal(self):
        generator = fractal_generator.FractalGenerator()
        fractal = generator.generate_cube_fractal(iterations=1)
        translated_fractal = generator.translate_fractal(fractal, translation=[1, 2, 3])
        self.assertIsInstance(translated_fractal, np.ndarray)
        self.assertEqual(translated_fractal.shape, fractal.shape)

if __name__ == '__main__':
    unittest.main()