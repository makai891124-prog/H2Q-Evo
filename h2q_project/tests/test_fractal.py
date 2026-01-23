import unittest
import numpy as np
from h2q_project.fractal import mandelbrot

class TestFractal(unittest.TestCase):

    def test_mandelbrot_basic(self):
        # Test a point inside the Mandelbrot set
        c = 0 + 0j
        self.assertEqual(mandelbrot(c, max_iter=25), 25)

    def test_mandelbrot_outside(self):
        # Test a point outside the Mandelbrot set
        c = 2 + 0j
        self.assertLess(mandelbrot(c, max_iter=25), 25)

    def test_mandelbrot_edge(self):
        # Test a point near the edge of the Mandelbrot set
        c = -0.5 + 0.5j
        result = mandelbrot(c, max_iter=25)
        self.assertTrue(0 <= result <= 25)


if __name__ == '__main__':
    unittest.main()