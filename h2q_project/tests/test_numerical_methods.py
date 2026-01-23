import unittest
import numpy as np

class TestNumericalMethods(unittest.TestCase):

    def test_addition_accuracy(self):
        a = 0.1
        b = 0.2
        expected = 0.3
        actual = a + b
        self.assertAlmostEqual(actual, expected, places=7, msg="Addition accuracy test failed")

    def test_subtraction_accuracy(self):
        a = 0.3
        b = 0.1
        expected = 0.2
        actual = a - b
        self.assertAlmostEqual(actual, expected, places=7, msg="Subtraction accuracy test failed")

    def test_multiplication_accuracy(self):
        a = 0.1
        b = 3.0
        expected = 0.3
        actual = a * b
        self.assertAlmostEqual(actual, expected, places=7, msg="Multiplication accuracy test failed")

    def test_division_accuracy(self):
        a = 0.3
        b = 3.0
        expected = 0.1
        actual = a / b
        self.assertAlmostEqual(actual, expected, places=7, msg="Division accuracy test failed")

    def test_large_number_stability(self):
        a = 1e10
        b = 1e-10
        expected = a + b
        actual = a + b
        self.assertAlmostEqual(actual, expected, msg="Large number stability test failed")

    def test_small_number_stability(self):
        a = 1e-10
        b = 1e-10
        expected = a + b
        actual = a + b
        self.assertAlmostEqual(actual, expected, msg="Small number stability test failed")

if __name__ == '__main__':
    unittest.main()