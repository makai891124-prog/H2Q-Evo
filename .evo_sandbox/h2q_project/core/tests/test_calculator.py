import unittest
from h2q_project.core.calculator import Calculator


class TestCalculator(unittest.TestCase):

    def setUp(self):  # Added setUp method for initialization
        self.calculator = Calculator()

    def test_add(self):
        self.assertEqual(self.calculator.add(1, 2), 3)
        self.assertEqual(self.calculator.add(-1, 1), 0)
        self.assertEqual(self.calculator.add(-1, -1), -2)

    def test_subtract(self):
        self.assertEqual(self.calculator.subtract(5, 3), 2)
        self.assertEqual(self.calculator.subtract(3, 5), -2)
        self.assertEqual(self.calculator.subtract(0, 0), 0)

    def test_multiply(self):
        self.assertEqual(self.calculator.multiply(2, 3), 6)
        self.assertEqual(self.calculator.multiply(-2, 3), -6)
        self.assertEqual(self.calculator.multiply(2, -3), -6)
        self.assertEqual(self.calculator.multiply(-2, -3), 6)
        self.assertEqual(self.calculator.multiply(0, 5), 0)

    def test_divide(self):
        self.assertEqual(self.calculator.divide(6, 2), 3)
        self.assertEqual(self.calculator.divide(-6, 2), -3)
        self.assertEqual(self.calculator.divide(6, -2), -3)
        self.assertEqual(self.calculator.divide(-6, -2), 3)

        with self.assertRaises(ValueError):
            self.calculator.divide(5, 0)

    def test_power(self):
        self.assertEqual(self.calculator.power(2, 3), 8)
        self.assertEqual(self.calculator.power(5, 0), 1)
        self.assertEqual(self.calculator.power(2, -1), 0.5)
        self.assertEqual(self.calculator.power(-2, 3), -8)

    def test_sqrt(self):
       self.assertEqual(self.calculator.sqrt(9), 3)
       self.assertEqual(self.calculator.sqrt(0), 0)
       with self.assertRaises(ValueError):
            self.calculator.sqrt(-1)

if __name__ == '__main__':
    unittest.main()