import unittest

class Calculator:
    def add(self, x, y):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Inputs must be numbers")
        return x + y

    def subtract(self, x, y):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Inputs must be numbers")
        return x - y

    def multiply(self, x, y):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Inputs must be numbers")
        return x * y

    def divide(self, x, y):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Inputs must be numbers")
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y


class TestCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = Calculator()

    def test_add_positive_numbers(self):
        self.assertEqual(self.calculator.add(2, 3), 5)

    def test_add_negative_numbers(self):
        self.assertEqual(self.calculator.add(-2, -3), -5)

    def test_add_mixed_numbers(self):
        self.assertEqual(self.calculator.add(2, -3), -1)

    def test_add_zero(self):
        self.assertEqual(self.calculator.add(2, 0), 2)

    def test_add_floats(self):
        self.assertEqual(self.calculator.add(2.5, 3.5), 6.0)

    def test_add_invalid_input(self):
        with self.assertRaises(TypeError):
            self.calculator.add(2, "3")
        with self.assertRaises(TypeError):
            self.calculator.add("2", 3)

    def test_subtract_positive_numbers(self):
        self.assertEqual(self.calculator.subtract(5, 2), 3)

    def test_subtract_negative_numbers(self):
        self.assertEqual(self.calculator.subtract(-5, -2), -3)

    def test_subtract_mixed_numbers(self):
        self.assertEqual(self.calculator.subtract(5, -2), 7)

    def test_subtract_zero(self):
        self.assertEqual(self.calculator.subtract(5, 0), 5)

    def test_subtract_floats(self):
        self.assertEqual(self.calculator.subtract(5.5, 2.5), 3.0)

    def test_subtract_invalid_input(self):
        with self.assertRaises(TypeError):
            self.calculator.subtract(5, "2")
        with self.assertRaises(TypeError):
            self.calculator.subtract("5", 2)

    def test_multiply_positive_numbers(self):
        self.assertEqual(self.calculator.multiply(2, 3), 6)

    def test_multiply_negative_numbers(self):
        self.assertEqual(self.calculator.multiply(-2, 3), -6)

    def test_multiply_mixed_numbers(self):
        self.assertEqual(self.calculator.multiply(-2, -3), 6)

    def test_multiply_zero(self):
        self.assertEqual(self.calculator.multiply(2, 0), 0)

    def test_multiply_floats(self):
        self.assertEqual(self.calculator.multiply(2.5, 3.0), 7.5)

    def test_multiply_invalid_input(self):
        with self.assertRaises(TypeError):
            self.calculator.multiply(2, "3")
        with self.assertRaises(TypeError):
            self.calculator.multiply("2", 3)

    def test_divide_positive_numbers(self):
        self.assertEqual(self.calculator.divide(6, 2), 3)

    def test_divide_negative_numbers(self):
        self.assertEqual(self.calculator.divide(-6, 2), -3)

    def test_divide_mixed_numbers(self):
        self.assertEqual(self.calculator.divide(6, -2), -3)

    def test_divide_zero(self):
        with self.assertRaises(ValueError):
            self.calculator.divide(6, 0)

    def test_divide_floats(self):
        self.assertEqual(self.calculator.divide(7.5, 2.5), 3.0)

    def test_divide_invalid_input(self):
        with self.assertRaises(TypeError):
            self.calculator.divide(6, "2")
        with self.assertRaises(TypeError):
            self.calculator.divide("6", 2)


if __name__ == '__main__':
    unittest.main()