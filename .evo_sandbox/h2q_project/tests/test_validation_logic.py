import unittest
from h2q_project.validation_logic import validate_input

class TestValidationLogic(unittest.TestCase):

    def test_valid_input(self):
        data = {"name": "John Doe", "age": 30, "email": "john.doe@example.com"}
        self.assertTrue(validate_input(data))

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            validate_input("not a dictionary")

    def test_missing_name(self):
        data = {"age": 30, "email": "john.doe@example.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_invalid_name_type(self):
        data = {"name": 123, "age": 30, "email": "john.doe@example.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_empty_name(self):
        data = {"name": "", "age": 30, "email": "john.doe@example.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_missing_age(self):
        data = {"name": "John Doe", "email": "john.doe@example.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_invalid_age_type(self):
        data = {"name": "John Doe", "age": "30", "email": "john.doe@example.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_negative_age(self):
        data = {"name": "John Doe", "age": -1, "email": "john.doe@example.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_zero_age(self):
        data = {"name": "John Doe", "age": 0, "email": "john.doe@example.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_missing_email(self):
        data = {"name": "John Doe", "age": 30}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_invalid_email_type(self):
        data = {"name": "John Doe", "age": 30, "email": 123}
        with self.assertRaises(ValueError):
            validate_input(data)

    def test_invalid_email_format(self):
        data = {"name": "John Doe", "age": 30, "email": "john.doeexample.com"}
        with self.assertRaises(ValueError):
            validate_input(data)

if __name__ == '__main__':
    unittest.main()
