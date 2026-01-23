import unittest
from h2q_project.core import core_module

class TestCoreModule(unittest.TestCase):

    def test_core_function_basic(self):
        self.assertEqual(core_module.core_function(2, 3), 5)

    def test_core_function_negative(self):
        self.assertEqual(core_module.core_function(-2, 3), 1)

    def test_core_function_zero(self):
        self.assertEqual(core_module.core_function(0, 0), 0)

    def test_another_function_positive(self):
        self.assertEqual(core_module.another_function(5), 6)

    def test_another_function_negative(self):
        self.assertEqual(core_module.another_function(-5), -4)

    def test_another_function_zero(self):
        self.assertEqual(core_module.another_function(0), 1)

if __name__ == '__main__':
    unittest.main()