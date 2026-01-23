import unittest
from h2q_project.utils import reflection

class TestReflection(unittest.TestCase):

    def test_get_class_from_string(self):
        cls = reflection.get_class_from_string('h2q_project.utils.reflection', 'TestReflection')
        self.assertEqual(cls, TestReflection)

        cls = reflection.get_class_from_string('h2q_project.utils.reflection', 'NonExistentClass')
        self.assertIsNone(cls)

    def test_get_function_from_string(self):
        func = reflection.get_function_from_string('h2q_project.utils.reflection', 'get_class_from_string')
        self.assertEqual(func, reflection.get_class_from_string)

        func = reflection.get_function_from_string('h2q_project.utils.reflection', 'NonExistentFunction')
        self.assertIsNone(func)

    def test_get_module_attribute(self):
        attr = reflection.get_module_attribute('h2q_project.utils.reflection', '__name__')
        self.assertEqual(attr, 'h2q_project.utils.reflection')

        attr = reflection.get_module_attribute('h2q_project.utils.reflection', 'NonExistentAttribute')
        self.assertIsNone(attr)

    def test_list_class_methods(self):
        methods = reflection.list_class_methods(TestReflection)
        self.assertIn('test_get_class_from_string', methods)
        self.assertIn('test_get_function_from_string', methods)
        self.assertIn('test_get_module_attribute', methods)
        self.assertIn('test_list_class_methods', methods)
