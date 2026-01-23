import unittest
from h2q_project.core import some_core_module
from h2q_project.core.memory_management import MemoryTracker

class TestCoreModule(unittest.TestCase):

    def setUp(self):
        self.tracker = MemoryTracker()

    def tearDown(self):
        self.assertTrue(not self.tracker.check_leaks(), "Memory leak detected in core module")

    def test_core_functionality(self):
        # Example test, replace with actual tests
        result = some_core_module.some_function(1, 2)
        self.assertEqual(result, 3)

    def test_another_function(self):
        #Another example, replace with actual tests
        result = some_core_module.another_function("a", "b")
        self.assertEqual(result, "ab")

if __name__ == '__main__':
    unittest.main()