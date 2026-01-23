import unittest
from h2q_project.core.geometry_kernel import GeometryKernel

class TestGeometryKernel(unittest.TestCase):

    def setUp(self):
        self.kernel = GeometryKernel()

    def test_perform_operation(self):
        result = self.kernel.perform_operation("intersection", [1, 2, 3])
        self.assertEqual(result, "Intersection Result")

        result = self.kernel.perform_operation("distance", [4, 5, 6])
        self.assertEqual(result, "Distance Result")

    def test_operations_count(self):
        self.assertEqual(self.kernel.get_operations_count(), 0)
        self.kernel.perform_operation("intersection", [1, 2, 3])
        self.assertEqual(self.kernel.get_operations_count(), 1)
        self.kernel.perform_operation("distance", [4, 5, 6])
        self.assertEqual(self.kernel.get_operations_count(), 2)

if __name__ == '__main__':
    unittest.main()