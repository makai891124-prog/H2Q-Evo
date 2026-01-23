import unittest
import os
import numpy as np
from h2q_project.core.mesh_processing import MeshProcessor
import logging

# Set up basic logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMeshProcessing(unittest.TestCase):
    def setUp(self):
        # Create a dummy mesh file for testing
        self.dummy_mesh_path = "test_mesh.stl"
        self.create_dummy_mesh(self.dummy_mesh_path)
        self.processor = MeshProcessor(self.dummy_mesh_path)

    def tearDown(self):
        # Clean up the dummy mesh file after testing
        if os.path.exists(self.dummy_mesh_path):
            os.remove(self.dummy_mesh_path)
        self.processor.clear_mesh()

    def create_dummy_mesh(self, file_path):
        # Create a simple cube mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                          [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
                          [0, 4, 7], [0, 7, 3], [1, 5, 6], [1, 6, 2]])

        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(file_path)
            logger.info(f"Dummy mesh created at {file_path}")
        except ImportError:
            logger.error("trimesh library not found. Skipping mesh creation.")

    def test_load_mesh(self):
        self.processor.load_mesh()
        self.assertIsNotNone(self.processor.mesh)

    def test_simplify_mesh(self):
        self.processor.load_mesh()
        simplified_mesh = self.processor.simplify_mesh()
        self.assertIsNotNone(simplified_mesh)

    def test_compute_mesh_properties(self):
        self.processor.load_mesh()
        properties = self.processor.compute_mesh_properties()
        self.assertIn("area", properties)
        self.assertIn("volume", properties)

    def test_process_large_dataset(self):
        large_data = list(range(1000))
        processed_data = self.processor.process_large_dataset(large_data)
        self.assertEqual(len(processed_data), 1000)

if __name__ == '__main__':
    unittest.main()