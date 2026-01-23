import numpy as np
import trimesh
import logging
from memory_profiler import profile

logger = logging.getLogger(__name__)

class MeshProcessor:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.mesh = None

    @profile
    def load_mesh(self):
        try:
            self.mesh = trimesh.load_mesh(self.mesh_path)
            logger.info(f"Mesh loaded successfully from {self.mesh_path}")
        except Exception as e:
            logger.error(f"Error loading mesh: {e}")
            self.mesh = None

    @profile
    def simplify_mesh(self, target_reduction=0.5):
        if self.mesh is None:
            logger.warning("No mesh loaded.  Please load a mesh first.")
            return None

        try:
            simplified_mesh = self.mesh.simplify_quadric_decimation(proportion=target_reduction)
            logger.info(f"Mesh simplified with target reduction of {target_reduction}")
            return simplified_mesh
        except Exception as e:
            logger.error(f"Error simplifying mesh: {e}")
            return None

    @profile
    def compute_mesh_properties(self):
        if self.mesh is None:
            logger.warning("No mesh loaded. Please load a mesh first.")
            return {}

        try:
            area = self.mesh.area
            volume = self.mesh.volume
            properties = {"area": area, "volume": volume}
            logger.info("Mesh properties computed successfully.")
            return properties
        except Exception as e:
            logger.error(f"Error computing mesh properties: {e}")
            return {}

    @profile
    def process_large_dataset(self, data):
        # Simulate a large dataset processing operation.
        processed_data = []
        for item in data:
            processed_item = item * 2  # Example operation
            processed_data.append(processed_item)
        logger.info("Large dataset processed.")
        return processed_data

    def clear_mesh(self):
      # Explicitly clear the mesh to release memory
      self.mesh = None
      logger.info("Mesh cleared from memory.")