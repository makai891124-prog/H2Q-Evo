import numpy as np
from pyquaternion import Quaternion

class CameraController:
    def __init__(self, position=np.array([0.0, 0.0, 0.0]), rotation=Quaternion(1, 0, 0, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation  # Quaternion

    def move(self, delta_position):
        self.position += delta_position

    def rotate(self, delta_rotation: Quaternion):
        self.rotation = delta_rotation * self.rotation
        self.rotation = self.rotation.normalised

    def set_position(self, position):
        self.position = np.array(position, dtype=np.float32)

    def set_rotation(self, rotation: Quaternion):
        self.rotation = rotation

    def get_position(self):
        return self.position

    def get_rotation(self):
        return self.rotation

    def get_world_matrix(self):
        # Convert quaternion to rotation matrix
        rotation_matrix = self.rotation.rotation_matrix

        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = self.position

        # Create rotation matrix
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = rotation_matrix

        # Combine rotation and translation
        world_matrix = translation_matrix @ rotation_matrix_4x4

        return world_matrix