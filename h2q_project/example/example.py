import numpy as np
from pyquaternion import Quaternion
from h2q_project.camera.camera_controller import CameraController

# Example usage:
if __name__ == '__main__':
    # Initialize camera controller
    camera = CameraController()

    # Move the camera
    camera.move(np.array([1.0, 0.0, 0.0]))
    print(f"Camera Position: {camera.get_position()}")

    # Rotate the camera
    rotation_quaternion = Quaternion(axis=np.array([0.0, 1.0, 0.0]), angle=np.pi/2) # Rotate 90 degrees around Y axis
    camera.rotate(rotation_quaternion)
    print(f"Camera Rotation: {camera.get_rotation()}")

    # Get world matrix
    world_matrix = camera.get_world_matrix()
    print(f"World Matrix:\n{world_matrix}")

    # Set new position and rotation
    camera.set_position([2.0, 3.0, 4.0])
    camera.set_rotation(Quaternion(axis=np.array([1.0, 0.0, 0.0]), angle=np.pi/4))
    print(f"New Camera Position: {camera.get_position()}")
    print(f"New Camera Rotation: {camera.get_rotation()}")

    world_matrix = camera.get_world_matrix()
    print(f"New World Matrix:\n{world_matrix}")