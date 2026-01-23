import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def visualize_quaternion(quaternion, vector=[1, 0, 0]):
    """Visualizes the effect of a quaternion rotation on a 3D vector.

    Args:
        quaternion (list or np.ndarray): A quaternion in the form [x, y, z, w].
        vector (list or np.ndarray): The vector to be rotated, defaults to [1, 0, 0].
    """
    quaternion = np.array(quaternion)
    vector = np.array(vector)

    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()

    # Rotate the vector
    rotated_vector = rotation.apply(vector)

    # Create the figure and axes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original vector
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', label='Original Vector')

    # Plot the rotated vector
    ax.quiver(0, 0, 0, rotated_vector[0], rotated_vector[1], rotated_vector[2], color='b', label='Rotated Vector')

    # Set the axis limits
    max_val = max(np.max(np.abs(vector)), np.max(np.abs(rotated_vector)))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Quaternion Rotation Visualization')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Example usage
    quaternion = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]  # Rotate 90 degrees around Z axis
    visualize_quaternion(quaternion)
