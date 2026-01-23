import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import quaternion

class QuaternionVisualizer:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quaternion Visualization')

    def plot_quaternion(self, q):
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion.as_rotation_matrix(q)

        # Define a vector to rotate (e.g., the x-axis)
        vector = np.array([1, 0, 0])

        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)

        # Plot the original and rotated vectors
        self.ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', label='Original Vector')
        self.ax.quiver(0, 0, 0, rotated_vector[0], rotated_vector[1], rotated_vector[2], color='b', label='Rotated Vector')

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quaternion Visualization')

    def show(self):
        self.ax.legend()
        plt.show()

if __name__ == '__main__':
    # Example usage
    visualizer = QuaternionVisualizer()

    # Define a quaternion (e.g., rotation of 90 degrees around the z-axis)
    q = quaternion.from_rotation_vector([0, 0, np.pi/2])

    # Plot the quaternion
    visualizer.plot_quaternion(q)

    # Show the plot
    visualizer.show()
