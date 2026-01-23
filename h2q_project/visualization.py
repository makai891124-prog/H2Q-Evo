import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Visualizer:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_fractal(self, fractal_points):
        x = fractal_points[:, 0]
        y = fractal_points[:, 1]
        z = fractal_points[:, 2]
        self.ax.plot(x, y, z)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Fractal Visualization')

    def show(self):
        plt.show()

    def clear(self):
        self.ax.clear()
