import numpy as np
import pyvista as pv

class QuaternionFractalVisualizer:
    def __init__(self, quaternion_function, fractal_iterations=10, resolution=100):
        self.quaternion_function = quaternion_function
        self.fractal_iterations = fractal_iterations
        self.resolution = resolution
        self.plotter = pv.Plotter()

    def generate_fractal_data(self):
        x = np.linspace(-2, 2, self.resolution)
        y = np.linspace(-2, 2, self.resolution)
        z = np.linspace(-2, 2, self.resolution)

        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)

        points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
        values = np.zeros(points.shape[0])

        for i in range(points.shape[0]):
            point = points[i]
            q = np.quaternion(0, point[0], point[1], point[2])
            
            for _ in range(self.fractal_iterations):
                q = self.quaternion_function(q)

            values[i] = np.linalg.norm(np.array([q.x, q.y, q.z])) # Use norm as a proxy for 'divergence'

        return points, values

    def visualize(self, colormap='viridis', opacity=1.0):
        points, values = self.generate_fractal_data()

        cloud = pv.PolyData(points)
        cloud['values'] = values

        self.plotter.add_mesh(cloud, scalars='values', cmap=colormap, opacity=opacity)
        self.plotter.show()

    def update_parameters(self, fractal_iterations=None, resolution=None):
        if fractal_iterations is not None:
            self.fractal_iterations = fractal_iterations
        if resolution is not None:
            self.resolution = resolution

    def show(self):
        self.plotter.show()

if __name__ == '__main__':
    # Example Usage
    def mandelbrot_quaternion(q):
        return q*q + np.quaternion(0.2, 0.3, 0.4, 0.5)

    visualizer = QuaternionFractalVisualizer(mandelbrot_quaternion)
    visualizer.visualize()

    # Example of updating parameters:
    # visualizer.update_parameters(fractal_iterations=20, resolution=150)
    # visualizer.visualize()
