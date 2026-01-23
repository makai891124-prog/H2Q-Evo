import numpy as np

class FractalGeometryGenerator:
    def __init__(self, max_iterations=100, escape_radius=2.0):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def mandelbrot(self, c, smooth_coloring=True):
        z = 0
        for i in range(self.max_iterations):
            z = z**2 + c
            if abs(z) > self.escape_radius:
                if smooth_coloring:
                    return i + 1 - np.log(np.log(abs(z))) / np.log(2)
                else:
                    return i
        return self.max_iterations

    def generate_fractal(self, width, height, x_min, x_max, y_min, y_max, smooth_coloring=True):
        image = np.zeros((height, width))
        x_range = np.linspace(x_min, x_max, width)
        y_range = np.linspace(y_min, y_max, height)

        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                c = complex(x, y)
                image[i, j] = self.mandelbrot(c, smooth_coloring)

        # Simple visual artifact check (example)
        if np.std(image) < 0.1: # Low standard deviation might indicate a problem
            print("Warning: Low image variance detected. Possible visual artifacts.")

        return image



