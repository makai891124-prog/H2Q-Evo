import numpy as np

class Mandelbrot:
    def __init__(self, max_iter=256):
        self.max_iter = max_iter

    def compute_set(self, x_min, x_max, y_min, y_max, width, height):
        x = np.linspace(x_min, x_max, width, dtype=np.float32)
        y = np.linspace(y_min, y_max, height, dtype=np.float32)
        c = x[np.newaxis, :] + 1j * y[:, np.newaxis]
        z = np.zeros(c.shape, dtype=np.complex64)
        divergence = np.zeros(c.shape, dtype=np.uint8)
        for i in range(self.max_iter):
            z = z**2 + c
            diverged = np.abs(z) > 2
            divergence[diverged & (divergence == 0)] = i
            z[diverged] = 2  # Optimization: Prevent further calculations on diverged points

        return divergence


if __name__ == '__main__':
    mandelbrot = Mandelbrot()
    result = mandelbrot.compute_set(-2.0, 1.0, -1.5, 1.5, 512, 512)
    print(f"Result shape: {result.shape}")