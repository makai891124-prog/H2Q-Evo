import numpy as np

class TrefoilKnot:
    def __init__(self, num_points=256):
        self.num_points = num_points

    def generate_points(self):
        t = np.linspace(0, 2 * np.pi, self.num_points)
        x = np.sin(t) + 2 * np.sin(2 * t)
        y = np.cos(t) - 2 * np.cos(2 * t)
        z = -np.sin(3 * t)
        return np.stack([x, y, z], axis=-1).astype(np.float32)


if __name__ == '__main__':
    trefoil = TrefoilKnot()
    points = trefoil.generate_points()
    print(f"Points shape: {points.shape}")