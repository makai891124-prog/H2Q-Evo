import numpy as np
import matplotlib.pyplot as plt
from h2q_project.core.quaternion import Quaternion, memory_usage
import tracemalloc

def mandelbrot(q, max_iter):
    c = q
    for n in range(max_iter):
        q = q * q + c
        if q.norm() > 2:
            return n
    return max_iter


def generate_mandelbrot_image(width, height, x_min, x_max, y_min, y_max, max_iter):
    image = np.zeros((height, width))
    x_range = np.linspace(x_min, x_max, width)
    y_range = np.linspace(y_min, y_max, height)

    for i in range(height):
        for j in range(width):
            x = x_range[j]
            y = y_range[i]
            q = Quaternion(0, x, y, 0)
            image[i, j] = mandelbrot(q, max_iter)
    return image


def plot_mandelbrot(image):
    plt.imshow(image, extent=[-2, 1, -1.5, 1.5])
    plt.gray()
    plt.show()


if __name__ == '__main__':
    tracemalloc.start()
    width, height = 256, 256
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    max_iter = 256

    image = generate_mandelbrot_image(width, height, x_min, x_max, y_min, y_max, max_iter)
    memory_usage()
    plot_mandelbrot(image)
    tracemalloc.stop()
