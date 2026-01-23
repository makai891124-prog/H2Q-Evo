import torch
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter):
    z = torch.zeros_like(c)
    n = torch.zeros_like(c, dtype=torch.int32)
    for i in range(max_iter):
        mask = torch.abs(z) <= 2
        z[mask] = z[mask] * z[mask] + c[mask]
        n[mask] += 1
    return n


def generate_fractal(width, height, x_min, x_max, y_min, y_max, max_iter):
    x = torch.linspace(x_min, x_max, width)
    y = torch.linspace(y_min, y_max, height)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    c = torch.complex(X, Y)
    n = mandelbrot(c, max_iter)
    return n.numpy()


if __name__ == '__main__':
    # Example usage
    width, height = 256, 256
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    max_iter = 50

    fractal = generate_fractal(width, height, x_min, x_max, y_min, y_max, max_iter)

    plt.imshow(fractal, extent=[x_min, x_max, y_min, y_max], cmap='hot')
    plt.colorbar()
    plt.title('Mandelbrot Set')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.show()