import numpy as np

def julia_set(width: int, height: int, c: complex, max_iter: int = 256) -> np.ndarray:
    """Generates a Julia set with given parameters.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        c (complex): Complex constant.
        max_iter (int): Maximum iterations.

    Returns:
        np.ndarray: A 2D numpy array representing the Julia set.
    """

    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    image = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            z = complex(x_min + x * (x_max - x_min) / width, y_min + y * (y_max - y_min) / height)
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z*z + c
                n += 1

            image[y, x] = int(n / max_iter * 255)

    return image
