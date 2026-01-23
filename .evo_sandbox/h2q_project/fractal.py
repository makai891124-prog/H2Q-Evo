def mandelbrot(c, max_iter):
    z = 0j
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter