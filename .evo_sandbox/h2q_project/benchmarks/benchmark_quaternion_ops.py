import timeit

import numpy as np

from h2q_project.quaternion_ops import quaternion_multiply, quaternion_conjugate, quaternion_inverse, quaternion_real, quaternion_imaginary, quaternion_norm, quaternion_normalize


def create_random_quaternion():
    return np.random.rand(4)


def benchmark_quaternion_multiply(n_iterations=1000):
    q1 = create_random_quaternion()
    q2 = create_random_quaternion()

    setup_code = f"from h2q_project.quaternion_ops import quaternion_multiply; import numpy as np; q1 = {q1.tolist()}; q2 = {q2.tolist()}"
    stmt = "quaternion_multiply(q1, q2)"
    time = timeit.timeit(stmt, setup=setup_code, number=n_iterations)
    print(f"quaternion_multiply: {time / n_iterations:.6f} seconds/iteration")


def benchmark_quaternion_conjugate(n_iterations=1000):
    q = create_random_quaternion()
    setup_code = f"from h2q_project.quaternion_ops import quaternion_conjugate; import numpy as np; q = {q.tolist()}"
    stmt = "quaternion_conjugate(q)"
    time = timeit.timeit(stmt, setup=setup_code, number=n_iterations)
    print(f"quaternion_conjugate: {time / n_iterations:.6f} seconds/iteration")


def benchmark_quaternion_inverse(n_iterations=1000):
    q = create_random_quaternion()
    setup_code = f"from h2q_project.quaternion_ops import quaternion_inverse; import numpy as np; q = {q.tolist()}"
    stmt = "quaternion_inverse(q)"
    time = timeit.timeit(stmt, setup=setup_code, number=n_iterations)
    print(f"quaternion_inverse: {time / n_iterations:.6f} seconds/iteration")


def benchmark_quaternion_real(n_iterations=1000):
    q = create_random_quaternion()
    setup_code = f"from h2q_project.quaternion_ops import quaternion_real; import numpy as np; q = {q.tolist()}"
    stmt = "quaternion_real(q)"
    time = timeit.timeit(stmt, setup=setup_code, number=n_iterations)
    print(f"quaternion_real: {time / n_iterations:.6f} seconds/iteration")


def benchmark_quaternion_imaginary(n_iterations=1000):
    q = create_random_quaternion()
    setup_code = f"from h2q_project.quaternion_ops import quaternion_imaginary; import numpy as np; q = {q.tolist()}"
    stmt = "quaternion_imaginary(q)"
    time = timeit.timeit(stmt, setup=setup_code, number=n_iterations)
    print(f"quaternion_imaginary: {time / n_iterations:.6f} seconds/iteration")


def benchmark_quaternion_norm(n_iterations=1000):
    q = create_random_quaternion()
    setup_code = f"from h2q_project.quaternion_ops import quaternion_norm; import numpy as np; q = {q.tolist()}"
    stmt = "quaternion_norm(q)"
    time = timeit.timeit(stmt, setup=setup_code, number=n_iterations)
    print(f"quaternion_norm: {time / n_iterations:.6f} seconds/iteration")


def benchmark_quaternion_normalize(n_iterations=1000):
    q = create_random_quaternion()
    setup_code = f"from h2q_project.quaternion_ops import quaternion_normalize; import numpy as np; q = {q.tolist()}"
    stmt = "quaternion_normalize(q)"
    time = timeit.timeit(stmt, setup=setup_code, number=n_iterations)
    print(f"quaternion_normalize: {time / n_iterations:.6f} seconds/iteration")


if __name__ == "__main__":
    benchmark_quaternion_multiply()
    benchmark_quaternion_conjugate()
    benchmark_quaternion_inverse()
    benchmark_quaternion_real()
    benchmark_quaternion_imaginary()
    benchmark_quaternion_norm()
    benchmark_quaternion_normalize()