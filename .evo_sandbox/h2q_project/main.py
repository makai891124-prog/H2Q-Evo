import torch
import time
from h2q_project.quaternion import Quaternion
from h2q_project.fractal_generator import generate_fractal

if hasattr(torch, 'compile'):
    compile_available = True
else:
    compile_available = False


def benchmark_quaternion():
    roll = torch.tensor(0.1)
    pitch = torch.tensor(0.2)
    yaw = torch.tensor(0.3)

    start_time = time.time()
    q1 = Quaternion.from_euler(roll, pitch, yaw)
    q2 = Quaternion.from_euler(0.4, 0.5, 0.6)
    q_mult = q1 * q2
    q_conj = q1.conjugate()
    end_time = time.time()
    return end_time - start_time


def benchmark_fractal():
    width, height = 256, 256
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    max_iter = 50

    start_time = time.time()
    fractal = generate_fractal(width, height, x_min, x_max, y_min, y_max, max_iter)
    end_time = time.time()
    return end_time - start_time


if __name__ == '__main__':
    if compile_available:
        print("torch.compile available, attempting to compile.")
        # Compile quaternion operations (example)
        compiled_quaternion = torch.compile(benchmark_quaternion, mode="reduce-overhead")
        # Compile fractal generation (example)
        compiled_fractal = torch.compile(benchmark_fractal, mode="reduce-overhead")

        # Benchmark compiled functions
        print("Benchmarking compiled quaternion operations...")
        quaternion_time_compiled = compiled_quaternion()
        print(f"Compiled Quaternion time: {quaternion_time_compiled:.4f} seconds")

        print("Benchmarking compiled fractal generation...")
        fractal_time_compiled = compiled_fractal()
        print(f"Compiled Fractal time: {fractal_time_compiled:.4f} seconds")

    else:
        print("torch.compile not available.")

    # Benchmark uncompiled functions
    print("Benchmarking uncompiled quaternion operations...")
    quaternion_time_uncompiled = benchmark_quaternion()
    print(f"Uncompiled Quaternion time: {quaternion_time_uncompiled:.4f} seconds")

    print("Benchmarking uncompiled fractal generation...")
    fractal_time_uncompiled = benchmark_fractal()
    print(f"Uncompiled Fractal time: {fractal_time_uncompiled:.4f} seconds")