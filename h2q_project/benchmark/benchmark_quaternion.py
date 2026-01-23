import timeit
import numpy as np
import cupy as cp
import pyquaternion

# Configuration
NUM_RUNS = 1000
ARRAY_SIZE = 1000

# CPU Implementations
def quaternion_multiply_numpy(q1, q2):
    return np.array([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    ])

def benchmark_numpy():
    q1_array = np.random.rand(ARRAY_SIZE, 4)
    q2_array = np.random.rand(ARRAY_SIZE, 4)

    def numpy_multiply():
        for i in range(ARRAY_SIZE):
            quaternion_multiply_numpy(q1_array[i], q2_array[i])

    time = timeit.timeit(numpy_multiply, number=NUM_RUNS)
    print(f"Numpy Multiplication: {time/NUM_RUNS:.6f} seconds per run")

# CuPy Implementations
quaternion_multiply_gpu_kernel = cp.ElementwiseKernel(
    'float32 w1, float32 x1, float32 y1, float32 z1, float32 w2, float32 x2, float32 y2, float32 z2',
    'float32 w3, float32 x3, float32 y3, float32 z3',
    '''
    w3 = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    x3 = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    y3 = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    z3 = w1*z2 + x1*y2 - y1*x2 + z1*w2;
    ''',
    'quaternion_multiply'
)

def benchmark_cupy():
    q1_array = cp.random.rand(ARRAY_SIZE, 4, dtype=cp.float32)
    q2_array = cp.random.rand(ARRAY_SIZE, 4, dtype=cp.float32)

    def cupy_multiply():
        quaternion_multiply_gpu_kernel(q1_array[:,0], q1_array[:,1], q1_array[:,2], q1_array[:,3],
                                        q2_array[:,0], q2_array[:,1], q2_array[:,2], q2_array[:,3],
                                        None, None, None, None)

    time = timeit.timeit(cupy_multiply, number=NUM_RUNS)
    print(f"CuPy Multiplication: {time/NUM_RUNS:.6f} seconds per run")

# pyquaternion Implementation
def benchmark_pyquaternion():
    q1_array = [pyquaternion.Quaternion(np.random.rand(4)) for _ in range(ARRAY_SIZE)]
    q2_array = [pyquaternion.Quaternion(np.random.rand(4)) for _ in range(ARRAY_SIZE)]

    def pyquaternion_multiply():
        for i in range(ARRAY_SIZE):
            q1_array[i] * q2_array[i]

    time = timeit.timeit(pyquaternion_multiply, number=NUM_RUNS)
    print(f"pyquaternion Multiplication: {time/NUM_RUNS:.6f} seconds per run")


if __name__ == "__main__":
    print("Benchmarking Quaternion Multiplication")
    benchmark_numpy()
    benchmark_cupy()
    benchmark_pyquaternion()
