import numpy as np
import quaternion
import time
import json

# Configuration
n_trials = 1000
quaternion_dtype = np.float64  # Or np.float32


def generate_random_quaternions(n, dtype):
    return quaternion.random.rand(n, dtype=dtype)


def evaluate_quaternion_operation(op_name, operation, quaternions1, quaternions2=None):
    start_time = time.time()
    if quaternions2 is None:
        results = operation(quaternions1)
    else:
        results = operation(quaternions1, quaternions2)
    end_time = time.time()
    duration = end_time - start_time
    norm_results = np.linalg.norm(results.imag, axis=-1)  # Only check norm of imaginary parts
    max_norm = np.max(norm_results)
    min_norm = np.min(norm_results)
    mean_norm = np.mean(norm_results)
    return {
        'operation': op_name,
        'duration': duration,
        'max_imaginary_norm': max_norm,
        'min_imaginary_norm': min_norm,
        'mean_imaginary_norm': mean_norm,
    }


if __name__ == '__main__':
    np.random.seed(42)  # for reproducibility
    quaternions1 = generate_random_quaternions(n_trials, quaternion_dtype)
    quaternions2 = generate_random_quaternions(n_trials, quaternion_dtype)

    results = []
    results.append(evaluate_quaternion_operation('addition', lambda q1, q2: q1 + q2, quaternions1, quaternions2))
    results.append(evaluate_quaternion_operation('subtraction', lambda q1, q2: q1 - q2, quaternions1, quaternions2))
    results.append(evaluate_quaternion_operation('multiplication', lambda q1, q2: q1 * q2, quaternions1, quaternions2))
    results.append(evaluate_quaternion_operation('conjugation', np.conjugate, quaternions1))
    results.append(evaluate_quaternion_operation('inverse', np.reciprocal, quaternions1))

    report = {
        'dtype': str(quaternion_dtype),
        'n_trials': n_trials,
        'results': results
    }

    print(json.dumps(report, indent=4))
