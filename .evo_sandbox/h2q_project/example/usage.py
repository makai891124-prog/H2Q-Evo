import numpy as np
from h2q_project.core.h2q_kernel import H2QKernel, SelfReflection

# Example Usage
if __name__ == "__main__":
    # Initialize the H2QKernel
    kernel = H2QKernel()

    # Define two points
    point1 = np.array([1.0, 2.0, 3.0])
    point2 = np.array([4.0, 5.0, 6.0])

    # Compute the distance between the points
    distance = kernel.compute_distance(point1, point2)
    print(f"The distance between point1 and point2 is: {distance}")

    # Check if the distance is within tolerance of an expected value
    expected_distance = 5.196  # Example expected distance
    is_within = kernel.is_within_tolerance(distance, expected_distance)
    print(f"Is the distance within tolerance of {expected_distance}? {is_within}")

    # Demonstrate SelfReflection
    self_reflection = SelfReflection(kernel)
    expected_distance_2 = np.linalg.norm(point1 - point2) # Calculate the expected distance for validation
    accuracy = self_reflection.reflect_on_distance_calculation(point1, point2, expected_distance_2)
    print(f"Is the distance calculation accurate based on self-reflection? {accuracy}")