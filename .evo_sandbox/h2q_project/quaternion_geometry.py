import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Quaternion(0, 0, 0, 0)  # Or raise an exception
        self.w /= mag
        self.x /= mag
        self.y /= mag
        self.z /= mag
        return self

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def to_rotation_matrix(self):
        # Convert quaternion to rotation matrix (using the formula)
        q = self.normalize() # Ensure it's a unit quaternion
        w, x, y, z = q.w, q.x, q.y, q.z
        
        rotation_matrix = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        return rotation_matrix

    def __repr__(self):
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"


class QuaternionSelfReflection:
    def __init__(self, quaternion):
        self.quaternion = quaternion

    def check_unit_quaternion(self, tolerance=1e-6):
        magnitude = self.quaternion.magnitude()
        if abs(magnitude - 1.0) > tolerance:
            print(f"Warning: Quaternion magnitude is {magnitude}, not close to 1.0.")
            return False
        return True

    def suggest_normalization(self):
        print("Suggestion: Normalizing the quaternion might resolve the issue.")
        normalized_quaternion = self.quaternion.normalize()
        print(f"Normalized Quaternion: {normalized_quaternion}")

    def reflect(self):
        is_unit = self.check_unit_quaternion()
        if not is_unit:
            self.suggest_normalization()


if __name__ == '__main__':
    # Example Usage
    q1 = Quaternion(1, 0.1, 0.2, 0.3)
    print(f"Original Quaternion: {q1}")
    q1_reflection = QuaternionSelfReflection(q1)
    q1_reflection.reflect()

    q2 = Quaternion(1, 0, 0, 0)
    print(f"Original Quaternion: {q2}")
    q2_reflection = QuaternionSelfReflection(q2)
    q2_reflection.reflect()

    # Example with rotation matrix conversion
    q3 = Quaternion(0.707, 0.707, 0, 0) # Represents a 90-degree rotation around X-axis
    rotation_matrix = q3.to_rotation_matrix()
    print("Rotation Matrix:\n", rotation_matrix)
