import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self._norm = None # Cache the norm
        self._conjugate = None # Cache the conjugate

    def __repr__(self):
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def normalize(self):
        norm = self.norm()
        if norm == 0:
            return Quaternion(1, 0, 0, 0)
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm
        self._norm = None # Reset cached norm
        self._conjugate = None # Reset cached conjugate
        return self

    def norm(self):
        if self._norm is None:
            self._norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        return self._norm

    def conjugate(self):
        if self._conjugate is None:
            self._conjugate = Quaternion(self.w, -self.x, -self.y, -self.z)
        return self._conjugate

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def rotate_vector(self, vector):
        # Convert vector to a quaternion
        q_vector = Quaternion(0, vector[0], vector[1], vector[2])
        # Perform rotation: q_rotated = q * q_vector * q_conjugate
        q_conjugate = self.conjugate()
        q_rotated = self * q_vector * q_conjugate
        return np.array([q_rotated.x, q_rotated.y, q_rotated.z])
