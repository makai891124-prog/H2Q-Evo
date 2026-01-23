import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):  # Changed to use __repr__
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        # Numerically stable normalization
        norm = self.norm()
        if norm == 0.0:
            return Quaternion(0.0, 0.0, 0.0, 0.0) # Or raise an exception
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def __mul__(self, other):
        # Numerically stable quaternion multiplication (Hamilton product)
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(w, x, y, z)

    def to_rotation_matrix(self):
        # Convert quaternion to rotation matrix
        q = self.normalize() # Normalize for accurate conversion
        w, x, y, z = q.w, q.x, q.y, q.z

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w

        rotation_matrix = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
        ])

        return rotation_matrix

    @staticmethod
    def from_rotation_matrix(matrix):
        # Convert rotation matrix to quaternion
        # Implementation from:  https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        trace = np.trace(matrix)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2 # S=4*qw 
            qw = 0.25 * S
            qx = (matrix[2, 1] - matrix[1, 2]) / S
            qy = (matrix[0, 2] - matrix[2, 0]) / S
            qz = (matrix[1, 0] - matrix[0, 1]) / S
        elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
            S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2 # S=4*qx
            qw = (matrix[2, 1] - matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (matrix[0, 1] + matrix[1, 0]) / S
            qz = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2 # S=4*qy
            qw = (matrix[0, 2] - matrix[2, 0]) / S
            qx = (matrix[0, 1] + matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (matrix[1, 2] + matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2 # S=4*qz
            qw = (matrix[1, 0] - matrix[0, 1]) / S
            qx = (matrix[0, 2] + matrix[2, 0]) / S
            qy = (matrix[1, 2] + matrix[2, 1]) / S
            qz = 0.25 * S
        
        return Quaternion(qw, qx, qy, qz)