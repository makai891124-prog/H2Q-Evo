import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def normalize(self):
        magnitude = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        self.w /= magnitude
        self.x /= magnitude
        self.y /= magnitude
        self.z /= magnitude

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def to_rotation_matrix(self):
        q = [self.w, self.x, self.y, self.z]
        q = [x / np.linalg.norm(q) for x in q]
        w, x, y, z = q
        rotation_matrix = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        return rotation_matrix

def quaternion_slerp(q1, q2, t):
    # Convert to numpy arrays
    q1_arr = np.array([q1.w, q1.x, q1.y, q1.z])
    q2_arr = np.array([q2.w, q2.x, q2.y, q2.z])

    # Normalize the quaternions
    q1_arr = q1_arr / np.linalg.norm(q1_arr)
    q2_arr = q2_arr / np.linalg.norm(q2_arr)

    dot = np.dot(q1_arr, q2_arr)

    # Ensure dot product is within the valid range
    if dot > 1.0: dot = 1.0
    if dot < -1.0: dot = -1.0

    theta = np.arccos(dot)
    if np.abs(theta) < 1e-8:  # Avoid division by zero
        return q1

    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    q_result = w1 * q1_arr + w2 * q2_arr
    q_result = q_result / np.linalg.norm(q_result)

    return Quaternion(q_result[0], q_result[1], q_result[2], q_result[3])


class Transformation:
    def __init__(self, translation=np.array([0, 0, 0]), rotation=Quaternion(1, 0, 0, 0)):
        self.translation = translation
        self.rotation = rotation

    def apply(self, point):
        rotated_point = self.rotation.to_rotation_matrix() @ point
        return rotated_point + self.translation

    def combine(self, other):
        new_translation = self.apply(other.translation)
        new_rotation = self.rotation * other.rotation
        new_rotation.normalize()
        return Transformation(new_translation, new_rotation)

    def interpolate(self, other, t):
      # Linear interpolation of translation
      interpolated_translation = (1 - t) * self.translation + t * other.translation

      # Spherical Linear Interpolation (SLERP) of rotation using quaternions
      interpolated_rotation = quaternion_slerp(self.rotation, other.rotation, t)

      return Transformation(interpolated_translation, interpolated_rotation)

# Example usage (can be removed or commented out in the final version)
if __name__ == '__main__':
    # Example Quaternions
    q1 = Quaternion(1, 0, 0, 0)  # Identity rotation
    q2 = Quaternion(0.707, 0.707, 0, 0) # 90-degree rotation around X-axis (approximately)

    # Interpolate between q1 and q2
    q_interp = quaternion_slerp(q1, q2, 0.5)
    print("Interpolated Quaternion: ", q_interp.w, q_interp.x, q_interp.y, q_interp.z)

    #Example Transformations
    transform1 = Transformation(translation=np.array([1, 2, 3]), rotation=Quaternion(1, 0, 0, 0))
    transform2 = Transformation(translation=np.array([4, 5, 6]), rotation=Quaternion(0.707, 0.707, 0, 0))

    #Interpolate between transformations
    interp_transform = transform1.interpolate(transform2, 0.5)

    #Example point
    point = np.array([1, 1, 1])
    transformed_point = interp_transform.apply(point)
    print("Transformed Point: ", transformed_point)
