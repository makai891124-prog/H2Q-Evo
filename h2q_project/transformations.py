import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normalize(self):
        magnitude = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if magnitude == 0:
            return Quaternion(0, 0, 0, 0)
        return Quaternion(self.w/magnitude, self.x/magnitude, self.y/magnitude, self.z/magnitude)

    def __mul__(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return Quaternion(w, x, y, z)

    def to_rotation_matrix(self):
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        
        rotation_matrix = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])

        return rotation_matrix

class Transformation:
    def __init__(self):
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = Quaternion(1.0, 0.0, 0.0, 0.0) # Identity quaternion
        self.scale = np.array([1.0, 1.0, 1.0])

    def set_translation(self, translation):
        self.translation = np.array(translation)

    def set_rotation(self, rotation):
        self.rotation = rotation

    def set_scale(self, scale):
        self.scale = np.array(scale)

    def get_matrix(self):
        # Translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = self.translation

        # Rotation matrix from quaternion
        rotation_matrix = self.rotation.to_rotation_matrix()
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = rotation_matrix

        # Scale matrix
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = self.scale[0]
        scale_matrix[1, 1] = self.scale[1]
        scale_matrix[2, 2] = self.scale[2]

        # Transformation matrix: T * R * S
        transformation_matrix = translation_matrix @ rotation_matrix_4x4 @ scale_matrix

        return transformation_matrix

    def transform_point(self, point):
        point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
        transformed_point_homogeneous = self.get_matrix() @ point_homogeneous
        transformed_point = transformed_point_homogeneous[:3] / transformed_point_homogeneous[3]
        return transformed_point


if __name__ == '__main__':
    # Example Usage
    transformation = Transformation()

    # Set translation
    transformation.set_translation([1, 2, 3])

    # Set rotation (example: rotate 90 degrees around the Z-axis)
    # Convert degrees to radians
    angle_radians = np.radians(90)
    # Create a quaternion representing the rotation
    rotation_quaternion = Quaternion(np.cos(angle_radians / 2), 0, 0, np.sin(angle_radians / 2))
    transformation.set_rotation(rotation_quaternion)

    # Set scale
    transformation.set_scale([2, 2, 2])

    # Define a point to transform
    point = [1, 1, 1]

    # Transform the point
    transformed_point = transformation.transform_point(point)

    print("Original Point:", point)
    print("Transformed Point:", transformed_point)
