import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        norm = self.norm()
        if norm == 0:
            return self # or raise an exception, depending on desired behavior
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, np.ndarray) and other.shape == (3,):
            # Rotate a vector by the quaternion
            q = self
            v = Quaternion(0, other[0], other[1], other[2])
            q_conj = self.conjugate()
            rotated_v = q * v * q_conj
            return np.array([rotated_v.x, rotated_v.y, rotated_v.z])
        else:
            raise TypeError("Unsupported operand type(s) for *: Quaternion and {}".format(type(other)))

    def to_rotation_matrix(self):
        # Convert quaternion to rotation matrix
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        rotation_matrix = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        return rotation_matrix


    @staticmethod
    def from_axis_angle(axis, angle):
        # Create quaternion from axis-angle representation
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        w = np.cos(half_angle)
        x = axis[0] * np.sin(half_angle)
        y = axis[1] * np.sin(half_angle)
        z = axis[2] * np.sin(half_angle)
        return Quaternion(w, x, y, z)


class Transformation:
    def __init__(self, rotation=None, translation=None, scale=None):
        self.rotation = rotation if rotation is not None else Quaternion(1, 0, 0, 0)
        self.translation = np.array(translation) if translation is not None else np.array([0.0, 0.0, 0.0])
        self.scale = scale if scale is not None else 1.0

    def __repr__(self):
         return f"Transformation(rotation={self.rotation}, translation={self.translation}, scale={self.scale})"

    def apply(self, point):
        # Apply the transformation to a point
        point = np.array(point)
        rotated_point = self.rotation * point  # Rotate the point
        scaled_point = rotated_point * self.scale # Scale the point
        translated_point = scaled_point + self.translation # Translate the point
        return translated_point


class GeometryEvaluator:
    def __init__(self, transformation):
        self.transformation = transformation

    def evaluate_rotation(self):
        # Evaluate the rotation component.  For now, just checks if it's a unit quaternion.
        rotation = self.transformation.rotation
        if not isinstance(rotation, Quaternion):
             return "Rotation is not a Quaternion."

        norm = rotation.norm()
        if not np.isclose(norm, 1.0): #Use isclose for floating point comparison
            return f"Rotation quaternion is not normalized. Norm = {norm}"
        else:
            return "Rotation is valid."

    def evaluate_translation(self):
        # Evaluate the translation component.  Checks if it's a numpy array of the correct size.
        translation = self.transformation.translation
        if not isinstance(translation, np.ndarray):
            return "Translation is not a numpy array."
        if translation.shape != (3,):
            return f"Translation has incorrect shape: {translation.shape}. Expected (3,)"
        return "Translation is valid."

    def evaluate_scale(self):
        #Evaluate scale component. Checks if it's a number
        scale = self.transformation.scale
        if not isinstance(scale, (int, float)): #Allow both int and float
            return "Scale is not a number."
        return "Scale is valid."

    def evaluate(self):
        #Performs a full evaluation of the transformation
        rotation_result = self.evaluate_rotation()
        translation_result = self.evaluate_translation()
        scale_result = self.evaluate_scale()

        return {
            "rotation": rotation_result,
            "translation": translation_result,
            "scale": scale_result
        }


if __name__ == '__main__':
    # Example Usage
    # Create a quaternion representing a 45-degree rotation around the Z-axis
    axis = np.array([0, 0, 1])
    angle = np.pi / 4  # 45 degrees in radians
    rotation_quaternion = Quaternion.from_axis_angle(axis, angle)

    # Create a translation vector
    translation_vector = np.array([1, 2, 3])

    # Create a transformation object
    transformation = Transformation(rotation=rotation_quaternion, translation=translation_vector, scale=2.0)

    # Apply the transformation to a point
    point = np.array([1, 1, 1])
    transformed_point = transformation.apply(point)
    print(f"Original point: {point}")
    print(f"Transformed point: {transformed_point}")

    # Evaluate the transformation
    evaluator = GeometryEvaluator(transformation)
    evaluation_results = evaluator.evaluate()
    print("\nTransformation Evaluation:")
    for component, result in evaluation_results.items():
        print(f"{component}: {result}")

    # Example of an invalid transformation (non-normalized quaternion)
    invalid_quaternion = Quaternion(2, 0, 0, 0)  # Not normalized
    invalid_transformation = Transformation(rotation=invalid_quaternion)
    invalid_evaluator = GeometryEvaluator(invalid_transformation)
    invalid_results = invalid_evaluator.evaluate()

    print("\nInvalid Transformation Evaluation:")
    for component, result in invalid_results.items():
        print(f"{component}: {result}")