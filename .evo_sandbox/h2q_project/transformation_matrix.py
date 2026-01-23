import numpy as np

class TransformationMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __str__(self):
        return f"TransformationMatrix:\n{self.matrix}"

    @classmethod
    def identity(cls):
        return cls(np.eye(4))

    def to_array(self):
        return self.matrix

    @classmethod
    def from_array(cls, arr):
        if arr.shape != (4, 4):
            raise ValueError("Array must have shape (4, 4)")
        return cls(arr)

    def serialize(self):
        """Serializes the transformation matrix to a list of lists.
        """
        return self.matrix.tolist()

    @classmethod
    def deserialize(cls, data):
        """Deserializes a transformation matrix from a list of lists.
        """
        return cls(np.array(data))

# Example Usage
if __name__ == '__main__':
    # Create an identity transformation matrix
    transform = TransformationMatrix.identity()
    print(transform)

    # Serialize it
    serialized_transform = transform.serialize()
    print(f"Serialized transformation matrix: {serialized_transform}")

    # Deserialize it
    deserialized_transform = TransformationMatrix.deserialize(serialized_transform)
    print(f"Deserialized transformation matrix:\n{deserialized_transform}")

    # Example using from_array and to_array
    arr = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    transform_from_array = TransformationMatrix.from_array(arr)
    print(f"Transformation Matrix from array:\n{transform_from_array}")
    arr_back = transform_from_array.to_array()
    print(f"Array from transformation matrix:\n{arr_back}")
