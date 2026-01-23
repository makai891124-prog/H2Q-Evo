import numpy as np

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]

    def get_vertex(self, index):
        return self.vertices[index]

    def get_face(self, index):
        return self.faces[index]

    def get_neighbors(self, vertex_index):
        # Efficiently find neighboring vertices using set operations
        neighbor_indices = set()
        for face_index in range(self.num_faces):
            if vertex_index in self.faces[face_index]:
                face = self.get_face(face_index)
                neighbor_indices.update(face)
        neighbor_indices.discard(vertex_index)
        return list(neighbor_indices)

    def simplify(self, target_num_vertices):
        # Placeholder for simplification algorithm.  
        # In a real implementation, this would use an algorithm like quadric error metrics.
        print(f"Simplifying mesh to {target_num_vertices} vertices (not fully implemented)")
        # For this example, we'll just return a dummy mesh with the target number of vertices.
        # and a single face.
        if target_num_vertices > 0:
            dummy_vertices = np.random.rand(target_num_vertices, 3).astype(np.float32)
            dummy_faces = np.array([[0, 1, 2]], dtype=np.int32) if target_num_vertices >= 3 else np.empty((0, 3), dtype=np.int32)
            return Mesh(dummy_vertices, dummy_faces)
        else:
            return Mesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32))

    def __repr__(self):
        return f"Mesh(vertices={self.num_vertices}, faces={self.num_faces})"
