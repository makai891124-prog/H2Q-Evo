import numpy as np

class GeometryKernel:
    def __init__(self, geometry_type='euclidean'):
        self.geometry_type = geometry_type

    def distance(self, point1, point2): #Distance function has been updated to support geometry type
        if self.geometry_type == 'euclidean':
            return self._euclidean_distance(point1, point2)
        elif self.geometry_type == 'spherical':
            return self._spherical_distance(point1, point2)
        else:
            raise ValueError("Unsupported geometry type")

    def _euclidean_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        return np.linalg.norm(point1 - point2)

    def _spherical_distance(self, point1, point2):
        # Haversine formula for distance on a sphere
        lat1, lon1 = np.radians(point1)
        lat2, lon2 = np.radians(point2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        radius = 6371  # Radius of earth in kilometers. Use appropriate radius for the sphere.
        return radius * c