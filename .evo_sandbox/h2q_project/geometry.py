import numpy as np

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def contains(self, point):
        return self.center.distance(point) <= self.radius