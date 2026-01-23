from h2q_project.core.geometry.point import Point
from h2q_project.core.geometry.profiler import profile

class Line:
    def __init__(self, start_point: Point, end_point: Point):
        self.start_point = start_point
        self.end_point = end_point

    @profile
    def length(self) -> float:
        return self.start_point.distance(self.end_point)

    def __repr__(self):
        return f"Line(start={self.start_point}, end={self.end_point})"