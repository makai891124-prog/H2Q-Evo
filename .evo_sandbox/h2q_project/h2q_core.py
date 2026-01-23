from h2q_project.core import Quaternion, FractalGeometryGenerator
from h2q_project.utils.reflection import ReflectionModule


class H2QCore:
    def __init__(self, enable_reflection=False, feedback_mechanism=None):
        self.reflection_module = ReflectionModule(feedback_mechanism=feedback_mechanism) if enable_reflection else None
        if enable_reflection:
            self.reflection_module.instrument_class(Quaternion)
            self.reflection_module.instrument_class(FractalGeometryGenerator)

    def create_quaternion(self, w, x, y, z):
        return Quaternion(w, x, y, z)

    def generate_fractal(self, width, height, x_min, x_max, y_min, y_max):
        generator = FractalGeometryGenerator()
        return generator.generate_fractal(width, height, x_min, x_max, y_min, y_max)

# Example usage (enable reflection):
# core = H2QCore(enable_reflection=True)
# q = core.create_quaternion(1, 0, 0, 0)
# fractal = core.generate_fractal(256, 256, -2, 1, -1.5, 1.5)

