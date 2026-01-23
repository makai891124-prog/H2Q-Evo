import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def magnitude(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Quaternion(0,0,0,0)
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def to_rotation_matrix(self):
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
    w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return Quaternion(w, x, y, z)

def quaternion_slerp(q1, q2, t):
    # q1 and q2 should be Quaternion objects
    q1 = q1.normalize()
    q2 = q2.normalize()

    dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

    if dot < 0.0:
        q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
        dot = -dot

    if dot > 0.9995:
        # If the quaternions are too close, linearly interpolate
        q = Quaternion(
            q1.w + t * (q2.w - q1.w),
            q1.x + t * (q2.x - q1.x),
            q1.y + t * (q2.y - q1.y),
            q1.z + t * (q2.z - q1.z)
        ).normalize()
        return q

    angle = np.arccos(dot)
    sin_angle = np.sin(angle)

    w1 = np.sin((1 - t) * angle) / sin_angle
    w2 = np.sin(t * angle) / sin_angle

    q = Quaternion(
        w1 * q1.w + w2 * q2.w,
        w1 * q1.x + w2 * q2.x,
        w1 * q1.y + w2 * q2.y,
        w1 * q1.z + w2 * q2.z
    ).normalize()

    return q