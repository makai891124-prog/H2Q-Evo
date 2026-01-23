import torch

class Quaternion(torch.Tensor):
    def __new__(cls, data):
        return super().__new__(cls, data)

    @staticmethod
    def from_euler(roll, pitch, yaw):
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(torch.tensor([w, x, y, z]))

    def to_euler(self):
        w, x, y, z = self[0], self[1], self[2], self[3]
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = torch.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = torch.atan2(t3, t4)

        return roll, pitch, yaw

    def conjugate(self):
        return Quaternion(torch.tensor([self[0], -self[1], -self[2], -self[3]]))

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self[0], self[1], self[2], self[3]
            w2, x2, y2, z2 = other[0], other[1], other[2], other[3]

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return Quaternion(torch.tensor([w, x, y, z]))
        else:
            return Quaternion(super().__mul__(other))

if __name__ == '__main__':
    # Example Usage
    import torch

    # Create a Quaternion from Euler angles
    roll = torch.tensor(0.1)
    pitch = torch.tensor(0.2)
    yaw = torch.tensor(0.3)
    q = Quaternion.from_euler(roll, pitch, yaw)
    print("Quaternion from Euler angles:", q)

    # Convert back to Euler angles
    roll_out, pitch_out, yaw_out = q.to_euler()
    print("Euler angles from Quaternion:", roll_out, pitch_out, yaw_out)

    # Quaternion multiplication
    q2 = Quaternion.from_euler(0.4, 0.5, 0.6)
    q_mult = q * q2
    print("Quaternion multiplication:", q_mult)

    # Quaternion conjugate
    q_conj = q.conjugate()
    print("Quaternion conjugate:", q_conj)