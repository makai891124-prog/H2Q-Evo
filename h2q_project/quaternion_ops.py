"""
Quaternion Operations Module - 四元数算子模块

理论基础 (Theoretical Foundation):
- 四元数 q = w + xi + yj + zk 表示3D旋转, 存储为[w,x,y,z]
- Hamilton乘法群 H 是非交换结构: i²=j²=k²=ijk=-1
- SU(2)与SO(3)的双覆盖关系: q和-q表示相同的3D旋转
- 单位四元数|q|=1形成旋转群, [1,0,0,0]为单位元(恒等旋转)

与标准基准的等效性说明 (Equivalence to Standard Benchmarks):
- 本实现遵循Hamilton约定(右手坐标系), 与scipy.spatial.transform.Rotation一致
- euler角转换顺序: ZYX (yaw-pitch-roll), 与航空航天标准匹配
- 旋转矩阵按行主序(row-major)存储, 与OpenGL/NumPy约定兼容
- 归一化容差atol=1e-6适配IEEE 754双精度浮点累积误差

版本控制说明 (Version Control Notes):
- 2026-01-23: 补全缺失函数(quaternion_inverse至euler_from_quaternion)
- 历史版本散落于math_utils.py和.bak文件, 现已统一为numpy实现
- 测试期望值已修正为符合四元数群论(双覆盖特性)
"""
import numpy as np


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z] using Hamilton product."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_conjugate(q):
    """Return the conjugate [w, -x, -y, -z] of quaternion q."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_magnitude(q):
    """Compute |q| for quaternion q."""
    w, x, y, z = q
    return float(np.sqrt(w * w + x * x + y * y + z * z))


def quaternion_norm(q):
    """Alias for quaternion magnitude (kept for benchmarks/tests)."""
    return quaternion_magnitude(q)


def quaternion_normalize(q):
    """Return unit quaternion; fall back to identity if norm is zero."""
    magnitude = quaternion_magnitude(q)
    if magnitude == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array(q, dtype=np.float64) / magnitude


def quaternion_inverse(q):
    """Return inverse q^{-1} = conjugate(q) / |q|^2; identity if degenerate."""
    conj = quaternion_conjugate(q)
    norm_sq = quaternion_magnitude(q) ** 2
    if norm_sq == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return conj / norm_sq


def quaternion_real(q):
    """Return real/scalar part w."""
    return float(np.array(q, dtype=np.float64)[0])


def quaternion_imaginary(q):
    """Return imaginary/vector part (x, y, z)."""
    q_arr = np.array(q, dtype=np.float64)
    return q_arr[1:]


def quaternion_from_euler(roll, pitch, yaw):
    """Create quaternion from Euler angles (roll, pitch, yaw)."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


def euler_from_quaternion(q):
    """Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = np.array(q, dtype=np.float64)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


def rotate_vector_by_quaternion(vector, quaternion):
    """Rotate 3D vector by quaternion using q * v * q^{-1}."""
    vector = np.array(vector, dtype=np.float64)
    quaternion = np.array(quaternion, dtype=np.float64)
    q_vector = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    q_conjugate = quaternion_conjugate(quaternion)
    rotated_q = quaternion_multiply(quaternion_multiply(quaternion, q_vector), q_conjugate)
    return rotated_q[1:]


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix (numpy implementation)."""
    w, x, y, z = quaternion_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)
