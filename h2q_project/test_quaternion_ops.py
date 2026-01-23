import pytest
import numpy as np
from h2q_project.quaternion_ops import quaternion_multiply, quaternion_conjugate, quaternion_norm, quaternion_normalize, quaternion_from_euler, euler_from_quaternion
from typing import Tuple, List


def test_quaternion_multiply():
    q1 = np.array([1, 0, 0, 0], dtype=np.float64)
    q2 = np.array([0, 1, 0, 0], dtype=np.float64)
    result = quaternion_multiply(q1, q2)
    expected = np.array([0, 1, 0, 0], dtype=np.float64)
    assert np.allclose(result, expected)

    # 四元数双覆盖特性说明 (Double Cover Property):
    # q3 ≈ 45°绕y轴旋转, q4 ≈ -45°绕y轴旋转
    # 标准物理期望: q3*q4应为单位元(恒等旋转)
    # 四元数群SU(2)是SO(3)的双覆盖, 单位旋转用[1,0,0,0]表示
    # 原期望[0,0,0,-1]假设π相位(180°), 但实际复合为0°旋转
    q3 = np.array([0.707, 0, 0.707, 0])
    q4 = np.array([0.707, 0, -0.707, 0])
    result2 = quaternion_multiply(q3, q4)
    # 修正期望: 应为单位四元数(恒等旋转), 允许归一化误差
    expected2 = np.array([1.0, 0, 0, 0], dtype=np.float64)
    assert np.allclose(result2, expected2, atol=1e-3)


def test_quaternion_conjugate():
    q = np.array([1, 2, 3, 4])
    result = quaternion_conjugate(q)
    expected = np.array([1, -2, -3, -4])
    assert np.allclose(result, expected)


def test_quaternion_norm():
    q = np.array([1, 2, 3, 4])
    result = quaternion_norm(q)
    expected = np.sqrt(30)
    assert np.isclose(result, expected)


def test_quaternion_normalize():
    q = np.array([1, 2, 3, 4])
    result = quaternion_normalize(q)
    norm = np.sqrt(30)
    expected = np.array([1/norm, 2/norm, 3/norm, 4/norm])
    assert np.allclose(result, expected)


def test_quaternion_from_euler():
    # Test with Euler angles (roll, pitch, yaw) = (0, 0, 0)
    roll, pitch, yaw = 0.0, 0.0, 0.0
    q = quaternion_from_euler(roll, pitch, yaw)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(q, expected)

    # Test with Euler angles (roll, pitch, yaw) = (np.pi/2, 0, 0)
    roll, pitch, yaw = np.pi / 2, 0.0, 0.0
    q = quaternion_from_euler(roll, pitch, yaw)
    expected = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
    assert np.allclose(q, expected)


def test_euler_from_quaternion():
    # Test with quaternion [1, 0, 0, 0] corresponding to (0, 0, 0)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    roll, pitch, yaw = euler_from_quaternion(q)
    assert np.isclose(roll, 0.0)
    assert np.isclose(pitch, 0.0)
    assert np.isclose(yaw, 0.0)

    # Test with quaternion corresponding to a rotation around the x-axis by pi/2
    q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    roll, pitch, yaw = euler_from_quaternion(q)
    assert np.isclose(roll, np.pi / 2)
    assert np.isclose(pitch, 0.0)
    assert np.isclose(yaw, 0.0)




