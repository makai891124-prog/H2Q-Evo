# h2q/quaternion_ops.py

import torch

def quaternion_mul(q1, q2):
    """
    实现四元数乘法 (Hamilton Product)
    q = [w, x, y, z] (Batch, ..., 4)
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack((w, x, y, z), dim=-1)

def quaternion_norm(q):
    """计算模长"""
    return torch.norm(q, p=2, dim=-1, keepdim=True)

def quaternion_normalize(q):
    """单位四元数化 (保持方向，归一化模长)"""
    return q / (quaternion_norm(q) + 1e-8)

def quaternion_stability(q):
    """
    计算稳定性指标：实部 w 的绝对值。
    根据你的理论，w -> 0 意味着纯向量态（稳定的三维纽结）。
    或者 w -> 1 意味着纯标量态。
    这里我们假设 w 越接近 0，表示它越像一个纯粹的物理结构（三维投影）。
    """
    w, _, _, _ = torch.unbind(q, dim=-1)
    return torch.abs(w)