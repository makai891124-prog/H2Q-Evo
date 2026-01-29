#!/usr/bin/env python3
"""
谱移计算验证脚本
"""

import torch
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), "h2q_project"))
sys.path.append(os.path.join(os.path.dirname(__file__), "h2q_project", "src"))

from h2q_project.src.h2q.core.sst import SpectralShiftTracker

def test_spectral_shift():
    """测试谱移计算"""
    print("🔬 测试谱移计算...")

    tracker = SpectralShiftTracker()

    # 创建测试矩阵
    # 简单的2x2复数矩阵
    S1 = torch.tensor([[1.0+0j, 0.5+0.5j], [0.5-0.5j, 1.0+0j]], dtype=torch.complex64)
    print("测试矩阵 S1:")
    print(S1)

    eta1 = tracker.compute_shift(S1)
    print("谱移 η1 = {:.6f}".format(eta1))

    # 另一个矩阵
    S2 = torch.tensor([[0.8+0.2j, 0.3+0.7j], [0.3-0.7j, 0.8-0.2j]], dtype=torch.complex64)
    print("\n测试矩阵 S2:")
    print(S2)

    eta2 = tracker.compute_shift(S2)
    print("谱移 η2 = {:.6f}".format(eta2))

    # 随机矩阵
    S3 = torch.randn(4, 4, dtype=torch.complex64)
    print("\n随机矩阵 S3 (4x4):")
    print("行列式:", torch.det(S3))

    eta3 = tracker.compute_shift(S3)
    print("谱移 η3 = {:.6f}".format(eta3))

    # 测试协方差矩阵
    print("\n测试协方差矩阵...")
    data = torch.randn(10, 5)  # 10个样本，5个特征
    cov = torch.cov(data.T)
    print("协方差矩阵:")
    print(cov)

    # 转换为复数矩阵（添加小的虚部）
    cov_complex = cov + 1j * torch.randn_like(cov) * 0.1
    eta_cov = tracker.compute_shift(cov_complex)
    print("协方差谱移 η_cov = {:.6f}".format(eta_cov))

if __name__ == "__main__":
    test_spectral_shift()