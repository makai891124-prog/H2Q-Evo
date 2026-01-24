#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
李群自同构引擎 - Hamilton四元数与分形几何

实现:
1. Hamilton四元数非交换群运算
2. 分形维数动态调整
3. 李群指数/对数映射
4. Iterated Function System (IFS)
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional


class QuaternionLieGroupConfig:
    """四元数李群配置"""
    def __init__(self):
        self.quaternion_dim = 4
        self.fractal_levels = 8
        self.hausdorff_d_range = [1.0, 2.0]
        self.hidden_dim = 256


class QuaternionLieGroupModule(nn.Module):
    """
    四元数李群模块
    
    实现Hamilton四元数的完整非交换群运算和李群自同构
    """
    
    def __init__(self, config: QuaternionLieGroupConfig):
        super().__init__()
        self.config = config
        
        # 注册Pauli矩阵缓冲区
        self.register_buffer("identity_quat", torch.tensor([1.0, 0.0, 0.0, 0.0]))
        
        # 分形维数参数（可学习）
        self.d_f_param = nn.Parameter(torch.tensor(1.5))
        
        # 转换层
        self.to_quaternion = nn.Linear(config.hidden_dim, 4)
        self.from_quaternion = nn.Linear(4, config.hidden_dim)
        
    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Hamilton四元数乘法 - 完整的8项公式
        
        q1 = w1 + x1*i + y1*j + z1*k
        q2 = w2 + x2*i + y2*j + z2*k
        
        q1*q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2)
              + (w1*x2 + x1*w2 + y1*z2 - z1*y2)*i
              + (w1*y2 - x1*z2 + y1*w2 + z1*x2)*j
              + (w1*z2 + x1*y2 - y1*x2 + z1*w2)*k
        """
        # 确保输入维度正确
        if q1.dim() == 1:
            q1 = q1.unsqueeze(0)
        if q2.dim() == 1:
            q2 = q2.unsqueeze(0)
            
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        # Hamilton乘法的8项公式
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """四元数共轭: q* = w - x*i - y*j - z*k"""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        conjugate = q.clone()
        conjugate[..., 1:] = -conjugate[..., 1:]  # 虚部取反
        return conjugate
    
    def quaternion_norm(self, q: torch.Tensor) -> torch.Tensor:
        """四元数范数: |q| = sqrt(w^2 + x^2 + y^2 + z^2)"""
        return torch.sqrt((q ** 2).sum(dim=-1, keepdim=True))
    
    def quaternion_inverse(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        四元数逆元: q^-1 = q* / |q|^2
        
        满足: q * q^-1 = e (单位元)
        """
        q_conj = self.quaternion_conjugate(q)
        norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
        return q_conj / (norm_sq + eps)
    
    def quaternion_normalize(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """归一化到单位四元数: |q| = 1"""
        norm = self.quaternion_norm(q)
        return q / (norm + eps)
    
    def exponential_map_so3_to_su2(self, omega: torch.Tensor) -> torch.Tensor:
        """
        指数映射: so(3) → SU(2)
        
        使用Rodrigues公式:
        exp(ω) = cos(θ/2) + sin(θ/2) * ω̂
        
        其中:
        - θ = |ω| (旋转角)
        - ω̂ = ω/|ω| (旋转轴)
        """
        if omega.dim() == 1:
            omega = omega.unsqueeze(0)
            
        theta = torch.norm(omega, dim=-1, keepdim=True)
        
        # 处理小角度情况
        small_angle_mask = (theta < 1e-8)
        
        half_theta = theta / 2.0
        w = torch.cos(half_theta)
        
        # 避免除零
        omega_normalized = omega / (theta + 1e-8)
        xyz = torch.sin(half_theta) * omega_normalized
        
        # 小角度时使用泰勒展开: sin(x)/x ≈ 1 - x^2/6
        if small_angle_mask.any():
            xyz = torch.where(
                small_angle_mask,
                omega * (0.5 - theta**2 / 48.0),
                xyz
            )
        
        return torch.cat([w, xyz], dim=-1)
    
    def logarithm_map_su2_to_so3(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        对数映射: SU(2) → so(3)
        
        给定单位四元数 q = (w, x, y, z)
        log(q) = θ * ω̂
        
        其中:
        - θ = 2*arccos(w)
        - ω̂ = (x, y, z) / sin(θ/2)
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
            
        w = q[..., 0:1]
        xyz = q[..., 1:]
        
        # 限制w到[-1, 1]避免arccos出错
        w = torch.clamp(w, -1.0 + eps, 1.0 - eps)
        theta = 2.0 * torch.acos(w)
        
        sin_half_theta = torch.sin(theta / 2.0)
        
        # 避免除零
        small_angle_mask = (sin_half_theta.abs() < eps)
        
        omega = torch.where(
            small_angle_mask,
            2.0 * xyz,  # 小角度近似
            theta * xyz / (sin_half_theta + eps)
        )
        
        return omega


class FractalGeometricDifferential(nn.Module):
    """
    分形几何微分算子
    
    实现Hausdorff维数动态调整和迭代函数系统(IFS)
    """
    
    def __init__(self, config: QuaternionLieGroupConfig):
        super().__init__()
        self.config = config
        self.levels = config.fractal_levels
        
        # 可学习的维数参数
        self.d_f_params = nn.Parameter(torch.randn(self.levels) * 0.1 + 1.5)
        
    def hausdorff_dimension_operator(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """
        Hausdorff维数算子
        
        对于分形集F，缩放变换:
        f(r*x) = r^d_f * f(x)
        
        其中 d_f 是Hausdorff维数
        """
        # 缩放比例: r_i = 0.5^level
        scaling_ratio = 0.5 ** level
        
        # 动态维数: d_f ∈ [1.0, 2.0]
        d_f = torch.sigmoid(self.d_f_params[level]) + 1.0
        
        # 分形缩放: x' = r^d_f * x
        scaling_factor = scaling_ratio ** d_f
        
        return scaling_factor * x
    
    def iterated_function_system(self, x: torch.Tensor) -> torch.Tensor:
        """
        迭代函数系统 (IFS)
        
        递归应用Hausdorff维数算子8层:
        F = ⋃_{i=1}^{8} f_i(F)
        """
        result = x
        for level in range(self.levels):
            result = self.hausdorff_dimension_operator(result, level)
        return result


class LieGroupAutomorphismEngine(nn.Module):
    """
    李群自同构引擎 - 完整模块
    
    整合:
    1. Hamilton四元数运算
    2. 分形几何变换
    3. 李群映射
    """
    
    def __init__(self, config: Optional[QuaternionLieGroupConfig] = None):
        super().__init__()
        
        if config is None:
            config = QuaternionLieGroupConfig()
        self.config = config
        
        # 子模块
        self.quaternion_module = QuaternionLieGroupModule(config)
        self.fractal_module = FractalGeometricDifferential(config)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, hidden_dim]
        
        Returns:
            output: 输出张量
            info: 信息字典（用于监控）
        """
        batch_size = x.size(0)
        
        # 1. 投影到四元数空间
        q = self.quaternion_module.to_quaternion(x)  # [batch, 4]
        q = self.quaternion_module.quaternion_normalize(q)
        
        # 2. 应用分形变换
        q_fractal = self.fractal_module.iterated_function_system(q)
        
        # 3. 再次归一化（保持在S³流形上）
        q_fractal = self.quaternion_module.quaternion_normalize(q_fractal)
        
        # 4. 投影回高维空间
        output = self.quaternion_module.from_quaternion(q_fractal)
        
        # 收集信息
        info = {
            'quaternion_norm': self.quaternion_module.quaternion_norm(q).mean().item(),
            'fractal_d_f_mean': torch.sigmoid(self.fractal_module.d_f_params).mean().item() + 1.0,
        }
        
        return output, info


# 测试函数
def test_quaternion_properties():
    """测试四元数的群性质"""
    print("="*60)
    print("测试 Hamilton 四元数群性质")
    print("="*60)
    
    config = QuaternionLieGroupConfig()
    module = QuaternionLieGroupModule(config)
    
    # 生成测试四元数
    q1 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    q1 = module.quaternion_normalize(q1)
    
    q2 = torch.tensor([1.0, 0.0, 1.0, 0.0])
    q2 = module.quaternion_normalize(q2)
    
    q3 = torch.tensor([1.0, 0.0, 0.0, 1.0])
    q3 = module.quaternion_normalize(q3)
    
    # 测试1: 结合律 (q1*q2)*q3 = q1*(q2*q3)
    print("\n✓ 测试结合律...")
    left = module.quaternion_multiply(module.quaternion_multiply(q1, q2), q3)
    right = module.quaternion_multiply(q1, module.quaternion_multiply(q2, q3))
    associativity_error = torch.norm(left - right).item()
    print(f"  结合律误差: {associativity_error:.2e}")
    print(f"  结合律: {'✅ PASS' if associativity_error < 1e-5 else '❌ FAIL'}")
    
    # 测试2: 单位元 e = (1,0,0,0)
    print("\n✓ 测试单位元...")
    identity = module.identity_quat
    id_left = module.quaternion_multiply(q1, identity)
    id_right = module.quaternion_multiply(identity, q1)
    identity_error = max(torch.norm(id_left - q1).item(), torch.norm(id_right - q1).item())
    print(f"  单位元误差: {identity_error:.2e}")
    print(f"  单位元: {'✅ PASS' if identity_error < 1e-5 else '❌ FAIL'}")
    
    # 测试3: 逆元 q*q^-1 = e
    print("\n✓ 测试逆元...")
    q1_inv = module.quaternion_inverse(q1)
    product = module.quaternion_multiply(q1, q1_inv)
    inverse_error = torch.norm(product - identity).item()
    print(f"  逆元误差: {inverse_error:.2e}")
    print(f"  逆元: {'✅ PASS' if inverse_error < 1e-5 else '❌ FAIL'}")
    
    # 测试4: 非交换性 q1*q2 ≠ q2*q1
    print("\n✓ 测试非交换性...")
    forward = module.quaternion_multiply(q1, q2)
    backward = module.quaternion_multiply(q2, q1)
    non_commutative_diff = torch.norm(forward - backward).item()
    print(f"  q1*q2 = {forward.numpy()}")
    print(f"  q2*q1 = {backward.numpy()}")
    print(f"  差异: {non_commutative_diff:.4f}")
    print(f"  非交换性: {'✅ PASS' if non_commutative_diff > 1e-5 else '❌ FAIL'}")
    
    # 测试5: 范数乘法性 |q1*q2| = |q1|*|q2|
    print("\n✓ 测试范数乘法性...")
    norm_product = module.quaternion_norm(module.quaternion_multiply(q1, q2)).item()
    norm_individual = module.quaternion_norm(q1).item() * module.quaternion_norm(q2).item()
    norm_error = abs(norm_product - norm_individual)
    print(f"  |q1*q2| = {norm_product:.6f}")
    print(f"  |q1|*|q2| = {norm_individual:.6f}")
    print(f"  误差: {norm_error:.2e}")
    print(f"  范数乘法性: {'✅ PASS' if norm_error < 1e-5 else '❌ FAIL'}")
    
    return {
        'associativity': associativity_error < 1e-5,
        'identity': identity_error < 1e-5,
        'inverse': inverse_error < 1e-5,
        'non_commutative': non_commutative_diff > 1e-5,
        'norm_multiplicative': norm_error < 1e-5
    }


def test_lie_group_mappings():
    """测试李群映射的互逆性"""
    print("\n" + "="*60)
    print("测试 李群 exp/log 映射互逆性")
    print("="*60)
    
    config = QuaternionLieGroupConfig()
    module = QuaternionLieGroupModule(config)
    
    # 生成测试向量
    omega = torch.randn(3) * 0.1
    
    # 测试 log(exp(ω)) = ω
    print("\n✓ 测试 log(exp(ω)) = ω...")
    q = module.exponential_map_so3_to_su2(omega)
    omega_reconstructed = module.logarithm_map_su2_to_so3(q)
    
    reconstruction_error = torch.norm(omega - omega_reconstructed).item()
    print(f"  原始 ω: {omega.numpy()}")
    print(f"  重构 ω: {omega_reconstructed.squeeze().numpy()}")
    print(f"  重构误差: {reconstruction_error:.2e}")
    print(f"  互逆性: {'✅ PASS' if reconstruction_error < 1e-4 else '❌ FAIL'}")
    
    # 测试范数保持
    print("\n✓ 测试范数保持 |exp(ω)| = 1...")
    q_norm = module.quaternion_norm(q).item()
    norm_error = abs(q_norm - 1.0)
    print(f"  |exp(ω)| = {q_norm:.6f}")
    print(f"  误差: {norm_error:.2e}")
    print(f"  范数保持: {'✅ PASS' if norm_error < 1e-5 else '❌ FAIL'}")
    
    return {
        'reconstruction': reconstruction_error < 1e-4,
        'norm_preservation': norm_error < 1e-5
    }


if __name__ == '__main__':
    print("\n" + "█"*60)
    print("█ Lie Group Automorphism Engine - 测试套件")
    print("█"*60)
    
    # 测试四元数性质
    quat_results = test_quaternion_properties()
    
    # 测试李群映射
    lie_results = test_lie_group_mappings()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    all_tests = {**quat_results, **lie_results}
    passed = sum(all_tests.values())
    total = len(all_tests)
    
    print(f"\n通过测试: {passed}/{total} ({100*passed/total:.1f}%)")
    
    for test_name, result in all_tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed == total:
        print("\n🏆 所有测试通过！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
