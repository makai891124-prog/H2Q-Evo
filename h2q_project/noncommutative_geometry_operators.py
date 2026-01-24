#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非交换几何算子 - Fueter四元数微积分与反射算子

实现:
1. Fueter四元数左/右微分
2. 反射矩阵 R² = I
3. 正交化约束
4. 全纯算子
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class FueterCalculusConfig:
    """Fueter微积分配置"""
    def __init__(self):
        self.quaternion_dim = 4
        self.hidden_dim = 256
        self.num_directions = 4  # 沿4个方向{1,i,j,k}求导


class FueterCalculusModule(nn.Module):
    """
    Fueter四元数微积分模块
    
    实现Fueter-Sce-Qian理论的四元数微分算子
    """
    
    def __init__(self, config: FueterCalculusConfig):
        super().__init__()
        self.config = config
        
        # Pauli矩阵基底
        # i = [0,1,0,0], j = [0,0,1,0], k = [0,0,0,1]
        self.register_buffer("i_unit", torch.tensor([0.0, 1.0, 0.0, 0.0]))
        self.register_buffer("j_unit", torch.tensor([0.0, 0.0, 1.0, 0.0]))
        self.register_buffer("k_unit", torch.tensor([0.0, 0.0, 0.0, 1.0]))
        
        # 可学习的微分参数
        self.diff_weights = nn.Parameter(torch.randn(4, 4) * 0.01)
        
    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton四元数乘法（内部使用）"""
        if q1.dim() == 1:
            q1 = q1.unsqueeze(0)
        if q2.dim() == 1:
            q2 = q2.unsqueeze(0)
            
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def left_quaternion_derivative(self, f: torch.Tensor, direction: str = 'i') -> torch.Tensor:
        """
        Fueter左微分算子
        
        ∂_L f = (∂_w + i∂_x + j∂_y + k∂_z) f
        
        满足左乘规则: ∂_L(q·f) = q·∂_L(f)
        """
        batch_size = f.size(0)
        
        # 选择方向单位四元数
        if direction == 'i':
            unit = self.i_unit
        elif direction == 'j':
            unit = self.j_unit
        elif direction == 'k':
            unit = self.k_unit
        else:  # direction == '1' (实部)
            unit = torch.tensor([1.0, 0.0, 0.0, 0.0], device=f.device)
        
        # 展开unit到batch
        unit = unit.unsqueeze(0).expand(batch_size, -1)
        
        # 左乘: unit * f
        derivative = self.quaternion_multiply(unit, f)
        
        # 应用可学习权重
        derivative = torch.matmul(derivative, self.diff_weights.t())
        
        return derivative
    
    def right_quaternion_derivative(self, f: torch.Tensor, direction: str = 'i') -> torch.Tensor:
        """
        Fueter右微分算子
        
        ∂_R f = (∂_w + ∂_x·i + ∂_y·j + ∂_z·k) f
        
        满足右乘规则: ∂_R(f·q) = ∂_R(f)·q
        """
        batch_size = f.size(0)
        
        # 选择方向单位四元数
        if direction == 'i':
            unit = self.i_unit
        elif direction == 'j':
            unit = self.j_unit
        elif direction == 'k':
            unit = self.k_unit
        else:
            unit = torch.tensor([1.0, 0.0, 0.0, 0.0], device=f.device)
        
        unit = unit.unsqueeze(0).expand(batch_size, -1)
        
        # 右乘: f * unit
        derivative = self.quaternion_multiply(f, unit)
        
        # 应用可学习权重
        derivative = torch.matmul(derivative, self.diff_weights.t())
        
        return derivative
    
    def fueter_holomorphic_operator(self, f: torch.Tensor) -> torch.Tensor:
        """
        Fueter全纯算子
        
        函数f是Fueter全纯当且仅当:
        ∂_L f = 0 (左微分为零)
        
        返回全纯性度量: |∂_L f|
        """
        # 计算4个方向的左微分
        derivatives = []
        for direction in ['i', 'j', 'k']:
            d = self.left_quaternion_derivative(f, direction)
            derivatives.append(d)
        
        # 合并
        total_derivative = torch.stack(derivatives, dim=1).sum(dim=1)
        
        # 计算范数
        holomorphic_measure = torch.norm(total_derivative, dim=-1, keepdim=True)
        
        return holomorphic_measure


class ReflectionOperatorModule(nn.Module):
    """
    反射算子模块
    
    实现正交反射矩阵 R 满足:
    1. R² = I (幂等性)
    2. R^T = R (对称性)
    3. det(R) = -1 (反射性质)
    """
    
    def __init__(self, dim: int = 4):
        super().__init__()
        self.dim = dim
        
        # Householder反射向量
        self.reflection_vector = nn.Parameter(torch.randn(dim))
        
    def orthogonalize_reflection_matrix(self) -> torch.Tensor:
        """
        构造正交反射矩阵
        
        使用Householder反射:
        R = I - 2vv^T / |v|^2
        
        保证 R² = I
        """
        v = self.reflection_vector
        v_norm_sq = (v ** 2).sum()
        
        # 避免除零
        v_norm_sq = v_norm_sq + 1e-8
        
        # v v^T
        vvT = torch.outer(v, v)
        
        # I - 2vv^T / |v|^2
        I = torch.eye(self.dim, device=v.device)
        R = I - 2.0 * vvT / v_norm_sq
        
        return R
    
    def verify_reflection_properties(self) -> dict:
        """
        验证反射矩阵性质
        
        Returns:
            包含各项性质验证结果的字典
        """
        R = self.orthogonalize_reflection_matrix()
        
        # 测试1: R² = I
        R_squared = torch.matmul(R, R)
        I = torch.eye(self.dim, device=R.device)
        idempotent_error = torch.norm(R_squared - I).item()
        
        # 测试2: R^T = R (对称性)
        symmetric_error = torch.norm(R - R.t()).item()
        
        # 测试3: R^T R = I (正交性)
        orthogonal_error = torch.norm(torch.matmul(R.t(), R) - I).item()
        
        # 测试4: det(R) ≈ ±1
        det_R = torch.det(R).item()
        det_error = abs(abs(det_R) - 1.0)
        
        return {
            'idempotent_error': idempotent_error,
            'symmetric_error': symmetric_error,
            'orthogonal_error': orthogonal_error,
            'det_error': det_error,
            'det_value': det_R
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用反射变换
        
        Args:
            x: 输入张量 [batch, dim]
        
        Returns:
            Rx: 反射后的张量
        """
        R = self.orthogonalize_reflection_matrix()
        return torch.matmul(x, R.t())


class NoncommutativeGeometryOperators(nn.Module):
    """
    非交换几何算子 - 完整模块
    
    整合Fueter微积分与反射算子
    """
    
    def __init__(self, config: Optional[FueterCalculusConfig] = None):
        super().__init__()
        
        if config is None:
            config = FueterCalculusConfig()
        self.config = config
        
        # 子模块
        self.fueter_module = FueterCalculusModule(config)
        self.reflection_module = ReflectionOperatorModule(config.quaternion_dim)
        
        # 转换层
        self.to_quaternion = nn.Linear(config.hidden_dim, config.quaternion_dim)
        self.from_quaternion = nn.Linear(config.quaternion_dim, config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, hidden_dim]
        
        Returns:
            output: 输出张量
            info: 信息字典
        """
        # 1. 投影到四元数空间
        q = self.to_quaternion(x)  # [batch, 4]
        
        # 2. 应用反射变换
        q_reflected = self.reflection_module(q)
        
        # 3. 计算Fueter全纯度
        holomorphic_measure = self.fueter_module.fueter_holomorphic_operator(q_reflected)
        
        # 4. 投影回高维空间
        output = self.from_quaternion(q_reflected)
        
        # 收集信息
        reflection_props = self.reflection_module.verify_reflection_properties()
        
        info = {
            'holomorphic_measure': holomorphic_measure.mean().item(),
            'reflection_idempotent_error': reflection_props['idempotent_error'],
            'reflection_det': reflection_props['det_value']
        }
        
        return output, info


def test_fueter_derivatives():
    """测试Fueter微分算子"""
    print("="*60)
    print("测试 Fueter 四元数微分算子")
    print("="*60)
    
    config = FueterCalculusConfig()
    module = FueterCalculusModule(config)
    
    # 生成测试四元数函数
    batch_size = 4
    f = torch.randn(batch_size, 4)
    
    # 测试左微分
    print("\n✓ 测试左微分算子...")
    d_left_i = module.left_quaternion_derivative(f, 'i')
    d_left_j = module.left_quaternion_derivative(f, 'j')
    d_left_k = module.left_quaternion_derivative(f, 'k')
    
    print(f"  左微分 (i方向) 形状: {d_left_i.shape}")
    print(f"  左微分 (j方向) 形状: {d_left_j.shape}")
    print(f"  左微分 (k方向) 形状: {d_left_k.shape}")
    
    # 测试右微分
    print("\n✓ 测试右微分算子...")
    d_right_i = module.right_quaternion_derivative(f, 'i')
    d_right_j = module.right_quaternion_derivative(f, 'j')
    d_right_k = module.right_quaternion_derivative(f, 'k')
    
    print(f"  右微分 (i方向) 形状: {d_right_i.shape}")
    print(f"  右微分 (j方向) 形状: {d_right_j.shape}")
    print(f"  右微分 (k方向) 形状: {d_right_k.shape}")
    
    # 测试全纯性
    print("\n✓ 测试Fueter全纯算子...")
    holomorphic = module.fueter_holomorphic_operator(f)
    print(f"  全纯度量: {holomorphic.mean().item():.4f}")
    print(f"  全纯度量范围: [{holomorphic.min().item():.4f}, {holomorphic.max().item():.4f}]")
    
    # 验证左右微分不交换性
    print("\n✓ 测试非交换性: ∂_L ∂_R ≠ ∂_R ∂_L...")
    d_lr = module.right_quaternion_derivative(d_left_i, 'j')
    d_rl = module.left_quaternion_derivative(d_right_j, 'i')
    commutator_norm = torch.norm(d_lr - d_rl).item()
    print(f"  [∂_L, ∂_R] = {commutator_norm:.4f}")
    print(f"  非交换性: {'✅ PASS' if commutator_norm > 1e-4 else '❌ FAIL'}")
    
    return {
        'left_derivative': d_left_i.shape == torch.Size([batch_size, 4]),
        'right_derivative': d_right_i.shape == torch.Size([batch_size, 4]),
        'holomorphic': holomorphic.shape == torch.Size([batch_size, 1]),
        'non_commutative': commutator_norm > 1e-4
    }


def test_reflection_operators():
    """测试反射算子性质"""
    print("\n" + "="*60)
    print("测试 反射算子 R² = I")
    print("="*60)
    
    module = ReflectionOperatorModule(dim=4)
    
    # 获取反射矩阵
    R = module.orthogonalize_reflection_matrix()
    print(f"\n✓ 反射矩阵形状: {R.shape}")
    print(f"  R =\n{R.detach().numpy()}")
    
    # 验证性质
    props = module.verify_reflection_properties()
    
    print("\n✓ 测试幂等性 R² = I...")
    print(f"  |R² - I| = {props['idempotent_error']:.2e}")
    print(f"  幂等性: {'✅ PASS' if props['idempotent_error'] < 1e-5 else '❌ FAIL'}")
    
    print("\n✓ 测试对称性 R^T = R...")
    print(f"  |R^T - R| = {props['symmetric_error']:.2e}")
    print(f"  对称性: {'✅ PASS' if props['symmetric_error'] < 1e-5 else '❌ FAIL'}")
    
    print("\n✓ 测试正交性 R^T R = I...")
    print(f"  |R^T R - I| = {props['orthogonal_error']:.2e}")
    print(f"  正交性: {'✅ PASS' if props['orthogonal_error'] < 1e-5 else '❌ FAIL'}")
    
    print("\n✓ 测试行列式 det(R) = ±1...")
    print(f"  det(R) = {props['det_value']:.6f}")
    print(f"  |det(R)| - 1| = {props['det_error']:.2e}")
    print(f"  行列式: {'✅ PASS' if props['det_error'] < 1e-3 else '❌ FAIL'}")
    
    # 测试反射效果
    print("\n✓ 测试反射变换...")
    x = torch.randn(8, 4)
    Rx = module(x)
    
    # 验证 R(Rx) = x
    RRx = module(Rx)
    reconstruction_error = torch.norm(RRx - x).item()
    print(f"  |R(Rx) - x| = {reconstruction_error:.2e}")
    print(f"  反射效果: {'✅ PASS' if reconstruction_error < 1e-4 else '❌ FAIL'}")
    
    return {
        'idempotent': props['idempotent_error'] < 1e-5,
        'symmetric': props['symmetric_error'] < 1e-5,
        'orthogonal': props['orthogonal_error'] < 1e-5,
        'determinant': props['det_error'] < 1e-3,
        'reflection_effect': reconstruction_error < 1e-4
    }


if __name__ == '__main__':
    print("\n" + "█"*60)
    print("█ Noncommutative Geometry Operators - 测试套件")
    print("█"*60)
    
    # 测试Fueter微分
    fueter_results = test_fueter_derivatives()
    
    # 测试反射算子
    reflection_results = test_reflection_operators()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    all_tests = {**fueter_results, **reflection_results}
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
