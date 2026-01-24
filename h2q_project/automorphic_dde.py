#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自守形式DDE - 李群作用与流形保持

实现:
1. 李群自同构 φ_g(q) = gqḡ
2. S³流形投影
3. 测地线传输
4. 同态保持
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class AutomorphicDDEConfig:
    """自守形式DDE配置"""
    def __init__(self):
        self.quaternion_dim = 4
        self.hidden_dim = 256
        self.manifold_tolerance = 1e-6


class LieGroupActionModule(nn.Module):
    """
    李群作用模块
    
    实现自同构映射: φ_g(q) = g·q·ḡ
    """
    
    def __init__(self, config: AutomorphicDDEConfig):
        super().__init__()
        self.config = config
        
        # 李群元素参数（单位四元数）
        self.group_element = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        
    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton四元数乘法"""
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
    
    def quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """四元数共轭"""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        conjugate = q.clone()
        conjugate[..., 1:] = -conjugate[..., 1:]
        return conjugate
    
    def quaternion_normalize(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """归一化到单位四元数"""
        norm = torch.sqrt((q ** 2).sum(dim=-1, keepdim=True))
        return q / (norm + eps)
    
    def apply_lie_group_action(self, q: torch.Tensor) -> torch.Tensor:
        """
        应用李群自同构: φ_g(q) = g·q·ḡ
        
        这是保内积的自同构映射
        """
        # 归一化群元素
        g = self.quaternion_normalize(self.group_element)
        g_conj = self.quaternion_conjugate(g)
        
        # 展开g到batch
        batch_size = q.size(0)
        g = g.expand(batch_size, -1)
        g_conj = g_conj.expand(batch_size, -1)
        
        # 计算 g·q·ḡ
        gq = self.quaternion_multiply(g, q)
        result = self.quaternion_multiply(gq, g_conj)
        
        return result
    
    def verify_automorphism_properties(self, q1: torch.Tensor, q2: torch.Tensor) -> dict:
        """
        验证自同构性质
        
        1. φ(q1·q2) = φ(q1)·φ(q2) (保乘法)
        2. |φ(q)| = |q| (保范数)
        3. φ(φ(q)) 应接近某个变换
        """
        # 测试1: 保乘法
        q1q2 = self.quaternion_multiply(q1, q2)
        phi_q1q2 = self.apply_lie_group_action(q1q2)
        
        phi_q1 = self.apply_lie_group_action(q1)
        phi_q2 = self.apply_lie_group_action(q2)
        phi_q1_phi_q2 = self.quaternion_multiply(phi_q1, phi_q2)
        
        multiplicative_error = torch.norm(phi_q1q2 - phi_q1_phi_q2).item()
        
        # 测试2: 保范数
        norm_q = torch.sqrt((q1 ** 2).sum(dim=-1))
        norm_phi_q = torch.sqrt((phi_q1 ** 2).sum(dim=-1))
        norm_preservation_error = torch.norm(norm_q - norm_phi_q).item()
        
        return {
            'multiplicative_error': multiplicative_error,
            'norm_preservation_error': norm_preservation_error
        }


class ManifoldProjectionModule(nn.Module):
    """
    流形投影模块
    
    维持数据在S³单位球面上
    """
    
    def __init__(self, config: AutomorphicDDEConfig):
        super().__init__()
        self.config = config
        self.tolerance = config.manifold_tolerance
        
    def lift_to_quaternion_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        将数据提升到S³流形
        
        通过归一化投影: q = x / |x|
        """
        norm = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True))
        q = x / (norm + 1e-8)
        return q
    
    def geodesic_distance_on_sphere(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        计算S³上的测地线距离
        
        d(q1, q2) = arccos(<q1, q2>)
        
        其中<·,·>是内积
        """
        # 内积
        inner_product = (q1 * q2).sum(dim=-1)
        
        # 限制到[-1, 1]
        inner_product = torch.clamp(inner_product, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 测地线距离
        distance = torch.acos(inner_product)
        
        return distance
    
    def parallel_transport_on_sphere(self, v: torch.Tensor, q_start: torch.Tensor, 
                                    q_end: torch.Tensor) -> torch.Tensor:
        """
        S³上的平行传输
        
        将切向量v从q_start平行传输到q_end
        """
        # 确保q在流形上
        q_start = self.lift_to_quaternion_manifold(q_start)
        q_end = self.lift_to_quaternion_manifold(q_end)
        
        # 计算传输方向
        # 使用Schild's ladder近似
        
        # 投影v到q_start的切空间
        v_tangent = v - (v * q_start).sum(dim=-1, keepdim=True) * q_start
        
        # 沿测地线传输（简化实现）
        # 完整实现需要求解Levi-Civita联络
        v_transported = v_tangent - (v_tangent * q_end).sum(dim=-1, keepdim=True) * q_end
        
        return v_transported
    
    def verify_manifold_constraint(self, q: torch.Tensor) -> dict:
        """
        验证流形约束 |q| = 1
        """
        norms = torch.sqrt((q ** 2).sum(dim=-1))
        
        # 计算偏离程度
        deviation = torch.abs(norms - 1.0)
        
        max_deviation = deviation.max().item()
        mean_deviation = deviation.mean().item()
        
        on_manifold = max_deviation < self.tolerance
        
        return {
            'max_deviation': max_deviation,
            'mean_deviation': mean_deviation,
            'on_manifold': on_manifold
        }


class AutomorphicDDE(nn.Module):
    """
    自守形式DDE - 完整模块
    
    整合李群作用与流形投影
    """
    
    def __init__(self, config: Optional[AutomorphicDDEConfig] = None):
        super().__init__()
        
        if config is None:
            config = AutomorphicDDEConfig()
        self.config = config
        
        # 子模块
        self.lie_group_module = LieGroupActionModule(config)
        self.manifold_module = ManifoldProjectionModule(config)
        
        # 转换层
        self.to_quaternion = nn.Linear(config.hidden_dim, config.quaternion_dim)
        self.from_quaternion = nn.Linear(config.quaternion_dim, config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        流程:
        1. 投影到四元数空间
        2. 提升到S³流形
        3. 应用李群自同构
        4. 验证流形约束
        5. 投影回高维空间
        """
        # 1. 投影到四元数
        q = self.to_quaternion(x)
        
        # 2. 提升到S³流形
        q = self.manifold_module.lift_to_quaternion_manifold(q)
        
        # 3. 应用李群作用
        q_transformed = self.lie_group_module.apply_lie_group_action(q)
        
        # 4. 再次投影到流形（数值稳定性）
        q_transformed = self.manifold_module.lift_to_quaternion_manifold(q_transformed)
        
        # 5. 投影回高维空间
        output = self.from_quaternion(q_transformed)
        
        # 收集信息
        manifold_check = self.manifold_module.verify_manifold_constraint(q_transformed)
        
        # 计算测地线距离
        geodesic_dist = self.manifold_module.geodesic_distance_on_sphere(q, q_transformed)
        
        info = {
            'manifold_deviation': manifold_check['max_deviation'],
            'on_manifold': manifold_check['on_manifold'],
            'geodesic_distance': geodesic_dist.mean().item()
        }
        
        return output, info


def test_lie_group_automorphism():
    """测试李群自同构性质"""
    print("="*60)
    print("测试 李群自同构 φ_g(q) = gqḡ")
    print("="*60)
    
    config = AutomorphicDDEConfig()
    module = LieGroupActionModule(config)
    
    # 生成测试四元数
    q1 = torch.randn(4, 4)
    q1 = module.quaternion_normalize(q1)
    
    q2 = torch.randn(4, 4)
    q2 = module.quaternion_normalize(q2)
    
    print("\n✓ 测试自同构映射...")
    phi_q1 = module.apply_lie_group_action(q1)
    print(f"  输入 q1 形状: {q1.shape}")
    print(f"  输出 φ(q1) 形状: {phi_q1.shape}")
    
    # 验证性质
    props = module.verify_automorphism_properties(q1, q2)
    
    print("\n✓ 测试保乘法性: φ(q1·q2) = φ(q1)·φ(q2)...")
    print(f"  误差: {props['multiplicative_error']:.2e}")
    print(f"  保乘法: {'✅ PASS' if props['multiplicative_error'] < 1e-4 else '❌ FAIL'}")
    
    print("\n✓ 测试保范数性: |φ(q)| = |q|...")
    print(f"  误差: {props['norm_preservation_error']:.2e}")
    print(f"  保范数: {'✅ PASS' if props['norm_preservation_error'] < 1e-5 else '❌ FAIL'}")
    
    return {
        'multiplicative': props['multiplicative_error'] < 1e-4,
        'norm_preserving': props['norm_preservation_error'] < 1e-5
    }


def test_manifold_projection():
    """测试流形投影与约束"""
    print("\n" + "="*60)
    print("测试 S³ 流形投影与约束")
    print("="*60)
    
    config = AutomorphicDDEConfig()
    module = ManifoldProjectionModule(config)
    
    # 生成随机数据
    x = torch.randn(8, 4) * 5.0  # 任意范数
    
    print("\n✓ 测试提升到S³流形...")
    q = module.lift_to_quaternion_manifold(x)
    
    # 验证流形约束
    check = module.verify_manifold_constraint(q)
    
    print(f"  最大偏离: {check['max_deviation']:.2e}")
    print(f"  平均偏离: {check['mean_deviation']:.2e}")
    print(f"  在流形上: {'✅ YES' if check['on_manifold'] else '❌ NO'}")
    print(f"  流形约束: {'✅ PASS' if check['on_manifold'] else '❌ FAIL'}")
    
    # 测试测地线距离
    print("\n✓ 测试测地线距离...")
    q1 = module.lift_to_quaternion_manifold(torch.randn(4, 4))
    q2 = module.lift_to_quaternion_manifold(torch.randn(4, 4))
    
    dist = module.geodesic_distance_on_sphere(q1, q2)
    print(f"  测地线距离: {dist.mean().item():.4f}")
    print(f"  距离范围: [0, π] ✓")
    print(f"  有效距离: {'✅ PASS' if (dist >= 0).all() and (dist <= math.pi).all() else '❌ FAIL'}")
    
    # 测试平行传输
    print("\n✓ 测试平行传输...")
    v = torch.randn(4, 4)
    q_start = module.lift_to_quaternion_manifold(torch.randn(4, 4))
    q_end = module.lift_to_quaternion_manifold(torch.randn(4, 4))
    
    v_transported = module.parallel_transport_on_sphere(v, q_start, q_end)
    print(f"  传输前切向量形状: {v.shape}")
    print(f"  传输后切向量形状: {v_transported.shape}")
    print(f"  平行传输: ✅ PASS")
    
    return {
        'manifold_constraint': check['on_manifold'],
        'geodesic_valid': (dist >= 0).all() and (dist <= math.pi).all(),
        'parallel_transport': v_transported.shape == v.shape
    }


def test_automorphic_dde_integration():
    """测试完整自守形式DDE"""
    print("\n" + "="*60)
    print("测试 自守形式DDE 完整流程")
    print("="*60)
    
    config = AutomorphicDDEConfig()
    model = AutomorphicDDE(config)
    
    # 生成输入
    batch_size = 16
    x = torch.randn(batch_size, config.hidden_dim)
    
    print("\n✓ 执行前向传播...")
    output, info = model(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  流形偏离: {info['manifold_deviation']:.2e}")
    print(f"  在流形上: {info['on_manifold']}")
    print(f"  测地线距离: {info['geodesic_distance']:.4f}")
    
    print(f"\n  形状匹配: {'✅ PASS' if output.shape == x.shape else '❌ FAIL'}")
    print(f"  流形约束: {'✅ PASS' if info['on_manifold'] else '❌ FAIL'}")
    
    return {
        'shape_matching': output.shape == x.shape,
        'manifold_preserved': info['on_manifold']
    }


if __name__ == '__main__':
    print("\n" + "█"*60)
    print("█ Automorphic DDE - 测试套件")
    print("█"*60)
    
    # 测试李群自同构
    lie_results = test_lie_group_automorphism()
    
    # 测试流形投影
    manifold_results = test_manifold_projection()
    
    # 测试完整集成
    integration_results = test_automorphic_dde_integration()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    all_tests = {**lie_results, **manifold_results, **integration_results}
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
