#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学统一性验证 - 综合性能测试

测试所有实现的数学模块并收集性能数据:
1. Hamilton四元数非交换群
2. 分形维数与IFS
3. Fueter四元数微积分
4. 反射算子 R² = I
5. 李群自同构
6. S³流形保持
"""

import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'h2q_project'))

# 导入已实现的模块
try:
    from lie_automorphism_engine import (
        QuaternionLieGroupModule,
        FractalGeometricDifferential,
        LieGroupAutomorphismEngine,
        QuaternionLieGroupConfig
    )
    LIE_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  无法导入 lie_automorphism_engine: {e}")
    LIE_MODULE_AVAILABLE = False

try:
    from noncommutative_geometry_operators import (
        FueterCalculusModule,
        ReflectionOperatorModule,
        NoncommutativeGeometryOperators,
        FueterCalculusConfig
    )
    FUETER_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  无法导入 noncommutative_geometry_operators: {e}")
    FUETER_MODULE_AVAILABLE = False

try:
    from automorphic_dde import (
        LieGroupActionModule,
        ManifoldProjectionModule,
        AutomorphicDDE,
        AutomorphicDDEConfig
    )
    AUTOMORPHIC_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  无法导入 automorphic_dde: {e}")
    AUTOMORPHIC_MODULE_AVAILABLE = False


class PerformanceBenchmark:
    """性能基准测试工具"""
    
    def __init__(self):
        self.results = {}
        
    def measure_time(self, func, *args, **kwargs) -> Tuple[float, any]:
        """测量函数执行时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        return elapsed, result
    
    def measure_memory(self, tensor: torch.Tensor) -> float:
        """测量张量内存使用（MB）"""
        return tensor.element_size() * tensor.nelement() / (1024 ** 2)
    
    def add_result(self, test_name: str, passed: bool, time_ms: float, 
                   memory_mb: float = 0.0, extra_info: dict = None):
        """添加测试结果"""
        self.results[test_name] = {
            'passed': passed,
            'time_ms': time_ms,
            'memory_mb': memory_mb,
            'extra_info': extra_info or {}
        }
    
    def print_summary(self):
        """打印测试总结"""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['passed'])
        total_time = sum(r['time_ms'] for r in self.results.values())
        
        print("\n" + "="*70)
        print("📊 综合性能测试总结")
        print("="*70)
        print(f"\n✅ 通过测试: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"⏱️  总耗时: {total_time:.2f} ms")
        print(f"💾 总内存: {sum(r['memory_mb'] for r in self.results.values()):.2f} MB")
        
        print("\n详细结果:")
        print("-"*70)
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {test_name:50s} {result['time_ms']:8.2f} ms")
        
        return passed, total


def test_quaternion_group_properties(benchmark: PerformanceBenchmark):
    """测试Hamilton四元数群性质"""
    print("\n" + "="*70)
    print("测试 1/6: Hamilton四元数非交换群")
    print("="*70)
    
    if not LIE_MODULE_AVAILABLE:
        print("❌ 模块不可用")
        return False
    
    config = QuaternionLieGroupConfig()
    module = QuaternionLieGroupModule(config)
    
    q1 = torch.randn(128, 4)
    q1 = module.quaternion_normalize(q1)
    q2 = torch.randn(128, 4)
    q2 = module.quaternion_normalize(q2)
    q3 = torch.randn(128, 4)
    q3 = module.quaternion_normalize(q3)
    
    # 测试结合律
    elapsed, _ = benchmark.measure_time(
        lambda: module.quaternion_multiply(module.quaternion_multiply(q1, q2), q3)
    )
    left = module.quaternion_multiply(module.quaternion_multiply(q1, q2), q3)
    right = module.quaternion_multiply(q1, module.quaternion_multiply(q2, q3))
    assoc_error = torch.norm(left - right).item()
    
    memory = benchmark.measure_memory(q1) * 3
    benchmark.add_result(
        "Hamilton四元数结合律",
        assoc_error < 1e-4,
        elapsed * 1000,
        memory,
        {'error': assoc_error}
    )
    
    # 测试非交换性
    elapsed, _ = benchmark.measure_time(
        lambda: module.quaternion_multiply(q1, q2)
    )
    forward = module.quaternion_multiply(q1, q2)
    backward = module.quaternion_multiply(q2, q1)
    non_comm = torch.norm(forward - backward).mean().item()
    
    benchmark.add_result(
        "Hamilton四元数非交换性",
        non_comm > 1e-4,
        elapsed * 1000,
        memory,
        {'non_commutative_measure': non_comm}
    )
    
    print(f"  ✓ 结合律误差: {assoc_error:.2e}")
    print(f"  ✓ 非交换性度量: {non_comm:.4f}")
    
    return True


def test_fractal_geometry(benchmark: PerformanceBenchmark):
    """测试分形几何变换"""
    print("\n" + "="*70)
    print("测试 2/6: 分形维数与迭代函数系统")
    print("="*70)
    
    if not LIE_MODULE_AVAILABLE:
        print("❌ 模块不可用")
        return False
    
    config = QuaternionLieGroupConfig()
    module = FractalGeometricDifferential(config)
    
    x = torch.randn(256, 4)
    
    # 测试IFS性能
    elapsed, result = benchmark.measure_time(
        module.iterated_function_system, x
    )
    
    # 验证分形维数范围
    d_f_values = torch.sigmoid(module.d_f_params) + 1.0
    d_f_valid = (d_f_values >= 1.0).all() and (d_f_values <= 2.0).all()
    
    memory = benchmark.measure_memory(x)
    benchmark.add_result(
        "分形维数约束 d_f ∈ [1,2]",
        d_f_valid,
        elapsed * 1000,
        memory,
        {'d_f_mean': d_f_values.mean().item()}
    )
    
    # 测试8层IFS
    benchmark.add_result(
        "8层迭代函数系统(IFS)",
        result.shape == x.shape,
        elapsed * 1000,
        memory,
        {'levels': config.fractal_levels}
    )
    
    print(f"  ✓ 分形维数均值: {d_f_values.mean().item():.4f}")
    print(f"  ✓ IFS层数: {config.fractal_levels}")
    
    return True


def test_fueter_calculus(benchmark: PerformanceBenchmark):
    """测试Fueter四元数微积分"""
    print("\n" + "="*70)
    print("测试 3/6: Fueter四元数微积分")
    print("="*70)
    
    if not FUETER_MODULE_AVAILABLE:
        print("❌ 模块不可用")
        return False
    
    config = FueterCalculusConfig()
    module = FueterCalculusModule(config)
    
    f = torch.randn(128, 4)
    
    # 测试左微分
    elapsed_left, d_left = benchmark.measure_time(
        module.left_quaternion_derivative, f, 'i'
    )
    
    # 测试右微分
    elapsed_right, d_right = benchmark.measure_time(
        module.right_quaternion_derivative, f, 'i'
    )
    
    # 测试非交换性
    d_lr = module.right_quaternion_derivative(d_left, 'j')
    d_rl = module.left_quaternion_derivative(d_right, 'j')
    commutator = torch.norm(d_lr - d_rl).item()
    
    memory = benchmark.measure_memory(f)
    benchmark.add_result(
        "Fueter左微分算子",
        d_left.shape == f.shape,
        elapsed_left * 1000,
        memory
    )
    
    benchmark.add_result(
        "Fueter右微分算子",
        d_right.shape == f.shape,
        elapsed_right * 1000,
        memory
    )
    
    benchmark.add_result(
        "Fueter微分非交换性 [∂_L, ∂_R] ≠ 0",
        commutator > 1e-5,
        (elapsed_left + elapsed_right) * 1000,
        memory * 2,
        {'commutator_norm': commutator}
    )
    
    # 测试全纯算子
    elapsed_holo, holo = benchmark.measure_time(
        module.fueter_holomorphic_operator, f
    )
    
    benchmark.add_result(
        "Fueter全纯算子",
        holo.shape[0] == f.shape[0],
        elapsed_holo * 1000,
        memory,
        {'holomorphic_measure': holo.mean().item()}
    )
    
    print(f"  ✓ 交换子范数: {commutator:.4f}")
    print(f"  ✓ 全纯度量: {holo.mean().item():.4f}")
    
    return True


def test_reflection_operators(benchmark: PerformanceBenchmark):
    """测试反射算子 R² = I"""
    print("\n" + "="*70)
    print("测试 4/6: 反射算子 R² = I")
    print("="*70)
    
    if not FUETER_MODULE_AVAILABLE:
        print("❌ 模块不可用")
        return False
    
    module = ReflectionOperatorModule(dim=4)
    
    # 测试反射矩阵生成
    elapsed, R = benchmark.measure_time(
        module.orthogonalize_reflection_matrix
    )
    
    # 验证性质
    props = module.verify_reflection_properties()
    
    memory = benchmark.measure_memory(R)
    benchmark.add_result(
        "反射矩阵幂等性 R² = I",
        props['idempotent_error'] < 1e-5,
        elapsed * 1000,
        memory,
        {'error': props['idempotent_error']}
    )
    
    benchmark.add_result(
        "反射矩阵对称性 R^T = R",
        props['symmetric_error'] < 1e-5,
        elapsed * 1000,
        memory,
        {'error': props['symmetric_error']}
    )
    
    benchmark.add_result(
        "反射矩阵正交性 R^T R = I",
        props['orthogonal_error'] < 1e-5,
        elapsed * 1000,
        memory,
        {'error': props['orthogonal_error']}
    )
    
    benchmark.add_result(
        "反射矩阵行列式 det(R) = ±1",
        props['det_error'] < 1e-3,
        elapsed * 1000,
        memory,
        {'det': props['det_value']}
    )
    
    print(f"  ✓ 幂等性误差: {props['idempotent_error']:.2e}")
    print(f"  ✓ 行列式: {props['det_value']:.6f}")
    
    return True


def test_lie_group_automorphism(benchmark: PerformanceBenchmark):
    """测试李群自同构"""
    print("\n" + "="*70)
    print("测试 5/6: 李群自同构 φ_g(q) = gqḡ")
    print("="*70)
    
    if not AUTOMORPHIC_MODULE_AVAILABLE:
        print("❌ 模块不可用")
        return False
    
    config = AutomorphicDDEConfig()
    module = LieGroupActionModule(config)
    
    q1 = torch.randn(128, 4)
    q1 = module.quaternion_normalize(q1)
    q2 = torch.randn(128, 4)
    q2 = module.quaternion_normalize(q2)
    
    # 测试自同构映射
    elapsed, phi_q1 = benchmark.measure_time(
        module.apply_lie_group_action, q1
    )
    
    # 验证性质
    props = module.verify_automorphism_properties(q1, q2)
    
    memory = benchmark.measure_memory(q1)
    benchmark.add_result(
        "李群自同构保乘法性 φ(q1·q2) = φ(q1)·φ(q2)",
        props['multiplicative_error'] < 1e-4,
        elapsed * 1000,
        memory,
        {'error': props['multiplicative_error']}
    )
    
    benchmark.add_result(
        "李群自同构保范数性 |φ(q)| = |q|",
        props['norm_preservation_error'] < 1e-5,
        elapsed * 1000,
        memory,
        {'error': props['norm_preservation_error']}
    )
    
    print(f"  ✓ 保乘法误差: {props['multiplicative_error']:.2e}")
    print(f"  ✓ 保范数误差: {props['norm_preservation_error']:.2e}")
    
    return True


def test_manifold_preservation(benchmark: PerformanceBenchmark):
    """测试S³流形保持"""
    print("\n" + "="*70)
    print("测试 6/6: S³ 单位球面流形保持")
    print("="*70)
    
    if not AUTOMORPHIC_MODULE_AVAILABLE:
        print("❌ 模块不可用")
        return False
    
    config = AutomorphicDDEConfig()
    module = ManifoldProjectionModule(config)
    
    # 生成随机数据
    x = torch.randn(256, 4) * 10.0
    
    # 测试投影到流形
    elapsed, q = benchmark.measure_time(
        module.lift_to_quaternion_manifold, x
    )
    
    # 验证流形约束
    check = module.verify_manifold_constraint(q)
    
    memory = benchmark.measure_memory(x)
    benchmark.add_result(
        "S³流形投影 |q| = 1",
        check['on_manifold'],
        elapsed * 1000,
        memory,
        {'max_deviation': check['max_deviation']}
    )
    
    # 测试测地线距离
    q1 = module.lift_to_quaternion_manifold(torch.randn(128, 4))
    q2 = module.lift_to_quaternion_manifold(torch.randn(128, 4))
    
    elapsed_geo, dist = benchmark.measure_time(
        module.geodesic_distance_on_sphere, q1, q2
    )
    
    dist_valid = (dist >= 0).all() and (dist <= 3.15).all()  # π ≈ 3.14
    
    benchmark.add_result(
        "S³测地线距离 d ∈ [0, π]",
        dist_valid,
        elapsed_geo * 1000,
        memory,
        {'mean_distance': dist.mean().item()}
    )
    
    # 测试平行传输
    v = torch.randn(128, 4)
    elapsed_transport, v_transported = benchmark.measure_time(
        module.parallel_transport_on_sphere, v, q1, q2
    )
    
    benchmark.add_result(
        "S³平行传输",
        v_transported.shape == v.shape,
        elapsed_transport * 1000,
        memory
    )
    
    print(f"  ✓ 流形最大偏离: {check['max_deviation']:.2e}")
    print(f"  ✓ 测地线距离: {dist.mean().item():.4f}")
    
    return True


def test_integrated_system(benchmark: PerformanceBenchmark):
    """测试集成系统"""
    print("\n" + "="*70)
    print("测试 集成: 完整数学统一架构")
    print("="*70)
    
    if not (LIE_MODULE_AVAILABLE and FUETER_MODULE_AVAILABLE and AUTOMORPHIC_MODULE_AVAILABLE):
        print("❌ 部分模块不可用")
        return False
    
    # 创建完整流程
    batch_size = 64
    hidden_dim = 256
    
    lie_engine = LieGroupAutomorphismEngine()
    fueter_ops = NoncommutativeGeometryOperators()
    automorphic_dde = AutomorphicDDE()
    
    x = torch.randn(batch_size, hidden_dim)
    
    # 测试完整前向传播
    start = time.time()
    
    out1, info1 = lie_engine(x)
    out2, info2 = fueter_ops(out1)
    out3, info3 = automorphic_dde(out2)
    
    end = time.time()
    elapsed = (end - start) * 1000
    
    # 验证形状保持
    shape_preserved = out3.shape == x.shape
    
    memory = benchmark.measure_memory(x) * 4
    benchmark.add_result(
        "完整流程: Lie → Fueter → Automorphic",
        shape_preserved,
        elapsed,
        memory,
        {
            'lie_fractal_d_f': info1['fractal_d_f_mean'],
            'fueter_holomorphic': info2['holomorphic_measure'],
            'manifold_deviation': info3['manifold_deviation']
        }
    )
    
    print(f"  ✓ 分形维数: {info1['fractal_d_f_mean']:.4f}")
    print(f"  ✓ Fueter全纯度: {info2['holomorphic_measure']:.4f}")
    print(f"  ✓ 流形偏离: {info3['manifold_deviation']:.2e}")
    
    return True


def generate_performance_report(benchmark: PerformanceBenchmark, output_file: str):
    """生成性能报告"""
    # 转换数据确保JSON可序列化
    def convert_to_json_serializable(obj):
        """递归转换对象为JSON可序列化格式"""
        if isinstance(obj, torch.Tensor):
            return float(obj.item()) if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(v) for v in obj]
        else:
            return str(obj)
    
    tests_data = {}
    for test_name, result in benchmark.results.items():
        tests_data[test_name] = {
            'passed': result['passed'],
            'time_ms': float(result['time_ms']),
            'memory_mb': float(result['memory_mb']),
            'extra_info': convert_to_json_serializable(result['extra_info'])
        }
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': len(benchmark.results),
        'passed_tests': sum(1 for r in benchmark.results.values() if r['passed']),
        'total_time_ms': float(sum(r['time_ms'] for r in benchmark.results.values())),
        'total_memory_mb': float(sum(r['memory_mb'] for r in benchmark.results.values())),
        'tests': tests_data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 性能报告已保存: {output_file}")
    
    return report


def main():
    """主测试函数"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  H2Q-Evo 数学统一性验证 - 综合性能测试套件".center(66) + "  █")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    benchmark = PerformanceBenchmark()
    
    # 运行所有测试
    tests = [
        test_quaternion_group_properties,
        test_fractal_geometry,
        test_fueter_calculus,
        test_reflection_operators,
        test_lie_group_automorphism,
        test_manifold_preservation,
        test_integrated_system
    ]
    
    for test_func in tests:
        try:
            test_func(benchmark)
        except Exception as e:
            print(f"❌ 测试失败: {test_func.__name__}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    passed, total = benchmark.print_summary()
    
    # 生成报告
    report_file = 'mathematical_performance_report.json'
    generate_performance_report(benchmark, report_file)
    
    # 最终评估
    print("\n" + "="*70)
    print("🎯 最终评估")
    print("="*70)
    
    pass_rate = 100 * passed / total if total > 0 else 0
    
    if pass_rate >= 95:
        grade = "🏆 Platinum"
        status = "优秀"
    elif pass_rate >= 85:
        grade = "🥇 Gold"
        status = "良好"
    elif pass_rate >= 70:
        grade = "🥈 Silver"
        status = "合格"
    else:
        grade = "🥉 Bronze"
        status = "需改进"
    
    print(f"\n等级: {grade}")
    print(f"状态: {status}")
    print(f"通过率: {pass_rate:.1f}%")
    print(f"总耗时: {sum(r['time_ms'] for r in benchmark.results.values()):.2f} ms")
    
    print("\n" + "█"*70)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
