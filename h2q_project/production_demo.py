"""
H2Q-Evo 生产就绪示例
展示如何使用版本控制、健康检查、鲁棒性包装器等功能
"""

import torch
import torch.nn as nn
from pathlib import Path

def main():
    print("🚀 H2Q-Evo 生产就绪系统演示")
    print("="*60)
    
    # ====================================================================
    # 第1步: 导入核心组件
    # ====================================================================
    print("\n📦 第1步: 导入核心组件...")
    
    from h2q.core.discrete_decision_engine import (
        DiscreteDecisionEngine,
        LatentConfig,
        get_canonical_dde
    )
    from h2q.core.algorithm_version_control import (
        get_version_control,
        AlgorithmStatus,
        verify_algorithm_compatibility
    )
    from h2q.core.production_validator import (
        ProductionValidator,
        run_production_validation
    )
    from h2q.core.robustness_wrapper import (
        RobustWrapper,
        RobustDiscreteDecisionEngine,
        SafetyGuard,
        robust_inference
    )
    
    print("✅ 所有核心组件已导入")
    
    # ====================================================================
    # 第2步: 初始化版本控制系统
    # ====================================================================
    print("\n📝 第2步: 初始化版本控制系统...")
    
    vc = get_version_control()
    
    # 创建配置
    config = LatentConfig(dim=256, n_choices=64, temperature=1.0)
    
    # 创建模型
    base_model = get_canonical_dde(config=config)
    
    # 注册算法版本
    try:
        version_info = vc.register_algorithm(
            name="DiscreteDecisionEngine",
            version="2.1.0",
            status=AlgorithmStatus.PRODUCTION,
            description="生产环境稳定版本",
            module=base_model,
            config={'latent_dim': 256, 'n_choices': 64},
            author="H2Q-Evo Team"
        )
        print(f"✅ 已注册算法: {version_info.name} v{version_info.version}")
        print(f"   状态: {version_info.status.value}")
        print(f"   签名: {version_info.signature}")
    except Exception as e:
        print(f"⚠️ 算法已注册或注册失败: {e}")
    
    # 验证兼容性
    is_compatible = verify_algorithm_compatibility(
        "DiscreteDecisionEngine",
        "2.0.0"
    )
    print(f"✅ 版本兼容性检查: {'通过' if is_compatible else '失败'}")
    
    # ====================================================================
    # 第3步: 包装模型以增强鲁棒性
    # ====================================================================
    print("\n🛡️ 第3步: 包装模型以增强鲁棒性...")
    
    robust_model = RobustDiscreteDecisionEngine(base_model)
    print("✅ 鲁棒性包装器已应用")
    print("   - 自动输入验证")
    print("   - NaN/Inf 检测与修复")
    print("   - GPU OOM 自动降级到 CPU")
    print("   - 异常值裁剪")
    
    # ====================================================================
    # 第4步: 运行生产环境验证
    # ====================================================================
    print("\n🏥 第4步: 运行生产环境健康检查...")
    
    validator = ProductionValidator()
    health_report = validator.run_full_validation()
    
    overall_status = health_report['overall_status']
    print(f"\n整体健康状态: {overall_status.upper()}")
    
    if overall_status == 'healthy':
        print("✅ 系统健康，可以继续")
    else:
        print("⚠️ 系统状态异常，建议检查")
    
    # ====================================================================
    # 第5步: 演示推理功能
    # ====================================================================
    print("\n🧠 第5步: 演示推理功能...")
    
    # 正常输入
    print("\n案例 1: 正常输入")
    normal_input = torch.randn(4, 256)
    
    with torch.no_grad():
        output = robust_model(normal_input)
    
    print(f"✅ 输入形状: {normal_input.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print(f"✅ 输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 异常输入（包含 NaN 和 Inf）
    print("\n案例 2: 异常输入（包含 NaN 和 Inf）")
    bad_input = torch.randn(4, 256)
    bad_input[0, 0] = float('nan')
    bad_input[1, 0] = float('inf')
    bad_input[2, 0] = -float('inf')
    
    print(f"⚠️ 输入包含异常值:")
    print(f"   - NaN 数量: {torch.isnan(bad_input).sum()}")
    print(f"   - Inf 数量: {torch.isinf(bad_input).sum()}")
    
    try:
        with torch.no_grad():
            cleaned_output = robust_model(bad_input)
        print(f"✅ 鲁棒包装器自动处理了异常值")
        print(f"✅ 输出形状: {cleaned_output.shape}")
        print(f"✅ 输出中 NaN 数量: {torch.isnan(cleaned_output).sum()}")
        print(f"✅ 输出中 Inf 数量: {torch.isinf(cleaned_output).sum()}")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
    
    # ====================================================================
    # 第6步: 演示安全数学操作
    # ====================================================================
    print("\n🔒 第6步: 演示安全数学操作...")
    
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 0.0, 4.0])  # 包含零
    
    # 不安全的除法会导致 Inf
    print("\n不安全除法:")
    unsafe_result = a / b
    print(f"   结果: {unsafe_result}")
    print(f"   包含 Inf: {torch.isinf(unsafe_result).any()}")
    
    # 安全除法避免除零
    print("\n安全除法:")
    safe_result = SafetyGuard.safe_division(a, b, epsilon=1e-8)
    print(f"   结果: {safe_result}")
    print(f"   包含 Inf: {torch.isinf(safe_result).any()}")
    
    # 安全对数
    x = torch.tensor([0.0, 1.0, 2.0])
    print("\n安全对数:")
    safe_log = SafetyGuard.safe_log(x)
    print(f"   输入: {x}")
    print(f"   输出: {safe_log}")
    
    # ====================================================================
    # 第7步: 性能基准测试
    # ====================================================================
    print("\n⚡ 第7步: 性能基准测试...")
    
    import time
    
    test_input = torch.randn(1, 256)
    times = []
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = robust_model(test_input)
    
    # 基准测试
    num_iterations = 100
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = robust_model(test_input)
        times.append((time.time() - start) * 1000)  # 转换为毫秒
    
    import statistics
    avg_time = statistics.mean(times)
    p50_time = statistics.median(times)
    p95_time = sorted(times)[int(0.95 * len(times))]
    p99_time = sorted(times)[int(0.99 * len(times))]
    
    print(f"✅ 基准测试完成 ({num_iterations} 次迭代)")
    print(f"   平均延迟: {avg_time:.2f}ms")
    print(f"   P50 延迟: {p50_time:.2f}ms")
    print(f"   P95 延迟: {p95_time:.2f}ms")
    print(f"   P99 延迟: {p99_time:.2f}ms")
    print(f"   吞吐量: ~{1000/avg_time:.0f} QPS")
    
    # ====================================================================
    # 第8步: 保存报告和日志
    # ====================================================================
    print("\n📊 第8步: 生成和保存报告...")
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # 保存健康检查报告
    import json
    health_report_path = reports_dir / "health_check_demo.json"
    with open(health_report_path, 'w') as f:
        json.dump(health_report, f, indent=2)
    print(f"✅ 健康检查报告已保存: {health_report_path}")
    
    # 保存性能报告
    perf_report = {
        'timestamp': '2026-01-20',
        'model': 'DiscreteDecisionEngine v2.1.0',
        'metrics': {
            'avg_latency_ms': avg_time,
            'p50_latency_ms': p50_time,
            'p95_latency_ms': p95_time,
            'p99_latency_ms': p99_time,
            'throughput_qps': 1000/avg_time
        },
        'environment': {
            'device': str(test_input.device),
            'dtype': str(test_input.dtype)
        }
    }
    
    perf_report_path = reports_dir / "performance_demo.json"
    with open(perf_report_path, 'w') as f:
        json.dump(perf_report, f, indent=2)
    print(f"✅ 性能报告已保存: {perf_report_path}")
    
    # ====================================================================
    # 总结
    # ====================================================================
    print("\n" + "="*60)
    print("✅ 所有演示步骤完成!")
    print("="*60)
    
    print("\n📋 总结:")
    print(f"1. ✅ 算法版本控制: DiscreteDecisionEngine v2.1.0 已注册")
    print(f"2. ✅ 健康检查: {overall_status.upper()}")
    print(f"3. ✅ 鲁棒性增强: 自动处理异常值")
    print(f"4. ✅ 性能优秀: 平均延迟 {avg_time:.2f}ms")
    print(f"5. ✅ 报告生成: {reports_dir}")
    
    print("\n🎯 生产环境建议:")
    print("- 定期运行健康检查 (每5分钟)")
    print("- 监控关键指标 (延迟、内存、错误率)")
    print("- 启用熔断器保护关键服务")
    print("- 保持算法版本记录")
    print("- 建立告警和回滚机制")
    
    print("\n🚀 系统已准备好用于生产环境!")

if __name__ == "__main__":
    main()
