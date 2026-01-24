#!/usr/bin/env python3
"""
完整的AGI能力测试与监督学习验证
包括:
- LLM标准基准测试
- 轨迹控制与流形稳定性分析
- 交叉验证
- 错误修正
- 自动测试发现
"""

import sys
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

import numpy as np


def test_supervised_learning():
    """测试监督学习系统."""
    print("=" * 70)
    print("🎓 监督学习系统测试")
    print("=" * 70)
    
    from h2q_project.h2q.agi.supervised_learning import (
        SupervisedLearningMonitor,
        TrajectoryController,
        LeanVerifier,
        CrossValidator,
        ErrorCorrector,
        AutoTestDiscovery
    )
    
    # 1. 测试轨迹控制器
    print("\n📊 1. 轨迹控制与流形稳定性分析")
    print("-" * 50)
    
    controller = TrajectoryController()
    
    # 模拟学习过程
    for epoch in range(20):
        loss = 1.0 / (1 + epoch * 0.1) + np.random.uniform(-0.05, 0.05)
        accuracy = 1 - loss + np.random.uniform(-0.02, 0.02)
        gradient_norm = np.random.uniform(0.5, 2.0) if epoch < 15 else np.random.uniform(0.01, 0.1)
        
        point = controller.record_point(
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            gradient_norm=gradient_norm,
            learning_rate=0.001
        )
    
    status = controller.get_status_report()
    print(f"  总epochs: {status['total_epochs']}")
    print(f"  当前损失: {status['current_loss']:.4f}")
    print(f"  流形稳定性: {status['stability_index']:.3f}")
    print(f"  流形曲率: {status['manifold_curvature']:.4f}")
    print(f"  损失趋势: {status['loss_trend']}")
    print(f"  检测到异常: {status['anomaly_count']}个")
    print(f"  建议学习率: {status['suggested_lr']:.6f}")
    
    # 2. 测试交叉验证
    print("\n🔄 2. 多源交叉验证")
    print("-" * 50)
    
    validator = CrossValidator()
    
    test_questions = [
        ("2 + 3 * 4 = ?", 14, "math"),
        ("秦始皇统一六国是哪年?", "公元前221年", "chinese"),
        ("What is 15 - 6?", 9, "arithmetic")
    ]
    
    for q, ans, cat in test_questions:
        results = validator.cross_validate(q, ans, cat)
        is_valid, confidence = validator.compute_consensus(results)
        print(f"  问题: {q[:30]}...")
        print(f"  验证结果: {'✅ 有效' if is_valid else '❌ 无效'}, 置信度: {confidence:.2f}")
    
    # 3. 测试错误修正
    print("\n🔧 3. 错误修正系统")
    print("-" * 50)
    
    corrector = ErrorCorrector()
    
    error_cases = [
        ("2 + 3 * 4 = ?", 11, 14, "math"),  # 运算顺序错误
        ("What is 100 - 25?", 65, 75, "arithmetic"),  # 计算错误
    ]
    
    for q, wrong, correct, cat in error_cases:
        analysis = corrector.analyze_and_correct(q, wrong, correct, cat)
        print(f"  问题: {q}")
        print(f"  错误答案: {wrong}, 正确答案: {correct}")
        print(f"  错误类型: {analysis['error_type']}")
        print(f"  修正策略: {analysis['correction_strategy']['type']}")
        print(f"  建议: {analysis['correction_strategy'].get('practice_recommendation', 'N/A')}")
        print()
    
    # 4. 测试Lean验证器
    print("🔬 4. 形式化验证 (Lean4)")
    print("-" * 50)
    
    verifier = LeanVerifier()
    print(f"  Lean4可用: {'是' if verifier.lean_available else '否 (使用Python回退)'}")
    
    # 测试算术验证
    result = verifier.verify_arithmetic("2 + 3 * 4", 14)
    print(f"  验证 2 + 3 * 4 = 14: {'✅' if result.is_valid else '❌'}")
    print(f"  验证方法: {result.method.value}")
    print(f"  置信度: {result.confidence}")
    
    # 5. 测试自动测试发现
    print("\n🔍 5. 自动测试发现")
    print("-" * 50)
    
    discovery = AutoTestDiscovery()
    
    # 模拟当前能力
    current_caps = {
        "math": 100.0,
        "logic": 100.0,
        "pattern": 100.0,
        "memory": 85.0
    }
    
    new_tests = discovery.discover_new_tests(current_caps)
    print(f"  发现 {len(new_tests)} 个新测试:")
    for test in new_tests[:5]:
        source = test.get('source', 'unknown')
        name = test.get('name', test.get('dataset', test.get('repo', 'Unknown')))
        area = test.get('area', 'general')
        difficulty = test.get('difficulty', 'standard')
        print(f"    [{source}] {name}: {area} ({difficulty})")
    
    return True


def test_llm_benchmarks():
    """测试LLM标准基准测试."""
    print("\n" + "=" * 70)
    print("🎯 LLM标准基准测试")
    print("=" * 70)
    
    from h2q_project.h2q.agi.llm_benchmarks import LLMBenchmarkSuite, BenchmarkType
    
    suite = LLMBenchmarkSuite()
    
    print("\n可用基准测试:")
    for bt in BenchmarkType:
        info = suite.get_benchmark_info(bt)
        if info['total_questions'] > 0:
            print(f"  • {bt.value.upper()}: {info['total_questions']}题")
    
    # 运行所有基准
    print("\n📊 运行基准测试:")
    print("-" * 50)
    
    results = suite.run_all_benchmarks(questions_per_benchmark=6)
    
    for name, data in results["benchmarks"].items():
        print(f"\n  {name.upper()}:")
        print(f"    准确率: {data['accuracy']:.1f}%")
        print(f"    正确数: {data['correct']}/{data['total']}")
    
    print("\n" + "=" * 50)
    print(f"📈 综合得分: {results['overall_score']:.1f}%")
    print(f"📋 等级: {results['grade']}")
    print("=" * 50)
    
    return results


def test_full_evaluation():
    """测试完整评估."""
    print("\n" + "=" * 70)
    print("🧪 完整能力评估 (基础 + LLM基准)")
    print("=" * 70)
    
    from h2q_project.h2q.agi.evolution_24h import CapabilityTester
    
    tester = CapabilityTester()
    results = tester.run_full_evaluation()
    
    return results


def main():
    print("=" * 70)
    print("🚀 H2Q-Evo AGI 能力测试与监督学习验证")
    print("=" * 70)
    
    # 1. 监督学习系统测试
    test_supervised_learning()
    
    # 2. LLM基准测试
    llm_results = test_llm_benchmarks()
    
    # 3. 完整评估
    full_results = test_full_evaluation()
    
    # 保存结果
    import json
    results_summary = {
        "timestamp": str(np.datetime64('now')),
        "llm_benchmark_score": llm_results['overall_score'],
        "full_evaluation_score": full_results['combined_score'],
        "grade": full_results['grade']
    }
    
    with open('/Users/imymm/H2Q-Evo/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成!")
    print(f"📁 结果已保存至: test_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
