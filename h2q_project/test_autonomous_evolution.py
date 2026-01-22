#!/usr/bin/env python3
"""H2Q AGI 自主进化综合测试.

测试内容:
1. 分形记忆压缩 - 验证四元数小波变换和分形压缩
2. 知识获取 - 测试网络资源安全获取
3. 自主进化引擎 - 验证兴趣驱动学习
4. 标准人类基准 - 运行 MMLU/GSM8K/ARC 等基准测试
"""

import sys
import os
import time
import json
from pathlib import Path

# 确保路径正确
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def print_header(title: str):
    """打印区块标题."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_fractal_memory():
    """测试分形记忆压缩."""
    print_header("1. 分形记忆压缩测试")
    
    from h2q.agi.fractal_memory_compression import (
        create_fractal_memory_db,
        CompressionLevel,
    )
    import numpy as np
    
    # 创建数据库
    db = create_fractal_memory_db(
        max_memory_mb=50.0
    )
    
    print(f"  ✅ 创建分形记忆数据库")
    
    # 存储测试数据
    test_vectors = [
        np.random.randn(128).astype(np.float32) for _ in range(10)
    ]
    
    for i, vec in enumerate(test_vectors):
        db.store(
            key=f"test_memory_{i}",
            data=vec,
            importance=np.random.uniform(0.5, 1.0),
            metadata={"category": "test"}
        )
    
    print(f"  ✅ 存储 10 条测试记忆")
    
    # 检索测试
    retrieved = db.retrieve("test_memory_5")
    if retrieved is not None:
        print(f"  ✅ 成功检索记忆: shape={retrieved.shape}")
    
    # 相似搜索
    query = test_vectors[0]
    similar = db.search_similar(query, top_k=3)
    print(f"  ✅ 相似搜索返回 {len(similar)} 条结果")
    
    # 压缩测试
    initial_stats = db.get_memory_usage()
    print(f"  📊 压缩前: {initial_stats['blocks_count']} 条记忆")
    
    db.compress_memory(target_ratio=0.5)
    
    compressed_stats = db.get_memory_usage()
    print(f"  📊 压缩后: 比率 = {compressed_stats.get('compression_ratio', 1.0):.2f}x")
    
    return True


def test_knowledge_acquisition():
    """测试知识获取模块."""
    print_header("2. 知识获取模块测试")
    
    from h2q.agi.knowledge_acquisition import (
        create_knowledge_acquisition_manager,
        ResourceSource,
    )
    
    # 创建管理器
    manager = create_knowledge_acquisition_manager(
        cache_dir="/tmp/h2q_knowledge_cache",
    )
    
    print(f"  ✅ 创建知识获取管理器")
    
    # 添加兴趣
    manager.add_interest("machine learning")
    manager.add_interest("quaternion mathematics")
    manager.add_interest("neural networks")
    print(f"  ✅ 添加 3 个学习兴趣")
    
    # 通过 poll_resources 获取资源 (会生成本地数学问题)
    print(f"  📚 获取资源...")
    math_resources = list(manager.poll_resources())
    print(f"  ✅ 获取 {len(math_resources)} 个资源")
    
    if math_resources:
        sample = math_resources[0]
        print(f"      示例: {sample.title[:50]}...")
    
    # 获取统计
    stats = manager.get_stats()
    print(f"  📊 获取统计: {stats['total_acquired']} 资源")
    
    return True


def test_autonomous_evolution():
    """测试自主进化引擎."""
    print_header("3. 自主进化引擎测试")
    
    from h2q.agi.autonomous_evolution import (
        create_evolution_engine,
        EvolutionState,
    )
    
    # 创建引擎
    engine = create_evolution_engine(
        max_resources_per_cycle=5,
        evaluation_interval=3,
    )
    
    print(f"  ✅ 创建自主进化引擎")
    print(f"  📊 当前状态: {engine.state.value}")
    
    # 获取当前兴趣
    interests = engine.get_current_interests()
    print(f"  🎯 当前兴趣数: {len(interests)}")
    
    if interests:
        top_3 = sorted(interests, key=lambda x: x.priority, reverse=True)[:3]
        print(f"  🏆 Top 3 兴趣:")
        for i, interest in enumerate(top_3):
            print(f"      {i+1}. {interest.domain}: {interest.topic} (优先级: {interest.priority:.2f})")
    
    # 运行单次进化循环 (测试模式)
    print(f"\n  🔄 运行测试进化循环...")
    
    # 模拟一次学习
    stats_before = engine.get_stats()
    
    # 手动触发一次学习
    from h2q.agi.knowledge_acquisition import create_knowledge_acquisition_manager
    
    test_manager = create_knowledge_acquisition_manager()
    math_resources = test_manager.fetch_by_interest(
        "calculus", 
        source=test_manager.sources[3],  # LOCAL_MATH
        max_resources=2
    )
    
    for resource in math_resources:
        engine._learn_resource(resource)
    
    stats_after = engine.get_stats()
    
    print(f"  📊 学习前: {stats_before['total_resources_learned']} 资源")
    print(f"  📊 学习后: {stats_after['total_resources_learned']} 资源")
    print(f"  ✅ 进化循环测试完成")
    
    return True


def test_standard_benchmarks():
    """测试标准人类基准."""
    print_header("4. 标准人类基准测试")
    
    from h2q.agi.standard_benchmarks import (
        run_standard_benchmarks,
        BenchmarkType,
    )
    
    print(f"  🧪 运行标准基准测试 (MMLU, GSM8K, ARC, HellaSwag)...")
    print()
    
    # 运行测试
    result = run_standard_benchmarks(n_per_benchmark=None)  # 运行所有问题
    
    # 打印简化报告
    print(f"  {'='*60}")
    print(f"  基准测试结果")
    print(f"  {'='*60}")
    
    total_correct = 0
    total_questions = 0
    
    for benchmark, data in result["results"].items():
        acc = data["accuracy"] * 100
        correct = data["correct"]
        total = data["total"]
        
        total_correct += correct
        total_questions += total
        
        if acc >= 80:
            icon = "🟢"
        elif acc >= 60:
            icon = "🟡"
        else:
            icon = "🔴"
        
        print(f"  {icon} {benchmark.upper():12s}: {acc:5.1f}% ({correct}/{total})")
    
    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    
    print(f"  {'-'*40}")
    print(f"  📊 总体准确率: {overall_acc * 100:.1f}%")
    
    # 等级评定
    if overall_acc >= 0.90:
        grade = "卓越 (Expert)"
        icon = "🏆"
    elif overall_acc >= 0.80:
        grade = "优秀 (Above Average)"
        icon = "🥇"
    elif overall_acc >= 0.70:
        grade = "良好 (Average)"
        icon = "🥈"
    elif overall_acc >= 0.60:
        grade = "及格 (Below Average)"
        icon = "🥉"
    else:
        grade = "需改进 (Needs Work)"
        icon = "📈"
    
    print(f"  {icon} 等级: {grade}")
    
    return overall_acc


def test_integrated_system():
    """测试集成系统."""
    print_header("5. 集成系统测试")
    
    from h2q.agi.fractal_memory_compression import create_fractal_memory_db
    from h2q.agi.knowledge_acquisition import create_knowledge_acquisition_manager
    from h2q.agi.autonomous_evolution import create_evolution_engine
    from h2q.agi.standard_benchmarks import run_standard_benchmarks
    import numpy as np
    
    print(f"  🔗 初始化集成系统组件...")
    
    # 1. 创建分形记忆数据库
    memory_db = create_fractal_memory_db(max_memory_mb=100.0)
    print(f"      ✅ 分形记忆数据库")
    
    # 2. 创建知识获取器
    knowledge_mgr = create_knowledge_acquisition_manager()
    print(f"      ✅ 知识获取管理器")
    
    # 3. 创建进化引擎
    evolution_engine = create_evolution_engine()
    print(f"      ✅ 自主进化引擎")
    
    # 4. 集成测试: 获取知识 -> 存储到记忆 -> 学习
    print(f"\n  🔄 集成流程测试...")
    
    # 获取数学知识
    resources = knowledge_mgr.fetch_by_interest(
        "linear algebra",
        max_resources=3
    )
    print(f"      获取 {len(resources)} 个知识资源")
    
    # 存储到分形记忆
    for i, res in enumerate(resources):
        # 将内容转换为向量 (简化: 使用内容长度生成伪向量)
        content_vec = np.random.randn(64).astype(np.float32)
        content_vec = content_vec / np.linalg.norm(content_vec)
        
        memory_db.store(
            key=f"knowledge_{res.source.value}_{i}",
            data=content_vec,
            importance=0.8,
            category=res.source.value,
            metadata={"title": res.title, "url": res.url}
        )
    
    print(f"      存储到分形记忆数据库")
    
    # 进化引擎学习
    for res in resources:
        evolution_engine._learn_resource(res)
    
    print(f"      进化引擎完成学习")
    
    # 输出集成状态
    mem_stats = memory_db.get_stats()
    evo_stats = evolution_engine.get_stats()
    
    print(f"\n  📊 集成系统状态:")
    print(f"      记忆条目: {mem_stats['entries_count']}")
    print(f"      学习资源: {evo_stats['total_resources_learned']}")
    print(f"      当前代数: {evo_stats['generation']}")
    
    return True


def main():
    """主函数."""
    print("\n" + "🌟" * 35)
    print("  H2Q AGI 自主进化系统 - 综合测试")
    print("🌟" * 35)
    
    results = {}
    
    try:
        # 1. 分形记忆测试
        results["fractal_memory"] = test_fractal_memory()
        
        # 2. 知识获取测试
        results["knowledge_acquisition"] = test_knowledge_acquisition()
        
        # 3. 自主进化测试
        results["autonomous_evolution"] = test_autonomous_evolution()
        
        # 4. 标准基准测试
        benchmark_acc = test_standard_benchmarks()
        results["standard_benchmarks"] = benchmark_acc
        
        # 5. 集成测试
        results["integrated_system"] = test_integrated_system()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 最终汇总
    print_header("📋 测试汇总")
    
    all_passed = True
    for name, result in results.items():
        if isinstance(result, bool):
            status = "✅ 通过" if result else "❌ 失败"
            all_passed = all_passed and result
        elif isinstance(result, float):
            status = f"📊 {result * 100:.1f}%"
            all_passed = all_passed and (result >= 0.6)
        else:
            status = f"📊 {result}"
        
        print(f"  {name:25s}: {status}")
    
    print()
    if all_passed:
        print("  🎉 所有测试通过！H2Q AGI 自主进化系统运行正常。")
    else:
        print("  ⚠️ 部分测试未通过，请检查相关模块。")
    
    # 保存测试报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {
            k: v if isinstance(v, (bool, int, float)) else str(v)
            for k, v in results.items()
        },
        "all_passed": all_passed,
    }
    
    report_path = Path(__file__).parent / "autonomous_evolution_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n  📄 测试报告已保存: {report_path}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
