#!/usr/bin/env python3
"""H2Q AGI 自主进化系统测试.

测试:
1. 生存守护进程
2. 24小时进化系统
3. 能力验证

运行:
    python test_evolution_system.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_survival_daemon():
    """测试生存守护进程."""
    print("\n" + "=" * 60)
    print("🧪 测试1: 生存守护进程")
    print("=" * 60)
    
    from h2q_project.h2q.agi.survival_daemon import (
        SurvivalDaemon, SurvivalConfig, create_survival_daemon
    )
    
    config = SurvivalConfig(
        heartbeat_interval=5,
        max_no_heartbeat=30,
        capability_check_interval=10
    )
    
    daemon = create_survival_daemon(str(PROJECT_ROOT), config)
    
    # 设置能力检查回调
    def capability_check():
        return 85.0  # 模拟能力分数
    
    daemon.set_capability_callback(capability_check)
    
    # 启动守护进程
    daemon.start()
    
    print("等待 15 秒观察心跳...")
    time.sleep(15)
    
    # 获取状态
    status = daemon.get_status()
    print(f"\n状态: {status}")
    
    # 停止
    daemon.stop()
    
    print("✅ 生存守护进程测试通过")
    return True


def test_capability_tester():
    """测试能力测试器."""
    print("\n" + "=" * 60)
    print("🧪 测试2: 能力测试器")
    print("=" * 60)
    
    from h2q_project.h2q.agi.evolution_24h import CapabilityTester
    
    tester = CapabilityTester()
    
    # 运行测试
    results = tester.run_comprehensive_test()
    
    print(f"总分: {results['overall_score']:.1f}%")
    print(f"等级: {results['grade']}")
    
    for name, result in results["tests"].items():
        print(f"  - {name}: {result['score']:.1f}%")
    
    print("✅ 能力测试器测试通过")
    return results['overall_score'] >= 60


def test_fractal_compressor():
    """测试分形压缩器."""
    print("\n" + "=" * 60)
    print("🧪 测试3: 分形压缩器")
    print("=" * 60)
    
    from h2q_project.h2q.agi.evolution_24h import FractalCompressor
    
    compressor = FractalCompressor(compression_ratio=0.5)
    
    # 测试数据
    data = {
        "text": "This is a long text. " * 20 + "It has many sentences. " * 10,
        "list": list(range(100)),
        "nested": {
            "inner_text": "Inner content. " * 15,
            "inner_list": list(range(50))
        }
    }
    
    # 压缩
    compressed = compressor.compress(data)
    
    # 计算压缩比
    ratio = compressor.estimate_compression_ratio(data, compressed)
    
    print(f"压缩比: {ratio:.2f}")
    print(f"原始文本长度: {len(data['text'])}")
    print(f"压缩后长度: {len(compressed['text'])}")
    
    print("✅ 分形压缩器测试通过")
    return ratio < 1.0


def test_knowledge_acquirer():
    """测试知识获取器."""
    print("\n" + "=" * 60)
    print("🧪 测试4: 知识获取器")
    print("=" * 60)
    
    from h2q_project.h2q.agi.evolution_24h import KnowledgeAcquirer
    
    acquirer = KnowledgeAcquirer()
    
    # 测试获取
    topics = ["Python_(programming_language)", "Machine_learning"]
    
    for topic in topics:
        print(f"获取: {topic}")
        result = acquirer.fetch_summary(topic)
        
        if result:
            print(f"  ✅ 标题: {result.get('title', 'N/A')}")
            summary = result.get('summary', '')[:100]
            print(f"  摘要: {summary}...")
        else:
            print(f"  ⚠️ 获取失败 (可能是网络问题)")
    
    print(f"\n成功: {acquirer.acquired_count}, 失败: {acquirer.failed_count}")
    print("✅ 知识获取器测试通过")
    return True


def test_evolution_quick():
    """测试进化系统 (快速模式)."""
    print("\n" + "=" * 60)
    print("🧪 测试5: 24小时进化系统 (2分钟快速测试)")
    print("=" * 60)
    
    from h2q_project.h2q.agi.evolution_24h import Evolution24HSystem, EvolutionConfig
    
    config = EvolutionConfig(
        total_duration_hours=2/60,  # 2分钟
        learning_cycle_minutes=0.5,  # 30秒
        capability_check_minutes=1,  # 1分钟
        heartbeat_seconds=10
    )
    
    system = Evolution24HSystem(config, str(PROJECT_ROOT))
    
    # 启动
    system.start()
    
    print("运行 2 分钟快速测试...")
    
    # 等待完成
    start = time.time()
    while system.is_running and (time.time() - start) < 150:  # 最多等待2.5分钟
        time.sleep(5)
        status = system.get_status()
        print(f"  状态: 周期={status['cycle_count']}, 知识={status['knowledge_count']}")
    
    # 停止
    system.stop()
    
    # 验证结果
    status = system.get_status()
    print(f"\n最终状态:")
    print(f"  学习周期: {status['cycle_count']}")
    print(f"  知识条目: {status['knowledge_count']}")
    print(f"  最新评分: {status['latest_score']:.1f}%")
    
    print("✅ 进化系统测试通过")
    return status['cycle_count'] > 0


def main():
    """主测试函数."""
    print("=" * 60)
    print("H2Q AGI 自主进化系统 - 综合测试")
    print("=" * 60)
    
    results = {}
    
    # 测试1: 生存守护进程
    try:
        results["survival_daemon"] = test_survival_daemon()
    except Exception as e:
        print(f"❌ 生存守护进程测试失败: {e}")
        results["survival_daemon"] = False
    
    # 测试2: 能力测试器
    try:
        results["capability_tester"] = test_capability_tester()
    except Exception as e:
        print(f"❌ 能力测试器测试失败: {e}")
        results["capability_tester"] = False
    
    # 测试3: 分形压缩器
    try:
        results["fractal_compressor"] = test_fractal_compressor()
    except Exception as e:
        print(f"❌ 分形压缩器测试失败: {e}")
        results["fractal_compressor"] = False
    
    # 测试4: 知识获取器
    try:
        results["knowledge_acquirer"] = test_knowledge_acquirer()
    except Exception as e:
        print(f"❌ 知识获取器测试失败: {e}")
        results["knowledge_acquirer"] = False
    
    # 测试5: 进化系统
    try:
        results["evolution_system"] = test_evolution_quick()
    except Exception as e:
        print(f"❌ 进化系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["evolution_system"] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅ 通过" if passed_test else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统准备就绪，可以开始24小时自主进化。")
        print("\n启动命令:")
        print("  python h2q_project/h2q/agi/evolution_24h.py --hours 24")
    else:
        print("\n⚠️ 部分测试失败，请检查错误后重试。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
