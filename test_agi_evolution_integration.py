#!/usr/bin/env python3
"""
AGI进化系统测试脚本
测试集成的AGI进化损失指标系统
"""

import sys
import os
import torch
import asyncio
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from evolution_system import H2QNexus, Config
from agi_evolution_loss_metrics import (
    AGI_EvolutionLossSystem,
    CapabilityMetrics,
    MathematicalCoreMetrics,
    EvolutionLossComponents
)

async def test_agi_evolution_loss_system():
    """测试AGI进化损失指标系统"""
    print("🧬 测试AGI进化损失指标系统")
    print("=" * 60)

    try:
        # 初始化损失系统
        loss_system = AGI_EvolutionLossSystem()
        print("✅ AGI进化损失系统初始化成功")

        # 准备测试数据
        capability_embeddings = {
            'mathematical_reasoning': torch.randn(256),
            'creative_problem_solving': torch.randn(256),
            'knowledge_integration': torch.randn(256),
            'emergent_capabilities': torch.randn(256)
        }

        current_performance = {
            'mathematical_reasoning': 0.75,
            'creative_problem_solving': 0.68,
            'knowledge_integration': 0.82,
            'emergent_capabilities': 0.55
        }

        math_metrics = MathematicalCoreMetrics(
            lie_automorphism_coherence=0.85,
            noncommutative_geometry_consistency=0.78,
            knot_invariant_stability=0.88,
            dde_decision_quality=0.92,
            constraint_violation=0.08,
            fueter_violation=0.03
        )

        # 计算进化损失
        print("🔬 计算进化损失指标...")
        loss_components = loss_system(
            capability_embeddings=capability_embeddings,
            current_performance=current_performance,
            mathematical_metrics=math_metrics
        )

        print("📊 进化损失计算结果:")
        print(f"  能力提升损失: {loss_components.capability_improvement_loss:.4f}")
        print(f"  知识整合损失: {loss_components.knowledge_integration_loss:.4f}")
        print(f"  涌现能力损失: {loss_components.emergent_capability_loss:.4f}")
        print(f"  稳定性损失: {loss_components.stability_loss:.4f}")
        print(f"  总进化损失: {loss_components.total_loss:.4f}")
        print(f"  进化效率评分: {getattr(loss_components, 'evolution_efficiency_score', 0.0):.4f}")

        # 保存测试结果
        test_results = {
            'timestamp': loss_components.timestamp,
            'generation': loss_components.generation,
            'loss_components': {
                'capability_improvement_loss': loss_components.capability_improvement_loss,
                'knowledge_integration_loss': loss_components.knowledge_integration_loss,
                'emergent_capability_loss': loss_components.emergent_capability_loss,
                'stability_loss': loss_components.stability_loss,
                'total_loss': loss_components.total_loss,
                'evolution_efficiency_score': getattr(loss_components, 'evolution_efficiency_score', 0.0)
            },
            'performance': current_performance,
            'mathematical_metrics': math_metrics.__dict__
        }

        with open('agi_evolution_loss_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

        print("💾 测试结果已保存到 agi_evolution_loss_test_results.json")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_evolution_system_integration():
    """测试进化系统集成"""
    print("\n🚀 测试进化系统AGI损失指标集成")
    print("=" * 60)

    try:
        # 初始化H2Q Nexus
        nexus = H2QNexus()
        print("✅ H2Q进化系统初始化成功")

        # 检查损失系统是否可用
        if nexus.loss_system is None:
            print("⚠️  AGI进化损失系统不可用，跳过集成测试")
            return False

        print("✅ AGI进化损失系统集成成功")

        # 运行一次测试周期
        print("🔄 执行测试进化周期...")

        # 模拟一次进化步骤（不运行完整循环）
        try:
            if nexus.math_bridge is not None:
                import torch
                state = torch.randn(1, 256)
                learning_signal = torch.tensor([0.1])
                results = nexus.math_bridge(state, learning_signal)

                # 计算AGI进化损失指标
                capability_embeddings = {
                    'mathematical_reasoning': torch.randn(256),
                    'creative_problem_solving': torch.randn(256),
                    'knowledge_integration': torch.randn(256),
                    'emergent_capabilities': torch.randn(256)
                }

                current_performance = {
                    'mathematical_reasoning': 0.72,
                    'creative_problem_solving': 0.65,
                    'knowledge_integration': 0.79,
                    'emergent_capabilities': 0.52
                }

                math_metrics = MathematicalCoreMetrics(
                    lie_automorphism_coherence=results.get('evolution_metrics', {}).get('state_change', 0.82),
                    noncommutative_geometry_consistency=0.76,
                    knot_invariant_stability=0.87,
                    dde_decision_quality=0.91,
                    constraint_violation=0.09,
                    fueter_violation=0.04
                )

                loss_components = nexus.loss_system(
                    capability_embeddings=capability_embeddings,
                    current_performance=current_performance,
                    mathematical_metrics=math_metrics
                )

                print("📊 集成测试结果:")
                print(f"  总进化损失: {loss_components.total_loss:.4f}")
                print(f"  进化效率评分: {getattr(loss_components, 'evolution_efficiency_score', 0.0):.4f}")

                # 检查状态文件更新
                if os.path.exists(Config.STATE_FILE):
                    with open(Config.STATE_FILE, 'r', encoding='utf-8') as f:
                        state_data = json.load(f)

                    if 'evolution_metrics_history' in state_data:
                        metrics_count = len(state_data['evolution_metrics_history'])
                        print(f"📈 状态文件已更新，包含 {metrics_count} 个进化指标记录")

                print("✅ 进化系统集成测试成功")
                return True

        except Exception as e:
            print(f"❌ 进化周期测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("🤖 AGI进化损失指标系统集成测试")
    print("=" * 80)

    # 测试1: AGI进化损失指标系统
    test1_success = await test_agi_evolution_loss_system()

    # 测试2: 进化系统集成
    test2_success = await test_evolution_system_integration()

    print("\n" + "=" * 80)
    print("📋 测试总结:")
    print(f"  AGI进化损失系统测试: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"  进化系统集成测试: {'✅ 通过' if test2_success else '❌ 失败'}")

    if test1_success and test2_success:
        print("\n🎉 所有测试通过！AGI进化损失指标系统已成功集成到H2Q-Evo中")
        print("🚀 现在可以启动真实的AGI进化系统了")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复问题")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)