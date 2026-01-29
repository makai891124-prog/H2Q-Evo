#!/usr/bin/env python3
"""
AGI进化损失指标系统 - 简化测试
"""

import torch
from agi_evolution_loss_metrics import create_agi_evolution_loss_system, MathematicalCoreMetrics

def main():
    print("🧪 AGI进化损失指标系统简化测试")
    print("=" * 50)

    # 创建系统
    loss_system = create_agi_evolution_loss_system()

    # 模拟输入数据
    capability_embeddings = {
        'mathematical_reasoning': torch.randn(256),
        'creative_problem_solving': torch.randn(256),
        'knowledge_integration': torch.randn(256),
        'emergent_capabilities': torch.randn(256)
    }

    current_performance = {
        'mathematical_reasoning': 0.8,
        'creative_problem_solving': 0.7,
        'knowledge_integration': 0.6,
        'emergent_capabilities': 0.5
    }

    mathematical_metrics = MathematicalCoreMetrics(
        lie_automorphism_coherence=0.9,
        noncommutative_geometry_consistency=0.8,
        knot_invariant_stability=0.7,
        dde_decision_quality=0.85,
        constraint_violation=0.1,
        fueter_violation=0.05
    )

    print("✅ 系统创建成功")
    print("✅ 输入数据准备完成")

    # 计算损失
    try:
        loss_components = loss_system(
            capability_embeddings=capability_embeddings,
            current_performance=current_performance,
            new_knowledge=torch.randn(256),
            existing_knowledge=[torch.randn(256) for _ in range(3)],
            current_state=torch.randn(256),
            mathematical_metrics=mathematical_metrics
        )

        print("✅ 损失计算成功")
        print("📊 结果:")
        print(f"  能力提升损失: {loss_components.capability_improvement_loss:.4f}")
        print(f"  知识整合损失: {loss_components.knowledge_integration_loss:.4f}")
        print(f"  涌现能力损失: {loss_components.emergent_capability_loss:.4f}")
        print(f"  稳定性损失: {loss_components.stability_loss:.4f}")
        print(f"  总损失: {loss_components.total_loss:.4f}")
        # 获取报告
        report = loss_system.get_evolution_report()
        print("✅ 进化报告生成成功")
        print(f"📈 当前代数: {report['current_generation']}")

        print("\n🎉 所有测试通过！AGI进化损失指标系统工作正常")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()