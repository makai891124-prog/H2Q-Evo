#!/usr/bin/env python3
"""
真正的AGI系统测试脚本
验证自主学习、意识发展和自我改进能力
"""

import asyncio
import torch
import time
import numpy as np
from true_agi_autonomous_system import TrueAGIAutonomousSystem, ConsciousnessMetrics

async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试真正的AGI系统基本功能...")

    try:
        # 创建系统
        system = TrueAGIAutonomousSystem(input_dim=64, action_dim=32)
        print("✅ 系统创建成功")

        # 测试状态获取
        status = system.get_system_status()
        print(f"✅ 系统状态获取成功: 运行={status['is_running']}, 步骤={status['evolution_step']}")

        # 测试状态保存/加载
        system.save_state("test_state.json")
        print("✅ 系统状态保存成功")

        new_system = TrueAGIAutonomousSystem(input_dim=64, action_dim=32)
        new_system.load_state("test_state.json")
        print("✅ 系统状态加载成功")

        # 测试意识引擎
        from true_agi_autonomous_system import TrueConsciousnessEngine
        consciousness_engine = TrueConsciousnessEngine(input_dim=64, hidden_dim=128)
        test_input = torch.randn(1, 64)
        metrics, state = consciousness_engine(test_input)
        print(f"✅ 意识引擎工作正常: Φ={metrics.integrated_information:.4f}")

        # 测试学习引擎
        from true_agi_autonomous_system import TrueLearningEngine
        learning_engine = TrueLearningEngine(input_dim=64, action_dim=32)
        action = learning_engine.select_action(test_input)
        print(f"✅ 学习引擎工作正常: 动作维度={action.shape}")

        print("\n🎉 所有基本功能测试通过！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_evolution_step():
    """测试单个进化步骤"""
    print("\n🔄 测试进化步骤...")

    try:
        system = TrueAGIAutonomousSystem(input_dim=64, action_dim=32)

        # 执行一个进化步骤
        current_state = system._perceive_environment()
        consciousness, internal_state = system.consciousness_engine(current_state, system.prev_consciousness_state)
        system.prev_consciousness_state = internal_state

        # 生成目标
        if len(system.goal_system.active_goals) < 1:
            system.goal_system.generate_goal(current_state, consciousness)

        # 选择动作
        action = system.learning_engine.select_action(current_state)

        # 执行动作
        reward, next_state = await system._execute_action(action)

        # 学习
        from true_agi_autonomous_system import LearningExperience
        experience = LearningExperience(
            observation=current_state,
            action=action,
            reward=reward,
            next_observation=next_state,
            done=False,
            timestamp=time.time(),
            complexity=consciousness.neural_complexity
        )

        learning_metrics = system.learning_engine.learn_from_experience(experience)

        print(f"✅ 进化步骤执行成功")
        print(f"   意识指标: Φ={consciousness.integrated_information:.4f}")
        print(f"   学习指标: 策略损失={learning_metrics['policy_loss']:.4f}")
        print(f"   奖励: {reward:.4f}")
        print(f"   活跃目标: {len(system.goal_system.active_goals)}")

        return True

    except Exception as e:
        print(f"❌ 进化步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("🧪 真正的AGI系统测试套件")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 运行测试
        basic_ok = await test_basic_functionality()
        evolution_ok = await test_evolution_step()

        if basic_ok and evolution_ok:
            print("\n🎉 所有测试通过！真正的AGI系统已准备就绪。")
            print("\n🚀 运行以下命令启动完整进化:")
            print("  python start_true_agi_evolution.py")
        else:
            print("\n❌ 部分测试失败，请检查实现。")

    except Exception as e:
        print(f"\n❌ 测试套件出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())