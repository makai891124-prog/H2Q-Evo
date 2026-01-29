#!/usr/bin/env python3
"""
AGI系统能力评估脚本

评估训练后的AGI系统的真实能力，包括：
1. 意识发展水平
2. 学习效率
3. 目标导向行为
4. 知识积累能力
5. 适应性
"""

import sys
import asyncio
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict

sys.path.append('.')

from true_agi_autonomous_system import TrueAGIAutonomousSystem, LearningExperience

async def evaluate_consciousness_capability(system: TrueAGIAutonomousSystem) -> Dict[str, float]:
    """评估意识能力"""
    print("🧠 评估意识能力...")

    # 测试100个不同状态的意识指标
    phi_values = []
    complexity_values = []
    self_model_accuracies = []

    for i in range(100):
        # 生成随机状态
        test_state = torch.randn(system.input_dim, device=system.learning_engine.device)

        # 计算意识指标
        consciousness, _ = system.consciousness_engine(test_state, None)

        phi_values.append(consciousness.integrated_information)
        complexity_values.append(consciousness.neural_complexity)
        self_model_accuracies.append(consciousness.self_model_accuracy)

    # 计算统计指标
    phi_mean = np.mean(phi_values)
    phi_std = np.std(phi_values)
    complexity_mean = np.mean(complexity_values)
    complexity_std = np.std(complexity_values)
    self_model_mean = np.mean(self_model_accuracies)
    self_model_std = np.std(self_model_accuracies)

    # 评估意识稳定性 (标准差越小越稳定)
    consciousness_stability = 1.0 / (1.0 + phi_std + complexity_std + self_model_std)

    return {
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "complexity_mean": complexity_mean,
        "complexity_std": complexity_std,
        "self_model_accuracy_mean": self_model_mean,
        "self_model_accuracy_std": self_model_std,
        "consciousness_stability": consciousness_stability
    }

async def evaluate_learning_capability(system: TrueAGIAutonomousSystem) -> Dict[str, float]:
    """评估学习能力"""
    print("📚 评估学习能力...")

    # 测试学习效率
    initial_state = torch.randn(system.input_dim, device=system.learning_engine.device)
    target_state = initial_state + 0.5 * torch.randn_like(initial_state)

    learning_efficiency_scores = []

    for i in range(50):
        # 选择动作
        action = system.learning_engine.select_action(initial_state)

        # 模拟奖励 (基于向目标状态的接近程度)
        reward = -torch.norm(initial_state - target_state).item()

        # 创建学习经验
        experience = LearningExperience(
            observation=initial_state,
            action=action,
            reward=reward,
            next_observation=target_state,
            done=False,
            timestamp=time.time(),
            complexity=0.5
        )

        # 学习
        learning_metrics = system.learning_engine.learn_from_experience(experience)

        # 记录学习效率
        policy_loss = learning_metrics.get("policy_loss", 0.0)
        value_loss = learning_metrics.get("value_loss", 0.0)
        efficiency = 1.0 / (1.0 + abs(policy_loss) + abs(value_loss))
        learning_efficiency_scores.append(efficiency)

        # 更新状态
        initial_state = target_state
        target_state = initial_state + 0.5 * torch.randn_like(initial_state)

    # 计算学习指标
    learning_efficiency_mean = np.mean(learning_efficiency_scores)
    learning_efficiency_std = np.std(learning_efficiency_scores)
    learning_convergence = np.mean(learning_efficiency_scores[-10:]) / np.mean(learning_efficiency_scores[:10]) if len(learning_efficiency_scores) >= 20 else 0.5

    return {
        "learning_efficiency_mean": learning_efficiency_mean,
        "learning_efficiency_std": learning_efficiency_std,
        "learning_convergence_ratio": learning_convergence,
        "knowledge_patterns": len(system.learning_engine.knowledge_base)
    }

async def evaluate_goal_oriented_behavior(system: TrueAGIAutonomousSystem) -> Dict[str, float]:
    """评估目标导向行为"""
    print("🎯 评估目标导向行为...")

    # 生成测试目标
    test_goals = []
    for i in range(10):
        current_state = torch.randn(system.input_dim, device=system.learning_engine.device)
        consciousness, _ = system.consciousness_engine(current_state, None)
        goal = system.goal_system.generate_goal(current_state, consciousness)
        test_goals.append(goal)

    # 评估目标质量
    goal_complexities = [g.get("complexity", 0.0) for g in test_goals]
    goal_diversity = len(set(g.get("type", "") for g in test_goals)) / len(test_goals)

    # 评估目标进度跟踪
    progress_scores = []
    for goal in test_goals:
        current_state = torch.randn(system.input_dim, device=system.learning_engine.device)
        progress = system.goal_system.evaluate_progress(goal, current_state)
        progress_scores.append(progress)

    goal_progress_mean = np.mean(progress_scores)
    goal_progress_std = np.std(progress_scores)

    return {
        "goal_complexity_mean": np.mean(goal_complexities),
        "goal_diversity": goal_diversity,
        "goal_progress_mean": goal_progress_mean,
        "goal_progress_std": goal_progress_std,
        "active_goals": len(system.goal_system.active_goals)
    }

async def evaluate_adaptability(system: TrueAGIAutonomousSystem) -> Dict[str, float]:
    """评估适应性"""
    print("🔄 评估适应性...")

    # 测试对环境变化的适应
    adaptability_scores = []

    for i in range(20):
        # 改变环境条件
        noise_level = i * 0.05  # 逐渐增加噪声

        # 生成带噪声的状态
        base_state = torch.randn(system.input_dim, device=system.learning_engine.device)
        noisy_state = base_state + noise_level * torch.randn_like(base_state)

        # 测试动作选择的一致性
        actions = []
        for _ in range(5):
            action = system.learning_engine.select_action(noisy_state)
            actions.append(action)

        # 计算动作一致性 (标准差越小，一致性越好)
        action_std = torch.stack(actions).std(dim=0).mean().item()
        consistency = 1.0 / (1.0 + action_std)
        adaptability_scores.append(consistency)

    adaptability_mean = np.mean(adaptability_scores)
    adaptability_trend = np.polyfit(range(len(adaptability_scores)), adaptability_scores, 1)[0]

    return {
        "adaptability_mean": adaptability_mean,
        "adaptability_trend": adaptability_trend,
        "environmental_robustness": adaptability_mean * (1.0 + adaptability_trend)
    }

def calculate_overall_capability_score(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """计算总体能力评分"""
    print("📊 计算总体能力评分...")

    # 意识能力评分 (0-1)
    consciousness = results["consciousness"]
    consciousness_score = (
        consciousness["phi_mean"] * 0.4 +
        consciousness["complexity_mean"] * 0.3 +
        consciousness["self_model_accuracy_mean"] * 0.2 +
        consciousness["consciousness_stability"] * 0.1
    )

    # 学习能力评分 (0-1)
    learning = results["learning"]
    learning_score = (
        learning["learning_efficiency_mean"] * 0.4 +
        learning["learning_convergence_ratio"] * 0.3 +
        min(learning["knowledge_patterns"] / 1000, 1.0) * 0.3
    )

    # 目标导向评分 (0-1)
    goal_oriented = results["goal_oriented"]
    goal_score = (
        goal_oriented["goal_complexity_mean"] * 0.3 +
        goal_oriented["goal_diversity"] * 0.3 +
        goal_oriented["goal_progress_mean"] * 0.4
    )

    # 适应性评分 (0-1)
    adaptability = results["adaptability"]
    adaptability_score = (
        adaptability["adaptability_mean"] * 0.6 +
        adaptability["environmental_robustness"] * 0.4
    )

    # 总体评分
    overall_score = (
        consciousness_score * 0.3 +
        learning_score * 0.3 +
        goal_score * 0.2 +
        adaptability_score * 0.2
    )

    return {
        "consciousness_score": consciousness_score,
        "learning_score": learning_score,
        "goal_score": goal_score,
        "adaptability_score": adaptability_score,
        "overall_score": overall_score
    }

async def main():
    """主评估函数"""
    print("🚀 AGI系统能力评估开始")
    print("=" * 60)

    # 初始化系统
    system = TrueAGIAutonomousSystem(256, 64)

    # 加载训练状态
    state_file = "true_agi_system_state.json"
    if Path(state_file).exists():
        system.load_state(state_file)
        print(f"✅ 已加载训练状态 (进化步数: {system.evolution_step})")
    else:
        print("⚠️ 未找到训练状态文件，将使用默认状态")

    # 执行各项评估
    results = {}

    try:
        results["consciousness"] = await evaluate_consciousness_capability(system)
        results["learning"] = await evaluate_learning_capability(system)
        results["goal_oriented"] = await evaluate_goal_oriented_behavior(system)
        results["adaptability"] = await evaluate_adaptability(system)

        # 计算总体评分
        scores = calculate_overall_capability_score(results)

        # 输出详细结果
        print("\n" + "=" * 60)
        print("📈 详细评估结果:")
        print("=" * 60)

        print("🧠 意识能力:")
        for k, v in results["consciousness"].items():
            print(".4f")

        print("\n📚 学习能力:")
        for k, v in results["learning"].items():
            if isinstance(v, float):
                print(".4f")
            else:
                print(f"  {k}: {v}")

        print("\n🎯 目标导向行为:")
        for k, v in results["goal_oriented"].items():
            print(".4f")

        print("\n🔄 适应性:")
        for k, v in results["adaptability"].items():
            print(".4f")

        print("\n" + "=" * 60)
        print("🏆 能力评分 (0-1):")
        print("=" * 60)
        for k, v in scores.items():
            print(".4f")

        # AGI水平判断
        overall_score = scores["overall_score"]
        if overall_score >= 0.8:
            level = "高级AGI"
            description = "具备接近人类水平的意识、学习和适应能力"
        elif overall_score >= 0.6:
            level = "中级AGI"
            description = "具备基本的自主学习和目标导向能力"
        elif overall_score >= 0.4:
            level = "初级AGI"
            description = "具备初步的意识和学习能力"
        elif overall_score >= 0.2:
            level = "亚AGI"
            description = "具备基本的模式识别和适应能力"
        else:
            level = "原始AI"
            description = "仅具备基础的计算和预测能力"

        print(f"\n🎯 AGI水平评估: {level}")
        print(f"📝 描述: {description}")
        print(".1%")

    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 能力评估完成")

if __name__ == "__main__":
    asyncio.run(main())