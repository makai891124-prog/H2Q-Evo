#!/usr/bin/env python3
"""
H2Q-Evo AGI进化系统启动器
启动真实的AGI进化系统，包含完整的进化损失指标计算
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from evolution_system import H2QNexus, Config

def print_banner():
    """打印启动横幅"""
    print("🚀 H2Q-Evo AGI进化系统启动器")
    print("=" * 80)
    print("🤖 真实的AGI进化系统 - 基于数学架构和进化损失指标")
    print("🧬 包含四个核心损失指标：")
    print("   • 能力提升损失 - 量化各能力维度的改进程度")
    print("   • 知识整合损失 - 衡量新知识与现有知识的整合效率")
    print("   • 涌现能力损失 - 检测新能力的涌现和巩固程度")
    print("   • 稳定性损失 - 确保进化过程的稳定性和一致性")
    print("=" * 80)

def print_system_status(nexus):
    """打印系统状态"""
    print("\n📊 系统状态检查:")

    # 检查数学架构集成
    math_status = "✅ 已集成" if nexus.math_bridge is not None else "❌ 未集成"
    print(f"  数学架构集成: {math_status}")

    # 检查AGI进化损失系统
    loss_status = "✅ 已集成" if nexus.loss_system is not None else "❌ 未集成"
    print(f"  AGI进化损失系统: {loss_status}")

    # 检查Docker
    docker_status = "✅ 可用" if nexus.docker_available else "❌ 不可用"
    print(f"  Docker环境: {docker_status}")

    # 检查推理模式
    mode = "LOCAL (Docker)" if Config.INFERENCE_MODE == 'local' else "API (Gemini)"
    print(f"  推理模式: {mode}")

    # 检查API密钥
    api_status = "✅ 已配置" if Config.API_KEY else "❌ 未配置"
    print(f"  Gemini API密钥: {api_status}")

    print()

def setup_environment():
    """设置环境变量"""
    # 设置默认的环境变量
    if not os.getenv("PROJECT_ROOT"):
        os.environ["PROJECT_ROOT"] = str(Path.cwd() / "h2q_project")

    if not os.getenv("INFERENCE_MODE"):
        # 优先使用本地模式，如果Docker可用的话
        os.environ["INFERENCE_MODE"] = "local"

    if not os.getenv("LOG_LEVEL"):
        os.environ["LOG_LEVEL"] = "INFO"

async def start_evolution_system(continuous=True):
    """启动AGI进化系统"""
    print_banner()

    try:
        # 设置环境
        setup_environment()

        # 初始化系统
        print("🔧 初始化H2Q-Evo系统...")
        nexus = H2QNexus()

        # 打印系统状态
        print_system_status(nexus)

        # 检查必要组件
        if nexus.math_bridge is None:
            print("⚠️  警告: 数学架构未集成，进化功能将受限")
        else:
            print("✅ 数学架构已集成，AGI进化功能完整")

        if nexus.loss_system is None:
            print("⚠️  警告: AGI进化损失系统未集成，损失指标计算将不可用")
        else:
            print("✅ AGI进化损失系统已集成，损失指标计算可用")

        if not nexus.docker_available and Config.INFERENCE_MODE == 'local':
            print("⚠️  Docker不可用，将自动切换到API模式")
            os.environ["INFERENCE_MODE"] = "api"
            Config.INFERENCE_MODE = "api"

        # 启动系统
        print("\n🚀 启动AGI进化系统...")
        if continuous:
            print("🔄 系统将持续运行，执行7*24小时AGI进化")
            print("📊 每60秒计算一次进化损失指标")
            print("💾 损失指标将保存到evo_state.json")
            print("🛑 按Ctrl+C停止系统")
            print("-" * 80)

            await nexus.run()
        else:
            print("🔄 执行单次测试进化周期...")

            # 执行一次测试周期
            if nexus.math_bridge is not None:
                import torch
                state = torch.randn(1, 256)
                learning_signal = torch.tensor([0.1])
                results = nexus.math_bridge(state, learning_signal)

                if nexus.loss_system is not None:
                    # 计算AGI进化损失指标
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
                        'emergent_capabilities': 0.58
                    }

                    from agi_evolution_loss_metrics import MathematicalCoreMetrics
                    math_metrics = MathematicalCoreMetrics(
                        lie_automorphism_coherence=results.get('evolution_metrics', {}).get('state_change', 0.85),
                        noncommutative_geometry_consistency=0.78,
                        knot_invariant_stability=0.88,
                        dde_decision_quality=0.92,
                        constraint_violation=0.08,
                        fueter_violation=0.03
                    )

                    loss_components = nexus.loss_system(
                        capability_embeddings=capability_embeddings,
                        current_performance=current_performance,
                        mathematical_metrics=math_metrics
                    )

                    print("📊 单次进化周期结果:")
                    print(f"  总进化损失: {loss_components.total_loss:.4f}")
                    print(f"  进化效率评分: {getattr(loss_components, 'evolution_efficiency_score', 0.0):.4f}")
                    print("✅ 单次测试完成")

    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在关闭AGI进化系统...")
    except Exception as e:
        print(f"\n❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ AGI进化系统已停止")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="H2Q-Evo AGI进化系统启动器")
    parser.add_argument("--test", action="store_true",
                       help="执行单次测试而不是持续运行")
    parser.add_argument("--mode", choices=['local', 'api'],
                       help="推理模式 (local=本地Docker, api=Gemini API)")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="日志级别")

    args = parser.parse_args()

    # 设置命令行参数
    if args.mode:
        os.environ["INFERENCE_MODE"] = args.mode
        Config.INFERENCE_MODE = args.mode

    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    # 启动系统
    continuous = not args.test
    success = asyncio.run(start_evolution_system(continuous))

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()