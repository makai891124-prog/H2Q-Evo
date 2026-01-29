#!/usr/bin/env python3
"""
自动AGI训练和性能改进脚本
持续运行多模态AGI进化，监控性能并自动调整参数
"""

import asyncio
import sys
import signal
import time
import json
from pathlib import Path
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from true_agi_autonomous_system import TrueAGIAutonomousSystem

class AutoAGITrainer:
    """自动AGI训练器"""

    def __init__(self):
        self.system = None
        self.performance_history = []
        self.best_performance = float('-inf')
        self.training_start_time = time.time()

    async def initialize_system(self):
        """初始化AGI系统"""
        print("🚀 初始化多模态AGI训练系统...")
        self.system = TrueAGIAutonomousSystem(input_dim=256, action_dim=256)
        print("✅ AGI系统初始化完成")

    async def run_continuous_training(self):
        """运行持续训练"""
        print("🎯 开始自动AGI训练和性能改进...")

        step = 0
        while True:
            try:
                await asyncio.sleep(1)
                step += 1

                # 每100步评估性能
                if step % 100 == 0:
                    await self._evaluate_performance()

                # 每500步保存检查点
                if step % 500 == 0:
                    await self._save_checkpoint(step)

            except Exception as e:
                print(f"❌ 训练出错: {e}")
                await asyncio.sleep(5)

    async def _evaluate_performance(self):
        """评估当前性能"""
        if not self.system or not self.system.performance_history:
            return

        recent_metrics = self.system.performance_history[-100:]
        avg_phi = np.mean([m.integrated_information for m in recent_metrics])
        avg_complexity = np.mean([m.neural_complexity for m in recent_metrics])
        avg_meta_cognition = np.mean([m.meta_cognition for m in recent_metrics])

        current_performance = avg_phi * 0.4 + avg_complexity * 0.3 + avg_meta_cognition * 0.3

        self.performance_history.append({
            'step': len(self.system.performance_history),
            'performance': current_performance,
            'phi': avg_phi,
            'complexity': avg_complexity,
            'meta_cognition': avg_meta_cognition,
            'timestamp': time.time()
        })

        if current_performance > self.best_performance:
            self.best_performance = current_performance
            print("🏆 新的最佳性能!")

        print(f"📊 性能评估 (步骤 {len(self.system.performance_history)}):")
        print(f"   综合性能: {current_performance:.4f}")
        print(f"   整合信息Φ: {avg_phi:.4f}")
        print(f"   神经复杂度: {avg_complexity:.4f}")
        print(f"   元认知意识: {avg_meta_cognition:.4f}")

    async def _save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_path = f"agi_checkpoint_step_{step}.pt"
        try:
            torch.save({
                'step': step,
                'performance_history': self.performance_history,
                'best_performance': self.best_performance,
                'timestamp': time.time()
            }, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")

async def main():
    """主函数"""
    trainer = AutoAGITrainer()

    try:
        await trainer.initialize_system()

        # 启动AGI进化系统
        evolution_task = asyncio.create_task(trainer.system.start_true_evolution())

        # 启动训练监控
        training_task = asyncio.create_task(trainer.run_continuous_training())

        # 同时运行两个任务
        await asyncio.gather(evolution_task, training_task)

    except KeyboardInterrupt:
        print("\n👋 自动训练已停止")
    except Exception as e:
        print(f"❌ 自动训练失败: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 自动训练已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)</content>
<parameter name="filePath">/Users/imymm/H2Q-Evo/auto_agi_trainer.py