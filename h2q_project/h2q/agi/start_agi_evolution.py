#!/usr/bin/env python3
"""
AGI 长时间自主进化循环启动器
AGI Long-Running Autonomous Evolution Launcher

此脚本启动一个持续的AGI进化循环，整合：
1. 真实基准测试数据集学习
2. 第三方Gemini审计验证
3. 自动代码生成和安全验证
4. 权重涌现检测
"""

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from real_agi_training_system import RealAGIEvolutionSystem


def main():
    print("\n" + "=" * 70)
    print("       AGI AUTONOMOUS EVOLUTION - LONG RUN")
    print("       (AGI Zi Zhu Jin Hua - Chang Qi Yun Xing)")
    print("=" * 70)
    print()
    print("Target: Continuous evolution with benchmark verification")
    print("Press Ctrl+C to safely stop and save checkpoint")
    print()
    print("-" * 70)
    
    # 创建系统
    system = RealAGIEvolutionSystem()
    
    # 运行长时间进化（20代，每代30个epoch）
    try:
        system.run_evolution_cycle(
            num_generations=20,
            epochs_per_gen=30
        )
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving final checkpoint...")
        system.save_checkpoint()
    except Exception as e:
        print(f"\n[Error] {e}")
        system.save_checkpoint()
        raise


if __name__ == "__main__":
    main()
