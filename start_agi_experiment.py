#!/usr/bin/env python3
"""
H2Q-Evo AGI实验启动脚本
启动内存优化后的AGI训练系统，开始安全的小规模实验
"""

import sys
import os
import logging
import time
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from agi_system_manager import AGISystemManager

def main():
    parser = argparse.ArgumentParser(description='H2Q-Evo AGI实验启动器')
    parser.add_argument('--max-generations', type=int, default=10,
                       help='最大进化代数 (默认: 10)')
    parser.add_argument('--memory-limit', type=float, default=3.0,
                       help='内存限制GB (默认: 3.0)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--experiment-name', default='agi_experiment_001',
                       help='实验名称')

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(f'./agi_experiment_{args.experiment_name}.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('AGI-Launcher')

    print("🚀 H2Q-Evo AGI实验启动器")
    print("=" * 50)
    print(f"实验名称: {args.experiment_name}")
    print(f"最大代数: {args.max_generations}")
    print(f"内存限制: {args.memory_limit}GB")
    print(f"日志级别: {args.log_level}")
    print("=" * 50)

    # 创建AGI系统管理器
    manager = AGISystemManager()

    try:
        # 启动系统
        logger.info("启动AGI系统...")
        manager.start_system()
        print("✅ AGI系统启动成功")

        # 运行实验
        logger.info(f"开始AGI实验: {args.experiment_name}")
        print(f"🔬 开始AGI实验，预计运行{args.max_generations}代...")

        generation = 0
        while generation < args.max_generations:
            print(f"\n📊 第 {generation + 1} 代进化")

            # 执行训练周期
            if manager.trainer:
                manager.trainer.run_training_cycle()
                print(f"✅ 第 {generation + 1} 代完成")
            else:
                print("❌ 训练器未初始化")
                break

            generation += 1

            # 检查是否应该停止
            if hasattr(manager.trainer, 'should_stop') and manager.trainer.should_stop:
                print("🎯 达到停止条件，提前结束实验")
                break

            # 小延迟避免CPU占用过高
            time.sleep(1)

        print(f"\n🎉 AGI实验完成！共运行 {generation} 代")

        # 显示实验结果
        if manager.trainer and hasattr(manager.trainer, 'state'):
            state = manager.trainer.state
            print("\n📈 实验结果总结:")
            print(f"  最终代数: {state.generation}")
            print(f"  最佳适应度: {state.best_fitness:.4f}")
            print(f"  平均损失: {state.average_loss:.4f}")
            print(f"  总训练时间: {state.total_training_time:.1f}秒")

    except KeyboardInterrupt:
        print("\n⏹️  收到中断信号，正在停止...")
    except Exception as e:
        logger.error(f"实验过程中出错: {e}")
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止系统
        logger.info("停止AGI系统...")
        manager.stop_system()
        print("✅ AGI系统已停止")

        print("\n📝 实验日志已保存到:")
        print(f"  ./agi_experiment_{args.experiment_name}.log")
        print(f"  ./evolution.log")
        print(f"  ./wandb/ (离线模式)")

if __name__ == "__main__":
    main()