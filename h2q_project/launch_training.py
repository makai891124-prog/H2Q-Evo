#!/usr/bin/env python3
"""
H2Q-Evo 训练启动器 - 支持自定义训练时长和模式

使用方式:
    python3 launch_training.py --duration 4.0 --mode enhanced
    python3 launch_training.py --duration 2.0 --mode quick
    python3 launch_training.py --help
"""

import sys
import argparse
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')


def launch_training(duration_hours: float = 4.0, mode: str = 'enhanced', 
                   verbose: bool = False):
    """启动训练"""
    
    print("\n" + "="*80)
    print("H2Q-Evo 训练启动器".center(80))
    print("="*80)
    print(f"模式: {mode}")
    print(f"训练时长: {duration_hours} 小时")
    print(f"日志详度: {'详细' if verbose else '标准'}")
    print("="*80 + "\n")
    
    if mode == 'enhanced':
        from train_enhanced_with_monitoring import EnhancedTrainingSession
        
        session = EnhancedTrainingSession(
            duration_hours=duration_hours,
            base_learning_rate=0.0001,
            initial_batch_size=32
        )
        monitor = session.run_training()
        
    elif mode == 'quick':
        from train_local_model_advanced import main as run_quick_training
        run_quick_training()
        
    else:
        print(f"未知的训练模式: {mode}")
        print("支持的模式: enhanced, quick")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("训练启动完成！".center(80))
    print("="*80)
    print("\n结果位置:")
    print("  • 进化数据: training_output/evolution_data_*.json")
    print("  • 训练报告: training_output/training_report_*.md")
    print("  • 日志文件: training_session.log")
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='H2Q-Evo 训练启动器 - 支持长时间训练和动态监控'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=4.0,
        help='训练时长（小时），默认 4.0'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['enhanced', 'quick'],
        default='enhanced',
        help='训练模式: enhanced (长时间+监控) 或 quick (快速测试)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志输出'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.duration <= 0:
        print("错误: 训练时长必须大于 0")
        sys.exit(1)
    
    if args.duration > 24:
        print("警告: 训练时长超过 24 小时，这需要很长时间")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 启动训练
    try:
        launch_training(
            duration_hours=args.duration,
            mode=args.mode,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
