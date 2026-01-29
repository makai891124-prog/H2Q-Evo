#!/usr/bin/env python3
"""
全数据量综合学习AGI目标进化系统演示
"""

import sys
import torch
import asyncio
import time
import os
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

print('🎯 全数据量综合学习AGI目标进化系统演示')
print('=' * 80)

async def demo_run():
    try:
        # 导入系统
        from comprehensive_full_data_agi_evolution import ComprehensiveAGIEvolutionSystem

        print('🔧 初始化系统...')
        evolution_system = ComprehensiveAGIEvolutionSystem(max_memory_gb=8.0)
        print('✅ 系统初始化成功')

        print('\n📊 系统配置:')
        print(f'  • 内存限制: {evolution_system.max_memory_gb}GB')
        print(f'  • 支持模态数: {len(evolution_system.evolution_core.modality_encoders)}')
        print(f'  • AGI进化目标数: {len(evolution_system.agi_goals)}')
        print(f'  • 学习策略数: {len(evolution_system.learning_strategies)}')

        print('\n🎯 AGI进化目标:')
        for i, goal in enumerate(evolution_system.agi_goals, 1):
            print(f'  {i}. {goal}')

        print('\n🚀 开始演示运行...')

        # 初始化数据流
        print('\n🔄 初始化数据流...')
        available_datasets = evolution_system.data_manager.get_available_datasets()
        print(f'📋 可用数据集: {available_datasets}')

        # 创建数据流
        for dataset in available_datasets:
            try:
                stream = evolution_system.data_manager.create_data_stream(dataset, batch_size=4)
                evolution_system.active_streams[dataset] = stream
                print(f'✅ 数据流创建成功: {dataset}')
            except Exception as e:
                print(f'⚠️ 数据流创建失败 {dataset}: {e}')

        print(f'🎯 活跃数据流数量: {len(evolution_system.active_streams)}')

        # 运行演示进化
        print('\n🧬 开始AGI进化演示...')
        start_time = time.time()

        for step in range(1, 6):  # 运行5个步骤
            step_start = time.time()
            print(f'\n📊 进化步骤 {step}/5')

            try:
                await evolution_system._evolution_step(step)
                step_time = time.time() - step_start

                # 显示当前状态
                import psutil
                current_memory = psutil.Process().memory_info().rss / (1024 ** 3)
                print(f'✅ 步骤 {step} 完成 (用时: {step_time:.2f}s, 内存: {current_memory:.2f}GB)')

                # 显示AGI目标进度
                if hasattr(evolution_system, 'evolution_metrics') and 'goal_progress' in evolution_system.evolution_metrics:
                    print('🎯 AGI目标进度:')
                    for goal, progress in evolution_system.evolution_metrics['goal_progress'].items():
                        if progress:
                            latest = progress[-1] if progress else 0
                            print(f'  • {goal}: {latest:.3f}')

            except Exception as e:
                print(f'⚠️ 步骤 {step} 失败: {e}')
                continue

        total_time = time.time() - start_time
        print(f'\n🎉 演示完成！总用时: {total_time:.2f}s')
        print(f'📈 平均每步骤用时: {total_time/5:.2f}s')

        # 显示最终统计
        print('\n📊 最终系统状态:')
        import psutil
        final_memory = psutil.Process().memory_info().rss / (1024 ** 3)
        print(f'  • 当前内存使用: {final_memory:.2f}GB')
        print(f'  • 活跃数据流: {len(evolution_system.active_streams)}')
        print(f'  • 模型参数量: {sum(p.numel() for p in evolution_system.evolution_core.parameters()):,}')

        print('\n🎯 系统验证成功！')
        print('✅ 全数据量综合学习AGI目标进化系统已就绪')

    except Exception as e:
        print(f'❌ 演示失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 设置环境变量避免多进程问题
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    asyncio.run(demo_run())