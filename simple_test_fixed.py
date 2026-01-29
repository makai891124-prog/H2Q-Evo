#!/usr/bin/env python3
"""
简化的全数据量AGI进化系统测试 - 修复版本
"""

import sys
import torch
import asyncio
import os
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

print('🎯 测试全数据量综合学习AGI目标进化系统')
print('=' * 60)

async def test_run():
    try:
        # 导入系统
        from comprehensive_full_data_agi_evolution import ComprehensiveAGIEvolutionSystem

        print('🔧 初始化系统...')
        evolution_system = ComprehensiveAGIEvolutionSystem(max_memory_gb=8.0)
        print('✅ 系统导入成功')

        print('📊 系统信息:')
        print(f'  • 内存限制: {evolution_system.max_memory_gb}GB')
        print(f'  • 支持模态数: {len(evolution_system.evolution_core.modality_encoders)}')
        print(f'  • AGI进化目标数: {len(evolution_system.agi_goals)}')

        print('🚀 开始简短测试运行...')

        # 初始化数据流
        print('🔄 初始化全数据量数据流...')
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

        # 运行几个进化步骤
        for step in range(1, 4):
            print(f'📊 测试步骤 {step}/3')
            try:
                await evolution_system._evolution_step(step)
                print(f'✅ 步骤 {step} 完成')
            except Exception as e:
                print(f'⚠️ 步骤 {step} 失败: {e}')
                continue

        print('🎯 测试完成！')

    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 设置环境变量避免多进程问题
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    asyncio.run(test_run())