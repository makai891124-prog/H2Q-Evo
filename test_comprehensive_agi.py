#!/usr/bin/env python3
"""
简化的全数据量AGI进化系统测试
"""

import asyncio
import sys
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

print('🎯 测试全数据量综合学习AGI目标进化系统')
print('=' * 60)

try:
    from comprehensive_full_data_agi_evolution import ComprehensiveAGIEvolutionSystem

    print('✅ 系统导入成功')

    # 创建简化版本进行测试
    evolution_system = ComprehensiveAGIEvolutionSystem(max_memory_gb=8.0)

    print('✅ 系统初始化成功')
    print(f'📊 内存限制: {evolution_system.max_memory_gb}GB')
    print(f'🎨 支持模态数: {len(evolution_system.evolution_core.modality_encoders)}')
    print(f'🎯 AGI进化目标数: {len(evolution_system.agi_goals)}')

    # 只运行几步测试
    print('🚀 开始简短测试运行...')

    async def test_run():
        await evolution_system._initialize_data_streams()
        print('✅ 数据流初始化成功')

        # 运行几步测试
        for step in range(3):
            print(f'📊 测试步骤 {step + 1}/3')
            try:
                await evolution_system._evolution_step(step)
                print(f'✅ 步骤 {step + 1} 完成')
            except Exception as e:
                print(f'⚠️ 步骤 {step + 1} 失败: {e}')
                continue

        print('🎯 测试完成！')

    asyncio.run(test_run())

except Exception as e:
    print(f'❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()