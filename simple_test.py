#!/usr/bin/env python3
"""
简化的全数据量AGI进化系统测试
"""

import sys
import torch
import torch.nn as nn
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

print('🎯 测试全数据量综合学习AGI目标进化系统')
print('=' * 60)

try:
    # 直接测试核心组件
    print('🔧 测试核心组件...')

    # 测试数据管理器
    from comprehensive_full_data_agi_evolution import ComprehensiveDataManager
    data_manager = ComprehensiveDataManager(max_memory_gb=8.0)
    print('✅ 数据管理器创建成功')

    available_datasets = data_manager.get_available_datasets()
    print(f'📋 可用数据集: {available_datasets}')

    # 测试进化核心
    from comprehensive_full_data_agi_evolution import ComprehensiveAGIEvolutionCore
    evolution_core = ComprehensiveAGIEvolutionCore(dim=1024, num_modalities=8)
    print('✅ 进化核心创建成功')

    # 测试基本张量操作
    test_input = {
        'text': torch.randn(2, 1024),
        'image': torch.randn(2, 3, 32, 32),
        'video': torch.randn(2, 3, 16, 64, 64)
    }

    print('🔄 测试前向传播...')
    evolved, agi_prob, strategy = evolution_core(test_input)
    print(f'✅ 前向传播成功 - 输出维度: {evolved.shape}, AGI概率: {agi_prob.shape}, 策略: {strategy.shape}')

    print('🎯 核心组件测试完成！')
    print('📊 系统架构验证:')
    print(f'  • 支持模态数: {len(evolution_core.modality_encoders)}')
    print(f'  • 模型参数量: {sum(p.numel() for p in evolution_core.parameters()):,}')
    print(f'  • 输出维度: {evolved.shape[1]}')

except Exception as e:
    print(f'❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()