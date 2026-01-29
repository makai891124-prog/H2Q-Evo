#!/usr/bin/env python3
"""
M24-DAS系统集成验证测试
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '.')
sys.path.insert(0, './h2q_project')

print('🔍 M24-DAS系统集成验证测试')
print('=' * 50)

# 1. 验证DAS核心
try:
    from h2q_project.das_core import DASCore
    das = DASCore(target_dimension=256)
    print('✅ DAS核心: 正常')
except Exception as e:
    print(f'❌ DAS核心: {e}')

# 2. 验证权重转换器
try:
    import m24_das_weight_converter
    print('✅ 权重转换器: 正常')
except Exception as e:
    print(f'❌ 权重转换器: {e}')

# 3. 验证推理引擎
try:
    import m24_das_m4_inference_benchmark
    print('✅ 推理引擎: 正常')
except Exception as e:
    print(f'❌ 推理引擎: {e}')

# 4. 验证模型文件
try:
    import torch
    model = torch.load('models/das_optimized_deepseek-coder-v2-236b.pth', map_location='cpu', weights_only=True)
    print(f'✅ 模型文件: 正常 ({len(model)} 个权重张量)')
except Exception as e:
    print(f'❌ 模型文件: {e}')

# 5. 验证基准测试结果
try:
    import json
    # 查找最新的基准测试结果文件
    import glob
    result_files = glob.glob('m4_benchmark_results_*.json')
    if result_files:
        latest_file = max(result_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            results = json.load(f)
        print(f'✅ 基准测试: 正常 (平均分数: {results["summary"]["average_score"]:.3f})')
    else:
        print('❌ 基准测试: 未找到结果文件')
except Exception as e:
    print(f'❌ 基准测试: {e}')

print('=' * 50)
print('🎉 M24-DAS Mac Mini M4 AGI系统集成验证完成！')