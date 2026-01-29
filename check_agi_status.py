#!/usr/bin/env python3
"""
检查当前AGI目标状态
"""

import json
from pathlib import Path

def check_agi_status():
    status_file = Path('realtime_training_status.json')
    if not status_file.exists():
        print('❌ 状态文件不存在')
        return

    with open(status_file, 'r') as f:
        status = json.load(f)

    geometric = status.get('geometric_metrics', {})
    perf = status.get('performance_metrics', {})

    print('🎯 当前AGI指标状态:')
    print(f'几何准确率: {geometric.get("geometric_accuracy", 0):.4f} (目标: 0.9)')
    print(f'谱移η实部: {geometric.get("spectral_shift_eta_real", 0):.4f} (目标: 0.5)')
    print(f'分形坍缩惩罚: {geometric.get("fractal_collapse_penalty", 0):.4f} (目标: ≤0.1)')
    print(f'分类F1分数: {geometric.get("classification_f1", 0):.4f} (目标: 0.85)')
    print(f'流形稳定性: {perf.get("manifold_stability", 0):.2f} (目标: 5.0)')

    # 检查目标达成情况
    targets = {
        'geometric_accuracy': geometric.get('geometric_accuracy', 0) >= 0.9,
        'spectral_shift_eta': geometric.get('spectral_shift_eta_real', 0) >= 0.5,
        'fractal_collapse_penalty': geometric.get('fractal_collapse_penalty', 0) <= 0.1,
        'classification_f1': geometric.get('classification_f1', 0) >= 0.85,
        'manifold_stability': perf.get('manifold_stability', 0) >= 5.0
    }

    achieved = all(targets.values())
    print(f'\nAGI目标达成: {"✅ 是" if achieved else "❌ 否"}')
    print(f'达成指标: {sum(targets.values())}/5')

    if achieved:
        print('\n🎉 AGI目标已达成！可以启动审计基准验收。')

if __name__ == "__main__":
    check_agi_status()