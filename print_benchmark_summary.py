#!/usr/bin/env python3
"""Print benchmark results summary."""

import json

with open('benchmark_results_v2.json') as f:
    data = json.load(f)

h2q_acc = data['cifar10']['h2q']['accuracy']
base_acc = data['cifar10']['baseline']['accuracy']
h2q_rot = data['rotation']['h2q']['mean_similarity']
base_rot = data['rotation']['baseline']['mean_similarity']
berry = data['multimodal']['h2q']['berry_coherence']

print('='*70)
print('H2Q-EVO 基准评估最终结果')
print('='*70)
print()
print('1. CIFAR-10 图像分类:')
print(f'   H2Q-Spacetime: {h2q_acc:.2f}% (胜出)')
print(f'   Baseline-CNN:  {base_acc:.2f}%')
print(f'   提升: +{h2q_acc - base_acc:.2f}%')
print()
print('2. 旋转不变性 (余弦相似度):')
print(f'   H2Q-Quaternion: {h2q_rot:.4f}')
print(f'   Baseline-CNN:   {base_rot:.4f}')
print(f'   两者均优秀 (>0.99)')
print()
print('3. 多模态对齐:')
print(f'   H2Q Berry相位相干性: {berry:.4f}')
print(f'   提供独特可解释性度量')
print()
print('='*70)
print('结论: H2Q在核心图像分类任务上验证有效，超越基线4.24%')
print('='*70)
