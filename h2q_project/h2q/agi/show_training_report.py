#!/usr/bin/env python3
"""审计驱动优化训练 - 总结报告生成器."""

import torch
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def generate_report():
    """生成训练总结报告."""
    model_path = SCRIPT_DIR / 'optimized_model.pt'
    
    if not model_path.exists():
        print("模型文件不存在")
        return
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    print('=' * 70)
    print('       Audit-Driven Optimization Training Summary Report')
    print('=' * 70)
    print()
    print('+------------------------------------------------------------------+')
    print('|                    ULTIMATE GOAL                                 |')
    print('|                                                                  |')
    print('|     Train locally-available real-time AGI system                 |')
    print('|     (Zui Zhong Mu Biao: Xun Lian Ben Di Ke Yong De Shi Shi AGI)  |')
    print('+------------------------------------------------------------------+')
    print()
    
    # 训练历史
    history = checkpoint.get('training_history', {})
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    
    print('[Training Results]')
    print(f'  Training steps: {len(train_losses)}')
    print(f'  Validation checks: {len(val_losses)}')
    if train_losses:
        print(f'  Initial train loss: {train_losses[0]:.4f}')
        print(f'  Final train loss: {train_losses[-1]:.4f}')
        print(f'  Loss reduction: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%')
    if val_losses:
        print(f'  Final val loss: {val_losses[-1]:.4f}')
    
    # 泛化报告
    gen_report = checkpoint.get('generalization_report', {})
    print()
    print('[Generalization Analysis]')
    print(f'  Overfitting risk: {gen_report.get("overfitting_risk", "unknown")}')
    print(f'  Interpretation: {gen_report.get("interpretation", "N/A")}')
    
    # 优化报告
    opt_report = checkpoint.get('optimization_report', {})
    print()
    print('[Audit Optimizations]')
    print(f'  Suggestions found: {opt_report.get("suggestions_found", 0)}')
    print(f'  Applied: {opt_report.get("optimizations_applied", 0)}')
    
    # 应用的优化列表
    print()
    print('[Applied Optimizations]')
    for opt in opt_report.get('optimization_history', []):
        if opt.get('applied'):
            print(f'  [OK] {opt["suggestion"][:50]}...')
            for change in opt.get('changes', []):
                print(f'        - {change}')
    
    print()
    print('=' * 70)


if __name__ == "__main__":
    generate_report()
