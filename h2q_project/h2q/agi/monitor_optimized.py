#!/usr/bin/env python3
"""
优化版5小时训练监控器
"""

import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / 'optimized_training.log'
CHECKPOINT_DIR = SCRIPT_DIR / 'optimized_checkpoints'
MODEL_DIR = SCRIPT_DIR / 'optimized_models'


def get_process():
    """获取进程信息"""
    result = subprocess.run(
        ['pgrep', '-f', 'optimized_5h_training.py'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        pid = result.stdout.strip().split('\n')[0]
        ps = subprocess.run(
            ['ps', '-p', pid, '-o', 'pid,%cpu,%mem,etime'],
            capture_output=True, text=True
        )
        return {'running': True, 'pid': pid, 'info': ps.stdout}
    return {'running': False}


def parse_latest_epoch(log_path):
    """解析最新的epoch信息"""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    epoch_info = {}
    
    for i, line in enumerate(lines):
        if 'Epoch' in line and '完成' in line:
            # 找到epoch完成信息
            for j in range(i, min(i + 10, len(lines))):
                l = lines[j].strip()
                if '训练 Loss:' in l:
                    parts = l.split('|')
                    for p in parts:
                        if 'Loss:' in p:
                            epoch_info['train_loss'] = p.split(':')[1].strip()
                        if 'Acc:' in p:
                            epoch_info['train_acc'] = p.split(':')[1].strip()
                elif '验证 Acc:' in l:
                    parts = l.split('|')
                    for p in parts:
                        if '验证 Acc:' in p:
                            epoch_info['val_acc'] = p.split(':')[1].strip()
                        if '最佳:' in p:
                            epoch_info['best_acc'] = p.split(':')[1].strip()
                elif '进度:' in l:
                    parts = l.split('|')
                    for p in parts:
                        if '进度:' in p:
                            epoch_info['progress'] = p.split(':')[1].strip()
                        if '已用:' in p:
                            epoch_info['elapsed'] = p.split(':')[1].strip()
                elif 'Epoch' in l and '完成' in l:
                    epoch_info['epoch'] = l.split()[1]
    
    return epoch_info if epoch_info else None


def get_latest_batch(log_path, n=5):
    """获取最新的batch日志"""
    if not log_path.exists():
        return []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    batch_lines = [l for l in lines if 'Batch' in l]
    return batch_lines[-n:] if batch_lines else []


def main():
    print("\n" + "=" * 70)
    print("   5小时真实AGI训练监控")
    print("=" * 70)
    print(f"   当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 进程状态
    print("\n📊 进程状态:")
    print("-" * 50)
    proc = get_process()
    if proc['running']:
        print(f"   状态: ✅ 运行中 (PID: {proc['pid']})")
        print(f"   {proc['info'].strip()}")
    else:
        print("   状态: ❌ 未运行")
    
    # Epoch信息
    print("\n📈 训练进度:")
    print("-" * 50)
    epoch_info = parse_latest_epoch(LOG_FILE)
    if epoch_info:
        print(f"   当前Epoch: {epoch_info.get('epoch', 'N/A')}")
        print(f"   训练Loss: {epoch_info.get('train_loss', 'N/A')}")
        print(f"   训练准确率: {epoch_info.get('train_acc', 'N/A')}")
        print(f"   验证准确率: {epoch_info.get('val_acc', 'N/A')}")
        print(f"   最佳准确率: {epoch_info.get('best_acc', 'N/A')}")
        print(f"   进度: {epoch_info.get('progress', 'N/A')}")
        print(f"   已用时间: {epoch_info.get('elapsed', 'N/A')}")
    else:
        print("   等待第一个Epoch完成...")
    
    # 最新Batch
    print("\n📝 最新训练Batch:")
    print("-" * 50)
    batches = get_latest_batch(LOG_FILE, 5)
    if batches:
        for b in batches:
            # 提取时间和内容
            parts = b.split('|')
            if len(parts) >= 2:
                print(f"   {parts[0].split()[-1]} | {' | '.join(parts[1:]).strip()}")
    else:
        print("   等待训练开始...")
    
    # 检查点
    print("\n💾 检查点:")
    print("-" * 50)
    ckpt = CHECKPOINT_DIR / 'checkpoint.pt'
    if ckpt.exists():
        import torch
        data = torch.load(ckpt, map_location='cpu')
        print(f"   Epoch: {data.get('epoch', 'N/A')}")
        stats = data.get('stats', {})
        print(f"   总样本: {stats.get('total_samples', 0):,}")
        print(f"   最佳准确率: {stats.get('best_accuracy', 0):.2%}")
        print(f"   文件大小: {ckpt.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("   尚未保存检查点")
    
    # 进度条
    if epoch_info and epoch_info.get('progress'):
        progress_str = epoch_info['progress'].replace('%', '').strip()
        try:
            progress = float(progress_str)
            bar_len = 40
            filled = int(bar_len * progress / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\n   [{bar}] {progress:.1f}%")
        except:
            pass
    
    print("\n" + "=" * 70)
    print("   使用方法:")
    print("   - 再次运行此脚本查看最新状态")
    print("   - 查看完整日志: tail -f optimized_training.log")
    print("   - 停止训练: kill <PID>")
    print("=" * 70)


if __name__ == "__main__":
    main()
