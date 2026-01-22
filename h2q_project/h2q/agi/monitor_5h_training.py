#!/usr/bin/env python3
"""
5小时真实训练监控器
"""

import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / 'real_training_logs'
MODEL_DIR = SCRIPT_DIR / 'real_trained_models'
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
OUTPUT_LOG = SCRIPT_DIR / 'real_training_output.log'


def get_process_info():
    """获取训练进程信息"""
    result = subprocess.run(
        ['pgrep', '-f', 'real_5h_training.py'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        # 获取进程详情
        ps_result = subprocess.run(
            ['ps', '-p', pids[0], '-o', 'pid,%cpu,%mem,etime'],
            capture_output=True, text=True
        )
        return {'running': True, 'pids': pids, 'details': ps_result.stdout}
    return {'running': False}


def get_latest_log_entries(n=20):
    """获取最新的日志条目"""
    # 查找最新的日志文件
    if LOG_DIR.exists():
        log_files = sorted(LOG_DIR.glob('training_*.log'), reverse=True)
        if log_files:
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
                return lines[-n:] if len(lines) >= n else lines
    
    # 尝试输出日志
    if OUTPUT_LOG.exists():
        with open(OUTPUT_LOG, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) >= n else lines
    
    return []


def get_checkpoint_info():
    """获取检查点信息"""
    checkpoint_path = CHECKPOINT_DIR / 'latest_checkpoint.pt'
    if checkpoint_path.exists():
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return {
            'exists': True,
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'total_samples': checkpoint.get('total_samples', 0),
            'best_accuracy': checkpoint.get('best_accuracy', 0),
            'total_time': checkpoint.get('total_time', 0),
            'file_size': checkpoint_path.stat().st_size / (1024 * 1024)
        }
    return {'exists': False}


def get_model_info():
    """获取模型信息"""
    model_path = MODEL_DIR / 'real_agi_model_latest.pt'
    if model_path.exists():
        import torch
        model_data = torch.load(model_path, map_location='cpu')
        return {
            'exists': True,
            'epoch': model_data.get('epoch', 0),
            'best_accuracy': model_data.get('best_accuracy', 0),
            'total_samples': model_data.get('total_samples', 0),
            'file_size': model_path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(model_path.stat().st_mtime)
        }
    return {'exists': False}


def main():
    print("\n" + "=" * 70)
    print("   5小时真实AGI训练监控")
    print("   Real 5-Hour AGI Training Monitor")
    print("=" * 70)
    print(f"   当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 进程状态
    print("\n📊 进程状态:")
    print("-" * 50)
    proc_info = get_process_info()
    if proc_info['running']:
        print(f"   状态: ✅ 运行中")
        print(f"   PID: {', '.join(proc_info['pids'])}")
        if 'details' in proc_info:
            print(f"   {proc_info['details'].strip()}")
    else:
        print("   状态: ❌ 未运行")
    
    # 检查点状态
    print("\n💾 检查点状态:")
    print("-" * 50)
    ckpt_info = get_checkpoint_info()
    if ckpt_info['exists']:
        print(f"   Epoch: {ckpt_info['epoch']}")
        print(f"   全局步数: {ckpt_info['global_step']:,}")
        print(f"   总样本: {ckpt_info['total_samples']:,}")
        print(f"   最佳准确率: {ckpt_info['best_accuracy']:.2%}")
        print(f"   训练时长: {timedelta(seconds=int(ckpt_info['total_time']))}")
        print(f"   文件大小: {ckpt_info['file_size']:.1f} MB")
    else:
        print("   尚未创建检查点")
    
    # 模型状态
    print("\n🤖 模型状态:")
    print("-" * 50)
    model_info = get_model_info()
    if model_info['exists']:
        print(f"   Epoch: {model_info['epoch']}")
        print(f"   最佳准确率: {model_info['best_accuracy']:.2%}")
        print(f"   总样本: {model_info['total_samples']:,}")
        print(f"   文件大小: {model_info['file_size']:.1f} MB")
        print(f"   最后更新: {model_info['modified'].strftime('%H:%M:%S')}")
    else:
        print("   尚未保存模型")
    
    # 最新日志
    print("\n📝 最新训练日志:")
    print("-" * 50)
    log_entries = get_latest_log_entries(15)
    if log_entries:
        for line in log_entries:
            print(f"   {line.rstrip()}")
    else:
        print("   暂无日志")
    
    # 进度估算
    print("\n⏱️ 进度估算:")
    print("-" * 50)
    if ckpt_info.get('exists') and ckpt_info.get('total_time', 0) > 0:
        elapsed_hours = ckpt_info['total_time'] / 3600
        progress = elapsed_hours / 5.0 * 100
        remaining_hours = max(0, 5.0 - elapsed_hours)
        eta = datetime.now() + timedelta(hours=remaining_hours)
        
        print(f"   已训练: {elapsed_hours:.2f} 小时")
        print(f"   进度: {progress:.1f}%")
        print(f"   预计剩余: {remaining_hours:.2f} 小时")
        print(f"   预计完成: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 绘制进度条
        bar_len = 40
        filled = int(bar_len * progress / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n   [{bar}] {progress:.1f}%")
    else:
        print("   等待训练数据...")
    
    print("\n" + "=" * 70)
    print("   提示: 再次运行此脚本查看最新状态")
    print("   停止训练: kill <PID>")
    print("=" * 70)


if __name__ == "__main__":
    main()
