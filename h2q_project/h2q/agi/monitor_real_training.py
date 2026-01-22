#!/usr/bin/env python3
"""
真实AGI训练监控
Real AGI Training Monitor
"""

import os
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / 'real_logs'


def get_latest_log():
    """获取最新日志文件"""
    logs = sorted(LOG_DIR.glob('training_*.log'), reverse=True)
    return logs[0] if logs else None


def parse_log(log_path):
    """解析日志"""
    if not log_path or not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 解析step信息
    step_pattern = r'Step\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+LR:\s+([\d.e+-]+)\s+\|\s+Tokens:\s+([\d,]+)\s+\|\s+Speed:\s+([\d.]+)\s+tok/s\s+\|\s+Progress:\s+([\d.]+)%'
    
    steps = re.findall(step_pattern, content)
    
    if not steps:
        return None
    
    latest = steps[-1]
    
    return {
        'step': int(latest[0]),
        'loss': float(latest[1]),
        'lr': float(latest[2]),
        'tokens': int(latest[3].replace(',', '')),
        'speed': float(latest[4]),
        'progress': float(latest[5])
    }


def get_process_info():
    """获取进程信息"""
    result = subprocess.run(
        ['pgrep', '-f', 'real_agi_training.py'],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return None
    
    pid = result.stdout.strip().split('\n')[0]
    
    ps = subprocess.run(
        ['ps', '-p', pid, '-o', 'etime=,pcpu=,pmem=,rss='],
        capture_output=True, text=True
    )
    
    if ps.returncode == 0:
        parts = ps.stdout.strip().split()
        return {
            'pid': pid,
            'elapsed': parts[0] if len(parts) > 0 else '-',
            'cpu': parts[1] if len(parts) > 1 else '-',
            'mem': parts[2] if len(parts) > 2 else '-',
            'rss': parts[3] if len(parts) > 3 else '-'
        }
    
    return {'pid': pid}


def display_status():
    """显示状态"""
    print("\033[2J\033[H")  # 清屏
    
    print("=" * 70)
    print("  🤖 真实AGI训练监控 - Real AGI Training Monitor")
    print("=" * 70)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 进程状态
    proc = get_process_info()
    if proc:
        print("  📊 进程状态:")
        print(f"     PID: {proc['pid']}")
        print(f"     运行时间: {proc.get('elapsed', '-')}")
        print(f"     CPU: {proc.get('cpu', '-')}%")
        print(f"     内存: {proc.get('mem', '-')}% ({int(int(proc.get('rss', 0))/1024)}MB)")
        status = "🟢 运行中"
    else:
        status = "🔴 已停止"
    
    print(f"\n  状态: {status}")
    
    # 训练指标
    log_path = get_latest_log()
    metrics = parse_log(log_path) if log_path else None
    
    if metrics:
        print("\n  📈 训练指标:")
        print(f"     Step: {metrics['step']:,}")
        print(f"     Loss: {metrics['loss']:.4f}")
        print(f"     学习率: {metrics['lr']:.2e}")
        print(f"     已处理Tokens: {metrics['tokens']:,}")
        print(f"     速度: {metrics['speed']:.0f} tok/s")
        print()
        
        # 进度条
        progress = metrics['progress']
        bar_width = 50
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  进度: [{bar}] {progress:.1f}%")
        
        # 预估
        if progress > 0 and proc:
            elapsed_parts = proc.get('elapsed', '0:00').split(':')
            if len(elapsed_parts) >= 2:
                try:
                    if len(elapsed_parts) == 2:
                        elapsed_min = int(elapsed_parts[0]) + int(elapsed_parts[1])/60
                    else:
                        elapsed_min = int(elapsed_parts[0])*60 + int(elapsed_parts[1]) + int(elapsed_parts[2])/60
                    
                    total_min = elapsed_min / (progress / 100)
                    remaining_min = total_min - elapsed_min
                    
                    print(f"  预计剩余: {remaining_min:.0f} 分钟 ({remaining_min/60:.1f} 小时)")
                except:
                    pass
    
    print("\n" + "=" * 70)
    print("  按 Ctrl+C 退出监控")
    print("=" * 70)


def main():
    """主函数"""
    print("启动监控...")
    
    try:
        while True:
            display_status()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n监控已停止")


if __name__ == "__main__":
    main()
