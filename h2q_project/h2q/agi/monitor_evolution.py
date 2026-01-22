#!/usr/bin/env python3
"""
进化循环监控工具
Evolution Loop Monitor
"""

import json
import sys
from pathlib import Path
from datetime import timedelta

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / 'evolution_state.json'
LOG_FILE = SCRIPT_DIR / 'evolution_10h.log'


def format_time(seconds: float) -> str:
    """格式化时间."""
    td = timedelta(seconds=int(seconds))
    return str(td)


def main():
    print("\n" + "=" * 60)
    print("       EVOLUTION LOOP MONITOR")
    print("=" * 60)
    
    if not STATE_FILE.exists():
        print("\n[ERROR] No evolution state found!")
        print("Start the evolution loop first.")
        return
    
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    # 计算进度
    elapsed = state['elapsed_seconds']
    target = state['target_seconds']
    progress = (elapsed / target) * 100 if target > 0 else 0
    remaining = target - elapsed
    
    # 进度条
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    print(f"\n[{bar}] {progress:.2f}%")
    print()
    print(f"  Elapsed:    {format_time(elapsed)}")
    print(f"  Remaining:  {format_time(remaining)}")
    print(f"  Target:     {format_time(target)}")
    print()
    print(f"  Generation: {state['current_generation']}")
    print(f"  Epochs:     {state['total_training_epochs']}")
    print()
    
    # 损失统计
    if state['loss_history']:
        initial_loss = state['loss_history'][0]
        current_loss = state['current_loss']
        best_loss = state['best_loss']
        improvement = (1 - current_loss / initial_loss) * 100 if initial_loss > 0 else 0
        
        print(f"  Initial Loss: {initial_loss:.4f}")
        print(f"  Current Loss: {current_loss:.4f}")
        print(f"  Best Loss:    {best_loss:.4f}")
        print(f"  Improvement:  {improvement:.1f}%")
    
    if state['val_loss_history']:
        print(f"  Val Loss:     {state['val_loss_history'][-1]:.4f}")
    
    print()
    print(f"  Checkpoints:  {state['checkpoint_count']}")
    print(f"  Audits:       {state['audit_count']}")
    
    # 显示最近的日志
    if LOG_FILE.exists():
        print("\n" + "-" * 60)
        print("Recent Log Entries:")
        print("-" * 60)
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
    
    print("\n" + "=" * 60)
    
    # 检查进程是否在运行
    import subprocess
    result = subprocess.run(
        ['pgrep', '-f', 'start_evolution.py'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        print(f"  Status: RUNNING (PID: {', '.join(pids)})")
    else:
        print("  Status: NOT RUNNING")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
