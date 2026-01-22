#!/usr/bin/env python3
"""
AGI 训练系统监控器
AGI Training System Monitor
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / 'agi_models' / 'evolution_system_state.json'
MODEL_FILE = SCRIPT_DIR / 'agi_models' / 'real_agi_evolved.pt'
LOG_FILE = SCRIPT_DIR / 'agi_evolution.log'


def main():
    print("\n" + "=" * 60)
    print("       AGI TRAINING SYSTEM MONITOR")
    print("=" * 60)
    
    # 检查进程
    result = subprocess.run(
        ['pgrep', '-f', 'start_agi_evolution.py'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        print(f"\n  Status: RUNNING (PID: {', '.join(pids)})")
    else:
        print("\n  Status: NOT RUNNING")
    
    # 检查状态文件
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        
        print(f"\n  Generation: {state.get('generation', 0)}")
        print(f"  Total Epochs: {state.get('total_epochs', 0)}")
        print(f"  Best Accuracy: {state.get('best_accuracy', 0):.2%}")
        print(f"  Emergence Events: {state.get('emergence_count', 0)}")
    else:
        print("\n  No state file found")
    
    # 检查模型文件
    if MODEL_FILE.exists():
        size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(MODEL_FILE.stat().st_mtime)
        print(f"\n  Model File: {MODEL_FILE.name}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Last Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示最近的日志（如果有）
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
        print("\n" + "-" * 60)
        print("Recent Log:")
        print("-" * 60)
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
