#!/usr/bin/env python3
"""
自动启动10小时进化循环 - 无需交互
Auto-start 10-Hour Evolution Loop - Non-interactive
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evolution_loop_10h import EvolutionLoop, EvolutionConfig

def main():
    print("\n" + "=" * 70)
    print("       AUTO-STARTING 10-HOUR EVOLUTION LOOP")
    print("=" * 70)
    print()
    
    config = EvolutionConfig()
    
    # 自动恢复（如果有检查点）
    resume = config.state_file.exists()
    
    if resume:
        print("[Auto-resuming from checkpoint]")
    else:
        print("[Starting fresh evolution]")
    
    print()
    print("Target: 10 hours of continuous learning")
    print("Press Ctrl+C to safely stop and save checkpoint")
    print()
    print("-" * 70)
    
    loop = EvolutionLoop(config, resume=resume)
    loop.run()


if __name__ == "__main__":
    main()
