#!/usr/bin/env python3
"""
快速生成若干证明工件样本：
 - 运行AGI守护进程的若干cycle
 - 运行Live AGI系统的demo
 - 导出bundle
"""
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # 运行daemon 3个cycle（通过直接调用其run_cycle若需更精细可扩展）
    # 这里用python -c方式调用一次run_cycle三次
    code = r"""
import agi_daemon
daemon = agi_daemon.AGIDaemon(interval=0)
for _ in range(3):
    daemon.run_cycle()
"""
    run([sys.executable, "-c", code])

    # 运行live系统demo一次（会生成多条推理及工件）
    code2 = r"""
import live_agi_system
agi = live_agi_system.LiveAGISystem()
agi._run_demo()
"""
    run([sys.executable, "-c", code2])

    # 导出bundle
    run([sys.executable, "export_knowledge_bundle.py"])

if __name__ == "__main__":
    main()
