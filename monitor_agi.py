#!/usr/bin/env python3
"""
H2Q-Evo AGI 实时监控面板
显示运行中AGI系统的状态
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

def clear_screen():
    """清屏"""
    os.system('clear' if os.name == 'posix' else 'cls')

def format_uptime(seconds):
    """格式化运行时长"""
    return str(timedelta(seconds=int(seconds)))

def load_status():
    """加载状态文件"""
    status_file = Path("agi_daemon_status.json")
    if not status_file.exists():
        return None
    
    try:
        with open(status_file) as f:
            return json.load(f)
    except:
        return None

def display_dashboard(status):
    """显示仪表板"""
    clear_screen()
    
    print("=" * 80)
    print("🎛️  H2Q-Evo AGI 实时监控面板".center(80))
    print("=" * 80)
    print()
    
    if status is None:
        print("⚠️  AGI守护进程未运行或状态文件不存在")
        print("\n启动守护进程: python3 agi_daemon.py [间隔秒数]")
        return
    
    # 基本信息
    print(f"📊 系统状态")
    print(f"   运行时长: {format_uptime(status['uptime_seconds'])}")
    print(f"   最后更新: {status['last_update']}")
    print()
    
    # 活动统计
    print(f"🔬 活动统计")
    print(f"   总查询数: {status['query_count']}")
    print(f"   进化周期: {status['evolution_cycles']}")
    print(f"   知识总量: {status['knowledge_total']} 条")
    
    # 计算速率
    if status['uptime_seconds'] > 0:
        qps = status['query_count'] / status['uptime_seconds']
        print(f"   查询速率: {qps*60:.2f} 次/分钟")
    print()
    
    # 知识分布
    print(f"🧠 知识库分布")
    domains = status['knowledge_by_domain']
    total = sum(domains.values())
    
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 5)
        print(f"   {domain:12s} │{bar:<20s}│ {count} 条 ({percentage:.1f}%)")
    
    print()
    print("=" * 80)
    print("💡 提示: 按 Ctrl+C 退出监控 | 查看日志: tail -f evolution.log")
    print("=" * 80)

def monitor_loop(refresh_interval=2):
    """监控循环"""
    try:
        while True:
            status = load_status()
            display_dashboard(status)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n\n👋 监控结束")

if __name__ == "__main__":
    import sys
    
    # 可选参数：刷新间隔（秒）
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    
    print(f"启动监控面板（刷新间隔：{interval}秒）...")
    time.sleep(1)
    
    monitor_loop(refresh_interval=interval)
