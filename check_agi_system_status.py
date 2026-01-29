#!/usr/bin/env python3
"""
AGI系统状态检查脚本

检查当前AGI系统的运行状态和监控数据
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.append('.')

def format_timestamp(ts):
    """格式化时间戳"""
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def check_system_status():
    """检查系统状态"""
    print("🔍 AGI系统状态检查")
    print("=" * 50)

    # 检查状态文件
    state_file = Path("true_agi_system_state.json")
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            print("✅ 系统状态文件存在")
            print(f"   进化步数: {state.get('evolution_step', 'N/A')}")
            print(f"   活跃目标: {state.get('active_goals_count', 'N/A')}")
            print(f"   已完成目标: {state.get('completed_goals_count', 'N/A')}")

            if state.get('last_consciousness'):
                phi = state['last_consciousness'].get('integrated_information', 'N/A')
                print(f"   最后Φ值: {phi}")

        except Exception as e:
            print(f"❌ 状态文件读取失败: {e}")
    else:
        print("❌ 系统状态文件不存在")

    # 检查监控数据
    monitor_file = Path("agi_monitoring_data.jsonl")
    if monitor_file.exists():
        try:
            with open(monitor_file, 'r') as f:
                lines = f.readlines()

            print(f"\n✅ 监控数据文件存在 ({len(lines)} 条记录)")

            if lines:
                # 显示最新记录
                latest = json.loads(lines[-1])
                print("📊 最新监控数据:")
                print(f"   时间: {format_timestamp(latest.get('timestamp', 0))}")
                print(f"   进化步数: {latest.get('evolution_step', 'N/A')}")
                print(f"   知识库大小: {latest.get('knowledge_base_size', 'N/A')}")
                print(f"   经验缓冲区: {latest.get('experience_buffer_total', 'N/A')}")
                print(f"   活跃目标: {latest.get('active_goals_count', 'N/A')}")

                # 显示学习率
                lrs = latest.get('learning_rates', {})
                if lrs:
                    print("   学习率:")
                    print(".2e")
                    print(".2e")
                    print(".2e")
                    print(".2e")

                # 显示最近指标
                if 'recent_phi_mean' in latest:
                    print(".4f")
                    print(".4f")
                    print(".4f")
        except Exception as e:
            print(f"❌ 监控数据读取失败: {e}")
    else:
        print("❌ 监控数据文件不存在")

    # 检查日志文件
    log_file = Path("true_agi_evolution.log")
    if log_file.exists():
        size = log_file.stat().st_size / 1024  # KB
        print(f"✅ 日志文件存在 ({size:.1f} KB)")
    else:
        print("❌ 日志文件不存在")

    # 检查权重文件
    weight_files = list(Path(".").glob("*.pt")) + list(Path(".").glob("*.pth"))
    if weight_files:
        print(f"\n✅ 找到 {len(weight_files)} 个权重文件")
        for wf in sorted(weight_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            size = wf.stat().st_size / (1024 * 1024)  # MB
            mtime = format_timestamp(wf.stat().st_mtime)
            print(f"   {wf.name}: {size:.1f} MB ({mtime})")
    else:
        print("❌ 未找到权重文件")

    print("\n" + "=" * 50)
    print("🎯 状态检查完成")

if __name__ == "__main__":
    check_system_status()