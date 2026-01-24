#!/usr/bin/env python3
"""
统一AGI系统状态监控脚本
整合所有状态文件，确保监控界面显示正确信息
"""

import os
import json
import time
import psutil
import socket
from pathlib import Path
from datetime import datetime

def check_network_connectivity():
    """检查网络连接状态"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except:
        return False

def get_system_info():
    """获取系统信息"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "internet_connected": check_network_connectivity()
    }

def update_unified_status():
    """更新统一状态文件"""
    status_dir = Path(".")

    # 读取现有状态文件
    system_status = {}
    training_status = {}
    system_report = {}

    # 读取agi_system_status.json
    status_file = status_dir / "agi_system_status.json"
    if status_file.exists():
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                system_status = json.load(f)
        except:
            pass

    # 读取realtime_training_status.json
    training_file = status_dir / "realtime_training_status.json"
    if training_file.exists():
        try:
            with open(training_file, 'r', encoding='utf-8') as f:
                training_status = json.load(f)
        except:
            pass

    # 读取agi_system_report.json
    report_file = status_dir / "agi_system_report.json"
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                system_report = json.load(f)
        except:
            pass

    # 获取当前系统信息
    current_system = get_system_info()

    # 检查是否有训练进程在运行
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('agi_realtime_training' in str(cmd) for cmd in proc.info['cmdline']):
                training_processes.append(proc.info['pid'])
        except:
            pass

    infrastructure_running = len(training_processes) > 0
    training_active = len(training_processes) > 0

    # 构建统一状态
    unified_status = {
        "timestamp": datetime.now().isoformat(),
        "infrastructure_running": infrastructure_running,
        "training_running": training_active,
        "training_active": training_active,
        "infrastructure_status": {
            "infrastructure_running": infrastructure_running
        },
        "environment": current_system,
        "network": {
            "internet_connected": current_system["internet_connected"]
        },
        "training_status": {
            "training_active": training_active,
            "hot_generation_active": training_active,
            "current_step": training_status.get('current_step', system_status.get('current_step', 0)),
            "best_loss": training_status.get('best_loss', system_status.get('best_loss', float('inf'))),
            "best_accuracy": training_status.get('best_accuracy', 0.0)
        },
        "performance_metrics": training_status.get('performance_metrics', {}),
        "system_health": {
            "overall_health": "healthy" if infrastructure_running else "warning"
        }
    }

    # 保存统一状态文件
    unified_file = status_dir / "agi_unified_status.json"
    with open(unified_file, 'w', encoding='utf-8') as f:
        json.dump(unified_status, f, indent=2, ensure_ascii=False)

    print(f"✅ 统一状态已更新: 基础设施={'运行中' if infrastructure_running else '已停止'}, 训练={'运行中' if training_active else '已停止'}")
    return unified_status

if __name__ == "__main__":
    update_unified_status()