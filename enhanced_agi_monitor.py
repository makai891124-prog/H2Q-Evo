#!/usr/bin/env python3
"""
增强AGI训练监控系统
显示高级谱稳定性控制器的实时指标
"""

import json
import time
import os
import sys
from datetime import datetime
import psutil

def get_system_stats():
    """获取系统统计信息"""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            'cpu_percent': cpu,
            'memory_percent': mem.percent,
            'disk_percent': disk.percent,
            'memory_used_gb': mem.used / (1024**3)
        }
    except:
        return {'cpu_percent': 0, 'memory_percent': 0, 'disk_percent': 0, 'memory_used_gb': 0}

def load_training_status():
    """加载训练状态"""
    try:
        with open('realtime_training_status.json', 'r') as f:
            return json.load(f)
    except:
        return None

def load_checkpoint():
    """加载断点信息"""
    try:
        with open('training_checkpoint.json', 'r') as f:
            return json.load(f)
    except:
        return None

def display_enhanced_monitoring():
    """显示增强监控界面"""
    print("\n" + "="*100)
    print("🎯 H2Q-Evo 增强AGI训练监控系统 - 基于黎曼谱稳定性控制")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    # 系统状态
    sys_stats = get_system_stats()
    print("\n🔧 系统状态 / System Status")
    print(f"🖥️  CPU使用率: {sys_stats['cpu_percent']:.1f}%")
    print(f"🧠 内存使用率: {sys_stats['memory_percent']:.1f}%")
    print(f"💾 磁盘使用率: {sys_stats['disk_percent']:.1f}%")
    print(f"📊 内存使用量: {sys_stats['memory_used_gb']:.2f} GB")

    # 训练状态
    status = load_training_status()
    checkpoint = load_checkpoint()

    if status:
        print("\n🎯 实时训练状态 / Real Training Status")
        print(f"📊 训练步骤: {status.get('current_step', 0):,}")
        print(f"🎯 最佳损失: {status.get('best_loss', 0):.4f}")
        print(f"🎯 最佳准确率: {status.get('best_accuracy', 0):.4f}")
        print(f"💚 系统健康: {status.get('system_health', 'unknown')}")

        geom = status.get('geometric_metrics', {})
        print("\n📈 几何指标 / Geometric Metrics")
        print(f"🔬 谱移η实部: {geom.get('spectral_shift_eta_real', 0):.6f}")
        print(f"🌌 分形坍缩惩罚: {geom.get('fractal_collapse_penalty', 0):.6f}")
        print(f"📐 几何准确率: {geom.get('geometric_accuracy', 0):.6f}")
        print(f"🎯 分类F1分数: {geom.get('classification_f1', 0):.6f}")

        perf = status.get('performance_metrics', {})
        print("\n⚡ 性能指标 / Performance Metrics")
        print(f"🧠 总样本数: {perf.get('total_samples_processed', 0):,}")
        print(f"📉 平均损失: {perf.get('average_loss', 0):.4f}")
        print(f"🎓 流形稳定性: {perf.get('manifold_stability', 0):.4f}")
        print(f"🧹 节流事件: {perf.get('throttle_events', 0)}")

    if checkpoint:
        print("\n💾 断点状态 / Checkpoint Status")
        print(f"📍 断点步骤: {checkpoint.get('current_step', 0):,}")
        print(f"💰 断点损失: {checkpoint.get('best_loss', 0):.4f}")
        print(f"🎯 断点准确率: {checkpoint.get('best_accuracy', 0):.4f}")

    # 检查进程状态
    print("\n🔄 进程状态 / Process Status")
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout

        training_running = 'memory_safe_training_launcher' in processes
        monitor_running = 'agi_monitor' in processes or 'enhanced_agi_monitor' in processes
        daemon_running = 'agi_daemon' in processes

        print(f"🤖 训练进程: {'🟢 运行中' if training_running else '🔴 未运行'}")
        print(f"📊 监控进程: {'🟢 运行中' if monitor_running else '🔴 未运行'}")
        print(f"👹 守护进程: {'🟢 运行中' if daemon_running else '🔴 未运行'}")

    except:
        print("❌ 无法检查进程状态")

    # AGI目标状态
    print("\n🎯 AGI目标状态 / AGI Targets Status")
    if status and status.get('training_active', False):
        print("🚀 AGI训练: 🟢 活跃进行中")
        print("🧠 高级谱控制: 🟢 已激活 (黎曼猜想基础)")
        print("📈 谱稳定性: 🔄 动态优化中")
        print("🎯 目标达成: ⏳ 持续进化中")
    else:
        print("🚀 AGI训练: 🔴 未激活")
        print("🧠 高级谱控制: 🔴 未激活")
        print("📈 谱稳定性: ❓ 未知")
        print("🎯 目标达成: ⏸️ 等待启动")

    print("\n" + "="*100)

def main():
    """主监控循环"""
    print("🎯 启动增强AGI训练监控系统...")
    print("💡 按 Ctrl+C 退出监控")

    try:
        while True:
            # 清除屏幕
            os.system('clear' if os.name == 'posix' else 'cls')

            # 显示监控信息
            display_enhanced_monitoring()

            # 等待一段时间
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n👋 监控系统已停止")
    except Exception as e:
        print(f"\n❌ 监控系统错误: {e}")

if __name__ == "__main__":
    main()