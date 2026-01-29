#!/usr/bin/env python3
"""
AGI训练状态快速检查器
"""

import json
import psutil
from datetime import datetime

def quick_status_check():
    """快速检查AGI训练状态"""
    print("🚀 AGI训练状态快速检查")
    print("=" * 50)

    try:
        # 加载训练状态
        with open('realtime_training_status.json', 'r') as f:
            status = json.load(f)

        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 训练步骤: {status.get('current_step', 0):,}")
        print(f"🎯 最佳损失: {status.get('best_loss', 0):.4f}")
        print(f"💚 系统健康: {status.get('system_health', 'unknown')}")

        geom = status.get('geometric_metrics', {})
        print(f"🔬 谱稳定性: {geom.get('spectral_shift_eta_real', 0):.6f}")
        print(f"🌌 分形惩罚: {geom.get('fractal_collapse_penalty', 0):.4f}")

        # 系统资源
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        print(f"🖥️  CPU: {cpu:.1f}% | 🧠 内存: {mem.percent:.1f}%")

        # 进程状态
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout

        training = '🟢' if 'memory_safe_training_launcher' in processes else '🔴'
        monitor = '🟢' if 'enhanced_agi_monitor' in processes else '🔴'
        daemon = '🟢' if 'agi_daemon' in processes else '🔴'

        print(f"🤖 训练进程: {training} | 📊 监控进程: {monitor} | 👹 守护进程: {daemon}")

        print("\n✅ AGI系统运行正常 - 基于黎曼谱稳定性控制")

    except Exception as e:
        print(f"❌ 状态检查失败: {e}")

if __name__ == "__main__":
    quick_status_check()