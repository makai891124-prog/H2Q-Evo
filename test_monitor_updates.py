#!/usr/bin/env python3
"""
监控界面数据更新测试脚本
"""

import time
import json
from pathlib import Path

def test_data_updates():
    """测试数据更新"""
    print("🔍 测试监控数据更新频率...")

    status_file = Path("agi_unified_status.json")
    training_file = Path("realtime_training_status.json")

    if not status_file.exists():
        print("❌ 状态文件不存在")
        return

    print("📊 监控数据更新 (按Ctrl+C退出)...")
    print("时间戳 | 训练步骤 | 最佳损失 | CPU% | 内存%")
    print("-" * 50)

    last_step = 0
    update_count = 0

    try:
        while True:
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)

                training = status.get('training_status', {})
                env = status.get('environment', {})

                current_step = training.get('current_step', 0)
                best_loss = training.get('best_loss', 0)
                cpu_percent = env.get('cpu_percent', 0)
                memory_percent = env.get('memory_percent', 0)

                timestamp = status.get('timestamp', '').split('T')[1][:8] if 'T' in status.get('timestamp', '') else 'N/A'

                if current_step != last_step:
                    update_count += 1
                    print("4d"
                last_step = current_step

                time.sleep(1)  # 1秒检查一次

            except Exception as e:
                print(f"❌ 读取错误: {e}")
                time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n✅ 测试结束，共检测到 {update_count} 次数据更新")

def show_update_intervals():
    """显示更新间隔信息"""
    print("\n📋 监控系统更新间隔说明:")
    print("=" * 40)
    print("• 监控界面刷新频率: 每2秒")
    print("• 训练状态更新频率: 每1秒")
    print("• 状态文件读取: 实时")
    print("• 界面重绘: 每次循环")
    print()
    print("🔧 如果数据不更新，可能是:")
    print("  1. 训练进程未运行")
    print("  2. 状态文件写入失败")
    print("  3. 文件权限问题")
    print("  4. 磁盘空间不足")

if __name__ == "__main__":
    show_update_intervals()
    test_data_updates()