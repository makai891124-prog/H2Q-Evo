#!/usr/bin/env python3
"""
详细监控数据更新间隔测试
"""

import time
import json
from pathlib import Path
from datetime import datetime

def detailed_update_test():
    """详细的更新测试"""
    print("🔬 详细监控数据更新测试")
    print("=" * 50)

    status_file = Path("agi_unified_status.json")

    if not status_file.exists():
        print("❌ 状态文件不存在")
        return

    print("📊 实时监控数据更新情况:")
    print("时间 | 训练步骤 | 最佳损失 | CPU% | 内存% | 更新间隔")
    print("-" * 60)

    last_update_time = time.time()
    last_step = 0
    update_intervals = []

    try:
        for i in range(20):  # 测试20次
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)

                training = status.get('training_status', {})
                env = status.get('environment', {})

                current_step = training.get('current_step', 0)
                best_loss = training.get('best_loss', 0)
                cpu_percent = env.get('cpu_percent', 0)
                memory_percent = env.get('memory_percent', 0)

                current_time = time.time()
                interval = current_time - last_update_time
                update_intervals.append(interval)
                last_update_time = current_time

                timestamp = datetime.now().strftime("%H:%M:%S")

                step_changed = "🔄" if current_step != last_step else "  "
                print("6.3f"
                last_step = current_step

            except Exception as e:
                print(f"{datetime.now().strftime('%H:%M:%S')} | ❌ 错误: {e}")

            time.sleep(1)  # 1秒间隔

    except KeyboardInterrupt:
        pass

    # 统计结果
    if update_intervals:
        avg_interval = sum(update_intervals[1:]) / len(update_intervals[1:])  # 排除第一次
        min_interval = min(update_intervals[1:])
        max_interval = max(update_intervals[1:])

        print("\n📈 统计结果:")
        print(".3f"        print(".3f"        print(".3f"
    print("\n✅ 测试完成")

def show_monitor_config():
    """显示监控配置"""
    print("\n⚙️  监控系统配置:")
    print("=" * 30)
    print("• 监控界面更新间隔: 2秒")
    print("• 训练进程更新间隔: 1秒")
    print("• 状态文件: agi_unified_status.json")
    print("• 监控进程: agi_monitor.py")
    print()
    print("🔧 故障排除:")
    print("1. 检查训练进程是否运行: ps aux | grep memory_safe_training")
    print("2. 检查状态文件更新: stat agi_unified_status.json")
    print("3. 检查日志: tail -f memory_safe_training.log")

if __name__ == "__main__":
    show_monitor_config()
    detailed_update_test()