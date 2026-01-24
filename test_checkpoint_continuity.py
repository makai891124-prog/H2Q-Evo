#!/usr/bin/env python3
"""
断点续连功能测试脚本
"""

import time
import json
from pathlib import Path
from datetime import datetime

def test_checkpoint_continuity():
    """测试断点续连功能"""
    print("🔄 测试断点续连功能")
    print("=" * 40)

    checkpoint_file = Path("training_checkpoint.json")
    status_file = Path("agi_unified_status.json")

    if not checkpoint_file.exists():
        print("❌ 没有找到断点文件")
        return

    # 读取断点
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        checkpoint_step = checkpoint.get('current_step', 0)
        print(f"📊 断点记录的步骤: {checkpoint_step}")

    except Exception as e:
        print(f"❌ 读取断点失败: {e}")
        return

    # 等待几秒让训练继续
    print("⏳ 等待训练继续运行...")
    time.sleep(5)

    # 检查状态文件
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)

            current_step = status.get('training_status', {}).get('current_step', 0)
            print(f"📈 当前训练步骤: {current_step}")

            if current_step > checkpoint_step:
                print("✅ 断点续连成功: 训练从断点继续")
                print(f"   续连进度: {current_step - checkpoint_step} 步")
            elif current_step == checkpoint_step:
                print("⚠️  训练可能刚刚开始或暂停")
            else:
                print("❌ 断点续连可能失败")

        except Exception as e:
            print(f"❌ 读取状态失败: {e}")
    else:
        print("❌ 状态文件不存在")

def test_checkpoint_backup():
    """测试断点备份功能"""
    print("\n💾 测试断点备份功能")
    print("=" * 30)

    import subprocess
    import sys

    try:
        # 备份断点
        result = subprocess.run([sys.executable, "checkpoint_manager.py", "backup", "test_backup"],
                              capture_output=True, text=True, cwd=".")

        if result.returncode == 0:
            print("✅ 断点备份成功")
            print(result.stdout.strip())
        else:
            print("❌ 断点备份失败")
            print(result.stderr.strip())

        # 列出断点
        result = subprocess.run([sys.executable, "checkpoint_manager.py", "list"],
                              capture_output=True, text=True, cwd=".")

        if result.returncode == 0:
            print("📁 当前断点列表:")
            print(result.stdout.strip())

    except Exception as e:
        print(f"❌ 测试失败: {e}")

def show_checkpoint_features():
    """显示断点功能特性"""
    print("\n🎯 断点续连功能特性")
    print("=" * 35)
    print("• 自动断点保存: 每50步自动保存")
    print("• 程序中断保护: Ctrl+C时保存断点")
    print("• 原子性写入: 防止断点文件损坏")
    print("• 断点验证: 启动时验证断点完整性")
    print("• 备份管理: 支持多版本断点备份")
    print("• 统计信息: 记录训练时长和节流次数")
    print()
    print("📋 使用方法:")
    print("1. 正常启动训练: python3 memory_safe_training_launcher.py")
    print("2. 中断后恢复: 重新运行启动命令即可自动续连")
    print("3. 手动备份: python3 checkpoint_manager.py backup")
    print("4. 查看断点: python3 checkpoint_manager.py info")

if __name__ == "__main__":
    show_checkpoint_features()
    test_checkpoint_continuity()
    test_checkpoint_backup()

    print("\n✅ 断点续连功能测试完成")