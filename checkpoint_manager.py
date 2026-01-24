#!/usr/bin/env python3
"""
训练断点管理器
用于管理训练断点、备份和恢复
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Checkpoint-Manager")

class CheckpointManager:
    """断点管理器"""

    def __init__(self):
        self.checkpoint_file = Path("training_checkpoint.json")
        self.backup_dir = Path("training_checkpoints")
        self.backup_dir.mkdir(exist_ok=True)

    def list_checkpoints(self):
        """列出所有断点"""
        checkpoints = []
        if self.checkpoint_file.exists():
            checkpoints.append(("current", self.checkpoint_file))

        # 列出备份断点
        for backup_file in self.backup_dir.glob("training_checkpoint_*.json"):
            timestamp = backup_file.stem.split("_")[-1]
            checkpoints.append((timestamp, backup_file))

        return checkpoints

    def show_checkpoint_info(self, checkpoint_path):
        """显示断点信息"""
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            print(f"📁 断点文件: {checkpoint_path}")
            print(f"📊 当前步骤: {data.get('current_step', 0)}")
            print(f"🎯 最佳损失: {data.get('best_loss', 0):.4f}")
            print(f"📈 总样本数: {data.get('total_samples', 0)}")
            print(f"⏱️  训练时长: {data.get('training_duration', 'N/A')}")
            print(f"💾 保存时间: {data.get('last_save_time', 'N/A')}")
            print(f"🛑 节流次数: {data.get('throttle_count', 0)}")

        except Exception as e:
            print(f"❌ 读取断点失败: {e}")

    def backup_checkpoint(self, name=None):
        """备份当前断点"""
        if not self.checkpoint_file.exists():
            print("❌ 没有找到当前断点文件")
            return False

        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_file = self.backup_dir / f"training_checkpoint_{name}.json"

        try:
            shutil.copy2(self.checkpoint_file, backup_file)
            print(f"✅ 断点已备份到: {backup_file}")
            return True
        except Exception as e:
            print(f"❌ 备份失败: {e}")
            return False

    def restore_checkpoint(self, checkpoint_path):
        """恢复断点"""
        try:
            if not Path(checkpoint_path).exists():
                print(f"❌ 断点文件不存在: {checkpoint_path}")
                return False

            shutil.copy2(checkpoint_path, self.checkpoint_file)
            print(f"✅ 断点已恢复: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"❌ 恢复失败: {e}")
            return False

    def clean_old_checkpoints(self, keep_recent=5):
        """清理旧的断点文件"""
        backup_files = sorted(self.backup_dir.glob("training_checkpoint_*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)

        if len(backup_files) <= keep_recent:
            print(f"ℹ️  没有需要清理的断点文件 (保留最近 {keep_recent} 个)")
            return

        files_to_remove = backup_files[keep_recent:]
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                print(f"🗑️  已删除旧断点: {file_path}")
            except Exception as e:
                print(f"❌ 删除失败 {file_path}: {e}")

    def get_checkpoint_stats(self):
        """获取断点统计信息"""
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            print("❌ 没有找到任何断点文件")
            return

        print(f"📊 断点统计: 共 {len(checkpoints)} 个断点")
        print("-" * 50)

        for name, path in checkpoints:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                step = data.get('current_step', 0)
                loss = data.get('best_loss', 0)
                duration = data.get('training_duration', 'N/A')

                print("12s")
            except Exception as e:
                print("12s")

def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python3 checkpoint_manager.py <命令> [参数]")
        print("\n命令:")
        print("  list          - 列出所有断点")
        print("  info [name]   - 显示断点信息 (默认current)")
        print("  backup [name] - 备份当前断点")
        print("  restore <path>- 恢复断点")
        print("  clean [num]   - 清理旧断点 (默认保留5个)")
        print("  stats         - 显示断点统计")
        return

    manager = CheckpointManager()
    command = sys.argv[1]

    try:
        if command == "list":
            checkpoints = manager.list_checkpoints()
            print("📁 可用的断点:")
            for name, path in checkpoints:
                print(f"  {name}: {path}")

        elif command == "info":
            name = sys.argv[2] if len(sys.argv) > 2 else "current"
            checkpoints = manager.list_checkpoints()
            checkpoint_path = None

            for n, path in checkpoints:
                if n == name:
                    checkpoint_path = path
                    break

            if checkpoint_path:
                manager.show_checkpoint_info(checkpoint_path)
            else:
                print(f"❌ 找不到断点: {name}")

        elif command == "backup":
            name = sys.argv[2] if len(sys.argv) > 2 else None
            manager.backup_checkpoint(name)

        elif command == "restore":
            if len(sys.argv) < 3:
                print("❌ 请指定要恢复的断点路径")
                return
            manager.restore_checkpoint(sys.argv[2])

        elif command == "clean":
            keep = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            manager.clean_old_checkpoints(keep)

        elif command == "stats":
            manager.get_checkpoint_stats()

        else:
            print(f"❌ 未知命令: {command}")

    except Exception as e:
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()