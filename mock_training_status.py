#!/usr/bin/env python3
"""
模拟AGI训练状态生成器
用于测试监控界面和生成状态数据
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime

class MockTrainingStatus:
    """模拟训练状态"""

    def __init__(self):
        self.current_step = 0
        self.best_loss = 2.5
        self.best_accuracy = 0.0
        self.total_samples = 0

    def update_status(self):
        """更新训练状态"""
        self.current_step += random.randint(1, 5)
        self.total_samples += random.randint(100, 500)

        # 模拟损失下降
        if random.random() < 0.3:
            self.best_loss = max(0.1, self.best_loss - random.uniform(0.01, 0.1))

        # 模拟准确率提升
        if random.random() < 0.2:
            self.best_accuracy = min(0.95, self.best_accuracy + random.uniform(0.001, 0.01))

        return {
            "timestamp": datetime.now().isoformat(),
            "training_active": True,
            "current_step": self.current_step,
            "current_epoch": 1,
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "system_health": "healthy",
            "cpu_percent": random.uniform(10, 60),
            "memory_percent": random.uniform(70, 85),
            "performance_metrics": {
                "training_steps": self.current_step,
                "total_samples_processed": self.total_samples,
                "average_loss": self.best_loss + random.uniform(0.1, 0.5),
                "learning_rate": 0.001,
                "throttle_events": random.randint(0, 10),
                "recovery_events": 0
            }
        }

def generate_mock_status():
    """生成模拟状态"""
    status_dir = Path(".")
    mock_trainer = MockTrainingStatus()

    print("🎭 启动模拟训练状态生成器...")

    try:
        while True:
            # 生成训练状态
            training_status = mock_trainer.update_status()

            # 保存到文件
            training_file = status_dir / "realtime_training_status.json"
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_status, f, indent=2, ensure_ascii=False)

            # 更新统一状态
            unified_status = {
                "timestamp": datetime.now().isoformat(),
                "infrastructure_running": True,
                "training_running": True,
                "training_active": True,
                "infrastructure_status": {
                    "infrastructure_running": True
                },
                "environment": {
                    "cpu_percent": training_status["cpu_percent"],
                    "memory_percent": training_status["memory_percent"],
                    "disk_percent": 15.0,
                    "internet_connected": True
                },
                "network": {
                    "internet_connected": True
                },
                "training_status": {
                    "training_active": True,
                    "hot_generation_active": True,
                    "current_step": training_status["current_step"],
                    "best_loss": training_status["best_loss"],
                    "best_accuracy": training_status["best_accuracy"]
                },
                "performance_metrics": training_status["performance_metrics"],
                "system_health": {
                    "overall_health": "healthy"
                }
            }

            unified_file = status_dir / "agi_unified_status.json"
            with open(unified_file, 'w', encoding='utf-8') as f:
                json.dump(unified_status, f, indent=2, ensure_ascii=False)

            print(f"📊 状态已更新: 步骤 {training_status['current_step']}, 损失 {training_status['best_loss']:.4f}")

            time.sleep(2)  # 每2秒更新一次

    except KeyboardInterrupt:
        print("\n🛑 模拟训练状态生成器已停止")

if __name__ == "__main__":
    generate_mock_status()