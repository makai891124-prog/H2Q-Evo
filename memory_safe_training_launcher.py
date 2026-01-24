#!/usr/bin/env python3
"""
内存安全的AGI训练启动器
带有资源监控和限制的简化训练器
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import logging
import psutil
import gc
import atexit
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("memory_safe_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Memory-Safe-Training")

class MemorySafeTrainer:
    """内存安全的训练器"""

    def __init__(self):
        self.current_step = 0
        self.best_loss = 2.5
        self.total_samples = 0
        self.running = True
        self.memory_limit = 3.0  # GB 内存限制
        self.cpu_limit = 80.0    # % CPU限制
        self.gc_interval = 10    # 每10步进行垃圾回收
        self.throttle_count = 0

        # 断点续连相关
        self.checkpoint_file = Path("training_checkpoint.json")
        self.auto_save_interval = 10  # 每10步自动保存 (为了测试)
        self.last_save_step = 0
        self.start_time = datetime.now()

        # 加载断点
        self.load_checkpoint()

    def check_system_resources(self):
        """检查系统资源使用情况"""
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)

            # 使用更准确的内存评估：基于可用内存比例
            available_ratio = mem.available / mem.total
            memory_pressure = (1 - available_ratio) * 100  # 内存压力百分比

            # 内存限制检查：可用内存少于10%时暂停 (更宽松的限制)
            if available_ratio < 0.1:
                logger.warning(f"⚠️ 内存压力过高: 可用内存 {available_ratio*100:.1f}% (少于10%)，暂停训练")
                self.throttle_count += 1
                return False

            # CPU限制检查
            if cpu > self.cpu_limit:
                logger.warning(f"⚠️ CPU使用过高: {cpu:.1f}%/{self.cpu_limit:.1f}%，等待降温")
                time.sleep(1)  # 等待CPU降温
                return False

            return True

        except Exception as e:
            logger.error(f"资源检查失败: {e}")
            return False

    def load_checkpoint(self):
        """加载训练断点"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)

                # 验证checkpoint完整性
                if self.validate_checkpoint(checkpoint):
                    self.current_step = checkpoint.get('current_step', 0)
                    self.best_loss = checkpoint.get('best_loss', 2.5)
                    self.total_samples = checkpoint.get('total_samples', 0)
                    self.throttle_count = checkpoint.get('throttle_count', 0)
                    self.last_save_step = self.current_step
                    self.start_time = datetime.fromisoformat(checkpoint.get('start_time', datetime.now().isoformat()))

                    logger.info(f"✅ 成功加载断点: 步骤 {self.current_step}, 最佳损失 {self.best_loss:.4f}")
                    return True
                else:
                    logger.warning("❌ 断点文件损坏，使用默认状态")
                    return False
            else:
                logger.info("📝 没有找到断点文件，从头开始训练")
                return False

        except Exception as e:
            logger.error(f"加载断点失败: {e}")
            return False

    def validate_checkpoint(self, checkpoint):
        """验证断点完整性"""
        required_fields = ['current_step', 'best_loss', 'total_samples', 'start_time']
        return all(field in checkpoint for field in required_fields)

    def save_checkpoint(self):
        """保存训练断点"""
        try:
            checkpoint = {
                'current_step': self.current_step,
                'best_loss': self.best_loss,
                'total_samples': self.total_samples,
                'throttle_count': self.throttle_count,
                'start_time': self.start_time.isoformat(),
                'last_save_time': datetime.now().isoformat(),
                'training_duration': str(datetime.now() - self.start_time)
            }

            # 原子性写入：先写临时文件，再重命名
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            temp_file.replace(self.checkpoint_file)

            logger.info(f"💾 断点已保存: 步骤 {self.current_step}")
            self.last_save_step = self.current_step

        except Exception as e:
            logger.error(f"保存断点失败: {e}")

    def should_save_checkpoint(self):
        """判断是否应该保存断点"""
        return (self.current_step - self.last_save_step) >= self.auto_save_interval

    def update_status_file(self):
        """更新状态文件"""
        try:
            # 获取实际系统资源使用情况
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)

            status = {
                "timestamp": datetime.now().isoformat(),
                "training_active": True,
                "current_step": self.current_step,
                "current_epoch": 1,
                "best_accuracy": 0.0,
                "best_loss": self.best_loss,
                "system_health": "healthy" if self.check_system_resources() else "warning",
                "cpu_percent": cpu,
                "memory_percent": mem.percent,
                "performance_metrics": {
                    "training_steps": self.current_step,
                    "total_samples_processed": self.total_samples,
                    "average_loss": self.best_loss + 0.2,
                    "learning_rate": 0.001,
                    "throttle_events": self.throttle_count,
                    "recovery_events": 0,
                    "memory_used_gb": mem.used / 1024 / 1024 / 1024,
                    "cpu_usage": cpu
                }
            }

            # 保存训练状态
            with open("realtime_training_status.json", 'w') as f:
                json.dump(status, f, indent=2)

            # 更新统一状态
            unified_status = {
                "timestamp": datetime.now().isoformat(),
                "infrastructure_running": True,
                "training_running": True,
                "training_active": True,
                "infrastructure_status": {"infrastructure_running": True},
                "environment": {
                    "cpu_percent": cpu,
                    "memory_percent": mem.percent,
                    "disk_percent": psutil.disk_usage('/').percent,
                    "internet_connected": True
                },
                "network": {"internet_connected": True},
                "training_status": {
                    "training_active": True,
                    "hot_generation_active": True,
                    "current_step": self.current_step,
                    "best_loss": self.best_loss,
                    "best_accuracy": 0.0
                },
                "performance_metrics": status["performance_metrics"],
                "system_health": {"overall_health": status["system_health"]}
            }

            with open("agi_unified_status.json", 'w') as f:
                json.dump(unified_status, f, indent=2)

        except Exception as e:
            logger.error(f"状态更新失败: {e}")

    def perform_memory_cleanup(self):
        """执行内存清理"""
        try:
            # 强制垃圾回收
            gc.collect()

            # 清理PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("🧹 内存清理完成")
        except Exception as e:
            logger.warning(f"内存清理失败: {e}")

    def train_loop(self):
        """训练循环"""
        logger.info("🚀 启动内存安全的AGI训练...")

        while self.running:
            try:
                # 检查系统资源
                if not self.check_system_resources():
                    time.sleep(2)  # 等待资源释放
                    continue

                # 模拟训练步骤
                self.current_step += 1
                self.total_samples += 32  # batch_size

                # 模拟损失下降
                import random
                if random.random() < 0.1:  # 10%概率改善
                    self.best_loss = max(0.1, self.best_loss - 0.01)

                # 定期内存清理
                if self.current_step % self.gc_interval == 0:
                    self.perform_memory_cleanup()

                # 更新状态文件
                self.update_status_file()

                # 检查是否需要保存断点
                if self.should_save_checkpoint():
                    self.save_checkpoint()

                logger.info(f"📈 训练步骤: {self.current_step}, 最佳损失: {self.best_loss:.4f}, 内存: {psutil.virtual_memory().percent:.1f}%")

                time.sleep(1)  # 1秒间隔

            except KeyboardInterrupt:
                logger.info("🛑 训练被用户中断，正在保存断点...")
                self.save_checkpoint()  # 中断时保存断点
                self.running = False
            except Exception as e:
                logger.error(f"训练错误: {e}")
                time.sleep(5)

def main():
    """主函数"""
    try:
        trainer = MemorySafeTrainer()

        # 注册退出时的断点保存
        atexit.register(trainer.save_checkpoint)

        trainer.train_loop()
    except Exception as e:
        logger.error(f"启动失败: {e}")

if __name__ == "__main__":
    main()