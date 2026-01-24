#!/usr/bin/env python3
"""
H2Q-Evo AGI 自主系统集成器
整合所有组件：训练、监控、授权和自主操作
"""

import os
import sys
import time
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('agi_autonomous_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AGI-Autonomous')

class AGIAutonomousSystem:
    """AGI自主系统集成器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.authorization_manager = None
        self.training_system = None
        self.monitoring_system = None
        self.is_running = False

        # 加载配置
        self.load_configuration()

    def load_configuration(self):
        """加载系统配置"""
        config_file = self.project_root / "agi_autonomous_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "auto_start_training": True,
                "auto_start_monitoring": True,
                "health_check_interval": 30,
                "backup_interval": 3600,
                "max_training_steps": 10000,
                "resource_limits": {
                    "cpu_percent": 95,
                    "memory_percent": 90,
                    "disk_percent": 95
                }
            }
            self.save_configuration()

    def save_configuration(self):
        """保存系统配置"""
        config_file = self.project_root / "agi_autonomous_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def verify_authorization(self) -> bool:
        """验证AGI授权"""
        try:
            sys.path.insert(0, str(self.project_root))
            from agi_authorization import AGIAuthorizationManager

            self.authorization_manager = AGIAuthorizationManager()
            status = self.authorization_manager.get_authorization_status()

            if status.get('authorized', False):
                logger.info("✅ AGI系统授权验证成功")
                return True
            else:
                logger.error("❌ AGI系统未获得授权")
                return False

        except ImportError as e:
            logger.error(f"无法加载授权管理器: {e}")
            return False
        except Exception as e:
            logger.error(f"授权验证失败: {e}")
            return False

    def start_training_system(self):
        """启动训练系统"""
        try:
            logger.info("启动AGI训练系统...")

            # 导入训练模块
            from agi_realtime_training import H2QRealtimeTrainer

            # 创建训练器实例
            self.training_system = H2QRealtimeTrainer()

            # 在后台线程中启动训练
            training_thread = threading.Thread(
                target=self._run_training_loop,
                daemon=True
            )
            training_thread.start()

            logger.info("✅ AGI训练系统已启动")

        except Exception as e:
            logger.error(f"启动训练系统失败: {e}")

    def _run_training_loop(self):
        """运行训练循环"""
        try:
            if self.training_system:
                # 启动实时训练
                self.training_system.start_realtime_training()

                # 保持训练运行
                while self.is_running and self.training_system.running:
                    time.sleep(10)  # 每10秒检查一次

        except Exception as e:
            logger.error(f"训练循环异常: {e}")

    def start_monitoring_system(self):
        """启动监控系统"""
        try:
            logger.info("启动AGI监控系统...")

            # 导入监控模块
            from agi_monitor import AGIMonitor

            # 创建监控器实例
            self.monitoring_system = AGIMonitor()

            # 在后台线程中启动监控
            monitoring_thread = threading.Thread(
                target=self._run_monitoring_loop,
                daemon=True
            )
            monitoring_thread.start()

            logger.info("✅ AGI监控系统已启动")

        except Exception as e:
            logger.error(f"启动监控系统失败: {e}")

    def _run_monitoring_loop(self):
        """运行监控循环"""
        try:
            while self.is_running:
                if self.monitoring_system:
                    self.monitoring_system.update_display()
                time.sleep(self.config.get('health_check_interval', 30))
        except Exception as e:
            logger.error(f"监控循环异常: {e}")

    def start_health_monitoring(self):
        """启动健康监控"""
        try:
            logger.info("启动健康监控...")

            health_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            health_thread.start()

            logger.info("✅ 健康监控已启动")

        except Exception as e:
            logger.error(f"启动健康监控失败: {e}")

    def _health_check_loop(self):
        """健康检查循环"""
        last_backup = time.time()

        while self.is_running:
            try:
                # 检查系统资源
                if self.authorization_manager:
                    resources, violations = self.authorization_manager.monitor_system_resources()

                    if violations:
                        logger.warning("系统资源违规:")
                        for violation in violations:
                            logger.warning(f"  - {violation}")

                # 定期备份
                current_time = time.time()
                if current_time - last_backup > self.config.get('backup_interval', 3600):
                    self.perform_system_backup()
                    last_backup = current_time

                time.sleep(60)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"健康检查异常: {e}")
                time.sleep(60)

    def perform_system_backup(self):
        """执行系统备份"""
        try:
            if self.authorization_manager:
                backup_name = f"auto_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = self.authorization_manager.create_system_backup(backup_name)
                logger.info(f"系统自动备份已创建: {backup_path}")
        except Exception as e:
            logger.error(f"系统备份失败: {e}")

    def start_autonomous_operation(self):
        """启动自主操作"""
        logger.info("🚀 启动AGI自主操作系统")

        if not self.verify_authorization():
            logger.error("无法启动：AGI系统未获得授权")
            return False

        self.is_running = True

        # 启动各个子系统
        if self.config.get('auto_start_training', True):
            self.start_training_system()

        if self.config.get('auto_start_monitoring', True):
            self.start_monitoring_system()

        # 启动健康监控
        self.start_health_monitoring()

        logger.info("🎉 AGI自主系统已完全启动并运行")
        logger.info("📊 系统状态：")
        logger.info("  - 训练系统：运行中" if self.training_system else "  - 训练系统：未启动")
        logger.info("  - 监控系统：运行中" if self.monitoring_system else "  - 监控系统：未启动")
        logger.info("  - 健康监控：运行中")
        logger.info("  - 授权状态：已验证")

        return True

    def stop_autonomous_operation(self):
        """停止自主操作"""
        logger.info("🛑 正在停止AGI自主系统...")

        self.is_running = False

        # 等待线程结束
        time.sleep(2)

        logger.info("✅ AGI自主系统已停止")

    def get_system_status(self) -> dict:
        """获取系统状态"""
        status = {
            'is_running': self.is_running,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'training': self.training_system is not None,
                'monitoring': self.monitoring_system is not None,
                'authorization': self.authorization_manager is not None
            },
            'config': self.config
        }

        # 获取授权状态
        if self.authorization_manager:
            auth_status = self.authorization_manager.get_authorization_status()
            status['authorization'] = {
                'authorized': auth_status.get('authorized', False),
                'granted_at': auth_status.get('granted_at'),
                'granted_by': auth_status.get('granted_by')
            }

        # 获取资源状态
        if self.authorization_manager:
            try:
                resources = self.authorization_manager.get_system_resources()
                status['resources'] = resources
            except Exception as e:
                status['resources'] = {'error': str(e)}

        return status

def main():
    """主函数"""
    print("H2Q-Evo AGI 自主系统集成器")
    print("=" * 50)

    system = AGIAutonomousSystem()

    try:
        # 启动自主系统
        if system.start_autonomous_operation():
            print("\n🎯 AGI自主系统运行中...")
            print("系统将在后台持续运行")
            print("使用 Ctrl+C 停止系统")

            # 在主线程中保持运行状态检查
            while system.is_running:
                time.sleep(5)
                # 每5秒打印一次状态摘要
                status = system.get_system_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 系统运行中 - 训练:{status['components']['training']} 监控:{status['components']['monitoring']}")

        else:
            print("❌ AGI自主系统启动失败")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 收到停止信号，正在关闭系统...")
        system.stop_autonomous_operation()
        print("✅ 系统已安全关闭")

    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        system.stop_autonomous_operation()
        sys.exit(1)

if __name__ == "__main__":
    main()