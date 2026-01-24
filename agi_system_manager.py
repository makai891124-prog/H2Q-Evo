#!/usr/bin/env python3
"""
H2Q-Evo AGI系统集成管理器
统一管理所有AGI组件的启动、监控和协调
"""

import os
import sys
import json
import time
import logging
import subprocess
import signal
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import argparse
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入AGI组件
try:
    from agi_persistent_evolution import PersistentAGITrainer
    from agi_training_monitor import AGITrainingMonitor
    from agi_data_generator import AGIDataGenerator
    from agi_evolution_monitor import AGIEvolutionMonitor
    from agi_manifold_encoder import LogarithmicManifoldEncoder
except ImportError as e:
    print(f"⚠️  警告: 无法导入AGI组件: {e}")
    print("请确保所有AGI组件文件都在同一目录中")

logger = logging.getLogger('AGI-SystemManager')

class AGISystemManager:
    """AGI系统管理器"""

    def __init__(self, config_path: str = "./agi_training_config.ini"):
        self.config_path = config_path
        self.config = self._load_config()

        # 系统组件
        self.trainer = None
        self.monitor = None
        self.data_generator = None
        self.evolution_monitor = None

        # 进程管理
        self.processes = {}
        self.threads = {}

        # 系统状态
        self.is_running = False
        self.start_time = None

        # 工作目录
        self.working_dir = Path("./agi_persistent_training")
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # 日志设置
        self._setup_logging()

        logger.info("AGI系统管理器初始化完成")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        config = {}

        if os.path.exists(self.config_path):
            try:
                import configparser
                parser = configparser.ConfigParser()
                parser.read(self.config_path)

                # 读取所有配置
                for section in parser.sections():
                    config[section] = dict(parser[section])

            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")

        # 默认配置
        config.setdefault('system', {})
        config['system'].setdefault('auto_restart', 'true')
        config['system'].setdefault('max_restarts', '3')
        config['system'].setdefault('health_check_interval', '30')

        config.setdefault('training', {})
        config['training'].setdefault('enabled', 'true')
        config['training'].setdefault('batch_size', '8')
        config['training'].setdefault('learning_rate', '0.001')

        config.setdefault('monitoring', {})
        config['monitoring'].setdefault('enabled', 'true')
        config['monitoring'].setdefault('update_interval', '5')

        config.setdefault('data_generation', {})
        config['data_generation'].setdefault('enabled', 'true')
        config['data_generation'].setdefault('generation_interval', '3600')  # 1小时

        return config

    def _setup_logging(self):
        """设置日志"""
        log_dir = self.working_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"agi_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def start_system(self) -> bool:
        """启动AGI系统"""
        if self.is_running:
            logger.warning("AGI系统已在运行")
            return False

        logger.info("🚀 启动H2Q-Evo AGI系统...")

        try:
            self.start_time = datetime.now()
            self.is_running = True

            # 启动各个组件
            self._start_components()

            # 启动系统监控
            self._start_system_monitoring()

            # 启动健康检查
            self._start_health_monitoring()

            logger.info("✅ AGI系统启动成功")
            return True

        except Exception as e:
            logger.error(f"AGI系统启动失败: {e}")
            self.stop_system()
            return False

    def stop_system(self):
        """停止AGI系统"""
        if not self.is_running:
            logger.info("AGI系统未在运行")
            return

        logger.info("🛑 停止AGI系统...")

        self.is_running = False

        # 停止所有组件
        self._stop_components()

        # 停止所有进程和线程
        self._cleanup_processes()

        logger.info("✅ AGI系统已停止")

    def _start_components(self):
        """启动各个组件"""
        logger.info("启动系统组件...")

        # 启动进化监控器
        if self.config.get('monitoring', {}).get('enabled', 'true').lower() == 'true':
            try:
                self.evolution_monitor = AGIEvolutionMonitor(self.config_path)
                self.evolution_monitor.start_monitoring(background=True)
                logger.info("✅ 进化监控器启动成功")
            except Exception as e:
                logger.error(f"进化监控器启动失败: {e}")

        # 启动训练监控器
        if self.config.get('training', {}).get('enabled', 'true').lower() == 'true':
            try:
                self.monitor = AGITrainingMonitor(self.config_path)
                self.monitor.start_monitoring()
                logger.info("✅ 训练监控器启动成功")
            except Exception as e:
                logger.error(f"训练监控器启动失败: {e}")

        # 启动数据生成器 (定时任务)
        if self.config.get('data_generation', {}).get('enabled', 'true').lower() == 'true':
            try:
                self.data_generator = AGIDataGenerator(self.config_path)
                self._start_data_generation_scheduler()
                logger.info("✅ 数据生成器启动成功")
            except Exception as e:
                logger.error(f"数据生成器启动失败: {e}")

        # 启动持久训练器
        if self.config.get('training', {}).get('enabled', 'true').lower() == 'true':
            try:
                # 创建PersistentAGIConfig对象
                from agi_persistent_evolution import PersistentAGIConfig
                trainer_config = PersistentAGIConfig()
                # 可以在这里根据self.config调整trainer_config

                self.trainer = PersistentAGITrainer(trainer_config)

                # 启动持久化训练
                self.trainer.start_persistent_training()

                trainer_thread = threading.Thread(target=self._run_trainer, daemon=True)
                trainer_thread.start()
                self.threads['trainer'] = trainer_thread
                logger.info("✅ 持久训练器启动成功")
            except Exception as e:
                logger.error(f"持久训练器启动失败: {e}")

    def _stop_components(self):
        """停止各个组件"""
        logger.info("停止系统组件...")

        # 停止训练器
        if self.trainer:
            try:
                self.trainer.stop_training()
                logger.info("✅ 持久训练器已停止")
            except Exception as e:
                logger.error(f"停止持久训练器失败: {e}")

        # 停止监控器
        if self.monitor:
            try:
                self.monitor.stop_monitoring()
                logger.info("✅ 训练监控器已停止")
            except Exception as e:
                logger.error(f"停止训练监控器失败: {e}")

        # 停止进化监控器
        if self.evolution_monitor:
            try:
                self.evolution_monitor.stop_monitoring()
                logger.info("✅ 进化监控器已停止")
            except Exception as e:
                logger.error(f"停止进化监控器失败: {e}")

    def _run_trainer(self):
        """运行训练器"""
        try:
            while self.is_running:
                if self.trainer:
                    self.trainer.run_training_cycle()
                time.sleep(1)  # 短暂休眠避免CPU占用过高
        except Exception as e:
            logger.error(f"训练器运行错误: {e}")

    def _start_data_generation_scheduler(self):
        """启动数据生成调度器"""
        def data_generation_worker():
            interval = int(self.config.get('data_generation', {}).get('generation_interval', '3600'))

            while self.is_running:
                try:
                    # 生成新数据
                    if self.data_generator:
                        evolution_gen = self._get_current_generation()
                        output_file = f"./agi_persistent_training/data/generated_data_gen_{evolution_gen}.jsonl"
                        self.data_generator.generate_incremental_data(evolution_gen, output_file)
                        logger.info(f"✅ 已生成第{evolution_gen}代增量数据")

                    # 等待下次生成
                    time.sleep(interval)

                except Exception as e:
                    logger.error(f"数据生成调度错误: {e}")
                    time.sleep(60)  # 出错时等待1分钟

        thread = threading.Thread(target=data_generation_worker, daemon=True)
        thread.start()
        self.threads['data_generator'] = thread

    def _start_system_monitoring(self):
        """启动系统级监控"""
        def system_monitor_worker():
            while self.is_running:
                try:
                    # 系统资源监控
                    self._check_system_resources()

                    # 组件健康检查
                    self._check_component_health()

                    time.sleep(30)  # 每30秒检查一次

                except Exception as e:
                    logger.error(f"系统监控错误: {e}")
                    time.sleep(10)

        thread = threading.Thread(target=system_monitor_worker, daemon=True)
        thread.start()
        self.threads['system_monitor'] = thread

    def _start_health_monitoring(self):
        """启动健康监控"""
        def health_monitor_worker():
            consecutive_failures = 0
            max_restarts = int(self.config.get('system', {}).get('max_restarts', '3'))

            while self.is_running:
                try:
                    if not self._perform_health_check():
                        consecutive_failures += 1
                        logger.warning(f"健康检查失败 {consecutive_failures}/{max_restarts}")

                        if consecutive_failures >= max_restarts:
                            logger.error("健康检查连续失败，尝试重启系统...")
                            self._restart_system()
                            consecutive_failures = 0
                    else:
                        consecutive_failures = 0

                    time.sleep(int(self.config.get('system', {}).get('health_check_interval', '30')))

                except Exception as e:
                    logger.error(f"健康监控错误: {e}")
                    time.sleep(10)

        thread = threading.Thread(target=health_monitor_worker, daemon=True)
        thread.start()
        self.threads['health_monitor'] = thread

    def _check_system_resources(self):
        """检查系统资源"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                logger.warning(f"⚠️  CPU使用率过高: {cpu_percent}%")

            # 内存使用率
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"⚠️  内存使用率过高: {memory.percent}%")

            # 磁盘空间
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                logger.warning(f"⚠️  磁盘空间不足: {disk.percent}%")

        except Exception as e:
            logger.error(f"系统资源检查失败: {e}")

    def _check_component_health(self):
        """检查组件健康状态"""
        health_status = {}

        # 检查训练器
        if self.trainer:
            health_status['trainer'] = self.trainer.is_training
        else:
            health_status['trainer'] = False

        # 检查监控器
        if self.monitor:
            health_status['monitor'] = self.monitor.is_monitoring
        else:
            health_status['monitor'] = False

        # 检查进化监控器
        if self.evolution_monitor:
            health_status['evolution_monitor'] = self.evolution_monitor.is_monitoring
        else:
            health_status['evolution_monitor'] = False

        # 记录不健康组件
        unhealthy = [comp for comp, healthy in health_status.items() if not healthy]
        if unhealthy:
            logger.warning(f"⚠️  不健康组件: {', '.join(unhealthy)}")

        return health_status

    def _check_critical_processes(self) -> bool:
        """检查关键进程是否在运行"""
        try:
            # 检查训练器进程
            if self.trainer:
                # 检查训练器是否在运行
                if hasattr(self.trainer, 'is_training') and self.trainer.is_training:
                    pass  # 训练器正在运行
                elif hasattr(self.trainer, 'is_running') and self.trainer.is_running:
                    pass  # 训练器正在运行
                else:
                    logger.warning("训练器进程未在运行")
                    return False

            # 检查监控器进程
            if self.monitor:
                if hasattr(self.monitor, 'is_monitoring') and not self.monitor.is_monitoring:
                    logger.warning("训练监控器进程未在运行")
                    return False

            # 检查进化监控器进程
            if self.evolution_monitor:
                if hasattr(self.evolution_monitor, 'is_monitoring') and not self.evolution_monitor.is_monitoring:
                    logger.warning("进化监控器进程未在运行")
                    return False

            # 检查数据生成器
            if self.data_generator:
                # 数据生成器通常是按需运行的，这里检查线程是否存在
                if 'data_generator' in self.threads:
                    if not self.threads['data_generator'].is_alive():
                        logger.warning("数据生成器线程未在运行")
                        # 这不是致命错误，继续检查

            return True

        except Exception as e:
            logger.error(f"检查关键进程失败: {e}")
            return False

    def _perform_health_check(self) -> bool:
        """执行健康检查 - 确保使用我们的核心算法"""
        try:
            # 检查配置文件是否存在
            if not os.path.exists(self.config_path):
                return False

            # 检查工作目录
            if not self.working_dir.exists():
                return False

            # 检查关键进程
            if not self._check_critical_processes():
                return False

            # 检查核心算法使用情况 - 这是关键验证
            if not self._verify_core_algorithm_usage():
                logger.error("❌ 核心算法验证失败 - 系统未正确使用对数流形编码")
                return False

            # 检查最近的日志活动
            log_dir = self.working_dir / "logs"
            if log_dir.exists():
                latest_log = max(log_dir.glob("*.log"), key=os.path.getmtime, default=None)
                if latest_log:
                    log_age = time.time() - os.path.getmtime(latest_log)
                    if log_age > 3600:  # 1小时没有日志更新
                        logger.warning("日志活动不活跃")
                        return False

            return True

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def _verify_core_algorithm_usage(self) -> bool:
        """验证核心算法使用情况 - 确保诚实的AGI实验"""
        try:
            # 检查训练数据文件
            data_dir = self.working_dir / "data"
            if not data_dir.exists():
                logger.warning("训练数据目录不存在，跳过算法验证")
                return True  # 初始状态下数据不存在是正常的

            # 查找最新的训练数据文件
            data_files = list(data_dir.glob("*.jsonl"))
            if not data_files:
                logger.warning("未找到训练数据文件，跳过算法验证")
                return True

            latest_data_file = max(data_files, key=os.path.getmtime)

            # 检查数据样本是否使用了我们的算法
            algorithm_used_count = 0
            total_samples = 0

            with open(latest_data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= 10:  # 只检查前10个样本
                        break

                    try:
                        sample = json.loads(line.strip())
                        total_samples += 1

                        # 检查是否标记使用了我们的算法
                        if sample.get('algorithm_used') == 'logarithmic_manifold_encoding':
                            algorithm_used_count += 1

                        # 检查是否有编码特征
                        if 'encoded_features' in sample and sample['encoded_features']:
                            algorithm_used_count += 1

                        # 检查压缩率是否合理 (我们的算法应该显示压缩)
                        compression_ratio = sample.get('compression_ratio', 1.0)
                        if compression_ratio < 0.9:  # 压缩率小于0.9表示使用了压缩算法
                            algorithm_used_count += 1

                    except json.JSONDecodeError:
                        continue

            # 计算算法使用率
            if total_samples > 0:
                usage_rate = algorithm_used_count / (total_samples * 3)  # 每个样本有3个检查点
                logger.info(f"核心算法使用验证: {usage_rate:.2f} (检查了{total_samples}个样本)")

                if usage_rate < 0.5:  # 少于50%的样本使用了算法
                    logger.error(f"❌ 核心算法使用不足: 只有{usage_rate:.1%}的数据使用了对数流形编码")
                    return False
                else:
                    logger.info(f"✅ 核心算法使用正常: {usage_rate:.1%}的数据使用了我们的编码算法")
                    return True
            else:
                logger.warning("没有找到有效的训练数据样本")
                return False

        except Exception as e:
            logger.error(f"核心算法验证失败: {e}")
            return False

    def _restart_system(self):
        """重启系统"""
        logger.info("🔄 重启AGI系统...")

        self.stop_system()
        time.sleep(5)  # 等待清理完成

        success = self.start_system()
        if success:
            logger.info("✅ 系统重启成功")
        else:
            logger.error("❌ 系统重启失败")

    def _get_current_generation(self) -> int:
        """获取当前进化代数"""
        try:
            if self.trainer and hasattr(self.trainer, 'evolution_engine'):
                return self.trainer.evolution_engine.generation
            elif os.path.exists("./evo_state.json"):
                with open("./evo_state.json", 'r') as f:
                    state = json.load(f)
                    return state.get('generation', 0)
        except:
            pass
        return 0

    def _cleanup_processes(self):
        """清理进程和线程"""
        # 停止所有线程
        for name, thread in self.threads.items():
            try:
                if thread.is_alive():
                    logger.info(f"等待线程 {name} 停止...")
                    thread.join(timeout=10)
            except Exception as e:
                logger.error(f"停止线程 {name} 失败: {e}")

        self.threads.clear()

        # 终止所有相关进程
        for name, proc in self.processes.items():
            try:
                if proc.poll() is None:  # 进程仍在运行
                    logger.info(f"终止进程 {name}...")
                    proc.terminate()
                    proc.wait(timeout=10)
            except Exception as e:
                logger.error(f"终止进程 {name} 失败: {e}")

        self.processes.clear()

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'components': {},
            'system_resources': {},
            'current_generation': self._get_current_generation()
        }

        # 组件状态
        status['components'] = self._check_component_health()

        # 系统资源
        try:
            status['system_resources'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        except:
            pass

        return status

    def generate_system_report(self) -> str:
        """生成系统报告"""
        status = self.get_system_status()

        report = f"""# H2Q-Evo AGI系统状态报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 系统概览
- 运行状态: {'运行中' if status['is_running'] else '已停止'}
- 启动时间: {status['start_time']}
- 运行时长: {status['uptime']}
- 当前进化代数: {status['current_generation']}

## 组件状态
"""

        for comp, healthy in status['components'].items():
            status_icon = "✅" if healthy else "❌"
            report += f"- {comp}: {status_icon} {'正常' if healthy else '异常'}\n"

        report += "\n## 系统资源\n"
        resources = status['system_resources']
        report += f"- CPU使用率: {resources.get('cpu_percent', 'N/A')}%\n"
        report += f"- 内存使用率: {resources.get('memory_percent', 'N/A')}%\n"
        report += f"- 磁盘使用率: {resources.get('disk_percent', 'N/A')}%\n"

        # 保存报告
        report_file = self.working_dir / "reports" / f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        return str(report_file)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='H2Q-Evo AGI系统管理器')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'report', 'restart'],
                       help='执行操作')
    parser.add_argument('--config', default='./agi_training_config.ini',
                       help='配置文件路径')
    parser.add_argument('--background', action='store_true',
                       help='后台运行')

    args = parser.parse_args()

    # 创建系统管理器
    manager = AGISystemManager(args.config)

    try:
        if args.action == 'start':
            if manager.start_system():
                print("✅ AGI系统启动成功")

                if not args.background:
                    print("按 Ctrl+C 停止系统...")
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n🛑 正在停止AGI系统...")
                        manager.stop_system()
                        print("✅ AGI系统已停止")
            else:
                print("❌ AGI系统启动失败")
                sys.exit(1)

        elif args.action == 'stop':
            manager.stop_system()
            print("✅ AGI系统已停止")

        elif args.action == 'status':
            status = manager.get_system_status()
            print("📊 AGI系统状态:")
            print(json.dumps(status, indent=2, ensure_ascii=False))

        elif args.action == 'report':
            report_file = manager.generate_system_report()
            print(f"📋 系统报告已生成: {report_file}")

        elif args.action == 'restart':
            print("🔄 重启AGI系统...")
            manager.stop_system()
            time.sleep(2)

            if manager.start_system():
                print("✅ AGI系统重启成功")
            else:
                print("❌ AGI系统重启失败")
                sys.exit(1)

    except Exception as e:
        logger.error(f"系统管理器运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()