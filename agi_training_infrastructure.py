#!/usr/bin/env python3
"""
H2Q-Evo AGI训练前置系统
提供完整的训练基础设施：数据管道、备份系统、环境感知、热重载等
"""

import os
import sys
import json
import time
import psutil
import threading
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import shutil
import hashlib
import importlib
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor
import socket
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('agi_training_infrastructure.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AGI-Infrastructure')

class SystemEnvironmentMonitor:
    """系统环境监控器"""

    def __init__(self):
        self.system_info = {}
        self.network_status = {}
        self.hardware_limits = {}
        self.update_interval = 30  # 30秒更新一次

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统基本信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_total': psutil.disk_usage('/').total,
            'disk_free': psutil.disk_usage('/').free,
            'disk_percent': psutil.disk_usage('/').percent,
            'boot_time': psutil.boot_time(),
            'platform': sys.platform,
            'python_version': sys.version,
            'process_count': len(psutil.pids())
        }

    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态"""
        try:
            # 检查网络连接
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            internet_connected = True
        except OSError:
            internet_connected = False

        return {
            'internet_connected': internet_connected,
            'hostname': socket.gethostname(),
            'local_ip': socket.gethostbyname(socket.gethostname())
        }

    def get_hardware_limits(self) -> Dict[str, Any]:
        """获取硬件限制"""
        return {
            'max_memory_gb': 16,  # 假设16GB内存限制
            'max_cpu_percent': 80,  # CPU使用率不超过80%
            'max_disk_percent': 90,  # 磁盘使用率不超过90%
            'recommended_batch_size': self._calculate_optimal_batch_size()
        }

    def _calculate_optimal_batch_size(self) -> int:
        """根据系统资源计算最优批次大小"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()

        # 基于经验公式计算批次大小
        base_batch = 4
        memory_factor = min(memory_gb / 8, 4)  # 8GB内存为基础
        cpu_factor = cpu_count / 4  # 4核为基础

        optimal_batch = int(base_batch * memory_factor * cpu_factor)
        return max(1, min(optimal_batch, 64))  # 限制在1-64之间

    def update_environment_info(self):
        """更新环境信息"""
        self.system_info = self.get_system_info()
        self.network_status = self.get_network_status()
        self.hardware_limits = self.get_hardware_limits()

        logger.info(f"环境更新: CPU={self.system_info['cpu_percent']}%, "
                   f"内存={self.system_info['memory_percent']}%, "
                   f"网络={'连接' if self.network_status['internet_connected'] else '断开'}")

    def should_throttle_training(self) -> bool:
        """判断是否应该限制训练"""
        if self.system_info.get('cpu_percent', 0) > self.hardware_limits.get('max_cpu_percent', 80):
            return True
        if self.system_info.get('memory_percent', 0) > 70:  # 内存使用率超过70%
            return True
        if self.system_info.get('disk_percent', 0) > self.hardware_limits.get('max_disk_percent', 90):
            return True
        return False

class DynamicBackupSystem:
    """动态备份系统"""

    def __init__(self, backup_dir: str = "agi_backups", max_backups: int = 24):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.max_backups = max_backups
        self.backup_interval = 3600  # 1小时备份一次
        self.last_backup = 0

    def create_backup(self, data: Dict[str, Any], backup_type: str) -> str:
        """创建备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{backup_type}_{timestamp}.json"

        backup_path = self.backup_dir / backup_name
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 压缩备份
        compressed_path = backup_path.with_suffix('.json.gz')
        import gzip
        with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 删除原始文件
        backup_path.unlink()

        logger.info(f"备份创建: {compressed_path}")
        return str(compressed_path)

    def cleanup_old_backups(self):
        """清理旧备份，保持滚动窗口"""
        backup_files = list(self.backup_dir.glob("*.json.gz"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if len(backup_files) > self.max_backups:
            files_to_delete = backup_files[self.max_backups:]
            for file_path in files_to_delete:
                file_path.unlink()
                logger.info(f"删除旧备份: {file_path}")

    def restore_from_backup(self, backup_type: str, hours_back: int = 1) -> Optional[Dict[str, Any]]:
        """从备份恢复数据"""
        backup_files = list(self.backup_dir.glob(f"{backup_type}_*.json.gz"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        target_time = datetime.now() - timedelta(hours=hours_back)

        for backup_file in backup_files:
            # 解析文件名中的时间戳
            timestamp_str = backup_file.stem.split('_', 1)[1]
            try:
                backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if backup_time <= target_time:
                    import gzip
                    with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"从备份恢复: {backup_file}")
                    return data
            except ValueError:
                continue

        logger.warning(f"未找到合适的{backup_type}备份")
        return None

    def should_backup(self) -> bool:
        """判断是否应该备份"""
        current_time = time.time()
        return current_time - self.last_backup > self.backup_interval

    def perform_backup_cycle(self, system_state: Dict[str, Any]):
        """执行备份周期"""
        if not self.should_backup():
            return

        # 备份关键数据
        self.create_backup(system_state, "system_state")
        self.create_backup(system_state.get('evolution_state', {}), "evolution_state")
        self.create_backup(system_state.get('model_weights', {}), "model_weights")

        # 清理旧备份
        self.cleanup_old_backups()

        self.last_backup = time.time()
        logger.info("备份周期完成")

class TrainingDataPipeline:
    """训练数据管道"""

    def __init__(self, data_dirs: List[str] = None):
        self.data_dirs = data_dirs or ["data", "h2q_project/data"]
        self.data_cache = {}
        self.data_stats = {}

    def discover_data_sources(self) -> Dict[str, Any]:
        """发现可用的数据源"""
        data_sources = {}

        for data_dir in self.data_dirs:
            dir_path = Path(data_dir)
            if dir_path.exists():
                # 扫描数据文件
                json_files = list(dir_path.glob("**/*.json"))
                jsonl_files = list(dir_path.glob("**/*.jsonl"))
                txt_files = list(dir_path.glob("**/*.txt"))

                data_sources[str(dir_path)] = {
                    'json_files': len(json_files),
                    'jsonl_files': len(jsonl_files),
                    'txt_files': len(txt_files),
                    'total_files': len(json_files) + len(jsonl_files) + len(txt_files)
                }

        return data_sources

    def load_training_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """加载训练批次"""
        # 这里应该实现实际的数据加载逻辑
        # 目前返回模拟数据
        batch = []
        for i in range(batch_size):
            sample = {
                'input': f"Sample input {i}",
                'target': f"Sample target {i}",
                'metadata': {'index': i, 'timestamp': time.time()}
            }
            batch.append(sample)

        return batch

    def validate_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证数据质量"""
        if not data:
            return {'valid': False, 'error': '空数据'}

        stats = {
            'total_samples': len(data),
            'avg_input_length': sum(len(str(s.get('input', ''))) for s in data) / len(data),
            'avg_target_length': sum(len(str(s.get('target', ''))) for s in data) / len(data),
            'has_metadata': all('metadata' in s for s in data),
            'valid': True
        }

        return stats

class HotReloadManager:
    """热重载管理器"""

    def __init__(self):
        self.loaded_modules = {}
        self.module_timestamps = {}
        self.reload_interval = 60  # 60秒检查一次

    def register_module(self, module_name: str, module_path: str):
        """注册需要热重载的模块"""
        self.loaded_modules[module_name] = module_path
        if os.path.exists(module_path):
            self.module_timestamps[module_name] = os.path.getmtime(module_path)

    def check_for_updates(self) -> List[str]:
        """检查模块更新"""
        updated_modules = []

        for module_name, module_path in self.loaded_modules.items():
            if os.path.exists(module_path):
                current_mtime = os.path.getmtime(module_path)
                last_mtime = self.module_timestamps.get(module_name, 0)

                if current_mtime > last_mtime:
                    updated_modules.append(module_name)
                    self.module_timestamps[module_name] = current_mtime

        return updated_modules

    def reload_module(self, module_name: str) -> bool:
        """重载模块"""
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                logger.info(f"模块重载成功: {module_name}")
                return True
            else:
                logger.warning(f"模块未加载: {module_name}")
                return False
        except Exception as e:
            logger.error(f"模块重载失败 {module_name}: {e}")
            return False

    def perform_hot_reload(self):
        """执行热重载检查"""
        updated_modules = self.check_for_updates()
        for module_name in updated_modules:
            self.reload_module(module_name)

class ResourceManager:
    """资源管理器"""

    def __init__(self):
        self.resource_limits = {
            'max_memory_gb': 8,
            'max_cpu_percent': 80,
            'max_gpu_memory_gb': 4
        }
        self.current_usage = {}

    def get_resource_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_gb': psutil.virtual_memory().used / (1024**3),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }

    def enforce_limits(self) -> bool:
        """强制执行资源限制"""
        usage = self.get_resource_usage()

        if usage['memory_gb'] > self.resource_limits['max_memory_gb']:
            logger.warning(f"内存使用超限: {usage['memory_gb']:.2f}GB / {self.resource_limits['max_memory_gb']}GB")
            return False

        if usage['cpu_percent'] > self.resource_limits['max_cpu_percent']:
            logger.warning(f"CPU使用超限: {usage['cpu_percent']:.1f}% / {self.resource_limits['max_cpu_percent']}%")
            return False

        return True

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {}
        self.metric_history = []
        self.max_history_size = 1000

    def record_metric(self, name: str, value: float, timestamp: float = None):
        """记录性能指标"""
        if timestamp is None:
            timestamp = time.time()

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append((timestamp, value))

        # 限制历史记录大小
        if len(self.metrics[name]) > self.max_history_size:
            self.metrics[name] = self.metrics[name][-self.max_history_size:]

    def get_metric_stats(self, name: str, window_seconds: int = 300) -> Dict[str, Any]:
        """获取指标统计"""
        if name not in self.metrics:
            return {}

        current_time = time.time()
        window_start = current_time - window_seconds

        recent_values = [v for t, v in self.metrics[name] if t >= window_start]

        if not recent_values:
            return {}

        return {
            'count': len(recent_values),
            'mean': sum(recent_values) / len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'latest': recent_values[-1]
        }

class AGIInfrastructure:
    """AGI基础设施主控制器"""

    def __init__(self):
        self.environment_monitor = SystemEnvironmentMonitor()
        self.backup_system = DynamicBackupSystem()
        self.data_pipeline = TrainingDataPipeline()
        self.hot_reload_manager = HotReloadManager()
        self.resource_manager = ResourceManager()
        self.performance_monitor = PerformanceMonitor()

        self.running = False
        self.monitor_thread = None
        self.backup_thread = None

        # 注册关键模块用于热重载
        self._register_hot_reload_modules()

        # 注册退出处理
        atexit.register(self.graceful_shutdown)

    def _register_hot_reload_modules(self):
        """注册需要热重载的模块"""
        key_modules = [
            ('evolution_system', 'evolution_system.py'),
            ('h2q_server', 'h2q_project/h2q_server.py'),
            ('real_benchmark_test', 'real_benchmark_test.py')
        ]

        for module_name, module_path in key_modules:
            if os.path.exists(module_path):
                self.hot_reload_manager.register_module(module_name, module_path)

    def start_infrastructure(self):
        """启动基础设施"""
        logger.info("启动AGI训练基础设施...")

        self.running = True

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        # 启动备份线程
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self.backup_thread.start()

        # 启动热重载检查
        self._start_hot_reload()

        logger.info("AGI基础设施启动完成")

    def stop_infrastructure(self):
        """停止基础设施"""
        logger.info("停止AGI训练基础设施...")
        self.running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.backup_thread:
            self.backup_thread.join(timeout=5)

        logger.info("AGI基础设施已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 更新环境信息
                self.environment_monitor.update_environment_info()

                # 记录性能指标
                usage = self.environment_monitor.system_info
                self.performance_monitor.record_metric('cpu_percent', usage['cpu_percent'])
                self.performance_monitor.record_metric('memory_percent', usage['memory_percent'])
                self.performance_monitor.record_metric('disk_percent', usage['disk_percent'])

                # 检查资源限制
                if not self.resource_manager.enforce_limits():
                    logger.warning("资源使用超限，可能需要调整训练参数")

                # 检查热重载
                self.hot_reload_manager.perform_hot_reload()

                time.sleep(self.environment_monitor.update_interval)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(10)

    def _backup_loop(self):
        """备份循环"""
        while self.running:
            try:
                # 收集系统状态
                system_state = {
                    'timestamp': datetime.now().isoformat(),
                    'environment': self.environment_monitor.system_info,
                    'performance_metrics': self.performance_monitor.metrics,
                    'data_sources': self.data_pipeline.discover_data_sources()
                }

                # 执行备份
                self.backup_system.perform_backup_cycle(system_state)

                time.sleep(self.backup_system.backup_interval)

            except Exception as e:
                logger.error(f"备份循环异常: {e}")
                time.sleep(300)  # 5分钟后重试

    def _start_hot_reload(self):
        """启动热重载检查"""
        def hot_reload_loop():
            while self.running:
                try:
                    self.hot_reload_manager.perform_hot_reload()
                    time.sleep(self.hot_reload_manager.reload_interval)
                except Exception as e:
                    logger.error(f"热重载异常: {e}")
                    time.sleep(30)

        reload_thread = threading.Thread(target=hot_reload_loop, daemon=True)
        reload_thread.start()

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'infrastructure_running': self.running,
            'environment': self.environment_monitor.system_info,
            'network': self.environment_monitor.network_status,
            'resources': self.resource_manager.get_resource_usage(),
            'performance': {
                name: self.performance_monitor.get_metric_stats(name)
                for name in self.performance_monitor.metrics.keys()
            },
            'data_sources': self.data_pipeline.discover_data_sources(),
            'backup_status': {
                'last_backup': self.backup_system.last_backup,
                'backup_count': len(list(self.backup_system.backup_dir.glob("*.json.gz")))
            }
        }

    def graceful_shutdown(self):
        """优雅关闭"""
        logger.info("执行优雅关闭...")
        self.stop_infrastructure()

        # 最后一次备份
        try:
            system_state = self.get_system_status()
            self.backup_system.create_backup(system_state, "shutdown_backup")
            logger.info("关闭备份已创建")
        except Exception as e:
            logger.error(f"关闭备份失败: {e}")

# 全局基础设施实例
infrastructure = AGIInfrastructure()

def start_agi_infrastructure():
    """启动AGI基础设施"""
    infrastructure.start_infrastructure()
    return infrastructure

def get_infrastructure_status():
    """获取基础设施状态"""
    return infrastructure.get_system_status()

if __name__ == "__main__":
    # 启动基础设施
    infra = start_agi_infrastructure()

    # 保持运行
    try:
        while True:
            time.sleep(60)
            status = infra.get_system_status()
            print(f"基础设施状态: 运行中, CPU: {status['environment']['cpu_percent']}%, "
                  f"内存: {status['environment']['memory_percent']}%")

    except KeyboardInterrupt:
        print("\n正在关闭AGI基础设施...")
        infra.stop_infrastructure()

if __name__ == "__main__":
    # 启动基础设施
    infra = start_agi_infrastructure()

    # 保持运行
    try:
        while True:
            time.sleep(60)
            status = infra.get_system_status()
            print(f"基础设施状态: 运行中, CPU: {status['environment']['cpu_percent']}%, "
                  f"内存: {status['environment']['memory_percent']}%")

    except KeyboardInterrupt:
        print("\n正在关闭AGI基础设施...")
        infra.stop_infrastructure()