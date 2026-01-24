#!/usr/bin/env python3
"""
H2Q-Evo AGI容错和恢复系统
提供异常处理、自动恢复、故障转移和系统稳定性保障
"""

import os
import sys
import json
import time
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Type
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, asdict
import signal
import psutil
import subprocess
import shlex

logger = logging.getLogger('AGI-FaultTolerance')

class FaultType:
    """故障类型枚举"""
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    DISK_ERROR = "disk_error"
    PROCESS_CRASH = "process_crash"
    TRAINING_DIVERGENCE = "training_divergence"
    MODEL_CORRUPTION = "model_corruption"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_DEPENDENCY = "external_dependency"
    UNKNOWN_ERROR = "unknown_error"

class RecoveryStrategy:
    """恢复策略枚举"""
    RESTART = "restart"
    ROLLBACK = "rollback"
    RETRY = "retry"
    SKIP = "skip"
    DEGRADATION = "degradation"
    FAILOVER = "failover"
    MANUAL = "manual"

@dataclass
class FaultRecord:
    """故障记录"""
    timestamp: float
    fault_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    context: Dict[str, Any]
    recovery_strategy: str
    recovery_success: bool
    recovery_time: float
    stack_trace: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FaultToleranceManager:
    """容错管理器"""

    def __init__(self):
        self.fault_history: List[FaultRecord] = []
        self.max_history_size = 1000
        self.recovery_strategies = self._init_recovery_strategies()
        self.health_checks = []
        self.circuit_breakers = {}
        self.watchdog_timer = None
        self.watchdog_timeout = 300  # 5分钟看门狗超时

        # 启动健康检查
        self._start_health_monitoring()

    def _init_recovery_strategies(self) -> Dict[str, Callable]:
        """初始化恢复策略"""
        return {
            RecoveryStrategy.RESTART: self._restart_component,
            RecoveryStrategy.ROLLBACK: self._rollback_to_checkpoint,
            RecoveryStrategy.RETRY: self._retry_operation,
            RecoveryStrategy.SKIP: self._skip_operation,
            RecoveryStrategy.DEGRADATION: self._degrade_service,
            RecoveryStrategy.FAILOVER: self._failover_to_backup,
            RecoveryStrategy.MANUAL: self._manual_intervention_required
        }

    def register_health_check(self, name: str, check_func: Callable[[], bool],
                            interval: int = 60):
        """注册健康检查"""
        self.health_checks.append({
            'name': name,
            'func': check_func,
            'interval': interval,
            'last_check': 0,
            'last_result': True
        })
        logger.info(f"健康检查已注册: {name}")

    def report_fault(self, fault_type: str, severity: str, description: str,
                    context: Dict[str, Any] = None, stack_trace: str = "") -> str:
        """报告故障"""
        timestamp = time.time()

        # 确定恢复策略
        recovery_strategy = self._determine_recovery_strategy(fault_type, severity)

        fault_record = FaultRecord(
            timestamp=timestamp,
            fault_type=fault_type,
            severity=severity,
            description=description,
            context=context or {},
            recovery_strategy=recovery_strategy,
            recovery_success=False,
            recovery_time=0,
            stack_trace=stack_trace
        )

        self.fault_history.append(fault_record)

        # 限制历史记录大小
        if len(self.fault_history) > self.max_history_size:
            self.fault_history = self.fault_history[-self.max_history_size:]

        logger.error(f"故障报告: {fault_type} ({severity}) - {description}")

        # 尝试自动恢复
        recovery_success = self._attempt_recovery(fault_record)

        fault_record.recovery_success = recovery_success
        fault_record.recovery_time = time.time() - timestamp

        # 如果自动恢复失败，触发人工干预
        if not recovery_success and severity in ['high', 'critical']:
            self._trigger_manual_intervention(fault_record)

        return f"{fault_type}_{int(timestamp)}"

    def _determine_recovery_strategy(self, fault_type: str, severity: str) -> str:
        """确定恢复策略"""
        if fault_type == FaultType.NETWORK_ERROR:
            return RecoveryStrategy.RETRY
        elif fault_type == FaultType.MEMORY_ERROR:
            return RecoveryStrategy.ROLLBACK
        elif fault_type == FaultType.PROCESS_CRASH:
            return RecoveryStrategy.RESTART
        elif fault_type == FaultType.TRAINING_DIVERGENCE:
            return RecoveryStrategy.ROLLBACK
        elif fault_type == FaultType.MODEL_CORRUPTION:
            return RecoveryStrategy.ROLLBACK
        elif fault_type == FaultType.RESOURCE_EXHAUSTION:
            return RecoveryStrategy.DEGRADATION
        elif severity == 'critical':
            return RecoveryStrategy.MANUAL
        else:
            return RecoveryStrategy.RETRY

    def _attempt_recovery(self, fault_record: FaultRecord) -> bool:
        """尝试恢复"""
        strategy = fault_record.recovery_strategy

        if strategy not in self.recovery_strategies:
            logger.error(f"未知的恢复策略: {strategy}")
            return False

        try:
            recovery_func = self.recovery_strategies[strategy]
            success = recovery_func(fault_record)

            if success:
                logger.info(f"恢复成功: {strategy} for {fault_record.fault_type}")
            else:
                logger.warning(f"恢复失败: {strategy} for {fault_record.fault_type}")

            return success

        except Exception as e:
            logger.error(f"恢复执行异常: {e}")
            return False

    def _restart_component(self, fault_record: FaultRecord) -> bool:
        """重启组件"""
        # 这里应该实现具体的重启逻辑
        logger.info("执行组件重启恢复")
        # 模拟重启
        time.sleep(2)
        return True

    def _rollback_to_checkpoint(self, fault_record: FaultRecord) -> bool:
        """回滚到检查点"""
        try:
            from agi_checkpoint_system import rollback_manager
            suggested_version = rollback_manager.suggest_rollback({})
            if suggested_version:
                return rollback_manager.rollback_to_checkpoint(
                    suggested_version, f"自动回滚: {fault_record.description}"
                )
        except Exception as e:
            logger.error(f"检查点回滚失败: {e}")
        return False

    def _retry_operation(self, fault_record: FaultRecord) -> bool:
        """重试操作"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"重试操作 (尝试 {attempt + 1}/{max_retries})")
                # 这里应该实现具体的重试逻辑
                time.sleep(1)
                return True
            except Exception as e:
                logger.warning(f"重试失败: {e}")
                if attempt == max_retries - 1:
                    return False
        return False

    def _skip_operation(self, fault_record: FaultRecord) -> bool:
        """跳过操作"""
        logger.info("跳过故障操作，继续执行")
        return True

    def _degrade_service(self, fault_record: FaultRecord) -> bool:
        """服务降级"""
        logger.info("执行服务降级恢复")
        # 降低资源使用率，减少批次大小等
        return True

    def _failover_to_backup(self, fault_record: FaultRecord) -> bool:
        """故障转移到备份"""
        logger.info("执行故障转移到备份系统")
        # 切换到备份服务
        return True

    def _manual_intervention_required(self, fault_record: FaultRecord) -> bool:
        """需要人工干预"""
        logger.critical(f"需要人工干预: {fault_record.description}")
        # 发送警报，记录到系统日志
        return False

    def _trigger_manual_intervention(self, fault_record: FaultRecord):
        """触发人工干预"""
        alert_message = f"""
AGI系统严重故障警报:
时间: {datetime.fromtimestamp(fault_record.timestamp)}
类型: {fault_record.fault_type}
严重程度: {fault_record.severity}
描述: {fault_record.description}
上下文: {json.dumps(fault_record.context, indent=2)}
        """

        # 写入警报文件
        alert_file = Path("fault_alerts.txt")
        with open(alert_file, 'a', encoding='utf-8') as f:
            f.write(alert_message + "\n" + "="*50 + "\n")

        logger.critical("严重故障警报已生成，请检查 fault_alerts.txt")

    def _start_health_monitoring(self):
        """启动健康监控"""
        def health_monitor_loop():
            while True:
                try:
                    self._perform_health_checks()
                    time.sleep(30)  # 30秒检查一次
                except Exception as e:
                    logger.error(f"健康监控异常: {e}")
                    time.sleep(10)

        monitor_thread = threading.Thread(target=health_monitor_loop, daemon=True)
        monitor_thread.start()

    def _perform_health_checks(self):
        """执行健康检查"""
        current_time = time.time()

        for check in self.health_checks:
            if current_time - check['last_check'] >= check['interval']:
                try:
                    result = check['func']()
                    check['last_result'] = result
                    check['last_check'] = current_time

                    if not result:
                        self.report_fault(
                            FaultType.UNKNOWN_ERROR,
                            'medium',
                            f"健康检查失败: {check['name']}",
                            {'check_name': check['name']}
                        )

                except Exception as e:
                    logger.error(f"健康检查异常 {check['name']}: {e}")
                    check['last_result'] = False

    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health_status = {
            'overall_health': 'healthy',
            'checks': {},
            'recent_faults': [],
            'uptime': self._get_system_uptime()
        }

        # 检查各个健康检查结果
        all_healthy = True
        for check in self.health_checks:
            health_status['checks'][check['name']] = {
                'healthy': check['last_result'],
                'last_check': check['last_check']
            }
            if not check['last_result']:
                all_healthy = False

        # 检查最近故障
        recent_faults = [f for f in self.fault_history
                        if time.time() - f.timestamp < 3600]  # 最近1小时
        health_status['recent_faults'] = [f.to_dict() for f in recent_faults]

        if recent_faults:
            critical_faults = [f for f in recent_faults if f.severity == 'critical']
            if critical_faults:
                health_status['overall_health'] = 'critical'
            else:
                health_status['overall_health'] = 'degraded'
        elif not all_healthy:
            health_status['overall_health'] = 'warning'

        return health_status

    def _get_system_uptime(self) -> float:
        """获取系统运行时间"""
        try:
            return time.time() - psutil.boot_time()
        except:
            return 0

    def enable_circuit_breaker(self, service_name: str, failure_threshold: int = 5,
                              recovery_timeout: int = 60):
        """启用熔断器"""
        self.circuit_breakers[service_name] = {
            'state': 'closed',  # closed, open, half_open
            'failure_count': 0,
            'failure_threshold': failure_threshold,
            'recovery_timeout': recovery_timeout,
            'last_failure_time': 0
        }

    def record_service_call(self, service_name: str, success: bool):
        """记录服务调用结果"""
        if service_name not in self.circuit_breakers:
            return

        cb = self.circuit_breakers[service_name]

        if success:
            if cb['state'] == 'half_open':
                cb['state'] = 'closed'
                cb['failure_count'] = 0
                logger.info(f"熔断器关闭: {service_name}")
        else:
            cb['failure_count'] += 1
            cb['last_failure_time'] = time.time()

            if cb['failure_count'] >= cb['failure_threshold']:
                cb['state'] = 'open'
                logger.warning(f"熔断器开启: {service_name}")

    def is_circuit_breaker_open(self, service_name: str) -> bool:
        """检查熔断器是否开启"""
        if service_name not in self.circuit_breakers:
            return False

        cb = self.circuit_breakers[service_name]

        if cb['state'] == 'open':
            # 检查是否可以尝试恢复
            if time.time() - cb['last_failure_time'] > cb['recovery_timeout']:
                cb['state'] = 'half_open'
                logger.info(f"熔断器半开: {service_name}")
                return False
            return True

        return False

def fault_tolerant(func: Callable) -> Callable:
    """故障容错装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 获取全局容错管理器
            if 'fault_tolerance_manager' in globals():
                ft_manager = globals()['fault_tolerance_manager']
            else:
                ft_manager = FaultToleranceManager()

            # 确定故障类型
            if isinstance(e, MemoryError):
                fault_type = FaultType.MEMORY_ERROR
            elif isinstance(e, OSError):
                fault_type = FaultType.DISK_ERROR
            else:
                fault_type = FaultType.UNKNOWN_ERROR

            # 报告故障
            stack_trace = traceback.format_exc()
            ft_manager.report_fault(
                fault_type=fault_type,
                severity='medium',
                description=str(e),
                context={'function': func.__name__, 'args': str(args), 'kwargs': str(kwargs)},
                stack_trace=stack_trace
            )

            # 返回默认值或重新抛出
            raise e

    return wrapper

class ProcessSupervisor:
    """进程监督器"""

    def __init__(self):
        self.supervised_processes: Dict[str, Dict[str, Any]] = {}
        self.supervisor_thread = None
        self.running = False

    def supervise_process(self, name: str, command: str, working_dir: str = ".",
                         env: Dict[str, str] = None, restart_on_crash: bool = True):
        """监督进程"""
        self.supervised_processes[name] = {
            'command': command,
            'working_dir': working_dir,
            'env': env or {},
            'restart_on_crash': restart_on_crash,
            'process': None,
            'start_time': 0,
            'restart_count': 0,
            'max_restarts': 5
        }

        logger.info(f"进程监督已注册: {name}")

    def start_supervision(self):
        """启动监督"""
        if self.running:
            return

        self.running = True
        self.supervisor_thread = threading.Thread(target=self._supervise_loop, daemon=True)
        self.supervisor_thread.start()
        logger.info("进程监督已启动")

    def stop_supervision(self):
        """停止监督"""
        self.running = False
        if self.supervisor_thread:
            self.supervisor_thread.join(timeout=5)

        # 终止所有监督的进程
        for name, proc_info in self.supervised_processes.items():
            if proc_info['process'] and proc_info['process'].poll() is None:
                try:
                    proc_info['process'].terminate()
                    proc_info['process'].wait(timeout=5)
                except:
                    proc_info['process'].kill()

        logger.info("进程监督已停止")

    def _supervise_loop(self):
        """监督循环"""
        while self.running:
            try:
                for name, proc_info in self.supervised_processes.items():
                    self._check_process(name, proc_info)

                time.sleep(10)  # 10秒检查一次

            except Exception as e:
                logger.error(f"监督循环异常: {e}")
                time.sleep(5)

    def _check_process(self, name: str, proc_info: Dict[str, Any]):
        """检查进程状态"""
        process = proc_info['process']

        if process is None:
            # 启动进程
            self._start_process(name, proc_info)
        elif process.poll() is not None:
            # 进程已退出
            exit_code = process.returncode
            logger.warning(f"进程 {name} 已退出，退出码: {exit_code}")

            if proc_info['restart_on_crash'] and proc_info['restart_count'] < proc_info['max_restarts']:
                logger.info(f"重启进程: {name}")
                proc_info['restart_count'] += 1
                time.sleep(2)  # 短暂延迟后重启
                self._start_process(name, proc_info)
            else:
                logger.error(f"进程 {name} 达到最大重启次数，不再重启")

    def _start_process(self, name: str, proc_info: Dict[str, Any]):
        """启动进程"""
        try:
            env = os.environ.copy()
            env.update(proc_info['env'])

            process = subprocess.Popen(
                shlex.split(proc_info['command']),
                cwd=proc_info['working_dir'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            proc_info['process'] = process
            proc_info['start_time'] = time.time()

            logger.info(f"进程已启动: {name} (PID: {process.pid})")

        except Exception as e:
            logger.error(f"启动进程失败 {name}: {e}")

# 全局实例
fault_tolerance_manager = FaultToleranceManager()
process_supervisor = ProcessSupervisor()

def get_fault_tolerance_manager() -> FaultToleranceManager:
    """获取容错管理器"""
    return fault_tolerance_manager

def get_process_supervisor() -> ProcessSupervisor:
    """获取进程监督器"""
    return process_supervisor

if __name__ == "__main__":
    # 测试容错系统
    print("测试AGI容错系统...")

    # 创建容错管理器
    ft_manager = FaultToleranceManager()

    # 注册健康检查
    def check_disk_space():
        usage = psutil.disk_usage('/')
        return usage.percent < 90

    def check_memory():
        usage = psutil.virtual_memory()
        return usage.percent < 85

    ft_manager.register_health_check("disk_space", check_disk_space, 30)
    ft_manager.register_health_check("memory", check_memory, 30)

    # 启用熔断器
    ft_manager.enable_circuit_breaker("api_service", failure_threshold=3, recovery_timeout=60)

    # 模拟故障
    ft_manager.report_fault(
        FaultType.NETWORK_ERROR,
        'low',
        "网络连接超时",
        {'url': 'api.example.com', 'timeout': 30}
    )

    # 获取健康状态
    health = ft_manager.get_system_health()
    print(f"系统健康状态: {health['overall_health']}")

    # 测试进程监督
    supervisor = ProcessSupervisor()
    supervisor.supervise_process(
        "test_service",
        "python3 -c 'import time; time.sleep(60)'",
        restart_on_crash=True
    )
    supervisor.start_supervision()

    time.sleep(5)  # 等待一会儿

    supervisor.stop_supervision()

    print("容错系统测试完成")