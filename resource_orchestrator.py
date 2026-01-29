"""
H2Q-Evo 资源编排器 (Resource Orchestrator)

实现完整的资源管理和智能调度：
1. 计算资源分配
2. 内存管理优化
3. 性能监控
4. 自适应调度
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional, Union
import time
import psutil
import threading
import os
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import GPUtil

from model_crystallization_engine import ModelCrystallizationEngine
from ollama_bridge import OllamaBridge, OllamaConfig
from hot_start_manager import HotStartManager, HotStartConfig


@dataclass
class ResourceConfig:
    """资源配置"""
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 2048
    max_gpu_memory_mb: int = 1024
    enable_gpu: bool = True
    monitoring_interval_seconds: float = 5.0
    adaptation_threshold: float = 0.8
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


class ResourceOrchestrator:
    """
    资源编排器

    统一管理所有计算资源：
    1. CPU/GPU内存分配
    2. 模型实例调度
    3. 性能监控和自适应调整
    4. 资源争用解决
    """

    def __init__(self, resource_config: ResourceConfig):
        self.config = resource_config

        # 资源状态
        self.cpu_usage = 0.0
        self.memory_usage_mb = 0.0
        self.gpu_memory_usage_mb = 0.0

        # 活跃任务
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue: List[Dict[str, Any]] = []

        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # 自适应参数
        self.adaptation_factors: Dict[str, float] = {
            "cpu_boost": 1.0,
            "memory_boost": 1.0,
            "gpu_boost": 1.0
        }

        # 性能历史
        self.performance_history: List[Dict[str, float]] = []

    def initialize_system(self) -> Dict[str, Any]:
        """初始化系统"""
        print("初始化H2Q-Evo资源编排器...")

        # 检查可用资源
        system_info = self._get_system_info()

        # 启动监控
        self.start_monitoring()

        return {
            "success": True,
            "system_info": system_info,
            "resource_limits": {
                "cpu_percent": self.config.max_cpu_percent,
                "memory_mb": self.config.max_memory_mb,
                "gpu_memory_mb": self.config.max_gpu_memory_mb
            }
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            # CPU信息
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # 内存信息
            memory = psutil.virtual_memory()

            # GPU信息（如果可用）
            gpu_info = {}
            if self.config.enable_gpu:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_info = {
                            "name": gpu.name,
                            "memory_total_mb": gpu.memoryTotal,
                            "memory_free_mb": gpu.memoryFree,
                            "temperature": gpu.temperature
                        }
                except:
                    gpu_info = {"available": False}

            return {
                "cpu_count": cpu_count,
                "cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
                "memory_total_mb": memory.total / (1024**2),
                "memory_available_mb": memory.available / (1024**2),
                "gpu_info": gpu_info,
                "platform": "macOS" if os.uname().sysname == "Darwin" else os.uname().sysname
            }

        except Exception as e:
            return {"error": f"获取系统信息失败: {e}"}

    def allocate_resources(self, task_name: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        分配资源

        Args:
            task_name: 任务名称
            requirements: 资源需求

        Returns:
            分配结果
        """
        # 检查当前资源状态
        current_resources = self._get_current_resources()

        # 计算需求
        required_memory = requirements.get("memory_mb", 512)
        required_cpu = requirements.get("cpu_percent", 20.0)
        required_gpu_memory = requirements.get("gpu_memory_mb", 0)

        # 检查资源可用性
        memory_available = self.config.max_memory_mb - current_resources["memory_mb"]
        cpu_available = self.config.max_cpu_percent - current_resources["cpu_percent"]
        gpu_available = self.config.max_gpu_memory_mb - current_resources["gpu_memory_mb"]

        if (required_memory <= memory_available and
            required_cpu <= cpu_available and
            required_gpu_memory <= gpu_available):

            # 分配资源
            allocation = {
                "task_name": task_name,
                "allocated_at": time.time(),
                "resources": {
                    "memory_mb": required_memory,
                    "cpu_percent": required_cpu,
                    "gpu_memory_mb": required_gpu_memory
                },
                "status": "active"
            }

            self.active_tasks[task_name] = allocation

            return {
                "success": True,
                "allocation": allocation
            }
        else:
            # 资源不足，加入队列
            queued_task = {
                "task_name": task_name,
                "requirements": requirements,
                "queued_at": time.time(),
                "priority": requirements.get("priority", 1)
            }

            self.task_queue.append(queued_task)
            self.task_queue.sort(key=lambda x: x["priority"], reverse=True)

            return {
                "success": False,
                "reason": "resources_unavailable",
                "queued": True,
                "queue_position": len(self.task_queue)
            }

    def release_resources(self, task_name: str) -> bool:
        """释放资源"""
        if task_name in self.active_tasks:
            del self.active_tasks[task_name]
            return True
        return False

    def _get_current_resources(self) -> Dict[str, float]:
        """获取当前资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # 内存使用
            memory = psutil.virtual_memory()
            memory_mb = (memory.total - memory.available) / (1024**2)

            # GPU内存（简化）
            gpu_memory_mb = 0.0
            if self.config.enable_gpu:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_memory_mb = gpus[0].memoryUsed
                except:
                    pass

            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "gpu_memory_mb": gpu_memory_mb
            }

        except Exception as e:
            print(f"获取资源状态失败: {e}")
            return {"cpu_percent": 0.0, "memory_mb": 0.0, "gpu_memory_mb": 0.0}

    def start_monitoring(self):
        """启动资源监控"""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                try:
                    # 获取当前资源状态
                    current = self._get_current_resources()

                    # 更新实例变量
                    self.cpu_usage = current["cpu_percent"]
                    self.memory_usage_mb = current["memory_mb"]
                    self.gpu_memory_usage_mb = current["gpu_memory_mb"]

                    # 记录历史
                    self.performance_history.append({
                        "timestamp": time.time(),
                        "cpu_percent": self.cpu_usage,
                        "memory_mb": self.memory_usage_mb,
                        "gpu_memory_mb": self.gpu_memory_usage_mb,
                        "active_tasks": len(self.active_tasks)
                    })

                    # 保持历史记录在合理范围内
                    if len(self.performance_history) > 100:
                        self.performance_history = self.performance_history[-100:]

                    # 自适应调整
                    self._adaptive_resource_adjustment(current)

                    # 处理任务队列
                    self._process_task_queue()

                    time.sleep(self.config.monitoring_interval_seconds)

                except Exception as e:
                    print(f"资源监控错误: {e}")
                    time.sleep(self.config.monitoring_interval_seconds)

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def _adaptive_resource_adjustment(self, current_resources: Dict[str, float]):
        """自适应资源调整"""
        # CPU调整
        if current_resources["cpu_percent"] > self.config.max_cpu_percent * self.config.adaptation_threshold:
            self.adaptation_factors["cpu_boost"] = max(0.5, self.adaptation_factors["cpu_boost"] * 0.9)
        else:
            self.adaptation_factors["cpu_boost"] = min(1.5, self.adaptation_factors["cpu_boost"] * 1.05)

        # 内存调整
        memory_ratio = current_resources["memory_mb"] / self.config.max_memory_mb
        if memory_ratio > self.config.adaptation_threshold:
            self.adaptation_factors["memory_boost"] = max(0.5, self.adaptation_factors["memory_boost"] * 0.95)
        else:
            self.adaptation_factors["memory_boost"] = min(1.2, self.adaptation_factors["memory_boost"] * 1.02)

        # GPU调整
        if self.config.enable_gpu:
            gpu_ratio = current_resources["gpu_memory_mb"] / self.config.max_gpu_memory_mb
            if gpu_ratio > self.config.adaptation_threshold:
                self.adaptation_factors["gpu_boost"] = max(0.5, self.adaptation_factors["gpu_boost"] * 0.9)
            else:
                self.adaptation_factors["gpu_boost"] = min(1.5, self.adaptation_factors["gpu_boost"] * 1.05)

    def _process_task_queue(self):
        """处理任务队列"""
        if not self.task_queue:
            return

        # 尝试分配队列中的任务
        remaining_tasks = []

        for task in self.task_queue:
            allocation_result = self.allocate_resources(
                task["task_name"],
                task["requirements"]
            )

            if not allocation_result["success"]:
                remaining_tasks.append(task)
            # 如果分配成功，任务会从队列中移除

        self.task_queue = remaining_tasks

    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        current = self._get_current_resources()

        return {
            "current_usage": current,
            "limits": {
                "cpu_percent": self.config.max_cpu_percent,
                "memory_mb": self.config.max_memory_mb,
                "gpu_memory_mb": self.config.max_gpu_memory_mb
            },
            "utilization_percent": {
                "cpu": (current["cpu_percent"] / self.config.max_cpu_percent) * 100,
                "memory": (current["memory_mb"] / self.config.max_memory_mb) * 100,
                "gpu": (current["gpu_memory_mb"] / self.config.max_gpu_memory_mb) * 100 if self.config.max_gpu_memory_mb > 0 else 0
            },
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "adaptation_factors": self.adaptation_factors,
            "monitoring_active": self.monitoring_active
        }

    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """优化资源分配"""
        # 分析性能历史
        if len(self.performance_history) < 10:
            return {"success": False, "reason": "insufficient_history"}

        # 计算平均使用率
        avg_cpu = np.mean([h["cpu_percent"] for h in self.performance_history[-10:]])
        avg_memory = np.mean([h["memory_mb"] for h in self.performance_history[-10:]])

        # 生成优化建议
        recommendations = []

        if avg_cpu > self.config.max_cpu_percent * 0.9:
            recommendations.append({
                "type": "cpu_optimization",
                "action": "reduce_concurrent_tasks",
                "expected_improvement": "降低CPU使用率20%"
            })

        if avg_memory > self.config.max_memory_mb * 0.9:
            recommendations.append({
                "type": "memory_optimization",
                "action": "enable_aggressive_gc",
                "expected_improvement": "释放内存15%"
            })

        return {
            "success": True,
            "avg_utilization": {
                "cpu_percent": avg_cpu,
                "memory_mb": avg_memory
            },
            "recommendations": recommendations
        }

    def emergency_resource_cleanup(self) -> Dict[str, Any]:
        """紧急资源清理"""
        print("执行紧急资源清理...")

        # 释放所有活跃任务
        released_tasks = list(self.active_tasks.keys())
        for task_name in released_tasks:
            self.release_resources(task_name)

        # 强制垃圾回收
        import gc
        gc.collect()

        # 清理任务队列
        cleared_queue = len(self.task_queue)
        self.task_queue.clear()

        return {
            "success": True,
            "released_tasks": released_tasks,
            "cleared_queue_items": cleared_queue,
            "gc_completed": True
        }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            memory = psutil.virtual_memory()
            return {
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024**2),
                "memory_available_mb": memory.available / (1024**2),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "active_tasks": len(self.active_tasks),
                "queued_tasks": len(self.task_queue)
            }
        except Exception as e:
            return {"error": f"获取系统状态失败: {e}"}

    def __del__(self):
        """清理资源"""
        self.stop_monitoring()