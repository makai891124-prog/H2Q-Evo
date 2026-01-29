"""
H2Q-Evo 生产环境健康检查系统
实时监控系统健康状态、性能指标和异常检测
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import traceback
from collections import deque
import json

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component_name: str
    status: HealthStatus
    message: str
    timestamp: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    throughput_qps: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    error_rate: float

class CircuitBreaker:
    """熔断器 - 防止级联故障"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.half_open_attempts = half_open_attempts
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.successful_attempts = 0
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数"""
        if self.state == "OPEN":
            # 检查是否应该进入半开状态
            if datetime.now() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                self.successful_attempts = 0
            else:
                raise Exception("熔断器开启 - 服务暂时不可用")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
            
    def _on_success(self):
        """成功调用处理"""
        if self.state == "HALF_OPEN":
            self.successful_attempts += 1
            if self.successful_attempts >= self.half_open_attempts:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
            
    def _on_failure(self):
        """失败调用处理"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class HealthMonitor:
    """健康监控系统"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics_history: Dict[str, deque] = {}
        self.alert_callbacks: List[Callable] = []
        
    def register_check(self, name: str, check_func: Callable):
        """注册健康检查"""
        self.checks[name] = check_func
        self.circuit_breakers[name] = CircuitBreaker()
        self.metrics_history[name] = deque(maxlen=100)  # 保留最近100次记录
        
    def run_check(self, name: str) -> HealthCheckResult:
        """运行单个健康检查"""
        if name not in self.checks:
            return HealthCheckResult(
                component_name=name,
                status=HealthStatus.UNKNOWN,
                message="未注册的健康检查",
                timestamp=datetime.now().isoformat()
            )
            
        try:
            # 通过熔断器执行检查
            result = self.circuit_breakers[name].call(self.checks[name])
            
            if isinstance(result, HealthCheckResult):
                self.results[name] = result
                self.metrics_history[name].append(result)
                return result
            else:
                # 如果返回布尔值，转换为结果对象
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                result = HealthCheckResult(
                    component_name=name,
                    status=status,
                    message="检查完成",
                    timestamp=datetime.now().isoformat()
                )
                self.results[name] = result
                self.metrics_history[name].append(result)
                return result
                
        except Exception as e:
            error_result = HealthCheckResult(
                component_name=name,
                status=HealthStatus.CRITICAL,
                message="健康检查失败",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            self.results[name] = error_result
            self.metrics_history[name].append(error_result)
            
            # 触发告警
            self._trigger_alerts(error_result)
            
            return error_result
            
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """运行所有健康检查"""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
        
    def get_overall_status(self) -> HealthStatus:
        """获取整体健康状态"""
        if not self.results:
            return HealthStatus.UNKNOWN
            
        statuses = [r.status for r in self.results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
            
    def register_alert(self, callback: Callable):
        """注册告警回调"""
        self.alert_callbacks.append(callback)
        
    def _trigger_alerts(self, result: HealthCheckResult):
        """触发告警"""
        for callback in self.alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"告警回调失败: {e}")
                
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}
        for name, history in self.metrics_history.items():
            if not history:
                continue
                
            recent_results = list(history)[-10:]  # 最近10次
            healthy_count = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
            
            summary[name] = {
                'success_rate': healthy_count / len(recent_results),
                'last_status': recent_results[-1].status.value,
                'last_check_time': recent_results[-1].timestamp,
                'circuit_breaker_state': self.circuit_breakers[name].state
            }
            
        return summary

class ProductionValidator:
    """生产环境验证器"""
    
    def __init__(self):
        self.monitor = HealthMonitor()
        self._register_core_checks()
        
    def _register_core_checks(self):
        """注册核心健康检查"""
        
        # 1. 模型加载检查
        def check_model_loading():
            try:
                from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
                config = LatentConfig(dim=256, n_choices=64)
                model = DiscreteDecisionEngine(config=config)
                return HealthCheckResult(
                    component_name="model_loading",
                    status=HealthStatus.HEALTHY,
                    message="模型加载正常",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                return HealthCheckResult(
                    component_name="model_loading",
                    status=HealthStatus.CRITICAL,
                    message="模型加载失败",
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                
        self.monitor.register_check("model_loading", check_model_loading)
        
        # 2. 推理性能检查
        def check_inference_performance():
            try:
                from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
                config = LatentConfig(dim=256, n_choices=64)
                model = DiscreteDecisionEngine(config=config)
                model.eval()
                
                # 测试推理时间
                x = torch.randn(1, 256)
                
                times = []
                for _ in range(10):
                    start = time.time()
                    with torch.no_grad():
                        _ = model(x)
                    times.append((time.time() - start) * 1000)  # 转换为毫秒
                    
                avg_time = sum(times) / len(times)
                
                # 性能阈值: 平均推理时间应小于100ms
                status = HealthStatus.HEALTHY if avg_time < 100 else HealthStatus.DEGRADED
                
                return HealthCheckResult(
                    component_name="inference_performance",
                    status=status,
                    message=f"平均推理时间: {avg_time:.2f}ms",
                    timestamp=datetime.now().isoformat(),
                    metrics={'avg_inference_time_ms': avg_time}
                )
            except Exception as e:
                return HealthCheckResult(
                    component_name="inference_performance",
                    status=HealthStatus.CRITICAL,
                    message="性能检查失败",
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                
        self.monitor.register_check("inference_performance", check_inference_performance)
        
        # 3. 内存使用检查
        def check_memory_usage():
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # 内存阈值: 应小于1GB
                status = HealthStatus.HEALTHY if memory_mb < 1024 else HealthStatus.DEGRADED
                
                return HealthCheckResult(
                    component_name="memory_usage",
                    status=status,
                    message=f"内存使用: {memory_mb:.1f}MB",
                    timestamp=datetime.now().isoformat(),
                    metrics={'memory_mb': memory_mb}
                )
            except ImportError:
                return HealthCheckResult(
                    component_name="memory_usage",
                    status=HealthStatus.UNKNOWN,
                    message="psutil 未安装",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                return HealthCheckResult(
                    component_name="memory_usage",
                    status=HealthStatus.CRITICAL,
                    message="内存检查失败",
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                
        self.monitor.register_check("memory_usage", check_memory_usage)
        
        # 4. 数学完整性检查
        def check_mathematical_integrity():
            try:
                from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
                config = LatentConfig(dim=256, n_choices=64)
                model = DiscreteDecisionEngine(config=config)
                
                # 测试数学运算的正确性
                x = torch.randn(1, 256)
                with torch.no_grad():
                    output = model(x)
                    
                # 检查输出是否包含 NaN 或 Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    return HealthCheckResult(
                        component_name="mathematical_integrity",
                        status=HealthStatus.CRITICAL,
                        message="检测到 NaN 或 Inf 值",
                        timestamp=datetime.now().isoformat()
                    )
                    
                return HealthCheckResult(
                    component_name="mathematical_integrity",
                    status=HealthStatus.HEALTHY,
                    message="数学运算正常",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                return HealthCheckResult(
                    component_name="mathematical_integrity",
                    status=HealthStatus.CRITICAL,
                    message="数学完整性检查失败",
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                
        self.monitor.register_check("mathematical_integrity", check_mathematical_integrity)
        
    def run_full_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        print("🔍 开始生产环境验证...")
        print("="*50)
        
        results = self.monitor.run_all_checks()
        overall_status = self.monitor.get_overall_status()
        metrics_summary = self.monitor.get_metrics_summary()
        
        # 打印结果
        for name, result in results.items():
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.CRITICAL: "❌",
                HealthStatus.UNKNOWN: "❓"
            }[result.status]
            
            print(f"{status_emoji} {name}: {result.message}")
            if result.error:
                print(f"   错误: {result.error}")
                
        print("="*50)
        print(f"整体状态: {overall_status.value.upper()}")
        print("="*50)
        
        return {
            'overall_status': overall_status.value,
            'checks': {name: result.status.value for name, result in results.items()},
            'metrics': metrics_summary,
            'timestamp': datetime.now().isoformat()
        }
        
    def export_report(self, output_path: str):
        """导出验证报告"""
        report = self.run_full_validation()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 验证报告已导出: {output_path}")

def run_production_validation():
    """运行生产环境验证"""
    validator = ProductionValidator()
    
    # 注册告警回调
    def alert_handler(result: HealthCheckResult):
        if result.status == HealthStatus.CRITICAL:
            print(f"🚨 CRITICAL ALERT: {result.component_name} - {result.message}")
            
    validator.monitor.register_alert(alert_handler)
    
    # 运行验证
    report = validator.run_full_validation()
    
    # 导出报告
    from pathlib import Path
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    validator.export_report(str(report_dir / "production_validation.json"))
    
    return report

if __name__ == "__main__":
    run_production_validation()
