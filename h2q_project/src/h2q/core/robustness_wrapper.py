"""
H2Q-Evo 鲁棒性增强包装器
为核心算法添加错误处理、输入验证、边界检查和降级策略
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Dict, Tuple, Union
from functools import wraps
import traceback
import time
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationError:
    """验证错误"""
    field_name: str
    error_message: str
    severity: ErrorSeverity
    actual_value: Any
    expected_range: Optional[Tuple[Any, Any]] = None

class RobustWrapper:
    """鲁棒性包装器"""
    
    def __init__(
        self,
        enable_validation: bool = True,
        enable_fallback: bool = True,
        enable_logging: bool = True,
        retry_attempts: int = 3
    ):
        self.enable_validation = enable_validation
        self.enable_fallback = enable_fallback
        self.enable_logging = enable_logging
        self.retry_attempts = retry_attempts
        self.error_count = 0
        self.total_calls = 0
        
    def validate_tensor_input(
        self,
        tensor: torch.Tensor,
        name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> list[ValidationError]:
        """验证张量输入"""
        errors = []
        
        # 检查是否为张量
        if not isinstance(tensor, torch.Tensor):
            errors.append(ValidationError(
                field_name=name,
                error_message=f"期望 torch.Tensor，实际是 {type(tensor)}",
                severity=ErrorSeverity.CRITICAL,
                actual_value=type(tensor)
            ))
            return errors
            
        # 检查形状
        if expected_shape is not None:
            if len(expected_shape) != len(tensor.shape):
                errors.append(ValidationError(
                    field_name=name,
                    error_message=f"形状维度不匹配: 期望 {len(expected_shape)}D，实际 {len(tensor.shape)}D",
                    severity=ErrorSeverity.HIGH,
                    actual_value=tensor.shape,
                    expected_range=(expected_shape, expected_shape)
                ))
            else:
                for i, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
                    if expected != -1 and expected != actual:
                        errors.append(ValidationError(
                            field_name=f"{name}.shape[{i}]",
                            error_message=f"维度 {i} 不匹配: 期望 {expected}，实际 {actual}",
                            severity=ErrorSeverity.HIGH,
                            actual_value=actual,
                            expected_range=(expected, expected)
                        ))
                        
        # 检查数据类型
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            errors.append(ValidationError(
                field_name=f"{name}.dtype",
                error_message=f"数据类型不匹配: 期望 {expected_dtype}，实际 {tensor.dtype}",
                severity=ErrorSeverity.MEDIUM,
                actual_value=tensor.dtype,
                expected_range=(expected_dtype, expected_dtype)
            ))
            
        # 检查 NaN
        if not allow_nan and torch.isnan(tensor).any():
            errors.append(ValidationError(
                field_name=name,
                error_message="包含 NaN 值",
                severity=ErrorSeverity.CRITICAL,
                actual_value="NaN detected"
            ))
            
        # 检查 Inf
        if not allow_inf and torch.isinf(tensor).any():
            errors.append(ValidationError(
                field_name=name,
                error_message="包含 Inf 值",
                severity=ErrorSeverity.CRITICAL,
                actual_value="Inf detected"
            ))
            
        # 检查值范围
        if min_value is not None:
            actual_min = tensor.min().item()
            if actual_min < min_value:
                errors.append(ValidationError(
                    field_name=f"{name}.min",
                    error_message=f"最小值低于阈值: {actual_min} < {min_value}",
                    severity=ErrorSeverity.MEDIUM,
                    actual_value=actual_min,
                    expected_range=(min_value, None)
                ))
                
        if max_value is not None:
            actual_max = tensor.max().item()
            if actual_max > max_value:
                errors.append(ValidationError(
                    field_name=f"{name}.max",
                    error_message=f"最大值超过阈值: {actual_max} > {max_value}",
                    severity=ErrorSeverity.MEDIUM,
                    actual_value=actual_max,
                    expected_range=(None, max_value)
                ))
                
        return errors
        
    def sanitize_tensor(
        self,
        tensor: torch.Tensor,
        replace_nan: Optional[float] = 0.0,
        replace_inf: Optional[float] = None,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ) -> torch.Tensor:
        """清理张量（替换异常值）"""
        tensor = tensor.clone()
        
        # 替换 NaN
        if replace_nan is not None:
            tensor = torch.nan_to_num(tensor, nan=replace_nan)
            
        # 替换 Inf
        if replace_inf is not None:
            tensor = torch.nan_to_num(tensor, posinf=replace_inf, neginf=-replace_inf)
            
        # 裁剪值范围
        if clip_min is not None or clip_max is not None:
            tensor = torch.clamp(tensor, min=clip_min, max=clip_max)
            
        return tensor

def robust_inference(
    validate_input: bool = True,
    sanitize_output: bool = True,
    fallback_value: Optional[Any] = None,
    max_retries: int = 3
):
    """鲁棒推理装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            wrapper_obj = RobustWrapper()
            wrapper_obj.total_calls += 1
            
            # 输入验证
            if validate_input:
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        errors = wrapper_obj.validate_tensor_input(
                            arg,
                            name="input_tensor",
                            allow_nan=False,
                            allow_inf=False
                        )
                        if errors:
                            critical_errors = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
                            if critical_errors:
                                raise ValueError(f"输入验证失败: {critical_errors[0].error_message}")
                                
            # 重试逻辑
            last_exception = None
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # 输出清理
                    if sanitize_output and isinstance(result, torch.Tensor):
                        result = wrapper_obj.sanitize_tensor(
                            result,
                            replace_nan=0.0,
                            replace_inf=1e6
                        )
                        
                    return result
                    
                except Exception as e:
                    last_exception = e
                    wrapper_obj.error_count += 1
                    
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))  # 指数退避
                        continue
                    else:
                        if fallback_value is not None:
                            print(f"⚠️ 使用降级值: {fallback_value}")
                            return fallback_value
                        else:
                            raise last_exception
                            
        return wrapper
    return decorator

class RobustDiscreteDecisionEngine(nn.Module):
    """增强鲁棒性的决策引擎包装器"""
    
    def __init__(self, base_engine: nn.Module):
        super().__init__()
        self.base_engine = base_engine
        self.wrapper = RobustWrapper()
        self.fallback_enabled = True
        self.performance_degraded = False
        
    @robust_inference(validate_input=True, sanitize_output=True, max_retries=3)
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播，带有鲁棒性检查"""
        # 输入验证
        errors = self.wrapper.validate_tensor_input(
            x,
            name="input",
            expected_shape=(-1, self.base_engine.config.latent_dim),
            allow_nan=False,
            allow_inf=False
        )
        
        if errors:
            critical = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
            if critical:
                # 尝试修复输入
                x = self.wrapper.sanitize_tensor(x, replace_nan=0.0, replace_inf=1e6)
                print(f"⚠️ 输入已清理: {len(critical)} 个严重错误")
                
        # 执行推理
        try:
            output = self.base_engine(x, **kwargs)
            
            # 输出验证
            output_errors = self.wrapper.validate_tensor_input(
                output,
                name="output",
                allow_nan=False,
                allow_inf=False
            )
            
            if output_errors:
                output = self.wrapper.sanitize_tensor(output, replace_nan=0.0, replace_inf=1e6)
                
            return output
            
        except RuntimeError as e:
            # GPU 内存不足时降级到 CPU
            if "out of memory" in str(e).lower():
                print("⚠️ GPU 内存不足，降级到 CPU")
                self.performance_degraded = True
                x_cpu = x.cpu()
                self.base_engine.cpu()
                output = self.base_engine(x_cpu, **kwargs)
                return output.to(x.device)
            else:
                raise e

def add_input_validation(
    tensor_name: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
):
    """添加输入验证的装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 查找名为 tensor_name 的参数
            if args and isinstance(args[0], torch.Tensor):
                tensor = args[0]
                wrapper_obj = RobustWrapper()
                errors = wrapper_obj.validate_tensor_input(
                    tensor,
                    name=tensor_name,
                    expected_shape=expected_shape,
                    min_value=min_value,
                    max_value=max_value,
                    allow_nan=False,
                    allow_inf=False
                )
                
                if errors:
                    critical = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
                    if critical:
                        raise ValueError(f"输入验证失败: {critical[0].error_message}")
                        
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class SafetyGuard:
    """安全防护层"""
    
    @staticmethod
    def check_numerical_stability(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """检查数值稳定性"""
        if torch.isnan(tensor).any():
            print(f"❌ {name} 包含 NaN")
            return False
        if torch.isinf(tensor).any():
            print(f"❌ {name} 包含 Inf")
            return False
        return True
        
    @staticmethod
    def safe_division(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """安全除法（避免除零）"""
        return a / (b + epsilon)
        
    @staticmethod
    def safe_log(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """安全对数（避免 log(0)）"""
        return torch.log(x + epsilon)
        
    @staticmethod
    def safe_sqrt(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """安全平方根（避免负数）"""
        return torch.sqrt(torch.clamp(x, min=epsilon))
        
    @staticmethod
    def gradient_clipping(tensor: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
        """梯度裁剪"""
        if tensor.grad is not None:
            torch.nn.utils.clip_grad_norm_([tensor], max_norm)
        return tensor

if __name__ == "__main__":
    print("🛡️ H2Q-Evo 鲁棒性增强系统")
    print("="*50)
    
    # 测试输入验证
    wrapper = RobustWrapper()
    
    # 正常输入
    normal_tensor = torch.randn(1, 256)
    errors = wrapper.validate_tensor_input(
        normal_tensor,
        name="test_tensor",
        expected_shape=(-1, 256),
        allow_nan=False
    )
    print(f"✅ 正常输入验证: {len(errors)} 个错误")
    
    # 异常输入
    bad_tensor = torch.tensor([float('nan'), float('inf'), 1.0, 2.0])
    errors = wrapper.validate_tensor_input(
        bad_tensor,
        name="bad_tensor",
        allow_nan=False,
        allow_inf=False
    )
    print(f"⚠️ 异常输入验证: {len(errors)} 个错误")
    for error in errors:
        print(f"   - {error.error_message}")
        
    # 测试清理
    cleaned = wrapper.sanitize_tensor(bad_tensor, replace_nan=0.0, replace_inf=1e6)
    print(f"✅ 清理后: {cleaned}")
    
    # 测试安全操作
    print("\n" + "="*50)
    print("🔒 安全操作测试:")
    print("="*50)
    
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 0.0, 4.0])
    
    safe_result = SafetyGuard.safe_division(a, b)
    print(f"✅ 安全除法: {safe_result}")
    
    x = torch.tensor([0.0, 1.0, 2.0])
    safe_log_result = SafetyGuard.safe_log(x)
    print(f"✅ 安全对数: {safe_log_result}")
    
    print("\n✅ 鲁棒性增强系统测试完成")
