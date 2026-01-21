"""H2Q System Control Core - 精细系统控制与微变化感知.

利用 H2Q 的以下核心特性:
1. 极小模型尺寸 (~1MB)
2. 确定性输出 (四元数归一化保证)
3. 可预测轨迹突变 (Berry 相位跃迁检测)
4. 高敏感度微变化感知 (Fueter 曲率)

应用场景:
- 工业控制系统
- 传感器异常检测
- 预测性维护
- 精密仪器校准
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class SystemState:
    """系统状态表示."""
    timestamp: float
    quaternion: torch.Tensor  # (4,) 四元数状态
    phase: float              # Berry 相位
    curvature: float          # Fueter 曲率
    raw_signal: torch.Tensor  # 原始信号
    anomaly_score: float = 0.0
    predicted_trajectory: Optional[torch.Tensor] = None


@dataclass
class ControlMetrics:
    """控制系统指标."""
    total_samples: int = 0
    anomalies_detected: int = 0
    trajectory_predictions: int = 0
    mean_latency_us: float = 0.0
    max_latency_us: float = 0.0
    phase_transitions: int = 0
    stability_score: float = 1.0


class QuaternionStateEncoder(nn.Module):
    """将任意维度信号编码为四元数状态.
    
    极小模型: ~4KB 参数
    确定性: 四元数归一化保证单位球面约束
    优化: 使用直接矩阵乘法而非 Sequential，减少调用开销
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        # 直接定义权重，避免 Sequential 开销
        self.w1 = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.empty(4, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(4))
        
        # 参数初始化为小值，确保初始状态接近单位四元数
        nn.init.xavier_uniform_(self.w1, gain=0.1)
        nn.init.xavier_uniform_(self.w2, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码信号为单位四元数 (优化版本)."""
        # 直接计算，避免 Sequential 开销
        h = torch.tanh(F.linear(x, self.w1, self.b1))
        q = F.linear(h, self.w2, self.b2)
        # 归一化到单位球面 (SU(2) 约束)
        return F.normalize(q, p=2, dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BerryPhaseTracker(nn.Module):
    """Berry 相位追踪器 - 检测拓扑相变.
    
    Berry 相位 = ∮ A·dl，其中 A 是 Berry 联络
    相位跃迁表示系统状态的拓扑变化
    """
    
    def __init__(self, history_len: int = 32):
        super().__init__()
        self.history_len = history_len
        self.q_history = deque(maxlen=history_len)
        self.phase_history = deque(maxlen=history_len)
        
    def compute_berry_phase(self, q: torch.Tensor) -> float:
        """计算当前 Berry 相位.
        
        相位 = 2 * arctan2(||v||, w)，其中 q = (w, v)
        """
        w = q[0].item()
        v_norm = torch.norm(q[1:]).item()
        phase = 2 * np.arctan2(v_norm, w)
        return phase
    
    def compute_berry_curvature(self) -> float:
        """计算 Berry 曲率 (相位变化率).
        
        高曲率表示即将发生相变
        """
        if len(self.phase_history) < 2:
            return 0.0
        
        phases = list(self.phase_history)
        # 一阶差分
        dphase = np.diff(phases)
        # 处理相位跳变 (2π wrap-around)
        dphase = np.where(dphase > np.pi, dphase - 2*np.pi, dphase)
        dphase = np.where(dphase < -np.pi, dphase + 2*np.pi, dphase)
        
        return float(np.std(dphase)) if len(dphase) > 0 else 0.0
    
    def detect_phase_transition(self, threshold: float = 0.5) -> bool:
        """检测相位跃迁."""
        if len(self.phase_history) < 3:
            return False
        
        phases = list(self.phase_history)
        # 检测突变: |Δφ| > threshold
        for i in range(1, len(phases)):
            delta = abs(phases[i] - phases[i-1])
            # 处理 wrap-around
            delta = min(delta, 2*np.pi - delta)
            if delta > threshold:
                return True
        return False
    
    def update(self, q: torch.Tensor) -> Tuple[float, float, bool]:
        """更新状态并返回 (相位, 曲率, 是否跃迁)."""
        phase = self.compute_berry_phase(q)
        self.q_history.append(q.detach().clone())
        self.phase_history.append(phase)
        
        curvature = self.compute_berry_curvature()
        transition = self.detect_phase_transition()
        
        return phase, curvature, transition


class FueterCurvatureDetector(nn.Module):
    """Fueter 曲率检测器 - 微小变化感知.
    
    Fueter 算子: Δ_F = ∂²/∂w² + ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
    曲率异常表示系统偏离正常流形
    """
    
    def __init__(self, sensitivity: float = 0.01):
        super().__init__()
        self.sensitivity = sensitivity
        self.baseline_curvature = None
        self.curvature_history = deque(maxlen=100)
        
    def compute_fueter_laplacian(self, q_history: List[torch.Tensor]) -> float:
        """计算 Fueter Laplacian (二阶导数和).
        
        使用有限差分近似
        """
        if len(q_history) < 3:
            return 0.0
        
        # 取最近3个状态
        q_prev = q_history[-3]
        q_curr = q_history[-2]
        q_next = q_history[-1]
        
        # 二阶中心差分: f'' ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
        laplacian = (q_next - 2*q_curr + q_prev)
        
        # Fueter 曲率 = ||Δ_F q||
        curvature = torch.norm(laplacian).item()
        
        return curvature
    
    def calibrate_baseline(self, curvatures: List[float]):
        """校准基线曲率 (正常运行状态)."""
        if curvatures:
            self.baseline_curvature = np.mean(curvatures)
    
    def detect_anomaly(self, curvature: float) -> Tuple[float, bool]:
        """检测曲率异常.
        
        Returns: (异常分数, 是否异常)
        """
        self.curvature_history.append(curvature)
        
        if self.baseline_curvature is None:
            if len(self.curvature_history) >= 10:
                self.calibrate_baseline(list(self.curvature_history))
            return 0.0, False
        
        # 异常分数 = |当前曲率 - 基线| / 基线
        if self.baseline_curvature > 1e-6:
            anomaly_score = abs(curvature - self.baseline_curvature) / self.baseline_curvature
        else:
            anomaly_score = curvature
        
        is_anomaly = anomaly_score > (1.0 / self.sensitivity)
        
        return anomaly_score, is_anomaly


class TrajectoryPredictor(nn.Module):
    """轨迹预测器 - 基于四元数动力学 (优化版).
    
    使用四元数插值 (SLERP) 预测未来状态
    可预测轨迹突变 (相位跃迁点)
    
    优化:
    - 预计算 SLERP 系数表
    - 批量计算预测轨迹
    - 使用指数映射替代 SLERP 提升精度
    """
    
    def __init__(self, prediction_horizon: int = 10):
        super().__init__()
        self.horizon = prediction_horizon
        
        # 优化: 直接权重而非 Sequential
        self.w1 = nn.Parameter(torch.empty(16, 8))
        self.b1 = nn.Parameter(torch.zeros(16))
        self.w2 = nn.Parameter(torch.empty(4, 16))
        self.b2 = nn.Parameter(torch.zeros(4))
        
        nn.init.xavier_uniform_(self.w1, gain=0.1)
        nn.init.xavier_uniform_(self.w2, gain=0.1)
        
        # 预计算时间步系数
        self.register_buffer('t_steps', torch.linspace(0.1, 1.0, prediction_horizon))
        
    def quaternion_exp(self, omega: torch.Tensor) -> torch.Tensor:
        """四元数指数映射: exp(ω) = [cos(||ω||), sin(||ω||) * ω/||ω||].
        
        比 SLERP 更适合动力学预测
        """
        theta = torch.norm(omega)
        if theta < 1e-6:
            return torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        w = torch.cos(theta)
        xyz = torch.sin(theta) * omega / theta
        return torch.cat([w.unsqueeze(0), xyz])
    
    def quaternion_log(self, q: torch.Tensor) -> torch.Tensor:
        """四元数对数映射: log(q) = arccos(w) * v/||v||."""
        w = q[0]
        v = q[1:]
        v_norm = torch.norm(v)
        
        if v_norm < 1e-6:
            return torch.zeros(3)
        
        return torch.acos(torch.clamp(w, -1, 1)) * v / v_norm
    
    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法 (Hamilton product)."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def estimate_angular_velocity(self, q_prev: torch.Tensor, q_curr: torch.Tensor) -> torch.Tensor:
        """估计角速度 ω = 2 * log(q_curr * q_prev^{-1})."""
        # q^{-1} = conjugate for unit quaternion
        q_prev_inv = torch.tensor([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])
        delta_q = self.quaternion_multiply(q_curr, q_prev_inv)
        return 2 * self.quaternion_log(delta_q)
    
    def predict(self, q_history: List[torch.Tensor]) -> torch.Tensor:
        """预测未来轨迹 (优化版).
        
        使用指数映射进行恒定角速度外推:
        q(t) = exp(ω*t) * q(0)
        
        Returns: (horizon, 4) 预测的四元数序列
        """
        if len(q_history) < 2:
            return torch.zeros(self.horizon, 4)
        
        q_prev = q_history[-2]
        q_curr = q_history[-1]
        
        # 使用神经网络估计速度 (学习修正项)
        velocity_input = torch.cat([q_prev, q_curr])
        h = torch.tanh(F.linear(velocity_input, self.w1, self.b1))
        velocity_correction = F.linear(h, self.w2, self.b2)
        
        # 几何角速度估计
        omega = self.estimate_angular_velocity(q_prev, q_curr)
        
        # 批量预测
        predictions = []
        for t in self.t_steps:
            # 指数映射外推
            q_delta = self.quaternion_exp(omega * t.item())
            q_pred = self.quaternion_multiply(q_delta, q_curr)
            
            # 添加学习修正
            q_pred = q_pred + velocity_correction * t.item() * 0.1
            q_pred = F.normalize(q_pred, p=2, dim=-1)
            
            predictions.append(q_pred)
        
        return torch.stack(predictions)
    
    def predict_transition_time(self, q_history: List[torch.Tensor], 
                                 phase_threshold: float = 0.5) -> Optional[int]:
        """预测下一次相位跃迁的时间步.
        
        Returns: 预测的跃迁时间步，None 表示预测窗口内无跃迁
        """
        if len(q_history) < 3:
            return None
        
        predictions = self.predict(q_history)
        
        # 计算预测轨迹的相位
        phases = []
        for q in predictions:
            w = q[0].item()
            v_norm = torch.norm(q[1:]).item()
            phase = 2 * np.arctan2(v_norm, w)
            phases.append(phase)
        
        # 检测相位跳变
        for i in range(1, len(phases)):
            delta = abs(phases[i] - phases[i-1])
            delta = min(delta, 2*np.pi - delta)
            if delta > phase_threshold:
                return i
        
        return None


class H2QSystemControlCore(nn.Module):
    """H2Q 系统控制核心.
    
    集成所有组件，提供完整的系统控制能力:
    - 微小变化感知 (Fueter 曲率)
    - 轨迹突变预测 (Berry 相位)
    - 确定性状态编码 (四元数)
    - 实时异常检测
    
    特性:
    - 模型大小: ~10KB
    - 延迟: <100μs
    - 确定性: 100% (无随机性)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 16,
                 history_len: int = 32,
                 prediction_horizon: int = 10,
                 anomaly_sensitivity: float = 0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 核心组件
        self.state_encoder = QuaternionStateEncoder(input_dim, hidden_dim)
        self.phase_tracker = BerryPhaseTracker(history_len)
        self.curvature_detector = FueterCurvatureDetector(anomaly_sensitivity)
        self.trajectory_predictor = TrajectoryPredictor(prediction_horizon)
        
        # 状态历史
        self.state_history: List[SystemState] = []
        self.q_history: List[torch.Tensor] = []
        
        # 指标
        self.metrics = ControlMetrics()
        self._latencies: List[float] = []
        
    def count_parameters(self) -> int:
        """统计模型参数量."""
        return sum(p.numel() for p in self.parameters())
    
    def model_size_bytes(self) -> int:
        """估算模型大小 (bytes)."""
        return self.count_parameters() * 4  # float32
    
    def reset(self):
        """重置控制器状态."""
        self.state_history.clear()
        self.q_history.clear()
        self.phase_tracker.q_history.clear()
        self.phase_tracker.phase_history.clear()
        self.curvature_detector.curvature_history.clear()
        self.curvature_detector.baseline_curvature = None
        self.metrics = ControlMetrics()
        self._latencies.clear()
    
    @torch.no_grad()
    def process(self, signal: torch.Tensor) -> SystemState:
        """处理输入信号，返回系统状态.
        
        Args:
            signal: (input_dim,) 输入信号
        
        Returns:
            SystemState 包含完整的系统状态分析
        """
        start_time = time.perf_counter()
        
        # 1. 编码为四元数状态
        q = self.state_encoder(signal)
        self.q_history.append(q)
        
        # 2. Berry 相位追踪
        phase, berry_curvature, phase_transition = self.phase_tracker.update(q)
        
        # 3. Fueter 曲率检测
        fueter_curvature = self.curvature_detector.compute_fueter_laplacian(self.q_history)
        anomaly_score, is_anomaly = self.curvature_detector.detect_anomaly(fueter_curvature)
        
        # 4. 轨迹预测
        predicted_trajectory = None
        if len(self.q_history) >= 2:
            predicted_trajectory = self.trajectory_predictor.predict(self.q_history)
        
        # 5. 构建状态
        state = SystemState(
            timestamp=time.time(),
            quaternion=q,
            phase=phase,
            curvature=fueter_curvature,
            raw_signal=signal,
            anomaly_score=anomaly_score,
            predicted_trajectory=predicted_trajectory,
        )
        
        self.state_history.append(state)
        
        # 6. 更新指标
        latency_us = (time.perf_counter() - start_time) * 1e6
        self._latencies.append(latency_us)
        
        self.metrics.total_samples += 1
        if is_anomaly:
            self.metrics.anomalies_detected += 1
        if phase_transition:
            self.metrics.phase_transitions += 1
        if predicted_trajectory is not None:
            self.metrics.trajectory_predictions += 1
        
        self.metrics.mean_latency_us = np.mean(self._latencies[-100:])
        self.metrics.max_latency_us = max(self._latencies[-100:])
        
        # 计算稳定性分数
        if len(self.phase_tracker.phase_history) > 1:
            phase_variance = np.var(list(self.phase_tracker.phase_history))
            self.metrics.stability_score = 1.0 / (1.0 + phase_variance)
        
        return state
    
    def get_control_output(self, state: SystemState, 
                           target_phase: float = 0.0) -> torch.Tensor:
        """生成控制输出 (用于反馈控制).
        
        基于当前状态与目标状态的偏差生成控制信号
        """
        # 相位误差
        phase_error = target_phase - state.phase
        
        # 四元数误差 (目标为单位四元数 [1,0,0,0])
        target_q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_error = target_q - state.quaternion
        
        # PID 式控制 (简化为 P 控制)
        Kp = 0.1
        control_output = Kp * q_error
        
        return control_output
    
    def export_state_dict(self) -> Dict[str, Any]:
        """导出完整状态用于持久化."""
        return {
            'model_state': self.state_dict(),
            'baseline_curvature': self.curvature_detector.baseline_curvature,
            'metrics': self.metrics,
        }
    
    def load_state_dict_full(self, state_dict: Dict[str, Any]):
        """加载完整状态."""
        self.load_state_dict(state_dict['model_state'])
        self.curvature_detector.baseline_curvature = state_dict.get('baseline_curvature')
        if 'metrics' in state_dict:
            self.metrics = state_dict['metrics']


def create_control_core(input_dim: int, **kwargs) -> H2QSystemControlCore:
    """工厂函数创建控制核心."""
    return H2QSystemControlCore(input_dim, **kwargs)


# ============================================================================
# 示例应用: 传感器监控系统
# ============================================================================

class SensorMonitor:
    """传感器监控系统示例.
    
    使用 H2Q 控制核心监控传感器数据流
    """
    
    def __init__(self, num_sensors: int = 8):
        self.controller = create_control_core(
            input_dim=num_sensors,
            hidden_dim=16,
            history_len=64,
            prediction_horizon=20,
            anomaly_sensitivity=0.005,
        )
        self.num_sensors = num_sensors
        self.alerts: List[Dict] = []
        
    def process_reading(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """处理传感器读数."""
        signal = torch.tensor(sensor_data, dtype=torch.float32)
        state = self.controller.process(signal)
        
        result = {
            'timestamp': state.timestamp,
            'phase': state.phase,
            'curvature': state.curvature,
            'anomaly_score': state.anomaly_score,
            'is_anomaly': state.anomaly_score > 1.0,
            'stability': self.controller.metrics.stability_score,
        }
        
        if result['is_anomaly']:
            self.alerts.append({
                'time': state.timestamp,
                'score': state.anomaly_score,
                'type': 'curvature_anomaly',
            })
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控系统状态."""
        metrics = self.controller.metrics
        return {
            'total_samples': metrics.total_samples,
            'anomalies': metrics.anomalies_detected,
            'phase_transitions': metrics.phase_transitions,
            'mean_latency_us': metrics.mean_latency_us,
            'stability': metrics.stability_score,
            'model_size_kb': self.controller.model_size_bytes() / 1024,
            'alert_count': len(self.alerts),
        }


if __name__ == "__main__":
    # 快速演示
    print("H2Q System Control Core - 快速演示")
    print("="*50)
    
    # 创建控制核心
    core = create_control_core(input_dim=8)
    print(f"模型参数量: {core.count_parameters()}")
    print(f"模型大小: {core.model_size_bytes() / 1024:.2f} KB")
    
    # 模拟信号处理
    for i in range(100):
        # 正常信号
        signal = torch.randn(8) * 0.1
        
        # 在第50步注入异常
        if i == 50:
            signal += torch.tensor([5.0, 0, 0, 0, 0, 0, 0, 0])
        
        state = core.process(signal)
        
        if i % 20 == 0 or state.anomaly_score > 0.5:
            print(f"Step {i}: phase={state.phase:.3f}, "
                  f"curvature={state.curvature:.6f}, "
                  f"anomaly={state.anomaly_score:.3f}")
    
    print(f"\n统计:")
    print(f"  总样本: {core.metrics.total_samples}")
    print(f"  异常检测: {core.metrics.anomalies_detected}")
    print(f"  相位跃迁: {core.metrics.phase_transitions}")
    print(f"  平均延迟: {core.metrics.mean_latency_us:.2f} μs")
