"""H2Q Lightweight Control Core - 超轻量高性能版本.

专为工业实时控制场景优化:
- 模型大小: <2KB
- 延迟: <50μs
- 确定性: 100%
- 微变化灵敏度: 可配置

优化策略:
1. 纯 NumPy 实现核心逻辑 (避免 PyTorch 开销)
2. 预分配数组 (避免动态内存分配)
3. 向量化计算 (避免 Python 循环)
4. 简化的相位追踪 (线性复杂度)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import time


@dataclass
class LightweightState:
    """轻量级系统状态."""
    timestamp: float
    quaternion: np.ndarray  # (4,)
    phase: float
    curvature: float
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    phase_transition: bool = False


@dataclass
class LightweightMetrics:
    """性能指标."""
    total_samples: int = 0
    anomalies_detected: int = 0
    phase_transitions: int = 0
    mean_latency_us: float = 0.0
    max_latency_us: float = 0.0


class LightweightEncoder:
    """超轻量四元数编码器 (纯 NumPy).
    
    参数量: input_dim * 16 + 16 * 4 = 16*input_dim + 64
    例: input_dim=8 → 192 参数 ≈ 768 bytes
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 16, seed: int = 42):
        np.random.seed(seed)
        
        # 初始化权重 (Xavier)
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim)) * 0.1
        scale2 = np.sqrt(2.0 / (hidden_dim + 4)) * 0.1
        
        self.w1 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(4, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(4, dtype=np.float32)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """编码信号为单位四元数."""
        # 两层全连接 + tanh + normalize
        h = np.tanh(self.w1 @ x + self.b1)
        q = self.w2 @ h + self.b2
        
        # 归一化到单位球面
        norm = np.sqrt(np.sum(q * q))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return q / norm
    
    def parameter_count(self) -> int:
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size
    
    def size_bytes(self) -> int:
        return self.parameter_count() * 4  # float32


class LightweightPhaseTracker:
    """轻量级 Berry 相位追踪器 (增强版).
    
    使用循环缓冲区避免动态内存分配
    增强相位跃迁检测灵敏度
    """
    
    def __init__(self, history_len: int = 32):
        self.history_len = history_len
        self.phases = np.zeros(history_len, dtype=np.float32)
        self.write_idx = 0
        self.count = 0
        
        # 跃迁检测参数
        self.transition_threshold = 0.3  # 降低阈值提高灵敏度
        
        # 自适应基线
        self.phase_mean = None
        self.phase_std = None
        self.calibration_count = 0
        
    def compute_phase(self, q: np.ndarray) -> float:
        """计算 Berry 相位 = 2 * arctan2(||v||, w)."""
        w = q[0]
        v_norm = np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
        return 2 * np.arctan2(v_norm, w)
    
    def update(self, q: np.ndarray) -> Tuple[float, float, bool]:
        """更新并返回 (相位, 曲率, 是否跃迁)."""
        phase = self.compute_phase(q)
        
        # 获取前一个相位用于比较
        prev_phase = self.phases[(self.write_idx - 1) % self.history_len] if self.count > 0 else phase
        
        # 存储到循环缓冲区
        self.phases[self.write_idx] = phase
        self.write_idx = (self.write_idx + 1) % self.history_len
        self.count = min(self.count + 1, self.history_len)
        
        # 计算曲率 (相位变化率的标准差)
        if self.count < 2:
            return phase, 0.0, False
        
        # 获取有效历史
        if self.count == self.history_len:
            phases = self.phases
        else:
            phases = self.phases[:self.count]
        
        # 相位差
        dphases = np.diff(phases)
        # 处理 wrap-around
        dphases = np.where(dphases > np.pi, dphases - 2*np.pi, dphases)
        dphases = np.where(dphases < -np.pi, dphases + 2*np.pi, dphases)
        
        curvature = float(np.std(dphases)) if len(dphases) > 0 else 0.0
        
        # 校准基线
        if self.count == 30 and self.phase_mean is None:
            self.phase_mean = float(np.mean(np.abs(dphases)))
            self.phase_std = float(np.std(np.abs(dphases)))
            if self.phase_std < 1e-6:
                self.phase_std = 1e-6
        
        # 检测跃迁 (当前相位变化是否异常大)
        transition = False
        if self.count > 1:
            # 计算当前相位变化
            delta = phase - prev_phase
            # 处理 wrap-around
            if delta > np.pi:
                delta -= 2*np.pi
            elif delta < -np.pi:
                delta += 2*np.pi
            
            abs_delta = abs(delta)
            
            # 方法1: 固定阈值
            if abs_delta > self.transition_threshold:
                transition = True
            
            # 方法2: 统计异常 (如果已校准)
            if self.phase_mean is not None and not transition:
                z_score = (abs_delta - self.phase_mean) / self.phase_std
                if z_score > 3.0:  # 3-sigma 规则
                    transition = True
        
        return phase, curvature, transition
    
    def reset(self):
        self.phases.fill(0)
        self.write_idx = 0
        self.count = 0
        self.phase_mean = None
        self.phase_std = None


class LightweightCurvatureDetector:
    """轻量级 Fueter 曲率检测器 (增强版).
    
    使用滑动窗口统计检测异常
    """
    
    def __init__(self, sensitivity: float = 0.01):
        self.sensitivity = sensitivity
        self.baseline_mean = None
        self.baseline_std = None
        self.curvatures = np.zeros(100, dtype=np.float32)
        self.write_idx = 0
        self.count = 0
        
        # 存储最近3个四元数用于 Laplacian 计算
        self.q_buffer = np.zeros((3, 4), dtype=np.float32)
        self.q_count = 0
        
        # Z-score 阈值: sensitivity 越高，阈值越低 (越敏感)
        self.z_threshold = max(1.5, 3.0 * (1.0 - sensitivity * 10))
        
    def update_q_buffer(self, q: np.ndarray):
        """更新四元数缓冲区."""
        self.q_buffer[:-1] = self.q_buffer[1:]
        self.q_buffer[-1] = q
        self.q_count = min(self.q_count + 1, 3)
    
    def compute_laplacian(self) -> float:
        """计算 Fueter Laplacian (二阶差分)."""
        if self.q_count < 3:
            return 0.0
        
        # 二阶中心差分
        laplacian = self.q_buffer[2] - 2*self.q_buffer[1] + self.q_buffer[0]
        return float(np.sqrt(np.sum(laplacian * laplacian)))
    
    def detect(self, q: np.ndarray) -> Tuple[float, bool]:
        """检测异常并返回 (异常分数, 是否异常).
        
        使用 Z-score 方法: z = (x - μ) / σ
        """
        self.update_q_buffer(q)
        curvature = self.compute_laplacian()
        
        # 存储曲率
        self.curvatures[self.write_idx] = curvature
        self.write_idx = (self.write_idx + 1) % 100
        self.count = min(self.count + 1, 100)
        
        # 校准基线 (使用前50个样本)
        if self.count == 50 and self.baseline_mean is None:
            valid = self.curvatures[:self.count]
            self.baseline_mean = float(np.mean(valid))
            self.baseline_std = float(np.std(valid))
            if self.baseline_std < 1e-8:
                self.baseline_std = self.baseline_mean * 0.1 if self.baseline_mean > 0 else 0.01
        
        if self.baseline_mean is None:
            return 0.0, False
        
        # 计算 Z-score
        z_score = abs(curvature - self.baseline_mean) / self.baseline_std
        
        # 异常分数 = Z-score / 阈值 (归一化到0-1范围)
        anomaly_score = min(1.0, z_score / self.z_threshold)
        is_anomaly = z_score > self.z_threshold
        
        return anomaly_score, is_anomaly
    
    def reset(self):
        self.baseline_mean = None
        self.baseline_std = None
        self.curvatures.fill(0)
        self.write_idx = 0
        self.count = 0
        self.q_buffer.fill(0)
        self.q_count = 0


class LightweightTrajectoryPredictor:
    """轻量级轨迹预测器 (几何方法).
    
    使用纯四元数几何进行预测，无需神经网络
    """
    
    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.t_steps = np.linspace(0.1, 1.0, horizon, dtype=np.float32)
        
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法 (Hamilton product)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float32)
    
    def quaternion_slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """球面线性插值."""
        dot = np.dot(q0, q1)
        
        # 确保最短路径
        if dot < 0:
            q1 = -q1
            dot = -dot
        
        if dot > 0.9995:
            # 近似线性插值
            result = q0 + t * (q1 - q0)
        else:
            theta = np.arccos(np.clip(dot, -1, 1))
            sin_theta = np.sin(theta)
            result = (np.sin((1-t)*theta)/sin_theta) * q0 + (np.sin(t*theta)/sin_theta) * q1
        
        # 归一化
        norm = np.sqrt(np.sum(result * result))
        return result / norm if norm > 1e-8 else q0
    
    def predict(self, q_prev: np.ndarray, q_curr: np.ndarray) -> np.ndarray:
        """基于恒定角速度预测轨迹.
        
        Returns: (horizon, 4) 预测轨迹
        """
        # 计算 delta = q_curr * q_prev^{-1}
        q_prev_inv = np.array([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])
        delta = self.quaternion_multiply(q_curr, q_prev_inv)
        
        # 外推: q(t) = slerp(q_curr, q_curr * delta, t)
        q_target = self.quaternion_multiply(q_curr, delta)
        
        predictions = np.zeros((self.horizon, 4), dtype=np.float32)
        for i, t in enumerate(self.t_steps):
            predictions[i] = self.quaternion_slerp(q_curr, q_target, t)
        
        return predictions


class H2QLightweightControl:
    """H2Q 超轻量控制核心.
    
    特性:
    - 模型大小: <2KB
    - 延迟: <50μs
    - 确定性: 100%
    - 纯 NumPy 实现
    """
    
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dim: int = 16,
                 history_len: int = 32,
                 prediction_horizon: int = 10,
                 anomaly_sensitivity: float = 0.01,
                 seed: int = 42):
        
        self.input_dim = input_dim
        
        # 组件
        self.encoder = LightweightEncoder(input_dim, hidden_dim, seed)
        self.phase_tracker = LightweightPhaseTracker(history_len)
        self.curvature_detector = LightweightCurvatureDetector(anomaly_sensitivity)
        self.trajectory_predictor = LightweightTrajectoryPredictor(prediction_horizon)
        
        # 状态
        self.prev_q = None
        self.metrics = LightweightMetrics()
        self._latencies: List[float] = []
        
    def reset(self):
        """重置控制器."""
        self.phase_tracker.reset()
        self.curvature_detector.reset()
        self.prev_q = None
        self.metrics = LightweightMetrics()
        self._latencies.clear()
    
    def process(self, signal: np.ndarray) -> LightweightState:
        """处理输入信号.
        
        Args:
            signal: (input_dim,) 输入信号，NumPy 数组
        
        Returns:
            LightweightState
        """
        start = time.perf_counter()
        
        # 1. 编码为四元数
        q = self.encoder.encode(signal.astype(np.float32))
        
        # 2. 相位追踪
        phase, curvature, phase_transition = self.phase_tracker.update(q)
        
        # 3. 曲率异常检测
        anomaly_score, is_anomaly = self.curvature_detector.detect(q)
        
        # 4. 构建状态
        state = LightweightState(
            timestamp=time.time(),
            quaternion=q,
            phase=phase,
            curvature=curvature,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            phase_transition=phase_transition,
        )
        
        # 更新指标
        self.metrics.total_samples += 1
        if is_anomaly:
            self.metrics.anomalies_detected += 1
        if phase_transition:
            self.metrics.phase_transitions += 1
        
        # 记录延迟
        latency_us = (time.perf_counter() - start) * 1e6
        self._latencies.append(latency_us)
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]
        
        self.metrics.mean_latency_us = np.mean(self._latencies)
        self.metrics.max_latency_us = max(self._latencies)
        
        self.prev_q = q
        return state
    
    def predict_trajectory(self) -> Optional[np.ndarray]:
        """预测未来轨迹.
        
        Returns: (horizon, 4) 或 None
        """
        if self.prev_q is None:
            return None
        
        # 使用当前和前一个状态预测
        # 由于我们只存储 prev_q，使用 identity 作为更早的状态
        q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        if self.phase_tracker.count < 2:
            return None
        
        return self.trajectory_predictor.predict(q_identity, self.prev_q)
    
    def model_size_bytes(self) -> int:
        """返回模型大小 (bytes)."""
        return self.encoder.size_bytes()
    
    def parameter_count(self) -> int:
        """返回参数数量."""
        return self.encoder.parameter_count()


def create_lightweight_control(
    input_dim: int = 8,
    hidden_dim: int = 16,
    history_len: int = 32,
    prediction_horizon: int = 10,
    anomaly_sensitivity: float = 0.01,
    seed: int = 42,
) -> H2QLightweightControl:
    """创建轻量级控制核心."""
    return H2QLightweightControl(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        history_len=history_len,
        prediction_horizon=prediction_horizon,
        anomaly_sensitivity=anomaly_sensitivity,
        seed=seed,
    )


# ============== 演示和测试 ==============

def demo_lightweight_control():
    """演示轻量级控制核心."""
    print("=" * 60)
    print("H2Q Lightweight Control Core Demo")
    print("=" * 60)
    
    # 创建控制器
    controller = create_lightweight_control(
        input_dim=8,
        anomaly_sensitivity=0.01,
    )
    
    print(f"模型大小: {controller.model_size_bytes()} bytes")
    print(f"参数数量: {controller.parameter_count()}")
    
    # 模拟正常信号
    print("\n1. 正常信号 (校准阶段)...")
    np.random.seed(123)
    
    for i in range(50):
        signal = np.random.randn(8) * 0.1
        state = controller.process(signal)
    
    print(f"   校准后基线曲率: {controller.curvature_detector.baseline:.6f}")
    
    # 模拟稳定运行
    print("\n2. 稳定运行...")
    anomaly_count = 0
    for i in range(100):
        signal = np.random.randn(8) * 0.1
        state = controller.process(signal)
        if state.is_anomaly:
            anomaly_count += 1
    
    print(f"   异常检测: {anomaly_count}/100")
    
    # 注入异常
    print("\n3. 注入异常...")
    for i in range(10):
        # 正常
        for _ in range(10):
            signal = np.random.randn(8) * 0.1
            controller.process(signal)
        
        # 异常 (突变)
        signal = np.random.randn(8) * 2.0
        state = controller.process(signal)
        print(f"   异常 {i+1}: score={state.anomaly_score:.4f}, detected={state.is_anomaly}")
    
    # 性能统计
    print("\n4. 性能统计:")
    print(f"   总样本: {controller.metrics.total_samples}")
    print(f"   检测到的异常: {controller.metrics.anomalies_detected}")
    print(f"   相位跃迁: {controller.metrics.phase_transitions}")
    print(f"   平均延迟: {controller.metrics.mean_latency_us:.2f} μs")
    print(f"   最大延迟: {controller.metrics.max_latency_us:.2f} μs")


if __name__ == "__main__":
    demo_lightweight_control()
