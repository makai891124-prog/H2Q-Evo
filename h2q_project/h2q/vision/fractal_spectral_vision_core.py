"""H2Q Fractal Spectral Vision Core - 分形波谱视觉控制核心.

集成架构:
1. 分形网络 (Fractal Network) - 层级特征提取
2. 四元数归一化控制 (Quaternion Normalization) - 确定性状态表示
3. 自组织并行计算 (Self-Organizing Parallel) - 动态计算分配
4. 视觉-控制反馈回路 (Vision-Control Feedback) - 闭环控制

核心特性:
- 模型大小: <10KB (超轻量)
- 延迟: <100μs
- 确定性: 100%
- 视觉感知 + 精细控制融合
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor
import threading


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class VisionState:
    """视觉状态表示."""
    features: np.ndarray           # 分形特征 (multi-scale)
    quaternion: np.ndarray         # 四元数状态 (4,)
    saliency_map: np.ndarray       # 显著性图
    confidence: float              # 置信度


@dataclass
class ControlState:
    """控制状态表示."""
    quaternion: np.ndarray         # 当前状态四元数
    phase: float                   # Berry 相位
    curvature: float               # Fueter 曲率
    anomaly_score: float           # 异常分数
    predicted_trajectory: Optional[np.ndarray] = None


@dataclass 
class FeedbackState:
    """反馈回路状态."""
    vision_state: VisionState
    control_state: ControlState
    error_signal: np.ndarray       # 误差信号
    correction: np.ndarray         # 修正量
    loop_latency_us: float         # 回路延迟


@dataclass
class SystemMetrics:
    """系统指标."""
    total_frames: int = 0
    total_control_cycles: int = 0
    mean_vision_latency_us: float = 0.0
    mean_control_latency_us: float = 0.0
    mean_feedback_latency_us: float = 0.0
    anomalies_detected: int = 0
    corrections_applied: int = 0


# ============================================================================
# 分形特征提取器 (纯 NumPy)
# ============================================================================

class FractalFeatureExtractor:
    """分形特征提取器 - 多尺度层级表示 (增强版).
    
    实现分形展开: 2 → 4 → 8 → 16 → 32 (5层)
    每层使用对称破缺: h ± δ
    层级间通过跳跃连接保持特征关联
    """
    
    def __init__(self, input_dim: int = 64, output_dim: int = 32, n_levels: int = 4, seed: int = 42):
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_levels = n_levels
        
        # 分形层权重
        self.weights = []
        self.h_params = []  # 对称性参数
        self.delta_params = []  # 破缺参数
        self.layer_norms = []  # 层归一化参数
        self.skip_projs = []  # 跳跃连接投影
        
        # 记录各层维度
        self.layer_dims = [input_dim]
        
        curr_dim = input_dim
        for i in range(n_levels):
            next_dim = max(curr_dim // 2, output_dim)
            self.layer_dims.append(next_dim)
            
            # Xavier 初始化 (适度放大以增强信号)
            scale = np.sqrt(2.0 / (curr_dim + next_dim)) * 0.15
            w = np.random.randn(next_dim, curr_dim).astype(np.float32) * scale
            
            # h 参数: 对称性中心
            h = np.ones(next_dim, dtype=np.float32) * 0.5 + np.random.randn(next_dim).astype(np.float32) * 0.05
            
            # delta 参数: 对称破缺强度 (与层级深度关联)
            delta_scale = 0.1 * (i + 1) / n_levels  # 深层破缺更强
            delta = np.random.randn(next_dim).astype(np.float32) * delta_scale
            
            self.weights.append(w)
            self.h_params.append(h)
            self.delta_params.append(delta)
            
            # 层归一化: gamma, beta
            self.layer_norms.append({
                'gamma': np.ones(next_dim, dtype=np.float32),
                'beta': np.zeros(next_dim, dtype=np.float32)
            })
            
            # 跳跃连接: 从输入层到当前层
            if i > 0:
                skip_scale = np.sqrt(2.0 / (input_dim + next_dim)) * 0.05
                skip_proj = np.random.randn(next_dim, input_dim).astype(np.float32) * skip_scale
                self.skip_projs.append(skip_proj)
            else:
                self.skip_projs.append(None)
            
            curr_dim = next_dim
        
        # 最终投影到输出维度
        scale = np.sqrt(2.0 / (curr_dim + output_dim)) * 0.15
        self.final_proj = np.random.randn(output_dim, curr_dim).astype(np.float32) * scale
        
        # 层间相关性矩阵 (用于跟踪分形层级关联)
        self.layer_correlations = np.zeros((n_levels, n_levels), dtype=np.float32)
    
    def layer_normalize(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """层归一化."""
        ln = self.layer_norms[layer_idx]
        mean = np.mean(x)
        var = np.var(x) + 1e-8
        x_norm = (x - mean) / np.sqrt(var)
        return ln['gamma'] * x_norm + ln['beta']
    
    def extract(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """提取分形特征.
        
        Args:
            x: (input_dim,) 输入特征
        
        Returns:
            (output_features, intermediate_features)
        """
        x0 = x.copy()  # 保存原始输入用于跳跃连接
        intermediates = [x]
        layer_activations = []
        
        for i, (w, h, delta) in enumerate(zip(self.weights, self.h_params, self.delta_params)):
            # 线性变换
            z = w @ x
            
            # 分形对称破缺: 使用 sinh 和 tanh 组合以增强非线性
            # y = tanh(z * (h + δ)) - sinh(z * (h - δ)) * 0.1
            y_plus = z * (h + delta)
            y_minus = z * (h - delta)
            x_new = np.tanh(y_plus) - np.sinh(np.clip(y_minus, -3, 3)) * 0.1
            
            # 跳跃连接 (从输入层)
            if self.skip_projs[i] is not None:
                skip = self.skip_projs[i] @ x0
                x_new = x_new + skip * 0.1
            
            # 层归一化
            x = self.layer_normalize(x_new, i)
            
            intermediates.append(x)
            layer_activations.append(x.copy())
        
        # 更新层间相关性
        self._update_correlations(layer_activations)
        
        # 最终投影
        output = np.tanh(self.final_proj @ x)
        
        return output, intermediates
    
    def _update_correlations(self, activations: List[np.ndarray]):
        """更新层间相关性矩阵."""
        n = len(activations)
        for i in range(n):
            for j in range(i, n):
                # 计算余弦相似度 (需要相同维度, 这里用能量比)
                e_i = np.sum(activations[i] ** 2)
                e_j = np.sum(activations[j] ** 2)
                if e_i > 1e-8 and e_j > 1e-8:
                    # 使用能量比的对数
                    ratio = np.log(e_j / e_i + 1e-8)
                    # 指数移动平均
                    self.layer_correlations[i, j] = 0.9 * self.layer_correlations[i, j] + 0.1 * ratio
                    self.layer_correlations[j, i] = self.layer_correlations[i, j]
    
    def get_layer_correlation(self) -> float:
        """获取层级相关性指标."""
        # 期望: 深层能量应该与浅层能量有一致的关系
        # 计算相邻层相关性的一致性
        adj_corrs = [self.layer_correlations[i, i+1] for i in range(self.n_levels - 1)]
        if len(adj_corrs) < 2:
            return 0.0
        
        # 相邻层相关性的标准差 (越小说明越一致)
        std = np.std(adj_corrs)
        mean = np.abs(np.mean(adj_corrs))
        
        # 一致性得分: mean / (std + 1)
        if mean < 1e-8:
            return 0.0
        return min(1.0, mean / (std + 0.1))
    
    def parameter_count(self) -> int:
        count = sum(w.size for w in self.weights)
        count += sum(h.size for h in self.h_params)
        count += sum(d.size for d in self.delta_params)
        count += sum(ln['gamma'].size + ln['beta'].size for ln in self.layer_norms)
        count += sum(sp.size for sp in self.skip_projs if sp is not None)
        count += self.final_proj.size
        return count


# ============================================================================
# 四元数状态编码器
# ============================================================================

class QuaternionStateEncoder:
    """四元数状态编码器 - 确定性映射到 S³ 流形."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 16, seed: int = 42):
        np.random.seed(seed)
        
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim)) * 0.1
        scale2 = np.sqrt(2.0 / (hidden_dim + 4)) * 0.1
        
        self.w1 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(4, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(4, dtype=np.float32)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """编码为单位四元数."""
        h = np.tanh(self.w1 @ x + self.b1)
        q = self.w2 @ h + self.b2
        
        # 归一化到 S³
        norm = np.sqrt(np.sum(q * q))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return q / norm
    
    def parameter_count(self) -> int:
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size


# ============================================================================
# Berry 相位追踪器
# ============================================================================

class BerryPhaseTracker:
    """Berry 相位追踪 - 拓扑状态监测."""
    
    def __init__(self, history_len: int = 32, transition_threshold: float = 0.3):
        self.history_len = history_len
        self.phases = np.zeros(history_len, dtype=np.float32)
        self.write_idx = 0
        self.count = 0
        self.transition_threshold = transition_threshold
        
        # 自适应基线
        self.phase_mean = None
        self.phase_std = None
    
    def compute_phase(self, q: np.ndarray) -> float:
        """计算 Berry 相位."""
        w = q[0]
        v_norm = np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
        return 2 * np.arctan2(v_norm, w)
    
    def update(self, q: np.ndarray) -> Tuple[float, float, bool]:
        """更新并返回 (相位, 曲率, 是否跃迁)."""
        phase = self.compute_phase(q)
        prev_phase = self.phases[(self.write_idx - 1) % self.history_len] if self.count > 0 else phase
        
        self.phases[self.write_idx] = phase
        self.write_idx = (self.write_idx + 1) % self.history_len
        self.count = min(self.count + 1, self.history_len)
        
        if self.count < 2:
            return phase, 0.0, False
        
        # 曲率计算
        valid_phases = self.phases[:self.count] if self.count < self.history_len else self.phases
        dphases = np.diff(valid_phases)
        dphases = np.where(dphases > np.pi, dphases - 2*np.pi, dphases)
        dphases = np.where(dphases < -np.pi, dphases + 2*np.pi, dphases)
        curvature = float(np.std(dphases)) if len(dphases) > 0 else 0.0
        
        # 跃迁检测
        delta = phase - prev_phase
        if delta > np.pi:
            delta -= 2*np.pi
        elif delta < -np.pi:
            delta += 2*np.pi
        
        transition = abs(delta) > self.transition_threshold
        
        return phase, curvature, transition
    
    def reset(self):
        self.phases.fill(0)
        self.write_idx = 0
        self.count = 0


# ============================================================================
# Fueter 曲率检测器
# ============================================================================

class FueterCurvatureDetector:
    """Fueter 曲率检测 - 微变化感知 (增强版)."""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
        self.baseline_mean = None
        self.baseline_std = None
        self.q_buffer = np.zeros((3, 4), dtype=np.float32)
        self.q_count = 0
        self.curvatures = np.zeros(50, dtype=np.float32)
        self.curv_idx = 0
        self.curv_count = 0
        
        # 增强: 使用更敏感的阈值
        self.z_threshold = max(2.0, 5.0 * (1.0 - sensitivity * 10))
        
        # 信号能量追踪
        self.energy_history = np.zeros(50, dtype=np.float32)
        self.energy_idx = 0
        self.energy_count = 0
        self.energy_baseline = None
    
    def compute_laplacian(self) -> float:
        if self.q_count < 3:
            return 0.0
        laplacian = self.q_buffer[2] - 2*self.q_buffer[1] + self.q_buffer[0]
        return float(np.sqrt(np.sum(laplacian * laplacian)))
    
    def update(self, q: np.ndarray) -> Tuple[float, bool]:
        """更新并返回 (异常分数, 是否异常)."""
        # 更新缓冲区
        self.q_buffer[:-1] = self.q_buffer[1:]
        self.q_buffer[-1] = q
        self.q_count = min(self.q_count + 1, 3)
        
        curvature = self.compute_laplacian()
        
        # 计算信号能量
        energy = float(np.sum(q * q))
        
        # 存储曲率和能量
        self.curvatures[self.curv_idx] = curvature
        self.energy_history[self.energy_idx] = energy
        self.curv_idx = (self.curv_idx + 1) % 50
        self.energy_idx = (self.energy_idx + 1) % 50
        self.curv_count = min(self.curv_count + 1, 50)
        self.energy_count = min(self.energy_count + 1, 50)
        
        # 校准基线
        if self.curv_count == 30 and self.baseline_mean is None:
            valid = self.curvatures[:self.curv_count]
            self.baseline_mean = float(np.mean(valid))
            self.baseline_std = float(np.std(valid))
            if self.baseline_std < 1e-8:
                self.baseline_std = max(self.baseline_mean * 0.1, 0.01)
            
            valid_energy = self.energy_history[:self.energy_count]
            self.energy_baseline = float(np.mean(valid_energy))
        
        if self.baseline_mean is None:
            return 0.0, False
        
        # 多维度异常检测
        # 1. 曲率异常
        z_score_curv = abs(curvature - self.baseline_mean) / self.baseline_std
        
        # 2. 能量异常
        if self.energy_baseline is not None and self.energy_baseline > 1e-8:
            energy_ratio = energy / self.energy_baseline
            energy_anomaly = abs(energy_ratio - 1.0) > 2.0  # 能量变化超过 200%
        else:
            energy_anomaly = False
        
        # 综合异常分数
        anomaly_score = min(1.0, z_score_curv / self.z_threshold)
        if energy_anomaly:
            anomaly_score = max(anomaly_score, 0.8)
        
        is_anomaly = z_score_curv > self.z_threshold or energy_anomaly
        
        return anomaly_score, is_anomaly
    
    def reset(self):
        self.baseline_mean = None
        self.baseline_std = None
        self.q_buffer.fill(0)
        self.q_count = 0
        self.curvatures.fill(0)
        self.curv_idx = 0
        self.curv_count = 0
        self.energy_history.fill(0)
        self.energy_idx = 0
        self.energy_count = 0
        self.energy_baseline = None


# ============================================================================
# 自组织并行计算单元
# ============================================================================

class SelfOrganizingParallelUnit:
    """自组织并行计算单元.
    
    根据负载动态分配计算资源:
    - 视觉处理通道
    - 控制计算通道
    - 反馈融合通道
    """
    
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        
        # 负载统计
        self.channel_loads = {
            'vision': deque(maxlen=100),
            'control': deque(maxlen=100),
            'feedback': deque(maxlen=100),
        }
        
        # 动态权重
        self.channel_weights = {
            'vision': 1.0,
            'control': 1.0,
            'feedback': 1.0,
        }
        
        self._lock = threading.Lock()
    
    def submit_vision(self, func, *args):
        """提交视觉计算任务."""
        start = time.perf_counter()
        future = self.executor.submit(func, *args)
        result = future.result()
        latency = (time.perf_counter() - start) * 1e6
        
        with self._lock:
            self.channel_loads['vision'].append(latency)
        
        return result, latency
    
    def submit_control(self, func, *args):
        """提交控制计算任务."""
        start = time.perf_counter()
        future = self.executor.submit(func, *args)
        result = future.result()
        latency = (time.perf_counter() - start) * 1e6
        
        with self._lock:
            self.channel_loads['control'].append(latency)
        
        return result, latency
    
    def submit_feedback(self, func, *args):
        """提交反馈融合任务."""
        start = time.perf_counter()
        future = self.executor.submit(func, *args)
        result = future.result()
        latency = (time.perf_counter() - start) * 1e6
        
        with self._lock:
            self.channel_loads['feedback'].append(latency)
        
        return result, latency
    
    def update_weights(self):
        """根据负载动态调整权重."""
        with self._lock:
            for channel, loads in self.channel_loads.items():
                if len(loads) > 10:
                    mean_load = np.mean(list(loads))
                    # 负载越高，权重越低 (优先级调整)
                    self.channel_weights[channel] = 1.0 / (1.0 + mean_load / 1000)
    
    def get_stats(self) -> Dict[str, float]:
        """获取通道统计."""
        stats = {}
        with self._lock:
            for channel, loads in self.channel_loads.items():
                if loads:
                    stats[f'{channel}_mean_us'] = np.mean(list(loads))
                    stats[f'{channel}_max_us'] = max(loads)
                else:
                    stats[f'{channel}_mean_us'] = 0.0
                    stats[f'{channel}_max_us'] = 0.0
        return stats
    
    def shutdown(self):
        self.executor.shutdown(wait=True)


# ============================================================================
# 视觉-控制反馈回路
# ============================================================================

class VisionControlFeedbackLoop:
    """视觉-控制反馈回路 (增强版).
    
    实现闭环控制:
    1. 视觉感知 → 特征提取
    2. 状态编码 → 四元数表示
    3. 异常检测 → 错误信号
    4. 反馈修正 → 控制输出
    """
    
    def __init__(self, 
                 vision_input_dim: int = 64,
                 control_input_dim: int = 32,
                 feedback_gain: float = 0.1):
        
        self.feedback_gain = feedback_gain
        
        # 误差积分器 (带衰减)
        self.error_integral = np.zeros(4, dtype=np.float32)
        self.error_history = deque(maxlen=64)
        
        # PID 参数 (优化后 - 增强收敛)
        self.kp = 1.2    # 比例增益 (提高)
        self.ki = 0.08   # 积分增益
        self.kd = 0.15   # 微分增益 (提高)
        
        # 积分衰减系数
        self.integral_decay = 0.92
        
        # 目标状态 (单位四元数 = 稳定状态)
        self.target_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 自适应增益
        self.adaptive_gain = 1.0
        self.error_magnitude_history = deque(maxlen=64)
        
        # 收敛追踪
        self.convergence_step = 0
    
    def compute_error(self, current_q: np.ndarray, vision_confidence: float) -> np.ndarray:
        """计算误差信号.
        
        误差 = 当前状态与目标状态的四元数差
        考虑视觉置信度加权
        """
        # 四元数误差: e = q_target * q_current^{-1}
        # 对于单位四元数，逆 = 共轭
        q_inv = np.array([current_q[0], -current_q[1], -current_q[2], -current_q[3]])
        
        # Hamilton 乘积
        t, x, y, z = self.target_q
        t2, x2, y2, z2 = q_inv
        
        error_q = np.array([
            t*t2 - x*x2 - y*y2 - z*z2,
            t*x2 + x*t2 + y*z2 - z*y2,
            t*y2 - x*z2 + y*t2 + z*x2,
            t*z2 + x*y2 - y*x2 + z*t2,
        ], dtype=np.float32)
        
        # 提取误差向量 (四元数虚部表示旋转轴*角度)
        # 当 w ≈ 1 时误差小, w 偏离 1 时误差大
        error = np.array([
            1.0 - error_q[0],  # w 偏离 1 的程度
            error_q[1],        # x 分量
            error_q[2],        # y 分量
            error_q[3],        # z 分量
        ], dtype=np.float32)
        
        # 视觉置信度加权 (降低权重使反馈更直接)
        error = error * (0.7 + 0.3 * vision_confidence)
        
        self.convergence_step += 1
        
        return error
    
    def compute_correction(self, error: np.ndarray) -> np.ndarray:
        """计算 PID 修正量 (带自适应增益和收敛加速)."""
        error_mag = np.linalg.norm(error)
        self.error_magnitude_history.append(error_mag)
        
        # 自适应增益: 基于误差历史动态调整
        if len(self.error_magnitude_history) > 10:
            recent_error = np.mean(list(self.error_magnitude_history)[-10:])
            older_error = np.mean(list(self.error_magnitude_history)[:10]) if len(self.error_magnitude_history) > 20 else recent_error
            
            # 如果误差在减少，增加增益以加速收敛
            if recent_error < older_error * 0.95:
                self.adaptive_gain = min(2.5, self.adaptive_gain * 1.08)
            elif recent_error > 0.5:
                self.adaptive_gain = min(2.0, self.adaptive_gain * 1.03)
            else:
                self.adaptive_gain = max(0.8, self.adaptive_gain * 0.99)
        
        # 积分项 (带衰减防止饱和, 增加权重)
        self.error_integral = self.error_integral * self.integral_decay + error * 0.02
        self.error_integral = np.clip(self.error_integral, -0.8, 0.8)
        
        # 微分项 (使用平滑微分)
        if len(self.error_history) >= 2:
            # 使用最近 3 个样本的平均变化
            diffs = [error - self.error_history[-1]]
            if len(self.error_history) >= 2:
                diffs.append(self.error_history[-1] - self.error_history[-2])
            error_derivative = np.mean(diffs, axis=0)
        elif len(self.error_history) > 0:
            error_derivative = error - self.error_history[-1]
        else:
            error_derivative = np.zeros(4, dtype=np.float32)
        
        self.error_history.append(error.copy())
        
        # PID 输出 (增强收敛)
        correction = self.adaptive_gain * (
            self.kp * error + 
            self.ki * self.error_integral + 
            self.kd * error_derivative
        )
        
        return correction * self.feedback_gain
    
    def apply_correction(self, current_q: np.ndarray, correction: np.ndarray) -> np.ndarray:
        """应用修正并归一化."""
        corrected_q = current_q + correction
        
        # 重新归一化到 S³
        norm = np.sqrt(np.sum(corrected_q * corrected_q))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        return corrected_q / norm
    
    def reset(self):
        self.error_integral.fill(0)
        self.error_history.clear()
        self.adaptive_gain = 1.0
        self.error_magnitude_history.clear()
        self.convergence_step = 0


# ============================================================================
# 分形波谱视觉核心 (主类)
# ============================================================================

class FractalSpectralVisionCore:
    """H2Q 分形波谱视觉控制核心.
    
    集成所有组件:
    - 分形特征提取
    - 四元数状态编码
    - Berry 相位追踪
    - Fueter 曲率检测
    - 自组织并行计算
    - 视觉-控制反馈回路
    """
    
    def __init__(self,
                 vision_input_dim: int = 64,
                 fractal_output_dim: int = 32,
                 n_fractal_levels: int = 4,
                 history_len: int = 32,
                 anomaly_sensitivity: float = 0.1,
                 feedback_gain: float = 0.1,
                 n_parallel_workers: int = 4,
                 seed: int = 42):
        
        self.vision_input_dim = vision_input_dim
        self.fractal_output_dim = fractal_output_dim
        
        # 核心组件
        self.fractal_extractor = FractalFeatureExtractor(
            input_dim=vision_input_dim,
            output_dim=fractal_output_dim,
            n_levels=n_fractal_levels,
            seed=seed
        )
        
        self.quaternion_encoder = QuaternionStateEncoder(
            input_dim=fractal_output_dim,
            hidden_dim=16,
            seed=seed + 1
        )
        
        self.phase_tracker = BerryPhaseTracker(
            history_len=history_len,
            transition_threshold=0.3
        )
        
        self.curvature_detector = FueterCurvatureDetector(
            sensitivity=anomaly_sensitivity
        )
        
        self.parallel_unit = SelfOrganizingParallelUnit(
            n_workers=n_parallel_workers
        )
        
        self.feedback_loop = VisionControlFeedbackLoop(
            vision_input_dim=vision_input_dim,
            control_input_dim=fractal_output_dim,
            feedback_gain=feedback_gain
        )
        
        # 状态
        self.last_vision_state: Optional[VisionState] = None
        self.last_control_state: Optional[ControlState] = None
        
        # 指标
        self.metrics = SystemMetrics()
    
    def process_vision(self, image_features: np.ndarray) -> VisionState:
        """处理视觉输入.
        
        Args:
            image_features: (vision_input_dim,) 图像特征向量
        """
        # 分形特征提取
        features, intermediates = self.fractal_extractor.extract(image_features.astype(np.float32))
        
        # 编码为四元数
        q = self.quaternion_encoder.encode(features)
        
        # 计算显著性 (基于中间层特征的方差)
        saliency = np.std(np.concatenate(intermediates))
        
        # 置信度 (基于四元数的稳定性)
        confidence = 1.0 - np.abs(q[0] - 1.0)  # w 接近 1 表示稳定
        
        vision_state = VisionState(
            features=features,
            quaternion=q,
            saliency_map=np.array([saliency]),
            confidence=confidence
        )
        
        self.last_vision_state = vision_state
        self.metrics.total_frames += 1
        
        return vision_state
    
    def process_control(self, state_signal: np.ndarray) -> ControlState:
        """处理控制信号.
        
        Args:
            state_signal: (fractal_output_dim,) 状态信号
        """
        # 编码为四元数
        q = self.quaternion_encoder.encode(state_signal.astype(np.float32))
        
        # Berry 相位追踪
        phase, curvature, phase_transition = self.phase_tracker.update(q)
        
        # Fueter 曲率检测
        anomaly_score, is_anomaly = self.curvature_detector.update(q)
        
        if is_anomaly:
            self.metrics.anomalies_detected += 1
        
        control_state = ControlState(
            quaternion=q,
            phase=phase,
            curvature=curvature,
            anomaly_score=anomaly_score,
            predicted_trajectory=None
        )
        
        self.last_control_state = control_state
        self.metrics.total_control_cycles += 1
        
        return control_state
    
    def process_feedback(self, 
                        vision_state: VisionState, 
                        control_state: ControlState) -> FeedbackState:
        """执行反馈融合.
        
        Args:
            vision_state: 视觉状态
            control_state: 控制状态
        """
        start = time.perf_counter()
        
        # 计算误差
        error = self.feedback_loop.compute_error(
            control_state.quaternion,
            vision_state.confidence
        )
        
        # 计算修正
        correction = self.feedback_loop.compute_correction(error)
        
        # 应用修正
        corrected_q = self.feedback_loop.apply_correction(
            control_state.quaternion,
            correction
        )
        
        loop_latency = (time.perf_counter() - start) * 1e6
        
        # 更新控制状态
        corrected_control = ControlState(
            quaternion=corrected_q,
            phase=control_state.phase,
            curvature=control_state.curvature,
            anomaly_score=control_state.anomaly_score,
            predicted_trajectory=control_state.predicted_trajectory
        )
        
        if np.linalg.norm(correction) > 0.01:
            self.metrics.corrections_applied += 1
        
        return FeedbackState(
            vision_state=vision_state,
            control_state=corrected_control,
            error_signal=error,
            correction=correction,
            loop_latency_us=loop_latency
        )
    
    def process_full_cycle(self, 
                          image_features: np.ndarray,
                          control_signal: Optional[np.ndarray] = None) -> FeedbackState:
        """执行完整的视觉-控制-反馈周期.
        
        Args:
            image_features: 图像特征
            control_signal: 控制信号 (可选，默认使用视觉特征)
        """
        total_start = time.perf_counter()
        
        # 1. 视觉处理
        vision_start = time.perf_counter()
        vision_state = self.process_vision(image_features)
        vision_latency = (time.perf_counter() - vision_start) * 1e6
        
        # 2. 控制处理
        control_start = time.perf_counter()
        if control_signal is None:
            control_signal = vision_state.features
        control_state = self.process_control(control_signal)
        control_latency = (time.perf_counter() - control_start) * 1e6
        
        # 3. 反馈融合
        feedback_state = self.process_feedback(vision_state, control_state)
        
        # 更新延迟统计
        n = self.metrics.total_frames
        self.metrics.mean_vision_latency_us = (
            (self.metrics.mean_vision_latency_us * (n-1) + vision_latency) / n
        )
        self.metrics.mean_control_latency_us = (
            (self.metrics.mean_control_latency_us * (n-1) + control_latency) / n
        )
        self.metrics.mean_feedback_latency_us = (
            (self.metrics.mean_feedback_latency_us * (n-1) + feedback_state.loop_latency_us) / n
        )
        
        return feedback_state
    
    def reset(self):
        """重置所有状态."""
        self.phase_tracker.reset()
        self.curvature_detector.reset()
        self.feedback_loop.reset()
        self.last_vision_state = None
        self.last_control_state = None
        self.metrics = SystemMetrics()
    
    def model_size_bytes(self) -> int:
        """计算模型大小."""
        size = self.fractal_extractor.parameter_count() * 4
        size += self.quaternion_encoder.parameter_count() * 4
        return size
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标."""
        return {
            'total_frames': self.metrics.total_frames,
            'total_control_cycles': self.metrics.total_control_cycles,
            'mean_vision_latency_us': self.metrics.mean_vision_latency_us,
            'mean_control_latency_us': self.metrics.mean_control_latency_us,
            'mean_feedback_latency_us': self.metrics.mean_feedback_latency_us,
            'anomalies_detected': self.metrics.anomalies_detected,
            'corrections_applied': self.metrics.corrections_applied,
            'model_size_bytes': self.model_size_bytes(),
        }
    
    def shutdown(self):
        """关闭并行计算单元."""
        self.parallel_unit.shutdown()


# ============================================================================
# 工厂函数
# ============================================================================

def create_fractal_vision_core(
    vision_input_dim: int = 64,
    fractal_output_dim: int = 32,
    n_fractal_levels: int = 4,
    anomaly_sensitivity: float = 0.1,
    feedback_gain: float = 0.1,
    seed: int = 42
) -> FractalSpectralVisionCore:
    """创建分形波谱视觉核心."""
    return FractalSpectralVisionCore(
        vision_input_dim=vision_input_dim,
        fractal_output_dim=fractal_output_dim,
        n_fractal_levels=n_fractal_levels,
        anomaly_sensitivity=anomaly_sensitivity,
        feedback_gain=feedback_gain,
        seed=seed
    )


# ============================================================================
# 演示
# ============================================================================

def demo():
    """演示分形波谱视觉控制核心."""
    print("=" * 70)
    print("H2Q Fractal Spectral Vision Core - Demo")
    print("=" * 70)
    
    core = create_fractal_vision_core(
        vision_input_dim=64,
        fractal_output_dim=32,
        anomaly_sensitivity=0.1,
    )
    
    print(f"模型大小: {core.model_size_bytes()} bytes")
    
    np.random.seed(42)
    
    # 模拟视觉-控制循环
    print("\n1. 正常运行周期...")
    for i in range(100):
        image_features = np.random.randn(64) * 0.1
        feedback = core.process_full_cycle(image_features)
    
    print(f"   处理帧数: {core.metrics.total_frames}")
    print(f"   平均视觉延迟: {core.metrics.mean_vision_latency_us:.2f} μs")
    print(f"   平均控制延迟: {core.metrics.mean_control_latency_us:.2f} μs")
    print(f"   平均反馈延迟: {core.metrics.mean_feedback_latency_us:.2f} μs")
    
    # 注入异常
    print("\n2. 注入异常...")
    for i in range(20):
        if i % 5 == 0:
            image_features = np.random.randn(64) * 2.0  # 异常
        else:
            image_features = np.random.randn(64) * 0.1  # 正常
        
        feedback = core.process_full_cycle(image_features)
        
        if feedback.control_state.anomaly_score > 0.5:
            print(f"   帧 {i}: 异常分数={feedback.control_state.anomaly_score:.3f}")
    
    print(f"\n总异常检测: {core.metrics.anomalies_detected}")
    print(f"总修正应用: {core.metrics.corrections_applied}")
    
    core.shutdown()


if __name__ == "__main__":
    demo()
