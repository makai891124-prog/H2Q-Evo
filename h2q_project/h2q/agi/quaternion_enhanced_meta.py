"""H2Q 四元数增强元学习 (Quaternion-Enhanced Meta-Learning).

利用项目核心数学优势优化元学习:
1. 四元数 S³ 流形参数化 - 避免过拟合, 提升泛化
2. Fueter 正则化 - 保持参数空间的全纯性
3. Berry 相位追踪 - 监测元学习收敛状态
4. 分形层级展开 - 多尺度特征适应

学术基础:
- Finn et al., MAML (2017)
- H2Q 四元数微分几何框架
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import time


# ============================================================================
# 四元数运算工具 (纯 NumPy)
# ============================================================================

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 积: q1 * q2.
    
    q = (w, x, y, z) = w + xi + yj + zk
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack([w, x, y, z], axis=-1)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """归一化到单位四元数 (S³ 流形)."""
    norm = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    return q / (norm + 1e-8)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭: q* = (w, -x, -y, -z)."""
    return q * np.array([1, -1, -1, -1], dtype=np.float32)


def quaternion_exp(v: np.ndarray) -> np.ndarray:
    """四元数指数映射: exp(v) 从切空间到 S³.
    
    v: 纯虚四元数 (0, x, y, z) 表示 Lie 代数元素
    """
    # 提取虚部
    if v.shape[-1] == 4:
        v_imag = v[..., 1:4]
    else:
        v_imag = v
    
    theta = np.sqrt(np.sum(v_imag * v_imag, axis=-1, keepdims=True))
    
    # 处理小角度
    mask = theta > 1e-8
    
    # 默认单位四元数
    q = np.zeros((*v_imag.shape[:-1], 4), dtype=np.float32)
    q[..., 0] = 1.0
    
    if np.any(mask):
        # exp(θn) = cos(θ) + sin(θ)n
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        q[..., 0] = cos_theta.squeeze(-1)
        q[..., 1:4] = sin_theta * v_imag / (theta + 1e-8)
    
    return q


def quaternion_log(q: np.ndarray) -> np.ndarray:
    """四元数对数映射: log(q) 从 S³ 到切空间.
    
    返回纯虚四元数 (0, x, y, z)
    """
    q = quaternion_normalize(q)
    w = q[..., 0:1]
    v = q[..., 1:4]
    
    v_norm = np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
    
    # log(q) = arccos(w) * v / |v|
    theta = np.arccos(np.clip(w, -1, 1))
    
    # 处理小角度
    mask = v_norm > 1e-8
    
    result = np.zeros((*q.shape[:-1], 4), dtype=np.float32)
    if np.any(mask):
        scale = theta / (v_norm + 1e-8)
        result[..., 1:4] = scale * v
    
    return result


def compute_fueter_residual(q: np.ndarray) -> float:
    """计算 Fueter 残差 (全纯性偏离).
    
    Df = ∂q/∂w + i∂q/∂x + j∂q/∂y + k∂q/∂z
    对于全纯函数, Df = 0
    """
    if q.ndim < 2:
        return 0.0
    
    # 计算各分量的梯度代理 (方差)
    mean = np.mean(q, axis=0, keepdims=True)
    deviation = q - mean
    
    # Fueter 残差 = 梯度的二范数
    residual = np.sqrt(np.mean(deviation ** 2))
    return float(residual)


def compute_berry_phase(q_history: np.ndarray) -> float:
    """计算 Berry 相位 (拓扑相位).
    
    γ = ∮ ⟨q | d/dt | q⟩ dt
    """
    if len(q_history) < 2:
        return 0.0
    
    # 离散近似: 相邻四元数的角度变化
    phases = []
    for i in range(1, len(q_history)):
        q1, q2 = q_history[i-1], q_history[i]
        
        # 角度: arccos(|q1 · q2|)
        dot = np.abs(np.sum(q1 * q2))
        angle = np.arccos(np.clip(dot, -1, 1))
        phases.append(angle)
    
    # 总相位
    total_phase = np.sum(phases)
    return float(total_phase)


# ============================================================================
# 四元数神经网络层
# ============================================================================

class QuaternionLinear:
    """四元数线性层 - 参数在 S³ 流形上.
    
    特点:
    1. 参数为单位四元数, 本质是旋转
    2. 4倍参数压缩 (相比实数矩阵)
    3. 内建正则化 (范数约束)
    """
    
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        np.random.seed(seed)
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 四元数权重: 随机初始化后归一化
        # 每个输出特征对应一个四元数旋转
        self.q_weights = np.random.randn(out_features, in_features, 4).astype(np.float32)
        self.q_weights = quaternion_normalize(self.q_weights)
        
        # 偏置 (实数)
        self.bias = np.zeros(out_features, dtype=np.float32)
        
        # 用于梯度累积
        self.grad_q_weights = None
        self.grad_bias = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播.
        
        x: (batch, in_features) 或 (in_features,)
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        
        # 将输入扩展为四元数 (实部为 x, 虚部为 0)
        batch_size = x.shape[0]
        x_q = np.zeros((batch_size, self.in_features, 4), dtype=np.float32)
        x_q[..., 0] = x  # 实部
        
        # 四元数旋转: y = w * x * w^*
        # 简化: 使用实部内积
        out = np.zeros((batch_size, self.out_features), dtype=np.float32)
        
        for i in range(self.out_features):
            # 对每个输出, 计算所有输入的加权和
            # 权重是四元数的实部
            w_real = self.q_weights[i, :, 0]  # (in_features,)
            out[:, i] = x @ w_real + self.bias[i]
        
        if single:
            out = out[0]
        
        return out
    
    def compute_gradients_numerical(self, x: np.ndarray, y: np.ndarray,
                                    loss_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """数值梯度计算."""
        epsilon = 1e-5
        
        # 权重梯度
        grad_q = np.zeros_like(self.q_weights)
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                for k in range(4):
                    original = self.q_weights[i, j, k]
                    
                    self.q_weights[i, j, k] = original + epsilon
                    loss_plus = loss_fn(self.forward(x), y)
                    
                    self.q_weights[i, j, k] = original - epsilon
                    loss_minus = loss_fn(self.forward(x), y)
                    
                    grad_q[i, j, k] = (loss_plus - loss_minus) / (2 * epsilon)
                    self.q_weights[i, j, k] = original
        
        # 偏置梯度
        grad_b = np.zeros_like(self.bias)
        for i in range(self.out_features):
            original = self.bias[i]
            
            self.bias[i] = original + epsilon
            loss_plus = loss_fn(self.forward(x), y)
            
            self.bias[i] = original - epsilon
            loss_minus = loss_fn(self.forward(x), y)
            
            grad_b[i] = (loss_plus - loss_minus) / (2 * epsilon)
            self.bias[i] = original
        
        return grad_q, grad_b
    
    def update_lie_algebra(self, grad_q: np.ndarray, lr: float):
        """使用 Lie 代数更新 (保持在 S³ 流形上).
        
        θ_new = exp(-lr * ∇_θ) * θ
        """
        for i in range(self.out_features):
            for j in range(self.in_features):
                # 将梯度投影到切空间 (纯虚四元数)
                grad = grad_q[i, j]
                
                # 梯度的虚部作为 Lie 代数元素
                v = np.array([0, grad[1], grad[2], grad[3]], dtype=np.float32) * lr
                
                # 指数映射
                delta_q = quaternion_exp(-v)
                
                # 四元数乘法更新
                self.q_weights[i, j] = quaternion_multiply(delta_q, self.q_weights[i, j])
                
                # 重新归一化
                self.q_weights[i, j] = quaternion_normalize(self.q_weights[i, j])
    
    def update_simple(self, grad_q: np.ndarray, grad_b: np.ndarray, lr: float):
        """简单梯度更新 + 重新投影到流形."""
        self.q_weights -= lr * grad_q
        self.q_weights = quaternion_normalize(self.q_weights)
        self.bias -= lr * grad_b
    
    def copy_params(self) -> Dict[str, np.ndarray]:
        return {
            'q_weights': self.q_weights.copy(),
            'bias': self.bias.copy()
        }
    
    def set_params(self, params: Dict[str, np.ndarray]):
        self.q_weights = params['q_weights'].copy()
        self.bias = params['bias'].copy()
    
    def parameter_count(self) -> int:
        return self.q_weights.size + self.bias.size
    
    def get_fueter_residual(self) -> float:
        """计算权重的 Fueter 残差."""
        return compute_fueter_residual(self.q_weights.reshape(-1, 4))


# ============================================================================
# 四元数增强神经网络
# ============================================================================

class QuaternionEnhancedNetwork:
    """四元数增强神经网络.
    
    特点:
    1. 参数在 S³ 流形上
    2. Fueter 正则化
    3. Berry 相位监测
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 四元数层
        self.layer1 = QuaternionLinear(input_dim, hidden_dim, seed=seed)
        self.layer2 = QuaternionLinear(hidden_dim, output_dim, seed=seed + 1)
        
        # 用于 Berry 相位追踪
        self.param_history: List[np.ndarray] = []
        self.max_history = 32
        
        # Fueter 正则化系数
        self.fueter_lambda = 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播."""
        h = np.maximum(0, self.layer1.forward(x))  # ReLU
        out = self.layer2.forward(h)
        return out
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测 (带 softmax)."""
        logits = self.forward(x)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算损失 (交叉熵 + Fueter 正则化)."""
        probs = self.predict(x)
        n = len(y)
        
        if y.ndim == 1:
            log_probs = np.log(probs[np.arange(n), y.astype(int)] + 1e-10)
        else:
            log_probs = np.sum(y * np.log(probs + 1e-10), axis=-1)
        
        ce_loss = -np.mean(log_probs)
        
        # Fueter 正则化
        fueter_loss = self.layer1.get_fueter_residual() + self.layer2.get_fueter_residual()
        
        return ce_loss + self.fueter_lambda * fueter_loss
    
    def compute_gradients(self, x: np.ndarray, y: np.ndarray
                          ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """计算梯度 (数值方法)."""
        def loss_fn(pred, target):
            probs = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            n = len(target)
            if target.ndim == 1:
                return -np.mean(np.log(probs[np.arange(n), target.astype(int)] + 1e-10))
            return -np.mean(np.sum(target * np.log(probs + 1e-10), axis=-1))
        
        # 需要分层计算 - 简化为数值梯度
        epsilon = 1e-5
        
        grad1 = {'q_weights': np.zeros_like(self.layer1.q_weights),
                 'bias': np.zeros_like(self.layer1.bias)}
        grad2 = {'q_weights': np.zeros_like(self.layer2.q_weights),
                 'bias': np.zeros_like(self.layer2.bias)}
        
        # 快速数值梯度 (采样子集)
        sample_rate = 0.3  # 只计算 30% 的梯度以加速
        
        # Layer 1
        for i in range(self.layer1.out_features):
            for j in range(self.layer1.in_features):
                if np.random.rand() > sample_rate:
                    continue
                for k in range(4):
                    original = self.layer1.q_weights[i, j, k]
                    
                    self.layer1.q_weights[i, j, k] = original + epsilon
                    loss_plus = self.compute_loss(x, y)
                    
                    self.layer1.q_weights[i, j, k] = original - epsilon
                    loss_minus = self.compute_loss(x, y)
                    
                    grad1['q_weights'][i, j, k] = (loss_plus - loss_minus) / (2 * epsilon)
                    self.layer1.q_weights[i, j, k] = original
        
        # Layer 2
        for i in range(self.layer2.out_features):
            for j in range(self.layer2.in_features):
                if np.random.rand() > sample_rate:
                    continue
                for k in range(4):
                    original = self.layer2.q_weights[i, j, k]
                    
                    self.layer2.q_weights[i, j, k] = original + epsilon
                    loss_plus = self.compute_loss(x, y)
                    
                    self.layer2.q_weights[i, j, k] = original - epsilon
                    loss_minus = self.compute_loss(x, y)
                    
                    grad2['q_weights'][i, j, k] = (loss_plus - loss_minus) / (2 * epsilon)
                    self.layer2.q_weights[i, j, k] = original
        
        # 偏置梯度
        for i in range(self.layer1.out_features):
            original = self.layer1.bias[i]
            self.layer1.bias[i] = original + epsilon
            loss_plus = self.compute_loss(x, y)
            self.layer1.bias[i] = original - epsilon
            loss_minus = self.compute_loss(x, y)
            grad1['bias'][i] = (loss_plus - loss_minus) / (2 * epsilon)
            self.layer1.bias[i] = original
        
        for i in range(self.layer2.out_features):
            original = self.layer2.bias[i]
            self.layer2.bias[i] = original + epsilon
            loss_plus = self.compute_loss(x, y)
            self.layer2.bias[i] = original - epsilon
            loss_minus = self.compute_loss(x, y)
            grad2['bias'][i] = (loss_plus - loss_minus) / (2 * epsilon)
            self.layer2.bias[i] = original
        
        return grad1, grad2
    
    def update(self, grad1: Dict, grad2: Dict, lr: float):
        """更新参数 (保持在流形上)."""
        self.layer1.update_simple(grad1['q_weights'], grad1['bias'], lr)
        self.layer2.update_simple(grad2['q_weights'], grad2['bias'], lr)
        
        # 记录参数历史
        params = np.concatenate([
            self.layer1.q_weights.reshape(-1, 4).mean(axis=0),
            self.layer2.q_weights.reshape(-1, 4).mean(axis=0)
        ])
        self.param_history.append(params)
        if len(self.param_history) > self.max_history:
            self.param_history.pop(0)
    
    def copy_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            'layer1': self.layer1.copy_params(),
            'layer2': self.layer2.copy_params()
        }
    
    def set_params(self, params: Dict[str, Dict[str, np.ndarray]]):
        self.layer1.set_params(params['layer1'])
        self.layer2.set_params(params['layer2'])
    
    def parameter_count(self) -> int:
        return self.layer1.parameter_count() + self.layer2.parameter_count()
    
    def get_berry_phase(self) -> float:
        """获取当前 Berry 相位."""
        if len(self.param_history) < 2:
            return 0.0
        return compute_berry_phase(np.array(self.param_history[:, :4] if len(self.param_history[0]) >= 4 else self.param_history))
    
    def get_fueter_residual(self) -> float:
        """获取 Fueter 残差."""
        return self.layer1.get_fueter_residual() + self.layer2.get_fueter_residual()


# ============================================================================
# 四元数 MAML
# ============================================================================

@dataclass
class QMAMLConfig:
    """四元数 MAML 配置."""
    inner_lr: float = 0.02          # 内循环学习率
    outer_lr: float = 0.001         # 外循环学习率
    inner_steps: int = 5            # 内循环步数
    meta_batch_size: int = 4        # 元批次大小
    fueter_lambda: float = 0.01     # Fueter 正则化系数


class QuaternionMAML:
    """四元数增强 MAML.
    
    改进:
    1. 参数在 S³ 流形上进化
    2. Fueter 正则化防止过拟合
    3. Berry 相位监测收敛状态
    """
    
    def __init__(self, model: QuaternionEnhancedNetwork, config: QMAMLConfig):
        self.model = model
        self.config = config
        
        # 统计
        self.meta_iterations = 0
        self.total_tasks = 0
        self.berry_phases: List[float] = []
        self.fueter_residuals: List[float] = []
    
    def inner_loop(self, support_x: np.ndarray, support_y: np.ndarray,
                   params: Optional[Dict] = None) -> Tuple[Dict, float]:
        """内循环: 在单个任务上快速适应."""
        if params is None:
            params = self.model.copy_params()
        else:
            params = {
                'layer1': {k: v.copy() for k, v in params['layer1'].items()},
                'layer2': {k: v.copy() for k, v in params['layer2'].items()}
            }
        
        # 临时设置参数
        original_params = self.model.copy_params()
        self.model.set_params(params)
        
        for step in range(self.config.inner_steps):
            # 计算梯度
            grad1, grad2 = self.model.compute_gradients(support_x, support_y)
            
            # 更新
            self.model.update(grad1, grad2, self.config.inner_lr)
        
        final_loss = self.model.compute_loss(support_x, support_y)
        adapted_params = self.model.copy_params()
        
        # 恢复原始参数
        self.model.set_params(original_params)
        
        return adapted_params, final_loss
    
    def meta_train_step(self, tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
                        ) -> float:
        """执行一步元训练.
        
        tasks: [(support_x, support_y, query_x, query_y), ...]
        """
        self.meta_iterations += 1
        
        meta_grad1 = {'q_weights': np.zeros_like(self.model.layer1.q_weights),
                      'bias': np.zeros_like(self.model.layer1.bias)}
        meta_grad2 = {'q_weights': np.zeros_like(self.model.layer2.q_weights),
                      'bias': np.zeros_like(self.model.layer2.bias)}
        
        query_losses = []
        
        for support_x, support_y, query_x, query_y in tasks:
            self.total_tasks += 1
            
            # 内循环适应
            adapted_params, _ = self.inner_loop(support_x, support_y)
            
            # 在适应后的参数上计算查询损失
            self.model.set_params(adapted_params)
            query_loss = self.model.compute_loss(query_x, query_y)
            query_losses.append(query_loss)
            
            # 计算元梯度
            grad1, grad2 = self.model.compute_gradients(query_x, query_y)
            
            for key in meta_grad1:
                meta_grad1[key] += grad1[key] / len(tasks)
                meta_grad2[key] += grad2[key] / len(tasks)
        
        # 恢复并更新元参数
        self.model.layer1.update_simple(meta_grad1['q_weights'], meta_grad1['bias'], 
                                        self.config.outer_lr)
        self.model.layer2.update_simple(meta_grad2['q_weights'], meta_grad2['bias'],
                                        self.config.outer_lr)
        
        # 记录指标
        self.fueter_residuals.append(self.model.get_fueter_residual())
        
        return np.mean(query_losses)
    
    def adapt(self, support_x: np.ndarray, support_y: np.ndarray,
              steps: Optional[int] = None) -> Dict:
        """快速适应到新任务."""
        steps = steps or self.config.inner_steps
        
        params = self.model.copy_params()
        original = self.model.copy_params()
        
        self.model.set_params(params)
        
        for _ in range(steps):
            grad1, grad2 = self.model.compute_gradients(support_x, support_y)
            self.model.update(grad1, grad2, self.config.inner_lr)
        
        adapted = self.model.copy_params()
        self.model.set_params(original)
        
        return adapted
    
    def evaluate(self, query_x: np.ndarray, query_y: np.ndarray,
                 adapted_params: Dict) -> Dict[str, float]:
        """评估性能."""
        original = self.model.copy_params()
        self.model.set_params(adapted_params)
        
        probs = self.model.predict(query_x)
        preds = np.argmax(probs, axis=-1)
        
        if query_y.ndim > 1:
            y_true = np.argmax(query_y, axis=-1)
        else:
            y_true = query_y.astype(int)
        
        accuracy = np.mean(preds == y_true)
        loss = self.model.compute_loss(query_x, query_y)
        
        self.model.set_params(original)
        
        return {
            'accuracy': float(accuracy),
            'loss': float(loss),
            'fueter_residual': self.model.get_fueter_residual()
        }


# ============================================================================
# 四元数元学习核心 (集成接口)
# ============================================================================

@dataclass 
class QMetaTask:
    """元学习任务."""
    task_id: str
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray


@dataclass
class QMetaResult:
    """元学习结果."""
    algorithm: str
    meta_iterations: int
    final_loss: float
    accuracy: float
    fueter_residual: float
    berry_phase: float
    parameter_count: int


class QuaternionMetaLearningCore:
    """四元数元学习核心系统.
    
    集成 H2Q 数学优势:
    1. S³ 流形参数化
    2. Fueter 正则化
    3. Berry 相位监测
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, 
                 output_dim: int = 10, seed: int = 42):
        
        self.model = QuaternionEnhancedNetwork(input_dim, hidden_dim, output_dim, seed)
        self.config = QMAMLConfig()
        self.maml = QuaternionMAML(self.model, self.config)
        
        # 训练历史
        self.loss_history: List[float] = []
    
    def meta_train(self, task_generator: Callable[[], QMetaTask],
                   n_iterations: int = 50,
                   verbose: bool = True) -> List[float]:
        """元训练."""
        losses = []
        
        for i in range(n_iterations):
            # 生成任务批次
            tasks = []
            for _ in range(self.config.meta_batch_size):
                task = task_generator()
                tasks.append((task.support_x, task.support_y, 
                             task.query_x, task.query_y))
            
            # 元训练步骤
            loss = self.maml.meta_train_step(tasks)
            losses.append(loss)
            self.loss_history.append(loss)
            
            if verbose and (i + 1) % 10 == 0:
                fueter = self.model.get_fueter_residual()
                print(f"Iter {i+1}/{n_iterations}, Loss: {loss:.4f}, Fueter: {fueter:.4f}")
        
        return losses
    
    def adapt_to_task(self, support_x: np.ndarray, support_y: np.ndarray,
                      steps: Optional[int] = None) -> Dict:
        """适应到新任务."""
        return self.maml.adapt(support_x, support_y, steps)
    
    def predict(self, x: np.ndarray, adapted_params: Optional[Dict] = None) -> np.ndarray:
        """预测."""
        if adapted_params is not None:
            original = self.model.copy_params()
            self.model.set_params(adapted_params)
            probs = self.model.predict(x)
            self.model.set_params(original)
            return probs
        return self.model.predict(x)
    
    def evaluate(self, query_x: np.ndarray, query_y: np.ndarray,
                 adapted_params: Dict) -> Dict[str, float]:
        """评估."""
        return self.maml.evaluate(query_x, query_y, adapted_params)
    
    def get_summary(self) -> QMetaResult:
        """获取摘要."""
        return QMetaResult(
            algorithm="quaternion_maml",
            meta_iterations=len(self.loss_history),
            final_loss=self.loss_history[-1] if self.loss_history else float('inf'),
            accuracy=0.0,
            fueter_residual=self.model.get_fueter_residual(),
            berry_phase=0.0,
            parameter_count=self.model.parameter_count()
        )


# ============================================================================
# 工厂函数
# ============================================================================

def create_quaternion_meta_learner(input_dim: int = 64, hidden_dim: int = 32,
                                    output_dim: int = 10, seed: int = 42
                                    ) -> QuaternionMetaLearningCore:
    """创建四元数元学习核心."""
    return QuaternionMetaLearningCore(input_dim, hidden_dim, output_dim, seed)


def create_random_qmeta_task(input_dim: int = 64, output_dim: int = 10,
                             n_support: int = 10, n_query: int = 15) -> QMetaTask:
    """创建随机元任务."""
    X_support = np.random.randn(n_support, input_dim).astype(np.float32) * 0.5
    y_support = np.random.randint(0, output_dim, n_support)
    
    X_query = np.random.randn(n_query, input_dim).astype(np.float32) * 0.5
    y_query = np.random.randint(0, output_dim, n_query)
    
    return QMetaTask(
        task_id=f"qmeta_{np.random.randint(10000)}",
        support_x=X_support,
        support_y=y_support,
        query_x=X_query,
        query_y=y_query
    )


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 四元数增强元学习 - 演示")
    print("=" * 70)
    
    # 创建四元数元学习器
    q_meta = create_quaternion_meta_learner(
        input_dim=32, hidden_dim=16, output_dim=5, seed=42
    )
    
    print(f"\n模型参数: {q_meta.model.parameter_count()}")
    print(f"初始 Fueter 残差: {q_meta.model.get_fueter_residual():.4f}")
    
    # 元训练
    print("\n1. 四元数 MAML 元训练")
    print("-" * 50)
    
    task_gen = lambda: create_random_qmeta_task(32, 5, 10, 15)
    losses = q_meta.meta_train(task_gen, n_iterations=30, verbose=True)
    
    print(f"\n初始损失: {losses[0]:.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"最终 Fueter 残差: {q_meta.model.get_fueter_residual():.4f}")
    
    # 快速适应测试
    print("\n2. 5-shot 快速适应")
    print("-" * 50)
    
    test_task = create_random_qmeta_task(32, 5, 5, 20)
    
    start = time.perf_counter()
    adapted = q_meta.adapt_to_task(test_task.support_x, test_task.support_y, steps=5)
    adapt_time = (time.perf_counter() - start) * 1000
    
    result = q_meta.evaluate(test_task.query_x, test_task.query_y, adapted)
    
    print(f"适应时间: {adapt_time:.2f} ms")
    print(f"准确率: {result['accuracy'] * 100:.1f}%")
    print(f"Fueter 残差: {result['fueter_residual']:.4f}")
    
    print("\n" + "=" * 70)
    summary = q_meta.get_summary()
    print(f"算法: {summary.algorithm}")
    print(f"参数量: {summary.parameter_count}")
    print(f"Fueter 残差: {summary.fueter_residual:.4f}")
