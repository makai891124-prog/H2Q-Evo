"""H2Q 元学习核心 (Meta-Learning Core).

实现AGI核心能力：
1. MAML (Model-Agnostic Meta-Learning)
2. Reptile 算法
3. 少样本学习
4. 快速任务适应

参考文献:
- Finn et al., "Model-Agnostic Meta-Learning" (2017)
- Nichol et al., "Reptile: A Scalable Meta-Learning Algorithm" (2018)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import deque
import time
import copy


# ============================================================================
# 基础组件
# ============================================================================

@dataclass
class Task:
    """元学习任务."""
    task_id: str
    support_set: Tuple[np.ndarray, np.ndarray]  # (X_support, y_support)
    query_set: Tuple[np.ndarray, np.ndarray]    # (X_query, y_query)
    task_type: str = "classification"  # classification, regression


@dataclass
class MetaLearningConfig:
    """元学习配置."""
    inner_lr: float = 0.01           # 内循环学习率
    outer_lr: float = 0.001          # 外循环学习率
    inner_steps: int = 5             # 内循环步数
    meta_batch_size: int = 4         # 元批次大小
    first_order: bool = False        # 是否使用一阶近似
    

class SimpleNetwork:
    """简单神经网络 (用于元学习)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        np.random.seed(seed)
        
        # Xavier 初始化
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
        
        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1,
            'b1': np.zeros(hidden_dim, dtype=np.float32),
            'W2': np.random.randn(hidden_dim, output_dim).astype(np.float32) * scale2,
            'b2': np.zeros(output_dim, dtype=np.float32),
        }
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, x: np.ndarray, params: Dict[str, np.ndarray] = None) -> np.ndarray:
        """前向传播."""
        if params is None:
            params = self.params
        
        h = np.maximum(0, x @ params['W1'] + params['b1'])  # ReLU
        out = h @ params['W2'] + params['b2']
        return out
    
    def predict(self, x: np.ndarray, params: Dict[str, np.ndarray] = None) -> np.ndarray:
        """预测 (带 softmax)."""
        logits = self.forward(x, params)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray, 
                     params: Dict[str, np.ndarray] = None) -> float:
        """计算交叉熵损失."""
        probs = self.predict(x, params)
        n = len(y)
        if y.ndim == 1:
            # 类别索引
            log_probs = np.log(probs[np.arange(n), y.astype(int)] + 1e-10)
        else:
            # one-hot
            log_probs = np.sum(y * np.log(probs + 1e-10), axis=-1)
        return -np.mean(log_probs)
    
    def compute_gradients(self, x: np.ndarray, y: np.ndarray,
                          params: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """计算梯度 (数值方法)."""
        if params is None:
            params = self.params
        
        epsilon = 1e-5
        grads = {}
        
        for key in params:
            grad = np.zeros_like(params[key])
            it = np.nditer(params[key], flags=['multi_index'])
            
            while not it.finished:
                idx = it.multi_index
                original = params[key][idx]
                
                # f(x + epsilon)
                params[key][idx] = original + epsilon
                loss_plus = self.compute_loss(x, y, params)
                
                # f(x - epsilon)
                params[key][idx] = original - epsilon
                loss_minus = self.compute_loss(x, y, params)
                
                # 梯度
                grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                
                params[key][idx] = original
                it.iternext()
            
            grads[key] = grad
        
        return grads
    
    def copy_params(self) -> Dict[str, np.ndarray]:
        """复制参数."""
        return {k: v.copy() for k, v in self.params.items()}
    
    def set_params(self, params: Dict[str, np.ndarray]):
        """设置参数."""
        for k, v in params.items():
            self.params[k] = v.copy()
    
    def parameter_count(self) -> int:
        """参数数量."""
        return sum(p.size for p in self.params.values())


# ============================================================================
# MAML 实现
# ============================================================================

class MAML:
    """Model-Agnostic Meta-Learning."""
    
    def __init__(self, model: SimpleNetwork, config: MetaLearningConfig):
        self.model = model
        self.config = config
        
        # 统计
        self.meta_iterations = 0
        self.total_tasks = 0
    
    def inner_loop(self, task: Task, params: Dict[str, np.ndarray] = None
                   ) -> Tuple[Dict[str, np.ndarray], float]:
        """内循环: 在单个任务上快速适应.
        
        Returns:
            (adapted_params, final_loss)
        """
        if params is None:
            params = self.model.copy_params()
        else:
            params = {k: v.copy() for k, v in params.items()}
        
        X_support, y_support = task.support_set
        
        for step in range(self.config.inner_steps):
            # 计算梯度
            grads = self.model.compute_gradients(X_support, y_support, params)
            
            # 梯度下降
            for key in params:
                params[key] = params[key] - self.config.inner_lr * grads[key]
        
        # 计算最终损失
        final_loss = self.model.compute_loss(X_support, y_support, params)
        
        return params, final_loss
    
    def outer_loop(self, task_batch: List[Task]) -> Tuple[Dict[str, np.ndarray], float]:
        """外循环: 元优化.
        
        Returns:
            (meta_gradients, mean_query_loss)
        """
        meta_grads = {k: np.zeros_like(v) for k, v in self.model.params.items()}
        query_losses = []
        
        for task in task_batch:
            self.total_tasks += 1
            
            # 内循环适应
            adapted_params, _ = self.inner_loop(task)
            
            # 在查询集上计算损失
            X_query, y_query = task.query_set
            query_loss = self.model.compute_loss(X_query, y_query, adapted_params)
            query_losses.append(query_loss)
            
            # 计算元梯度
            if self.config.first_order:
                # 一阶近似: 直接用适应后参数的梯度
                task_grads = self.model.compute_gradients(X_query, y_query, adapted_params)
            else:
                # 二阶: 需要计算 Hessian-vector 乘积 (简化为一阶)
                task_grads = self.model.compute_gradients(X_query, y_query, adapted_params)
            
            # 累加梯度
            for key in meta_grads:
                meta_grads[key] += task_grads[key] / len(task_batch)
        
        return meta_grads, np.mean(query_losses)
    
    def meta_train_step(self, task_batch: List[Task]) -> float:
        """执行一步元训练.
        
        Returns:
            mean_query_loss
        """
        self.meta_iterations += 1
        
        # 外循环
        meta_grads, mean_loss = self.outer_loop(task_batch)
        
        # 更新元参数
        for key in self.model.params:
            self.model.params[key] -= self.config.outer_lr * meta_grads[key]
        
        return mean_loss
    
    def adapt(self, support_x: np.ndarray, support_y: np.ndarray,
              inner_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """快速适应到新任务.
        
        Args:
            support_x: 支持集特征
            support_y: 支持集标签
            inner_steps: 适应步数 (默认使用配置)
        
        Returns:
            adapted_params
        """
        steps = inner_steps or self.config.inner_steps
        params = self.model.copy_params()
        
        for _ in range(steps):
            grads = self.model.compute_gradients(support_x, support_y, params)
            for key in params:
                params[key] = params[key] - self.config.inner_lr * grads[key]
        
        return params
    
    def evaluate(self, task: Task, adapted_params: Optional[Dict[str, np.ndarray]] = None
                ) -> Dict[str, float]:
        """评估任务性能."""
        X_query, y_query = task.query_set
        
        if adapted_params is None:
            adapted_params, _ = self.inner_loop(task)
        
        # 预测
        probs = self.model.predict(X_query, adapted_params)
        preds = np.argmax(probs, axis=-1)
        
        if y_query.ndim > 1:
            y_true = np.argmax(y_query, axis=-1)
        else:
            y_true = y_query.astype(int)
        
        accuracy = np.mean(preds == y_true)
        loss = self.model.compute_loss(X_query, y_query, adapted_params)
        
        return {
            "accuracy": float(accuracy),
            "loss": float(loss)
        }


# ============================================================================
# Reptile 实现
# ============================================================================

class Reptile:
    """Reptile: 简单高效的元学习算法."""
    
    def __init__(self, model: SimpleNetwork, config: MetaLearningConfig):
        self.model = model
        self.config = config
        
        # Reptile 特有参数
        self.epsilon = 0.1  # 插值系数
        
        # 统计
        self.meta_iterations = 0
    
    def task_train(self, task: Task, params: Dict[str, np.ndarray] = None
                   ) -> Dict[str, np.ndarray]:
        """在单个任务上训练多步."""
        if params is None:
            params = self.model.copy_params()
        else:
            params = {k: v.copy() for k, v in params.items()}
        
        X_support, y_support = task.support_set
        
        for _ in range(self.config.inner_steps):
            grads = self.model.compute_gradients(X_support, y_support, params)
            for key in params:
                params[key] = params[key] - self.config.inner_lr * grads[key]
        
        return params
    
    def meta_train_step(self, task_batch: List[Task]) -> float:
        """执行一步 Reptile 元训练."""
        self.meta_iterations += 1
        
        # 保存初始参数
        initial_params = self.model.copy_params()
        
        # 累积参数更新
        param_updates = {k: np.zeros_like(v) for k, v in initial_params.items()}
        losses = []
        
        for task in task_batch:
            # 在任务上训练
            task_params = self.task_train(task, initial_params)
            
            # 计算查询损失
            X_query, y_query = task.query_set
            loss = self.model.compute_loss(X_query, y_query, task_params)
            losses.append(loss)
            
            # 累积参数差异
            for key in param_updates:
                param_updates[key] += (task_params[key] - initial_params[key]) / len(task_batch)
        
        # Reptile 更新: θ = θ + ε * (θ' - θ)
        for key in self.model.params:
            self.model.params[key] += self.epsilon * param_updates[key]
        
        return np.mean(losses)
    
    def adapt(self, support_x: np.ndarray, support_y: np.ndarray,
              inner_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """快速适应到新任务."""
        steps = inner_steps or self.config.inner_steps
        params = self.model.copy_params()
        
        for _ in range(steps):
            grads = self.model.compute_gradients(support_x, support_y, params)
            for key in params:
                params[key] = params[key] - self.config.inner_lr * grads[key]
        
        return params


# ============================================================================
# 少样本学习器
# ============================================================================

@dataclass
class FewShotResult:
    """少样本学习结果."""
    n_way: int
    k_shot: int
    accuracy: float
    confidence_interval: Tuple[float, float]
    adaptation_time_ms: float
    tasks_evaluated: int


class FewShotLearner:
    """少样本学习器."""
    
    def __init__(self, meta_learner: MAML):
        self.meta_learner = meta_learner
        self.task_history: List[Dict[str, Any]] = []
    
    def create_few_shot_task(self, X: np.ndarray, y: np.ndarray,
                             n_way: int, k_shot: int, q_query: int = 15
                             ) -> Task:
        """创建 N-way K-shot 任务.
        
        Args:
            X: 特征矩阵
            y: 标签
            n_way: 类别数
            k_shot: 每类支持集样本数
            q_query: 每类查询集样本数
        """
        # 随机选择 n_way 个类
        unique_classes = np.unique(y)
        if len(unique_classes) < n_way:
            raise ValueError(f"Need at least {n_way} classes, got {len(unique_classes)}")
        
        selected_classes = np.random.choice(unique_classes, n_way, replace=False)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for new_label, cls in enumerate(selected_classes):
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            
            # 支持集
            support_indices = cls_indices[:k_shot]
            support_x.append(X[support_indices])
            support_y.extend([new_label] * k_shot)
            
            # 查询集
            query_indices = cls_indices[k_shot:k_shot + q_query]
            if len(query_indices) > 0:
                query_x.append(X[query_indices])
                query_y.extend([new_label] * len(query_indices))
        
        return Task(
            task_id=f"{n_way}way_{k_shot}shot",
            support_set=(np.vstack(support_x), np.array(support_y)),
            query_set=(np.vstack(query_x), np.array(query_y)),
            task_type="classification"
        )
    
    def evaluate_few_shot(self, X: np.ndarray, y: np.ndarray,
                          n_way: int, k_shot: int, 
                          n_episodes: int = 100) -> FewShotResult:
        """评估少样本学习性能.
        
        Args:
            X: 测试集特征
            y: 测试集标签
            n_way: 类别数
            k_shot: 每类样本数
            n_episodes: 评估回合数
        """
        accuracies = []
        total_time = 0
        
        for _ in range(n_episodes):
            # 创建任务
            task = self.create_few_shot_task(X, y, n_way, k_shot)
            
            # 适应并评估
            start = time.perf_counter()
            result = self.meta_learner.evaluate(task)
            total_time += (time.perf_counter() - start) * 1000
            
            accuracies.append(result["accuracy"])
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        ci_width = 1.96 * std_acc / np.sqrt(n_episodes)
        
        return FewShotResult(
            n_way=n_way,
            k_shot=k_shot,
            accuracy=float(mean_acc),
            confidence_interval=(float(mean_acc - ci_width), float(mean_acc + ci_width)),
            adaptation_time_ms=total_time / n_episodes,
            tasks_evaluated=n_episodes
        )
    
    def rapid_adapt(self, support_x: np.ndarray, support_y: np.ndarray,
                    max_steps: int = 10, target_loss: float = 0.1
                    ) -> Tuple[Dict[str, np.ndarray], int]:
        """快速适应直到收敛.
        
        Returns:
            (adapted_params, steps_taken)
        """
        params = self.meta_learner.model.copy_params()
        
        for step in range(max_steps):
            loss = self.meta_learner.model.compute_loss(support_x, support_y, params)
            
            if loss < target_loss:
                return params, step + 1
            
            grads = self.meta_learner.model.compute_gradients(support_x, support_y, params)
            for key in params:
                params[key] = params[key] - self.meta_learner.config.inner_lr * grads[key]
        
        return params, max_steps


# ============================================================================
# 元学习核心系统
# ============================================================================

@dataclass
class MetaLearningResult:
    """元学习系统结果."""
    algorithm: str
    meta_iterations: int
    final_meta_loss: float
    few_shot_accuracy: float
    adaptation_speed_ms: float
    model_params: int


class MetaLearningCore:
    """元学习核心系统."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, 
                 output_dim: int = 10, algorithm: str = "maml"):
        
        self.model = SimpleNetwork(input_dim, hidden_dim, output_dim)
        self.config = MetaLearningConfig()
        
        if algorithm == "maml":
            self.meta_learner = MAML(self.model, self.config)
        elif algorithm == "reptile":
            self.meta_learner = Reptile(self.model, self.config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.algorithm = algorithm
        self.few_shot_learner = FewShotLearner(self.meta_learner)
        
        # 训练历史
        self.loss_history: List[float] = []
    
    def meta_train(self, task_generator: Callable[[], Task], 
                   n_iterations: int = 100,
                   verbose: bool = True) -> List[float]:
        """执行元训练.
        
        Args:
            task_generator: 任务生成器
            n_iterations: 元迭代次数
            verbose: 是否打印进度
        """
        losses = []
        
        for i in range(n_iterations):
            # 生成任务批次
            task_batch = [task_generator() for _ in range(self.config.meta_batch_size)]
            
            # 元训练步骤
            loss = self.meta_learner.meta_train_step(task_batch)
            losses.append(loss)
            self.loss_history.append(loss)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.4f}")
        
        return losses
    
    def adapt_to_task(self, support_x: np.ndarray, support_y: np.ndarray,
                      steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """适应到新任务."""
        return self.meta_learner.adapt(support_x, support_y, steps)
    
    def predict(self, x: np.ndarray, 
                adapted_params: Optional[Dict[str, np.ndarray]] = None
                ) -> np.ndarray:
        """预测."""
        return self.model.predict(x, adapted_params)
    
    def evaluate_few_shot(self, X: np.ndarray, y: np.ndarray,
                          n_way: int = 5, k_shot: int = 5,
                          n_episodes: int = 100) -> FewShotResult:
        """评估少样本性能."""
        return self.few_shot_learner.evaluate_few_shot(X, y, n_way, k_shot, n_episodes)
    
    def get_summary(self) -> MetaLearningResult:
        """获取摘要."""
        return MetaLearningResult(
            algorithm=self.algorithm,
            meta_iterations=len(self.loss_history),
            final_meta_loss=self.loss_history[-1] if self.loss_history else float('inf'),
            few_shot_accuracy=0.0,  # 需要实际评估
            adaptation_speed_ms=0.0,
            model_params=self.model.parameter_count()
        )


# ============================================================================
# 工厂函数
# ============================================================================

def create_meta_learning_core(input_dim: int = 64, hidden_dim: int = 32,
                               output_dim: int = 10, algorithm: str = "maml"
                               ) -> MetaLearningCore:
    """创建元学习核心."""
    return MetaLearningCore(input_dim, hidden_dim, output_dim, algorithm)


def create_random_task(input_dim: int = 64, output_dim: int = 10,
                       n_support: int = 10, n_query: int = 15) -> Task:
    """创建随机任务 (用于测试)."""
    # 随机生成数据
    X_support = np.random.randn(n_support, input_dim).astype(np.float32)
    y_support = np.random.randint(0, output_dim, n_support)
    
    X_query = np.random.randn(n_query, input_dim).astype(np.float32)
    y_query = np.random.randint(0, output_dim, n_query)
    
    return Task(
        task_id=f"random_{np.random.randint(10000)}",
        support_set=(X_support, y_support),
        query_set=(X_query, y_query)
    )


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 元学习核心 - 演示")
    print("=" * 70)
    
    # 创建元学习器
    meta_core = create_meta_learning_core(
        input_dim=64, hidden_dim=32, output_dim=5, algorithm="maml"
    )
    
    # 元训练
    print("\n1. 元训练 (MAML)")
    print("-" * 50)
    
    task_gen = lambda: create_random_task(64, 5, 10, 15)
    losses = meta_core.meta_train(task_gen, n_iterations=50, verbose=True)
    
    print(f"\n初始损失: {losses[0]:.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"损失下降: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # 少样本适应测试
    print("\n2. 少样本适应测试")
    print("-" * 50)
    
    # 生成测试任务
    test_task = create_random_task(64, 5, 5, 20)  # 5-shot
    
    # 适应
    start = time.perf_counter()
    adapted_params = meta_core.adapt_to_task(*test_task.support_set, steps=5)
    adapt_time = (time.perf_counter() - start) * 1000
    
    # 评估
    probs = meta_core.predict(test_task.query_set[0], adapted_params)
    preds = np.argmax(probs, axis=-1)
    accuracy = np.mean(preds == test_task.query_set[1])
    
    print(f"5-shot 适应时间: {adapt_time:.2f} ms")
    print(f"查询集准确率: {accuracy * 100:.1f}%")
    print(f"模型参数: {meta_core.model.parameter_count()}")
    
    # 汇总
    print("\n" + "=" * 70)
    summary = meta_core.get_summary()
    print(f"算法: {summary.algorithm}")
    print(f"元迭代次数: {summary.meta_iterations}")
    print(f"最终元损失: {summary.final_meta_loss:.4f}")
