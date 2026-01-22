"""H2Q 持续学习模块 (Continual Learning Module).

实现AGI核心能力：
1. EWC (Elastic Weight Consolidation)
2. PackNet (逐步网络剪枝)
3. 记忆回放
4. 抗遗忘机制

参考文献:
- Kirkpatrick et al., "Overcoming catastrophic forgetting" (2017)
- Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network" (2018)
- Lopez-Paz & Ranzato, "Gradient Episodic Memory" (2017)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque
import time
import copy


# ============================================================================
# 基础组件
# ============================================================================

@dataclass
class ContinualTask:
    """持续学习任务."""
    task_id: str
    task_idx: int
    train_data: Tuple[np.ndarray, np.ndarray]  # (X, y)
    test_data: Tuple[np.ndarray, np.ndarray]
    n_classes: int
    description: str = ""


@dataclass
class ContinualConfig:
    """持续学习配置."""
    ewc_lambda: float = 1000.0       # EWC 正则化强度
    memory_size: int = 200           # 每任务记忆大小
    learning_rate: float = 0.01      # 学习率
    n_epochs: int = 10               # 每任务训练轮数
    batch_size: int = 32             # 批大小
    fisher_samples: int = 200        # Fisher 信息估计样本数


class ContinualNetwork:
    """持续学习神经网络."""
    
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
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播."""
        h = np.maximum(0, x @ self.params['W1'] + self.params['b1'])
        out = h @ self.params['W2'] + self.params['b2']
        return out
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """预测概率."""
        logits = self.forward(x)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测类别."""
        return np.argmax(self.predict_proba(x), axis=-1)
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算交叉熵损失."""
        probs = self.predict_proba(x)
        n = len(y)
        log_probs = np.log(probs[np.arange(n), y.astype(int)] + 1e-10)
        return -np.mean(log_probs)
    
    def compute_gradients(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """计算梯度."""
        epsilon = 1e-5
        grads = {}
        
        for key in self.params:
            grad = np.zeros_like(self.params[key])
            it = np.nditer(self.params[key], flags=['multi_index'])
            
            while not it.finished:
                idx = it.multi_index
                original = self.params[key][idx]
                
                self.params[key][idx] = original + epsilon
                loss_plus = self.compute_loss(x, y)
                
                self.params[key][idx] = original - epsilon
                loss_minus = self.compute_loss(x, y)
                
                grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                
                self.params[key][idx] = original
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
# EWC (Elastic Weight Consolidation)
# ============================================================================

@dataclass
class EWCMemory:
    """EWC 任务记忆."""
    task_id: str
    optimal_params: Dict[str, np.ndarray]
    fisher_matrix: Dict[str, np.ndarray]
    importance: float = 1.0


class EWC:
    """Elastic Weight Consolidation."""
    
    def __init__(self, network: ContinualNetwork, config: ContinualConfig):
        self.network = network
        self.config = config
        
        # 任务记忆
        self.task_memories: List[EWCMemory] = []
        self.tasks_learned: List[str] = []
    
    def compute_fisher_matrix(self, x: np.ndarray, y: np.ndarray,
                               n_samples: Optional[int] = None
                               ) -> Dict[str, np.ndarray]:
        """估计 Fisher 信息矩阵.
        
        Fisher 信息 = E[grad(log p(y|x))^2]
        """
        n_samples = n_samples or min(self.config.fisher_samples, len(x))
        indices = np.random.choice(len(x), n_samples, replace=False)
        
        fisher = {k: np.zeros_like(v) for k, v in self.network.params.items()}
        
        for idx in indices:
            xi = x[idx:idx+1]
            yi = y[idx:idx+1]
            
            # 计算梯度
            grads = self.network.compute_gradients(xi, yi)
            
            # Fisher = grad^2
            for key in fisher:
                fisher[key] += grads[key] ** 2
        
        # 平均
        for key in fisher:
            fisher[key] /= n_samples
        
        return fisher
    
    def ewc_penalty(self) -> float:
        """计算 EWC 惩罚项.
        
        penalty = sum_i (lambda/2) * F_i * (θ - θ*_i)^2
        """
        if not self.task_memories:
            return 0.0
        
        penalty = 0.0
        for memory in self.task_memories:
            for key in self.network.params:
                diff = self.network.params[key] - memory.optimal_params[key]
                penalty += np.sum(memory.fisher_matrix[key] * (diff ** 2))
        
        return (self.config.ewc_lambda / 2) * penalty
    
    def train_on_task(self, task: ContinualTask, verbose: bool = True) -> Dict[str, List[float]]:
        """在任务上训练 (带 EWC 正则化).
        
        Returns:
            训练历史
        """
        X, y = task.train_data
        n = len(X)
        
        history = {"loss": [], "ewc_penalty": [], "total_loss": []}
        
        for epoch in range(self.config.n_epochs):
            # 随机打乱
            indices = np.random.permutation(n)
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # 计算梯度
                grads = self.network.compute_gradients(X_batch, y_batch)
                
                # 添加 EWC 梯度
                for memory in self.task_memories:
                    for key in grads:
                        diff = self.network.params[key] - memory.optimal_params[key]
                        grads[key] += self.config.ewc_lambda * memory.fisher_matrix[key] * diff
                
                # 更新参数
                for key in self.network.params:
                    self.network.params[key] -= self.config.learning_rate * grads[key]
                
                batch_loss = self.network.compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            ewc_pen = self.ewc_penalty()
            total_loss = avg_loss + ewc_pen
            
            history["loss"].append(avg_loss)
            history["ewc_penalty"].append(ewc_pen)
            history["total_loss"].append(total_loss)
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{self.config.n_epochs}, "
                      f"Loss: {avg_loss:.4f}, EWC: {ewc_pen:.4f}")
        
        return history
    
    def consolidate(self, task: ContinualTask):
        """固化当前任务知识."""
        X, y = task.train_data
        
        # 计算 Fisher 信息
        fisher = self.compute_fisher_matrix(X, y)
        
        # 保存记忆
        memory = EWCMemory(
            task_id=task.task_id,
            optimal_params=self.network.copy_params(),
            fisher_matrix=fisher
        )
        
        self.task_memories.append(memory)
        self.tasks_learned.append(task.task_id)
    
    def evaluate(self, task: ContinualTask) -> Dict[str, float]:
        """评估任务."""
        X, y = task.test_data
        preds = self.network.predict(X)
        
        accuracy = np.mean(preds == y)
        loss = self.network.compute_loss(X, y)
        
        return {
            "accuracy": float(accuracy),
            "loss": float(loss)
        }


# ============================================================================
# 记忆回放 (Experience Replay)
# ============================================================================

@dataclass
class MemoryBuffer:
    """经验回放缓冲区."""
    task_id: str
    X: np.ndarray
    y: np.ndarray
    

class ExperienceReplay:
    """经验回放持续学习."""
    
    def __init__(self, network: ContinualNetwork, config: ContinualConfig):
        self.network = network
        self.config = config
        
        # 记忆缓冲区
        self.memory_buffers: List[MemoryBuffer] = []
        self.tasks_learned: List[str] = []
    
    def store_memory(self, task: ContinualTask):
        """存储任务记忆."""
        X, y = task.train_data
        
        # 随机采样
        n_store = min(self.config.memory_size, len(X))
        indices = np.random.choice(len(X), n_store, replace=False)
        
        buffer = MemoryBuffer(
            task_id=task.task_id,
            X=X[indices].copy(),
            y=y[indices].copy()
        )
        
        self.memory_buffers.append(buffer)
        self.tasks_learned.append(task.task_id)
    
    def sample_memory(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """从所有任务记忆中采样."""
        if not self.memory_buffers:
            return np.array([]), np.array([])
        
        # 每个任务平均采样
        per_task = max(1, n_samples // len(self.memory_buffers))
        
        X_all, y_all = [], []
        
        for buffer in self.memory_buffers:
            n_sample = min(per_task, len(buffer.X))
            indices = np.random.choice(len(buffer.X), n_sample, replace=False)
            X_all.append(buffer.X[indices])
            y_all.append(buffer.y[indices])
        
        return np.vstack(X_all), np.concatenate(y_all)
    
    def train_on_task(self, task: ContinualTask, replay_ratio: float = 0.5,
                      verbose: bool = True) -> Dict[str, List[float]]:
        """训练 (带记忆回放)."""
        X, y = task.train_data
        n = len(X)
        
        history = {"loss": [], "replay_loss": []}
        
        for epoch in range(self.config.n_epochs):
            indices = np.random.permutation(n)
            
            epoch_loss = 0.0
            replay_loss = 0.0
            n_batches = 0
            
            for i in range(0, n, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # 添加回放样本
                if self.memory_buffers and replay_ratio > 0:
                    n_replay = int(len(X_batch) * replay_ratio)
                    X_replay, y_replay = self.sample_memory(n_replay)
                    
                    if len(X_replay) > 0:
                        X_batch = np.vstack([X_batch, X_replay])
                        y_batch = np.concatenate([y_batch, y_replay])
                        replay_loss += self.network.compute_loss(X_replay, y_replay)
                
                # 计算梯度并更新
                grads = self.network.compute_gradients(X_batch, y_batch)
                for key in self.network.params:
                    self.network.params[key] -= self.config.learning_rate * grads[key]
                
                epoch_loss += self.network.compute_loss(X[batch_idx], y[batch_idx])
                n_batches += 1
            
            history["loss"].append(epoch_loss / n_batches)
            history["replay_loss"].append(replay_loss / max(1, n_batches))
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}, Loss: {history['loss'][-1]:.4f}")
        
        return history
    
    def evaluate(self, task: ContinualTask) -> Dict[str, float]:
        """评估."""
        X, y = task.test_data
        preds = self.network.predict(X)
        
        return {
            "accuracy": float(np.mean(preds == y)),
            "loss": float(self.network.compute_loss(X, y))
        }


# ============================================================================
# PackNet (渐进式剪枝)
# ============================================================================

@dataclass
class TaskMask:
    """任务参数掩码."""
    task_id: str
    masks: Dict[str, np.ndarray]  # 1 = 属于此任务, 0 = 不属于


class PackNet:
    """PackNet: 通过网络剪枝实现持续学习."""
    
    def __init__(self, network: ContinualNetwork, config: ContinualConfig):
        self.network = network
        self.config = config
        
        # 任务掩码
        self.task_masks: List[TaskMask] = []
        self.available_mask = {k: np.ones_like(v) for k, v in network.params.items()}
        
        # 剪枝率
        self.prune_rate = 0.5  # 每任务保留50%参数
    
    def prune_for_task(self, task_id: str):
        """为任务剪枝."""
        masks = {}
        
        for key, param in self.network.params.items():
            # 获取当前可用参数
            available = self.available_mask[key]
            available_indices = np.where(available.flatten() > 0)[0]
            
            if len(available_indices) == 0:
                masks[key] = np.zeros_like(param)
                continue
            
            # 按重要性排序 (使用参数绝对值)
            importance = np.abs(param.flatten()[available_indices])
            n_keep = max(1, int(len(available_indices) * self.prune_rate))
            
            keep_indices = available_indices[np.argsort(importance)[-n_keep:]]
            
            # 创建掩码
            mask = np.zeros_like(param).flatten()
            mask[keep_indices] = 1.0
            masks[key] = mask.reshape(param.shape)
            
            # 更新可用掩码
            self.available_mask[key] = available - masks[key]
        
        self.task_masks.append(TaskMask(task_id=task_id, masks=masks))
    
    def freeze_previous_tasks(self):
        """冻结之前任务的参数."""
        # 在训练新任务时，只更新未被占用的参数
        pass
    
    def train_on_task(self, task: ContinualTask, verbose: bool = True
                      ) -> Dict[str, List[float]]:
        """训练 (带参数冻结)."""
        X, y = task.train_data
        n = len(X)
        
        history = {"loss": []}
        
        for epoch in range(self.config.n_epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                grads = self.network.compute_gradients(X_batch, y_batch)
                
                # 只更新可用参数
                for key in self.network.params:
                    masked_grad = grads[key] * self.available_mask[key]
                    self.network.params[key] -= self.config.learning_rate * masked_grad
                
                epoch_loss += self.network.compute_loss(X_batch, y_batch)
                n_batches += 1
            
            history["loss"].append(epoch_loss / n_batches)
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}, Loss: {history['loss'][-1]:.4f}")
        
        return history
    
    def get_capacity_usage(self) -> float:
        """获取网络容量使用率."""
        total = sum(np.sum(self.available_mask[k] == 0) for k in self.available_mask)
        capacity = sum(m.size for m in self.available_mask.values())
        return total / capacity if capacity > 0 else 0.0
    
    def evaluate(self, task: ContinualTask) -> Dict[str, float]:
        """评估."""
        X, y = task.test_data
        preds = self.network.predict(X)
        
        return {
            "accuracy": float(np.mean(preds == y)),
            "loss": float(self.network.compute_loss(X, y))
        }


# ============================================================================
# 持续学习系统
# ============================================================================

@dataclass
class ContinualLearningResult:
    """持续学习结果."""
    method: str
    tasks_learned: List[str]
    final_accuracies: Dict[str, float]
    average_accuracy: float
    backward_transfer: float  # 学习新任务后旧任务性能变化
    forward_transfer: float   # 旧任务对新任务的帮助


class ContinualLearningSystem:
    """持续学习系统."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32,
                 output_dim: int = 10, method: str = "ewc"):
        
        self.network = ContinualNetwork(input_dim, hidden_dim, output_dim)
        self.config = ContinualConfig()
        
        if method == "ewc":
            self.learner = EWC(self.network, self.config)
        elif method == "replay":
            self.learner = ExperienceReplay(self.network, self.config)
        elif method == "packnet":
            self.learner = PackNet(self.network, self.config)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
        self.task_results: Dict[str, Dict[str, float]] = {}
    
    def learn_task(self, task: ContinualTask, verbose: bool = True) -> Dict[str, Any]:
        """学习新任务."""
        print(f"\n学习任务: {task.task_id}")
        print("-" * 40)
        
        # 训练
        history = self.learner.train_on_task(task, verbose=verbose)
        
        # 固化/存储
        if hasattr(self.learner, 'consolidate'):
            self.learner.consolidate(task)
        elif hasattr(self.learner, 'store_memory'):
            self.learner.store_memory(task)
        elif hasattr(self.learner, 'prune_for_task'):
            self.learner.prune_for_task(task.task_id)
        
        # 评估当前任务
        result = self.learner.evaluate(task)
        self.task_results[task.task_id] = result
        
        return history
    
    def evaluate_all_tasks(self, tasks: List[ContinualTask]) -> Dict[str, Dict[str, float]]:
        """评估所有任务."""
        results = {}
        
        for task in tasks:
            result = self.learner.evaluate(task)
            results[task.task_id] = result
        
        return results
    
    def compute_backward_transfer(self, tasks: List[ContinualTask]) -> float:
        """计算后向迁移 (遗忘程度).
        
        BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
        其中 R_{j,i} 是学完任务j后在任务i上的性能
        """
        if len(tasks) < 2:
            return 0.0
        
        current_results = self.evaluate_all_tasks(tasks)
        
        # 简化：与初始记录比较
        bwt = 0.0
        for task_id, initial_result in self.task_results.items():
            if task_id in current_results:
                diff = current_results[task_id]["accuracy"] - initial_result["accuracy"]
                bwt += diff
        
        return bwt / max(1, len(self.task_results) - 1)
    
    def get_summary(self, tasks: List[ContinualTask]) -> ContinualLearningResult:
        """获取摘要."""
        all_results = self.evaluate_all_tasks(tasks)
        
        accuracies = {tid: r["accuracy"] for tid, r in all_results.items()}
        avg_acc = np.mean(list(accuracies.values())) if accuracies else 0.0
        
        bwt = self.compute_backward_transfer(tasks)
        
        return ContinualLearningResult(
            method=self.method,
            tasks_learned=list(self.task_results.keys()),
            final_accuracies=accuracies,
            average_accuracy=float(avg_acc),
            backward_transfer=float(bwt),
            forward_transfer=0.0  # 需要更复杂的计算
        )


# ============================================================================
# 工厂函数
# ============================================================================

def create_continual_learning_system(input_dim: int = 64, hidden_dim: int = 32,
                                      output_dim: int = 10, method: str = "ewc"
                                      ) -> ContinualLearningSystem:
    """创建持续学习系统."""
    return ContinualLearningSystem(input_dim, hidden_dim, output_dim, method)


def create_random_task_sequence(n_tasks: int = 3, input_dim: int = 64,
                                 n_classes_per_task: int = 5,
                                 n_train: int = 200, n_test: int = 50
                                 ) -> List[ContinualTask]:
    """创建随机任务序列."""
    tasks = []
    
    for i in range(n_tasks):
        # 生成任务数据
        X_train = np.random.randn(n_train, input_dim).astype(np.float32)
        y_train = np.random.randint(0, n_classes_per_task, n_train)
        
        X_test = np.random.randn(n_test, input_dim).astype(np.float32)
        y_test = np.random.randint(0, n_classes_per_task, n_test)
        
        task = ContinualTask(
            task_id=f"task_{i+1}",
            task_idx=i,
            train_data=(X_train, y_train),
            test_data=(X_test, y_test),
            n_classes=n_classes_per_task,
            description=f"Random task {i+1}"
        )
        tasks.append(task)
    
    return tasks


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 持续学习模块 - 演示")
    print("=" * 70)
    
    # 创建任务序列
    print("\n1. 创建任务序列")
    print("-" * 50)
    
    tasks = create_random_task_sequence(
        n_tasks=3, input_dim=64, n_classes_per_task=5, 
        n_train=150, n_test=50
    )
    
    for task in tasks:
        print(f"  {task.task_id}: {len(task.train_data[0])} 训练, "
              f"{len(task.test_data[0])} 测试, {task.n_classes} 类")
    
    # EWC 方法
    print("\n2. EWC (Elastic Weight Consolidation) 方法")
    print("-" * 50)
    
    ewc_system = create_continual_learning_system(
        input_dim=64, hidden_dim=32, output_dim=5, method="ewc"
    )
    
    for task in tasks:
        ewc_system.learn_task(task, verbose=True)
    
    ewc_summary = ewc_system.get_summary(tasks)
    
    print(f"\nEWC 结果:")
    print(f"  学习任务数: {len(ewc_summary.tasks_learned)}")
    print(f"  平均准确率: {ewc_summary.average_accuracy * 100:.1f}%")
    print(f"  后向迁移: {ewc_summary.backward_transfer:.4f}")
    
    # 记忆回放方法
    print("\n3. 记忆回放方法")
    print("-" * 50)
    
    replay_system = create_continual_learning_system(
        input_dim=64, hidden_dim=32, output_dim=5, method="replay"
    )
    
    for task in tasks:
        replay_system.learn_task(task, verbose=True)
    
    replay_summary = replay_system.get_summary(tasks)
    
    print(f"\n记忆回放结果:")
    print(f"  学习任务数: {len(replay_summary.tasks_learned)}")
    print(f"  平均准确率: {replay_summary.average_accuracy * 100:.1f}%")
    
    # 比较
    print("\n" + "=" * 70)
    print("方法比较")
    print("=" * 70)
    print(f"{'方法':<15} {'平均准确率':>15} {'后向迁移':>15}")
    print("-" * 45)
    print(f"{'EWC':<15} {ewc_summary.average_accuracy*100:>14.1f}% {ewc_summary.backward_transfer:>14.4f}")
    print(f"{'记忆回放':<15} {replay_summary.average_accuracy*100:>14.1f}% {replay_summary.backward_transfer:>14.4f}")
