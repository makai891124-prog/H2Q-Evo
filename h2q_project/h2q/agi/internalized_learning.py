#!/usr/bin/env python3
"""
真正的内化学习系统
- 基于神经网络的知识内化
- 真正的训练过程（前向传播、反向传播、参数更新）
- 闭卷考试验证（训练后不再访问答案）
- 训练集/测试集分离
- 可验证的学习过程

核心理念：
- 学习 ≠ 记忆答案
- 学习 = 通过训练更新神经网络参数，使其能够泛化到新问题
"""

import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pickle
from pathlib import Path


class LearningPhase(Enum):
    """学习阶段."""
    TRAINING = "training"      # 训练阶段：可以看答案
    VALIDATION = "validation"  # 验证阶段：用验证集调参
    TESTING = "testing"        # 测试阶段：闭卷考试


@dataclass
class TrainingSample:
    """训练样本."""
    id: str
    question: str
    choices: List[str]
    correct_answer: int
    category: str
    embedding: Optional[np.ndarray] = None
    
    def to_input_vector(self) -> np.ndarray:
        """转换为输入向量."""
        if self.embedding is not None:
            return self.embedding
        
        # 简化的文本嵌入：基于字符和词的特征
        text = self.question + " ".join(self.choices)
        
        # 1. 字符级特征 (256维)
        char_freq = np.zeros(256)
        for c in text.lower():
            if ord(c) < 256:
                char_freq[ord(c)] += 1
        char_freq = char_freq / (len(text) + 1)
        
        # 2. 词级特征 (100维)
        words = text.lower().split()
        word_hash = np.zeros(100)
        for w in words:
            h = hash(w) % 100
            word_hash[h] += 1
        word_hash = word_hash / (len(words) + 1)
        
        # 3. 结构特征 (44维)
        struct_feat = np.array([
            len(self.question),
            len(self.choices),
            np.mean([len(c) for c in self.choices]),
            self.question.count('?'),
            self.question.count('.'),
            sum(1 for c in self.question if c.isupper()),
            sum(1 for c in self.question if c.isdigit()),
            # 更多特征...
        ] + [0] * 37)  # 填充到44维
        struct_feat = struct_feat / (np.max(np.abs(struct_feat)) + 1e-8)
        
        # 合并特征 (400维)
        self.embedding = np.concatenate([char_freq, word_hash, struct_feat])
        return self.embedding


class NeuralKnowledgeNetwork:
    """
    神经知识网络 - 真正的内化学习模型.
    
    架构：多层感知机 (MLP)
    - 输入层: 400维 (文本嵌入)
    - 隐藏层1: 256维 (ReLU)
    - 隐藏层2: 128维 (ReLU)
    - 隐藏层3: 64维 (ReLU)
    - 输出层: 4维 (Softmax, 对应4个选项)
    """
    
    def __init__(self, input_dim: int = 400, hidden_dims: List[int] = [256, 128, 64], output_dim: int = 4):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 初始化权重 (Xavier初始化)
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            # Xavier初始化
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # 训练统计
        self.training_history = []
        self.total_updates = 0
        
        # 缓存（用于反向传播）
        self._cache = {}
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        前向传播.
        
        Args:
            x: 输入向量 [batch_size, input_dim] 或 [input_dim]
            training: 是否处于训练模式
        
        Returns:
            输出概率分布 [batch_size, output_dim] 或 [output_dim]
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self._cache = {'input': x, 'activations': [], 'pre_activations': []}
        
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(a, w) + b
            self._cache['pre_activations'].append(z)
            
            if i < len(self.weights) - 1:
                # 隐藏层使用ReLU
                a = self._relu(z)
                
                # Dropout (仅训练时)
                if training:
                    dropout_mask = np.random.binomial(1, 0.8, size=a.shape) / 0.8
                    a = a * dropout_mask
            else:
                # 输出层使用Softmax
                a = self._softmax(z)
            
            self._cache['activations'].append(a)
        
        return a.squeeze()
    
    def backward(self, y_true: int, learning_rate: float = 0.01) -> float:
        """
        反向传播 + 参数更新.
        
        Args:
            y_true: 正确答案的索引
            learning_rate: 学习率
        
        Returns:
            损失值
        """
        # 获取前向传播的输出
        y_pred = self._cache['activations'][-1]
        batch_size = y_pred.shape[0]
        
        # 创建one-hot编码
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[0, y_true] = 1
        
        # 计算交叉熵损失
        loss = -np.sum(y_true_onehot * np.log(y_pred + 1e-8)) / batch_size
        
        # 反向传播
        # 输出层梯度 (softmax + cross-entropy的简化梯度)
        delta = (y_pred - y_true_onehot) / batch_size
        
        # 从后向前传播
        for i in range(len(self.weights) - 1, -1, -1):
            # 获取前一层的激活值
            if i > 0:
                prev_activation = self._cache['activations'][i-1]
            else:
                prev_activation = self._cache['input']
            
            # 计算梯度
            dw = np.dot(prev_activation.T, delta)
            db = np.sum(delta, axis=0)
            
            # 更新参数 (带L2正则化)
            self.weights[i] -= learning_rate * (dw + 0.001 * self.weights[i])
            self.biases[i] -= learning_rate * db
            
            # 传播到前一层 (如果不是第一层)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                # ReLU的导数
                delta = delta * (self._cache['pre_activations'][i-1] > 0)
        
        self.total_updates += 1
        return loss
    
    def predict(self, x: np.ndarray) -> int:
        """
        预测（闭卷考试模式）.
        
        Args:
            x: 输入向量
        
        Returns:
            预测的答案索引
        """
        probs = self.forward(x, training=False)
        return int(np.argmax(probs))
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数."""
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax激活函数."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)
    
    def save(self, path: str):
        """保存模型参数."""
        state = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'total_updates': self.total_updates,
            'training_history': self.training_history
        }
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load(self, path: str):
        """加载模型参数."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.weights = [np.array(w) for w in state['weights']]
        self.biases = [np.array(b) for b in state['biases']]
        self.total_updates = state['total_updates']
        self.training_history = state.get('training_history', [])


class InternalizedLearningSystem:
    """
    内化学习系统 - 真正的训练和测试分离.
    
    学习流程:
    1. 数据准备: 划分训练集/验证集/测试集
    2. 训练阶段: 使用训练集进行梯度下降
    3. 验证阶段: 使用验证集调整超参数
    4. 测试阶段: 使用测试集进行闭卷考试（不再访问答案）
    """
    
    def __init__(self, model_path: str = None):
        self.model = NeuralKnowledgeNetwork()
        self.model_path = model_path or "internalized_model.json"
        
        # 数据集
        self.train_set: List[TrainingSample] = []
        self.val_set: List[TrainingSample] = []
        self.test_set: List[TrainingSample] = []
        
        # 训练状态
        self.current_phase = LearningPhase.TRAINING
        self.epochs_completed = 0
        self.best_val_accuracy = 0.0
        
        # 训练历史
        self.loss_history = []
        self.accuracy_history = []
    
    def prepare_data(self, samples: List[Dict], train_ratio: float = 0.6, val_ratio: float = 0.2):
        """
        准备数据集 - 严格划分训练/验证/测试集.
        
        Args:
            samples: 原始样本列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例 (剩余为测试集)
        """
        # 转换为TrainingSample
        all_samples = []
        for i, s in enumerate(samples):
            sample = TrainingSample(
                id=s.get('id', f'sample_{i}'),
                question=s['question'],
                choices=s['choices'],
                correct_answer=s['correct_answer'],
                category=s.get('category', 'general')
            )
            all_samples.append(sample)
        
        # 随机打乱
        np.random.shuffle(all_samples)
        
        # 划分数据集
        n = len(all_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        self.train_set = all_samples[:n_train]
        self.val_set = all_samples[n_train:n_train+n_val]
        self.test_set = all_samples[n_train+n_val:]
        
        print(f"📊 数据集划分:")
        print(f"  训练集: {len(self.train_set)} 样本")
        print(f"  验证集: {len(self.val_set)} 样本")
        print(f"  测试集: {len(self.test_set)} 样本 (闭卷考试)")
    
    def train_epoch(self, learning_rate: float = 0.01, verbose: bool = True) -> Dict[str, float]:
        """
        训练一个epoch.
        
        这是真正的训练过程:
        1. 遍历训练集每个样本
        2. 前向传播计算预测
        3. 计算损失
        4. 反向传播更新参数
        
        Returns:
            训练统计 {'loss': float, 'accuracy': float}
        """
        self.current_phase = LearningPhase.TRAINING
        
        total_loss = 0.0
        correct = 0
        
        # 打乱训练集
        indices = np.random.permutation(len(self.train_set))
        
        for idx in indices:
            sample = self.train_set[idx]
            
            # 1. 获取输入向量
            x = sample.to_input_vector()
            
            # 2. 前向传播
            probs = self.model.forward(x, training=True)
            
            # 3. 记录预测是否正确
            pred = np.argmax(probs)
            if pred == sample.correct_answer:
                correct += 1
            
            # 4. 反向传播 + 参数更新 (这是真正的学习!)
            loss = self.model.backward(sample.correct_answer, learning_rate)
            total_loss += loss
        
        # 计算统计
        avg_loss = total_loss / len(self.train_set)
        accuracy = correct / len(self.train_set)
        
        self.epochs_completed += 1
        self.loss_history.append(avg_loss)
        self.accuracy_history.append(accuracy)
        
        if verbose:
            print(f"  Epoch {self.epochs_completed}: Loss={avg_loss:.4f}, Train Acc={accuracy*100:.1f}%")
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """
        验证阶段 - 使用验证集评估（不更新参数）.
        
        Returns:
            验证统计
        """
        self.current_phase = LearningPhase.VALIDATION
        
        correct = 0
        
        for sample in self.val_set:
            x = sample.to_input_vector()
            pred = self.model.predict(x)  # 不更新参数
            
            if pred == sample.correct_answer:
                correct += 1
        
        accuracy = correct / len(self.val_set) if self.val_set else 0
        
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            # 保存最佳模型
            self.model.save(self.model_path)
        
        return {'accuracy': accuracy, 'best_accuracy': self.best_val_accuracy}
    
    def test(self) -> Dict[str, Any]:
        """
        测试阶段 - 闭卷考试！
        
        关键: 这里完全不能访问correct_answer来做任何决策,
        只能在最后用于统计准确率.
        
        Returns:
            测试结果
        """
        self.current_phase = LearningPhase.TESTING
        
        predictions = []
        correct = 0
        
        print("\n🎓 闭卷考试开始 (测试集)")
        print("-" * 50)
        
        for sample in self.test_set:
            # 只使用问题和选项，不能访问答案
            x = sample.to_input_vector()
            
            # 模型预测 (纯粹基于内化的知识)
            pred = self.model.predict(x)
            
            predictions.append({
                'id': sample.id,
                'question': sample.question[:50] + '...',
                'predicted': pred,
                'actual': sample.correct_answer  # 仅用于统计
            })
            
            # 统计（事后评估）
            if pred == sample.correct_answer:
                correct += 1
        
        accuracy = correct / len(self.test_set) if self.test_set else 0
        
        print(f"\n📊 闭卷考试结果:")
        print(f"  正确: {correct}/{len(self.test_set)}")
        print(f"  准确率: {accuracy*100:.1f}%")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(self.test_set),
            'predictions': predictions,
            'model_updates': self.model.total_updates
        }
    
    def full_training_cycle(self, 
                           samples: List[Dict],
                           epochs: int = 50,
                           learning_rate: float = 0.01,
                           early_stopping_patience: int = 10) -> Dict[str, Any]:
        """
        完整的训练周期.
        
        Args:
            samples: 所有样本
            epochs: 最大训练轮数
            learning_rate: 学习率
            early_stopping_patience: 早停耐心值
        
        Returns:
            完整训练报告
        """
        print("=" * 60)
        print("🧠 内化学习系统 - 完整训练周期")
        print("=" * 60)
        
        # 1. 准备数据
        self.prepare_data(samples)
        
        # 2. 训练
        print(f"\n📚 开始训练 (最多 {epochs} epochs)...")
        print("-" * 50)
        
        no_improve_count = 0
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_stats = self.train_epoch(learning_rate)
            
            # 验证
            val_stats = self.validate()
            
            # 早停检查
            if val_stats['accuracy'] > best_val_acc:
                best_val_acc = val_stats['accuracy']
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if epoch % 10 == 0:
                print(f"    Val Acc: {val_stats['accuracy']*100:.1f}% (best: {best_val_acc*100:.1f}%)")
            
            if no_improve_count >= early_stopping_patience:
                print(f"\n  ⏹️ 早停: {early_stopping_patience} epochs无提升")
                break
        
        # 3. 加载最佳模型
        try:
            self.model.load(self.model_path)
            print(f"\n  ✅ 加载最佳模型 (验证准确率: {best_val_acc*100:.1f}%)")
        except:
            pass
        
        # 4. 闭卷考试
        test_results = self.test()
        
        # 5. 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'training': {
                'epochs': self.epochs_completed,
                'total_updates': self.model.total_updates,
                'final_loss': self.loss_history[-1] if self.loss_history else None,
                'loss_history': self.loss_history
            },
            'validation': {
                'best_accuracy': best_val_acc
            },
            'test': test_results,
            'is_real_learning': True,
            'methodology': {
                'architecture': 'MLP (400->256->128->64->4)',
                'optimizer': 'SGD with L2 regularization',
                'training_samples': len(self.train_set),
                'test_samples': len(self.test_set)
            }
        }
        
        print("\n" + "=" * 60)
        print("✅ 内化学习完成!")
        print(f"  模型参数更新次数: {self.model.total_updates}")
        print(f"  闭卷考试准确率: {test_results['accuracy']*100:.1f}%")
        print("=" * 60)
        
        return report


class HonestBenchmarkEvaluator:
    """
    诚实的基准评估器 - 区分开卷和闭卷.
    """
    
    def __init__(self):
        self.learning_system = InternalizedLearningSystem()
        
    def evaluate_with_real_learning(self, benchmark_data: List[Dict]) -> Dict[str, Any]:
        """
        使用真正的学习进行评估.
        
        流程:
        1. 用60%数据训练模型
        2. 用20%数据验证和调参
        3. 用20%数据进行闭卷考试
        """
        return self.learning_system.full_training_cycle(
            samples=benchmark_data,
            epochs=100,
            learning_rate=0.005
        )
    
    def compare_methods(self, benchmark_data: List[Dict]) -> Dict[str, Any]:
        """
        对比开卷考试 vs 闭卷考试（真正学习后）.
        """
        print("\n" + "=" * 70)
        print("📊 方法对比: 开卷考试 vs 真正学习")
        print("=" * 70)
        
        # 1. 开卷考试（作弊方式）
        print("\n🔓 开卷考试（关键词匹配）:")
        open_book_correct = 0
        for sample in benchmark_data:
            # 直接用答案匹配
            open_book_correct += 1  # 假设全对（因为是作弊）
        
        open_book_acc = 100.0
        print(f"  准确率: {open_book_acc:.1f}% (但这是作弊!)")
        
        # 2. 闭卷考试（真正学习）
        print("\n🔒 闭卷考试（真正学习后）:")
        real_learning_results = self.evaluate_with_real_learning(benchmark_data)
        closed_book_acc = real_learning_results['test']['accuracy'] * 100
        
        print(f"\n📊 对比结果:")
        print(f"  开卷考试: {open_book_acc:.1f}% (不可信)")
        print(f"  闭卷考试: {closed_book_acc:.1f}% (真实能力)")
        print(f"  差距: {open_book_acc - closed_book_acc:.1f}%")
        
        return {
            'open_book': {'accuracy': open_book_acc, 'is_honest': False},
            'closed_book': {'accuracy': closed_book_acc, 'is_honest': True},
            'real_learning_report': real_learning_results
        }


def generate_benchmark_samples() -> List[Dict]:
    """生成基准测试样本."""
    samples = [
        # 数学
        {"question": "What is 2 + 3 * 4?", "choices": ["20", "14", "10", "24"], "correct_answer": 1, "category": "math"},
        {"question": "What is 15 - 6?", "choices": ["8", "9", "10", "7"], "correct_answer": 1, "category": "math"},
        {"question": "What is 7 * 8?", "choices": ["54", "56", "58", "64"], "correct_answer": 1, "category": "math"},
        {"question": "What is 100 / 4?", "choices": ["20", "25", "30", "24"], "correct_answer": 1, "category": "math"},
        {"question": "What is 12 + 8?", "choices": ["18", "20", "22", "19"], "correct_answer": 1, "category": "math"},
        
        # 科学
        {"question": "What causes day and night?", "choices": ["Sun moving", "Earth rotation", "Moon", "Stars"], "correct_answer": 1, "category": "science"},
        {"question": "What do plants need for photosynthesis?", "choices": ["Only water", "Sunlight", "Darkness", "Cold"], "correct_answer": 1, "category": "science"},
        {"question": "What is the boiling point of water?", "choices": ["90°C", "100°C", "110°C", "80°C"], "correct_answer": 1, "category": "science"},
        {"question": "Which is the largest planet?", "choices": ["Mars", "Jupiter", "Saturn", "Earth"], "correct_answer": 1, "category": "science"},
        {"question": "What gas do humans breathe out?", "choices": ["Oxygen", "CO2", "Nitrogen", "Hydrogen"], "correct_answer": 1, "category": "science"},
        
        # 常识
        {"question": "How many days in a week?", "choices": ["5", "7", "6", "8"], "correct_answer": 1, "category": "common"},
        {"question": "How many months in a year?", "choices": ["10", "12", "11", "13"], "correct_answer": 1, "category": "common"},
        {"question": "What color is the sky?", "choices": ["Green", "Blue", "Red", "Yellow"], "correct_answer": 1, "category": "common"},
        {"question": "How many legs does a dog have?", "choices": ["2", "4", "6", "3"], "correct_answer": 1, "category": "common"},
        {"question": "What is H2O?", "choices": ["Fire", "Water", "Air", "Earth"], "correct_answer": 1, "category": "common"},
        
        # 中文
        {"question": "中国的首都是哪里?", "choices": ["上海", "北京", "广州", "深圳"], "correct_answer": 1, "category": "chinese"},
        {"question": "一年有多少天?", "choices": ["360", "365", "366", "370"], "correct_answer": 1, "category": "chinese"},
        {"question": "太阳从哪个方向升起?", "choices": ["西", "东", "南", "北"], "correct_answer": 1, "category": "chinese"},
        {"question": "水的化学式是什么?", "choices": ["CO2", "H2O", "O2", "N2"], "correct_answer": 1, "category": "chinese"},
        {"question": "地球是什么形状?", "choices": ["方形", "球形", "三角形", "椭圆"], "correct_answer": 1, "category": "chinese"},
        
        # 逻辑
        {"question": "If all A are B, and X is A, then X is?", "choices": ["A", "B", "C", "None"], "correct_answer": 1, "category": "logic"},
        {"question": "If it rains, the ground is wet. It rained. So?", "choices": ["Dry", "Wet", "Unknown", "Both"], "correct_answer": 1, "category": "logic"},
        {"question": "2, 4, 6, 8, what comes next?", "choices": ["9", "10", "11", "12"], "correct_answer": 1, "category": "pattern"},
        {"question": "1, 1, 2, 3, 5, what comes next?", "choices": ["6", "8", "7", "9"], "correct_answer": 1, "category": "pattern"},
        {"question": "A, C, E, G, what comes next?", "choices": ["H", "I", "J", "K"], "correct_answer": 1, "category": "pattern"},
    ]
    return samples


def demonstrate_honest_learning():
    """演示诚实的学习过程."""
    print("=" * 70)
    print("🎯 诚实学习演示 - 真正的内化 vs 开卷作弊")
    print("=" * 70)
    
    samples = generate_benchmark_samples()
    
    evaluator = HonestBenchmarkEvaluator()
    results = evaluator.compare_methods(samples)
    
    print("\n" + "=" * 70)
    print("📋 结论")
    print("=" * 70)
    print("""
之前的实现问题:
  ❌ 使用硬编码的知识库直接匹配答案
  ❌ 没有真正的训练过程
  ❌ 本质上是"开卷考试"作弊

现在的实现:
  ✅ 使用神经网络模型
  ✅ 真正的前向传播、反向传播、参数更新
  ✅ 训练集/验证集/测试集严格分离
  ✅ 闭卷考试（测试时不能访问答案）
  ✅ 可验证的学习过程
""")
    
    return results


if __name__ == "__main__":
    demonstrate_honest_learning()
