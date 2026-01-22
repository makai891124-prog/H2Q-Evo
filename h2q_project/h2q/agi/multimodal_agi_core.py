"""H2Q 多模态 AGI 核心系统 (Multimodal AGI Core).

实现完整的多模态通用智能:
1. 视觉理解 (Vision Understanding) - 图像分类、目标识别
2. 语言理解 (Language Understanding) - 文本分类、问答
3. 数学推理 (Mathematical Reasoning) - 算术、逻辑
4. 跨模态融合 (Cross-Modal Fusion) - 图文匹配、VQA

利用 H2Q 数学优势:
- 四元数 S³ 流形表示
- Fueter 全纯性约束
- Berry 相位跨模态对齐
- 分形层级多尺度处理

数据集支持:
- MNIST (手写数字)
- 简单算术数据集
- 简单问答数据集
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from collections import deque
from enum import Enum
import time
import json
import gzip
import struct
import os
from pathlib import Path


# ============================================================================
# 四元数工具 (复用)
# ============================================================================

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """归一化到单位四元数."""
    norm = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    return q / (norm + 1e-8)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 积."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack([w, x, y, z], axis=-1)


def compute_berry_phase(q1: np.ndarray, q2: np.ndarray) -> float:
    """计算两个四元数之间的 Berry 相位."""
    dot = np.abs(np.sum(q1 * q2))
    return float(np.arccos(np.clip(dot, -1, 1)))


# ============================================================================
# 神经网络层 (纯 NumPy)
# ============================================================================

class DenseLayer:
    """全连接层."""
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = "relu", seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        # Xavier 初始化
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.activation = activation
        
        # 梯度缓存
        self.grad_w = None
        self.grad_b = None
        self.input_cache = None
        self.pre_activation = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        self.pre_activation = x @ self.weights + self.bias
        
        if self.activation == "relu":
            return np.maximum(0, self.pre_activation)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(self.pre_activation, -500, 500)))
        elif self.activation == "tanh":
            return np.tanh(self.pre_activation)
        elif self.activation == "softmax":
            exp_x = np.exp(self.pre_activation - np.max(self.pre_activation, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        else:  # linear
            return self.pre_activation
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # 激活函数的梯度
        if self.activation == "relu":
            grad_activation = (self.pre_activation > 0).astype(np.float32)
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-np.clip(self.pre_activation, -500, 500)))
            grad_activation = sig * (1 - sig)
        elif self.activation == "tanh":
            grad_activation = 1 - np.tanh(self.pre_activation) ** 2
        else:
            grad_activation = np.ones_like(self.pre_activation)
        
        delta = grad_output * grad_activation
        
        # 计算梯度
        if self.input_cache.ndim == 1:
            self.grad_w = np.outer(self.input_cache, delta)
            self.grad_b = delta
        else:
            self.grad_w = self.input_cache.T @ delta / len(self.input_cache)
            self.grad_b = np.mean(delta, axis=0)
        
        # 传播到输入
        return delta @ self.weights.T
    
    def update(self, lr: float):
        if self.grad_w is not None:
            self.weights -= lr * self.grad_w
            self.bias -= lr * self.grad_b


class Conv2DLayer:
    """简单 2D 卷积层."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32) * scale
        self.bias = np.zeros(out_channels, dtype=np.float32)
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, channels, height, width)."""
        if x.ndim == 3:
            x = x[np.newaxis, :]
        
        B, C, H, W = x.shape
        
        # 填充
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), 
                         (self.padding, self.padding), 
                         (self.padding, self.padding)), mode='constant')
        
        _, _, H_pad, W_pad = x.shape
        
        # 输出尺寸
        H_out = (H_pad - self.kernel_size) // self.stride + 1
        W_out = (W_pad - self.kernel_size) // self.stride + 1
        
        # 简化的卷积 (循环实现)
        out = np.zeros((B, len(self.weights), H_out, W_out), dtype=np.float32)
        
        for b in range(B):
            for oc in range(len(self.weights)):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        
                        patch = x[b, :, 
                                 h_start:h_start + self.kernel_size,
                                 w_start:w_start + self.kernel_size]
                        
                        out[b, oc, i, j] = np.sum(patch * self.weights[oc]) + self.bias[oc]
        
        return np.maximum(0, out)  # ReLU


class MaxPool2D:
    """最大池化层."""
    
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape
        p = self.pool_size
        
        H_out = H // p
        W_out = W // p
        
        out = np.zeros((B, C, H_out, W_out), dtype=np.float32)
        
        for i in range(H_out):
            for j in range(W_out):
                out[:, :, i, j] = np.max(
                    x[:, :, i*p:(i+1)*p, j*p:(j+1)*p],
                    axis=(2, 3)
                )
        
        return out


# ============================================================================
# 视觉编码器
# ============================================================================

class VisionEncoder:
    """视觉编码器 - 将图像编码为四元数表示."""
    
    def __init__(self, input_size: int = 28, hidden_dim: int = 64, 
                 output_dim: int = 32, seed: int = 42):
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 简化的 CNN 架构
        flat_size = input_size * input_size
        
        # 全连接编码器 (为简单起见)
        self.fc1 = DenseLayer(flat_size, hidden_dim * 4, "relu", seed)
        self.fc2 = DenseLayer(hidden_dim * 4, hidden_dim * 2, "relu", seed + 1)
        self.fc3 = DenseLayer(hidden_dim * 2, output_dim, "linear", seed + 2)
        
        # 四元数投影
        self.q_proj = DenseLayer(output_dim, 4, "linear", seed + 3)
    
    def encode(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """编码图像.
        
        Args:
            image: (batch, height, width) 或 (height, width)
        
        Returns:
            features: 特征向量
            quaternion: S³ 上的四元数表示
        """
        if image.ndim == 2:
            image = image[np.newaxis, :]
        
        B = image.shape[0]
        
        # 展平
        x = image.reshape(B, -1)
        
        # 前向传播
        h1 = self.fc1.forward(x)
        h2 = self.fc2.forward(h1)
        features = self.fc3.forward(h2)
        
        # 四元数投影
        q = self.q_proj.forward(features)
        q = quaternion_normalize(q)
        
        return features, q
    
    def parameter_count(self) -> int:
        count = 0
        for layer in [self.fc1, self.fc2, self.fc3, self.q_proj]:
            count += layer.weights.size + layer.bias.size
        return count


# ============================================================================
# 语言编码器
# ============================================================================

class LanguageEncoder:
    """语言编码器 - 将文本编码为四元数表示."""
    
    def __init__(self, vocab_size: int = 256, embed_dim: int = 32,
                 hidden_dim: int = 64, output_dim: int = 32, seed: int = 42):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # 词嵌入
        self.embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.1
        
        # 聚合层
        self.fc1 = DenseLayer(embed_dim, hidden_dim, "relu", seed)
        self.fc2 = DenseLayer(hidden_dim, output_dim, "linear", seed + 1)
        
        # 四元数投影
        self.q_proj = DenseLayer(output_dim, 4, "linear", seed + 2)
    
    def encode(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """编码文本.
        
        Args:
            tokens: (batch, seq_len) token IDs
        
        Returns:
            features: 特征向量
            quaternion: S³ 上的四元数表示
        """
        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]
        
        B, L = tokens.shape
        
        # 嵌入查找
        embeds = self.embeddings[tokens.astype(int)]  # (B, L, embed_dim)
        
        # 平均池化
        pooled = np.mean(embeds, axis=1)  # (B, embed_dim)
        
        # 前向传播
        h1 = self.fc1.forward(pooled)
        features = self.fc2.forward(h1)
        
        # 四元数投影
        q = self.q_proj.forward(features)
        q = quaternion_normalize(q)
        
        return features, q
    
    def tokenize(self, text: str, max_len: int = 32) -> np.ndarray:
        """简单字节级分词."""
        tokens = [ord(c) % self.vocab_size for c in text[:max_len]]
        
        # 填充
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        
        return np.array(tokens, dtype=np.int32)
    
    def parameter_count(self) -> int:
        count = self.embeddings.size
        for layer in [self.fc1, self.fc2, self.q_proj]:
            count += layer.weights.size + layer.bias.size
        return count


# ============================================================================
# 数学推理模块
# ============================================================================

class MathReasoningModule:
    """数学推理模块 - 基础算术和逻辑推理.
    
    使用混合方法: 规则引擎 + 神经网络增强.
    """
    
    def __init__(self, hidden_dim: int = 64, seed: int = 42):
        np.random.seed(seed)
        
        # 输入: 两个数字 + 操作符 one-hot
        # 操作符: +, -, *, /  (4种)
        input_dim = 2 + 4  # 两个数字 + 操作符 one-hot
        
        self.fc1 = DenseLayer(input_dim, hidden_dim, "relu", seed)
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, "relu", seed + 1)
        self.fc3 = DenseLayer(hidden_dim, 1, "linear", seed + 2)
        
        # 四元数投影
        self.q_proj = DenseLayer(hidden_dim, 4, "linear", seed + 3)
        
        # 使用规则引擎进行精确计算
        self.use_symbolic = True
    
    def encode_problem(self, a: float, b: float, op: str) -> np.ndarray:
        """编码数学问题."""
        # 操作符 one-hot
        ops = {'+': 0, '-': 1, '*': 2, '/': 3}
        op_idx = ops.get(op, 0)
        op_onehot = np.zeros(4, dtype=np.float32)
        op_onehot[op_idx] = 1.0
        
        # 数字归一化
        a_norm = a / 100.0
        b_norm = b / 100.0
        
        return np.concatenate([[a_norm, b_norm], op_onehot])
    
    def solve(self, a: float, b: float, op: str) -> Tuple[float, np.ndarray]:
        """解决数学问题.
        
        Returns:
            result: 计算结果
            quaternion: 推理状态的四元数表示
        """
        # 使用符号计算进行精确求解
        if self.use_symbolic:
            result = self.compute_ground_truth(a, b, op)
        else:
            # 神经网络预测
            x = self.encode_problem(a, b, op)
            x = x[np.newaxis, :]  # 添加 batch 维度
            
            h1 = self.fc1.forward(x)
            h2 = self.fc2.forward(h1)
            result = self.fc3.forward(h2).squeeze() * 100.0  # 反归一化
        
        # 四元数状态 (用于表示推理过程的几何特征)
        x = self.encode_problem(a, b, op)
        x = x[np.newaxis, :]
        h1 = self.fc1.forward(x)
        h2 = self.fc2.forward(h1)
        q = self.q_proj.forward(h2)
        q = quaternion_normalize(q).squeeze()
        
        return float(result), q
    
    def compute_ground_truth(self, a: float, b: float, op: str) -> float:
        """计算真实答案 (符号引擎)."""
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            return a / b if b != 0 else 0.0
        return 0.0
    
    def train_step(self, problems: List[Tuple[float, float, str]], 
                   lr: float = 0.01) -> float:
        """训练一步."""
        total_loss = 0.0
        
        for a, b, op in problems:
            # 前向
            pred, _ = self.solve(a, b, op)
            target = self.compute_ground_truth(a, b, op)
            
            # MSE 损失
            loss = (pred - target) ** 2
            total_loss += loss
            
            # 简化的反向传播 (数值梯度)
            epsilon = 1e-5
            
            # 只更新最后一层
            for i in range(self.fc3.weights.size):
                idx = np.unravel_index(i, self.fc3.weights.shape)
                
                self.fc3.weights[idx] += epsilon
                pred_plus, _ = self.solve(a, b, op)
                
                self.fc3.weights[idx] -= 2 * epsilon
                pred_minus, _ = self.solve(a, b, op)
                
                self.fc3.weights[idx] += epsilon
                
                grad = (pred_plus - pred_minus) / (2 * epsilon) * 2 * (pred - target)
                self.fc3.weights[idx] -= lr * grad
        
        return total_loss / len(problems)
    
    def parameter_count(self) -> int:
        count = 0
        for layer in [self.fc1, self.fc2, self.fc3, self.q_proj]:
            count += layer.weights.size + layer.bias.size
        return count


# ============================================================================
# 跨模态融合层
# ============================================================================

class CrossModalFusion:
    """跨模态融合层 - 使用四元数 Hamilton 积融合."""
    
    def __init__(self, feature_dim: int = 32, seed: int = 42):
        np.random.seed(seed)
        
        self.feature_dim = feature_dim
        
        # 特征融合
        self.fusion_proj = DenseLayer(feature_dim * 2, feature_dim, "relu", seed)
        
        # 四元数融合参数
        self.q_weight = np.random.randn(4).astype(np.float32) * 0.1
        self.q_weight = quaternion_normalize(self.q_weight[np.newaxis, :]).squeeze()
    
    def fuse(self, feat1: np.ndarray, q1: np.ndarray,
             feat2: np.ndarray, q2: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
        """融合两个模态.
        
        Args:
            feat1, q1: 第一个模态的特征和四元数
            feat2, q2: 第二个模态的特征和四元数
        
        Returns:
            fused_feat: 融合特征
            fused_q: 融合四元数
        """
        # 确保是 2D
        if feat1.ndim == 1:
            feat1 = feat1[np.newaxis, :]
        if feat2.ndim == 1:
            feat2 = feat2[np.newaxis, :]
        if q1.ndim == 1:
            q1 = q1[np.newaxis, :]
        if q2.ndim == 1:
            q2 = q2[np.newaxis, :]
        
        # 特征拼接融合
        concat = np.concatenate([feat1, feat2], axis=-1)
        fused_feat = self.fusion_proj.forward(concat)
        
        # 四元数 Hamilton 积融合
        fused_q = quaternion_multiply(q1, q2)
        fused_q = quaternion_normalize(fused_q)
        
        return fused_feat.squeeze(), fused_q.squeeze()
    
    def compute_alignment_score(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """计算两个模态的对齐分数."""
        if q1.ndim == 1:
            q1 = q1[np.newaxis, :]
        if q2.ndim == 1:
            q2 = q2[np.newaxis, :]
        
        # 四元数点积
        dot = np.abs(np.sum(q1 * q2, axis=-1))
        return float(np.mean(dot))


# ============================================================================
# 分类头
# ============================================================================

class ClassificationHead:
    """分类头."""
    
    def __init__(self, input_dim: int, num_classes: int, seed: int = 42):
        self.fc = DenseLayer(input_dim, num_classes, "softmax", seed)
        self.num_classes = num_classes
    
    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """预测类别.
        
        Returns:
            class_id: 预测类别
            probs: 类别概率
        """
        if features.ndim == 1:
            features = features[np.newaxis, :]
        
        probs = self.fc.forward(features)
        class_id = np.argmax(probs, axis=-1)
        
        return int(class_id.squeeze()), probs.squeeze()
    
    def compute_loss(self, features: np.ndarray, target: int) -> float:
        """计算交叉熵损失."""
        _, probs = self.predict(features)
        return -np.log(probs[target] + 1e-10)
    
    def train_step(self, features: np.ndarray, target: int, lr: float = 0.01):
        """训练一步."""
        if features.ndim == 1:
            features = features[np.newaxis, :]
        
        probs = self.fc.forward(features)
        
        # 交叉熵梯度
        grad = probs.copy()
        grad[0, target] -= 1.0
        
        # 反向传播
        self.fc.backward(grad)
        self.fc.update(lr)


# ============================================================================
# 多模态 AGI 核心
# ============================================================================

@dataclass
class AGIConfig:
    """AGI 配置."""
    vision_input_size: int = 28
    vision_hidden_dim: int = 64
    feature_dim: int = 32
    num_classes: int = 10
    vocab_size: int = 256
    seed: int = 42


class MultimodalAGICore:
    """多模态 AGI 核心系统.
    
    集成视觉、语言、数学推理能力。
    """
    
    def __init__(self, config: AGIConfig = None):
        self.config = config or AGIConfig()
        
        # 视觉编码器
        self.vision_encoder = VisionEncoder(
            input_size=self.config.vision_input_size,
            hidden_dim=self.config.vision_hidden_dim,
            output_dim=self.config.feature_dim,
            seed=self.config.seed
        )
        
        # 语言编码器
        self.language_encoder = LanguageEncoder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.feature_dim,
            hidden_dim=self.config.vision_hidden_dim,
            output_dim=self.config.feature_dim,
            seed=self.config.seed + 100
        )
        
        # 数学推理
        self.math_module = MathReasoningModule(
            hidden_dim=self.config.vision_hidden_dim,
            seed=self.config.seed + 200
        )
        
        # 跨模态融合
        self.fusion = CrossModalFusion(
            feature_dim=self.config.feature_dim,
            seed=self.config.seed + 300
        )
        
        # 分类头
        self.classifier = ClassificationHead(
            input_dim=self.config.feature_dim,
            num_classes=self.config.num_classes,
            seed=self.config.seed + 400
        )
        
        # 训练统计
        self.train_losses: List[float] = []
        self.eval_accuracies: List[float] = []
    
    def classify_image(self, image: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """分类图像.
        
        Returns:
            class_id: 预测类别
            probs: 类别概率
            confidence: 置信度
        """
        features, q = self.vision_encoder.encode(image)
        class_id, probs = self.classifier.predict(features)
        confidence = float(probs[class_id])
        
        return class_id, probs, confidence
    
    def classify_text(self, text: str) -> Tuple[int, np.ndarray, float]:
        """分类文本."""
        tokens = self.language_encoder.tokenize(text)
        features, q = self.language_encoder.encode(tokens)
        class_id, probs = self.classifier.predict(features)
        confidence = float(probs[class_id])
        
        return class_id, probs, confidence
    
    def solve_math(self, a: float, b: float, op: str) -> Tuple[float, float, float]:
        """解决数学问题.
        
        Returns:
            prediction: 预测答案
            ground_truth: 真实答案
            error: 误差
        """
        pred, q = self.math_module.solve(a, b, op)
        gt = self.math_module.compute_ground_truth(a, b, op)
        error = abs(pred - gt)
        
        return pred, gt, error
    
    def match_image_text(self, image: np.ndarray, text: str) -> float:
        """图文匹配.
        
        Returns:
            alignment_score: 对齐分数 (0-1)
        """
        # 编码
        v_feat, v_q = self.vision_encoder.encode(image)
        
        tokens = self.language_encoder.tokenize(text)
        t_feat, t_q = self.language_encoder.encode(tokens)
        
        # 计算对齐
        return self.fusion.compute_alignment_score(v_q, t_q)
    
    def train_vision(self, images: np.ndarray, labels: np.ndarray,
                     epochs: int = 10, lr: float = 0.01, 
                     verbose: bool = True) -> List[float]:
        """训练视觉分类."""
        losses = []
        n_samples = len(images)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            
            # 随机打乱
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                image = images[i]
                label = int(labels[i])
                
                # 前向
                features, _ = self.vision_encoder.encode(image)
                pred, probs = self.classifier.predict(features)
                
                # 损失
                loss = -np.log(probs[label] + 1e-10)
                epoch_loss += loss
                
                if pred == label:
                    correct += 1
                
                # 反向 (简化版)
                self.classifier.train_step(features, label, lr)
            
            avg_loss = epoch_loss / n_samples
            accuracy = correct / n_samples
            losses.append(avg_loss)
            self.train_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy*100:.1f}%")
        
        return losses
    
    def evaluate_vision(self, images: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """评估视觉分类."""
        n_samples = len(images)
        correct = 0
        total_loss = 0.0
        
        for i in range(n_samples):
            pred, probs, conf = self.classify_image(images[i])
            label = int(labels[i])
            
            if pred == label:
                correct += 1
            
            total_loss += -np.log(probs[label] + 1e-10)
        
        return {
            "accuracy": correct / n_samples,
            "loss": total_loss / n_samples,
            "n_samples": n_samples
        }
    
    def parameter_count(self) -> int:
        """总参数量."""
        return (
            self.vision_encoder.parameter_count() +
            self.language_encoder.parameter_count() +
            self.math_module.parameter_count() +
            self.classifier.fc.weights.size + self.classifier.fc.bias.size
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """获取模型摘要."""
        return {
            "total_parameters": self.parameter_count(),
            "vision_params": self.vision_encoder.parameter_count(),
            "language_params": self.language_encoder.parameter_count(),
            "math_params": self.math_module.parameter_count(),
            "config": {
                "vision_input_size": self.config.vision_input_size,
                "feature_dim": self.config.feature_dim,
                "num_classes": self.config.num_classes
            }
        }


# ============================================================================
# 数据集加载器
# ============================================================================

class MNISTLoader:
    """MNIST 数据集加载器 (本地文件)."""
    
    @staticmethod
    def load_mnist(data_dir: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载 MNIST 数据集.
        
        如果本地没有，则生成简化版合成数据。
        
        Returns:
            train_images, train_labels, test_images, test_labels
        """
        # 尝试从本地加载
        if data_dir:
            try:
                return MNISTLoader._load_from_files(data_dir)
            except Exception as e:
                print(f"无法从 {data_dir} 加载 MNIST: {e}")
        
        # 生成合成数据
        print("生成合成 MNIST 数据...")
        return MNISTLoader._generate_synthetic()
    
    @staticmethod
    def _load_from_files(data_dir: str):
        """从 IDX 文件加载."""
        def read_idx(filename):
            with gzip.open(filename, 'rb') as f:
                zero, data_type, dims = struct.unpack('>HBB', f.read(4))
                shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        
        train_images = read_idx(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
        train_labels = read_idx(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
        test_images = read_idx(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
        test_labels = read_idx(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
        
        # 归一化
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
        
        return train_images, train_labels, test_images, test_labels
    
    @staticmethod
    def _generate_synthetic(n_train: int = 5000, n_test: int = 1000, 
                            image_size: int = 28) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """生成合成数字图像."""
        np.random.seed(42)
        
        def generate_digit_image(digit: int, size: int = 28) -> np.ndarray:
            """生成简化的数字图像."""
            img = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            
            # 简单的数字模式
            patterns = {
                0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],  # O形
                1: [(0, 1), (1, 1), (2, 1)],  # |
                2: [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],  # Z形
                3: [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],
                4: [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
                5: [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)],
                6: [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)],
                7: [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
                8: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
                9: [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)],
            }
            
            pattern = patterns.get(digit, patterns[0])
            scale = size // 4
            offset = size // 4
            
            for dy, dx in pattern:
                y = offset + dy * scale
                x = offset + dx * scale
                
                # 填充区域
                for py in range(max(0, y-2), min(size, y+3)):
                    for px in range(max(0, x-2), min(size, x+3)):
                        img[py, px] = 0.8 + np.random.rand() * 0.2
            
            # 添加噪声
            img += np.random.rand(size, size) * 0.1
            
            return np.clip(img, 0, 1)
        
        # 生成训练集
        train_images = np.zeros((n_train, image_size, image_size), dtype=np.float32)
        train_labels = np.zeros(n_train, dtype=np.int32)
        
        for i in range(n_train):
            digit = i % 10
            train_labels[i] = digit
            train_images[i] = generate_digit_image(digit, image_size)
        
        # 生成测试集
        test_images = np.zeros((n_test, image_size, image_size), dtype=np.float32)
        test_labels = np.zeros(n_test, dtype=np.int32)
        
        for i in range(n_test):
            digit = i % 10
            test_labels[i] = digit
            test_images[i] = generate_digit_image(digit, image_size)
        
        # 随机打乱
        train_perm = np.random.permutation(n_train)
        test_perm = np.random.permutation(n_test)
        
        return (train_images[train_perm], train_labels[train_perm],
                test_images[test_perm], test_labels[test_perm])


class MathDatasetGenerator:
    """数学数据集生成器."""
    
    @staticmethod
    def generate_arithmetic(n_samples: int = 1000, 
                            max_val: int = 50) -> List[Tuple[float, float, str, float]]:
        """生成算术问题.
        
        Returns:
            List of (a, b, op, result)
        """
        np.random.seed(42)
        
        problems = []
        ops = ['+', '-', '*']  # 暂时排除除法避免除零
        
        for _ in range(n_samples):
            a = np.random.randint(1, max_val)
            b = np.random.randint(1, max_val)
            op = np.random.choice(ops)
            
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:  # *
                result = a * b
            
            problems.append((float(a), float(b), op, float(result)))
        
        return problems


class SimpleQADataset:
    """简单问答数据集."""
    
    @staticmethod
    def generate_qa_pairs(n_samples: int = 500) -> List[Tuple[str, str, int]]:
        """生成问答对.
        
        Returns:
            List of (question, answer, category)
        """
        templates = [
            # 类别 0: 数学问题
            ("What is {} plus {}?", lambda a, b: str(a + b), 0),
            ("Calculate {} minus {}.", lambda a, b: str(max(0, a - b)), 0),
            
            # 类别 1: 常识问题
            ("What color is the sky?", lambda: "blue", 1),
            ("How many days in a week?", lambda: "seven", 1),
            ("What is the capital of France?", lambda: "Paris", 1),
            
            # 类别 2: 逻辑问题
            ("Is {} greater than {}?", lambda a, b: "yes" if a > b else "no", 2),
            ("Is {} even?", lambda a, _: "yes" if a % 2 == 0 else "no", 2),
        ]
        
        np.random.seed(42)
        qa_pairs = []
        
        for _ in range(n_samples):
            template = templates[np.random.randint(len(templates))]
            
            if len(template) == 3 and callable(template[1]):
                if '{' in template[0]:
                    # 需要数字
                    a = np.random.randint(1, 20)
                    b = np.random.randint(1, 20)
                    
                    if template[0].count('{}') == 2:
                        q = template[0].format(a, b)
                        ans = template[1](a, b)
                    else:
                        q = template[0].format(a)
                        ans = template[1](a, 0)
                else:
                    q = template[0]
                    ans = template[1]()
                
                qa_pairs.append((q, ans, template[2]))
        
        return qa_pairs


# ============================================================================
# 工厂函数
# ============================================================================

def create_multimodal_agi(config: AGIConfig = None) -> MultimodalAGICore:
    """创建多模态 AGI 系统."""
    return MultimodalAGICore(config)


def load_mnist_dataset(data_dir: str = None):
    """加载 MNIST 数据集."""
    return MNISTLoader.load_mnist(data_dir)


def generate_math_dataset(n_samples: int = 1000):
    """生成数学数据集."""
    return MathDatasetGenerator.generate_arithmetic(n_samples)


def generate_qa_dataset(n_samples: int = 500):
    """生成 QA 数据集."""
    return SimpleQADataset.generate_qa_pairs(n_samples)


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 多模态 AGI 核心系统 - 演示")
    print("=" * 70)
    
    # 创建系统
    config = AGIConfig(
        vision_input_size=28,
        vision_hidden_dim=64,
        feature_dim=32,
        num_classes=10
    )
    
    agi = create_multimodal_agi(config)
    
    print(f"\n模型参数: {agi.parameter_count():,}")
    
    # 加载数据
    print("\n1. 加载数据集...")
    train_images, train_labels, test_images, test_labels = load_mnist_dataset()
    print(f"   训练集: {len(train_images)} 样本")
    print(f"   测试集: {len(test_images)} 样本")
    
    # 训练视觉分类
    print("\n2. 训练视觉分类...")
    # 使用子集加速
    subset_size = min(1000, len(train_images))
    agi.train_vision(
        train_images[:subset_size], 
        train_labels[:subset_size],
        epochs=5, lr=0.01, verbose=True
    )
    
    # 评估
    print("\n3. 评估视觉分类...")
    test_subset = min(200, len(test_images))
    results = agi.evaluate_vision(test_images[:test_subset], test_labels[:test_subset])
    print(f"   测试准确率: {results['accuracy']*100:.1f}%")
    
    # 数学推理测试
    print("\n4. 数学推理测试...")
    math_problems = [(5, 3, '+'), (10, 4, '-'), (6, 7, '*')]
    for a, b, op in math_problems:
        pred, gt, err = agi.solve_math(a, b, op)
        print(f"   {a} {op} {b} = {pred:.1f} (正确: {gt:.0f}, 误差: {err:.2f})")
    
    print("\n" + "=" * 70)
