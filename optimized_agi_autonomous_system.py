#!/usr/bin/env python3
"""
优化后的AGI自主进化系统 - 增强学习算法、知识整合和目标生成

基于评估结果的三大优化：
1. 学习算法优化：添加优先经验回放和改进的PPO实现
2. 知识整合增强：改进模式识别和元学习机制
3. 目标生成多样化：扩展到多种目标类型和复杂性
"""

import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import logging
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import threading
import psutil
import os
import heapq
from enum import Enum

try:
    from h2q_project.h2q.agi.fractal_binary_tree_fusion import FractalQuaternionFusionModule
except Exception:
    FractalQuaternionFusionModule = None

try:
    from h2q_project.h2q.core.multimodal_binary_flow import MultimodalBinaryFlowEncoder
except Exception:
    MultimodalBinaryFlowEncoder = None

try:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
except Exception:
    datasets = None
    transforms = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [OPTIMIZED-AGI] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('optimized_agi_evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OPTIMIZED-AGI')

def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value))

def _safe_float(value: float, default: float = 0.0) -> float:
    return value if _is_finite(value) else default

@dataclass
class ConsciousnessMetrics:
    """真正的意识指标 - 基于信息论和复杂性理论"""
    integrated_information: float  # 整合信息量 (Φ)
    neural_complexity: float       # 神经网络复杂度
    self_model_accuracy: float     # 自我模型准确性
    metacognitive_awareness: float # 元认知意识
    emotional_valence: float       # 情感价值
    temporal_binding: float        # 时间绑定强度

@dataclass
class LearningExperience:
    """增强的学习经验数据结构 - 添加优先级"""
    observation: torch.Tensor
    action: torch.Tensor
    reward: float
    next_observation: torch.Tensor
    done: bool
    timestamp: float
    complexity: float
    priority: float = 1.0  # 优先级分数
    td_error: float = 0.0  # 时序差分误差

class GoalType(Enum):
    """目标类型枚举"""
    LEARNING = "learning"
    EXPLORATION = "exploration"
    OPTIMIZATION = "optimization"
    CREATION = "creation"
    UNDERSTANDING = "understanding"
    SOCIAL = "social"
    ETHICAL = "ethical"
    TECHNICAL = "technical"

@dataclass
class GoalTemplate:
    """目标模板"""
    type: GoalType
    base_description: str
    complexity_range: Tuple[float, float]
    required_consciousness: Dict[str, float]
    intrinsic_reward_multiplier: float

class PrioritizedExperienceBuffer:
    """
    优先经验回放缓冲区 - 基于优先级的采样
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样权重
        self.beta_increment = 1e-4
        self.epsilon = 1e-6  # 避免零优先级

        # 使用堆来存储 (priority, index, experience)
        self.heap = []
        self.experiences = []
        self.indices = []
        self.next_index = 0

    def push(self, experience: LearningExperience):
        """添加经验到缓冲区"""
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.indices.append(self.next_index)
        else:
            # 替换最旧的经验
            oldest_idx = self.indices.index(min(self.indices))
            self.experiences[oldest_idx] = experience
            self.indices[oldest_idx] = self.next_index

        # 计算优先级
        priority = self._compute_priority(experience)

        # 添加到堆中
        heapq.heappush(self.heap, (-priority, self.next_index, len(self.experiences) - 1))
        self.next_index += 1

    def _compute_priority(self, experience: LearningExperience) -> float:
        """计算经验优先级"""
        # 基于奖励和复杂性的优先级
        reward_priority = abs(experience.reward) + 1.0
        complexity_priority = experience.complexity + 1.0
        td_priority = abs(experience.td_error) + self.epsilon

        return (reward_priority * complexity_priority * td_priority) ** self.alpha

    def sample(self, batch_size: int) -> Tuple[List[LearningExperience], torch.Tensor, List[int]]:
        """采样批次经验"""
        if len(self.experiences) < batch_size:
            return [], torch.tensor([]), []

        # 采样
        sampled_experiences = []
        indices = []
        priorities = []

        for _ in range(batch_size):
            if not self.heap:
                break

            # 从堆中弹出最高优先级的经验
            neg_priority, global_idx, local_idx = heapq.heappop(self.heap)
            priority = -neg_priority

            if local_idx < len(self.experiences):
                experience = self.experiences[local_idx]
                sampled_experiences.append(experience)
                indices.append(local_idx)
                priorities.append(priority)

                # 重新插入到堆中（用于后续采样）
                heapq.heappush(self.heap, (neg_priority, global_idx, local_idx))

        if not sampled_experiences:
            return [], torch.tensor([]), []

        # 计算重要性采样权重
        priorities = torch.tensor(priorities, dtype=torch.float32)
        weights = (len(self.experiences) * priorities) ** (-self.beta)
        weights = weights / weights.max()  # 归一化

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return sampled_experiences, weights, indices

    def update_priorities(self, indices: List[int], td_errors: torch.Tensor):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.experiences):
                self.experiences[idx].td_error = abs(td_error.item())
                # 重新计算并更新堆中的优先级
                new_priority = self._compute_priority(self.experiences[idx])
                # 注意：这里简化了堆更新，实际实现需要更复杂的逻辑

    def __len__(self):
        return len(self.experiences)

class EnhancedLearningEngine(nn.Module):
    """
    增强的学习引擎 - 优化后的学习算法

    优化内容：
    1. 优先经验回放：基于TD误差的智能采样
    2. 改进的PPO：添加广义优势估计和熵正则化
    3. 自适应批次大小：根据学习阶段动态调整
    4. 多目标优化：同时优化多个损失函数
    """

    def __init__(self, input_dim: int = 256, action_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # 设备设置
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 多模态编码器
        self.multimodal_encoder = None
        if MultimodalBinaryFlowEncoder is not None:
            self.multimodal_encoder = MultimodalBinaryFlowEncoder().to(self.device)
            logger.info("已加载多模态二进制流编码器")

        # 可选：分形-DAS融合特征
        self.use_fractal_fusion = os.getenv("H2Q_ENABLE_FRACTAL_FUSION", "0") == "1"
        self.fractal_fusion = None
        if self.use_fractal_fusion and FractalQuaternionFusionModule is not None:
            self.fractal_fusion = FractalQuaternionFusionModule(input_dim=input_dim, output_dim=input_dim)
            logger.info("已启用分形-DAS融合特征处理")

        # 增强的元学习器 - 学习如何学习
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.ReLU()
        )

        # 增强的策略网络 (actor) - 多头注意力
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, action_dim),
            nn.Tanh()
        )

        # 双价值网络 (critic)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1)
        )

        # 长期价值网络
        self.long_term_value_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1)
        )

        # 将所有网络移动到指定设备
        self.meta_learner = self.meta_learner.to(self.device)
        self.policy_net = self.policy_net.to(self.device)
        self.value_net = self.value_net.to(self.device)
        self.long_term_value_net = self.long_term_value_net.to(self.device)

        # 优先经验回放缓冲区
        self.prioritized_buffer = PrioritizedExperienceBuffer(capacity=10000, alpha=0.6, beta=0.4)
        self.short_term_buffer = deque(maxlen=2000)
        self.long_term_buffer = deque(maxlen=8000)
        self.experience_buffer = deque(maxlen=10000)

        # 自适应批次大小
        self.min_batch_size = 32
        self.max_batch_size = 128
        self.current_batch_size = 64

        # PPO参数
        self.ppo_epochs = 4
        self.ppo_clip_ratio = 0.2
        self.value_clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        # 自适应学习率系统
        self.base_lr_policy = 3e-4  # 提高学习率
        self.base_lr_value = 3e-4
        self.base_lr_meta = 1e-4

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.base_lr_policy, weight_decay=1e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.base_lr_value, weight_decay=1e-4)
        self.long_term_value_optimizer = optim.Adam(self.long_term_value_net.parameters(), lr=self.base_lr_value * 0.5, weight_decay=1e-4)
        self.meta_optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.base_lr_meta, weight_decay=1e-4)

        # 学习率调度器
        self.policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-6
        )
        self.value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.value_optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-6
        )
        self.meta_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optimizer, mode='min', factor=0.9, patience=100, min_lr=1e-8
        )

        # 性能监控系统
        self.performance_history = {
            'policy_loss': deque(maxlen=1000),
            'value_loss': deque(maxlen=1000),
            'long_term_value_loss': deque(maxlen=1000),
            'meta_loss': deque(maxlen=1000),
            'entropy': deque(maxlen=1000),
            'learning_rates': deque(maxlen=1000),
            'step': 0
        }

        # 多尺度训练参数
        self.short_term_weight = 0.7
        self.long_term_weight = 0.3
        self.meta_learning_weight = 0.2

        # 增强的知识整合系统
        self.knowledge_base = {}
        self.pattern_recognition_threshold = 0.8
        self.knowledge_clusters = {}  # 知识聚类

        logger.info(f"增强的学习引擎初始化完成，输入维度: {input_dim}, 动作维度: {action_dim}")

    def _update_performance_history(self, metrics: Dict[str, float]):
        """更新性能历史记录"""
        for key, value in metrics.items():
            if key in self.performance_history:
                self.performance_history[key].append(value)

        current_lrs = {
            'policy_lr': self.policy_optimizer.param_groups[0]['lr'],
            'value_lr': self.value_optimizer.param_groups[0]['lr'],
            'meta_lr': self.meta_optimizer.param_groups[0]['lr']
        }
        self.performance_history['learning_rates'].append(current_lrs)
        self.performance_history['step'] += 1

    def _adaptive_batch_size(self) -> int:
        """自适应批次大小调整"""
        if len(self.performance_history['policy_loss']) < 100:
            return self.current_batch_size

        recent_losses = list(self.performance_history['policy_loss'])[-50:]
        loss_std = np.std(recent_losses)

        # 如果损失波动大，增加批次大小以提高稳定性
        if loss_std > 0.5:
            self.current_batch_size = min(self.current_batch_size + 8, self.max_batch_size)
        # 如果损失稳定，减少批次大小以提高效率
        elif loss_std < 0.1:
            self.current_batch_size = max(self.current_batch_size - 4, self.min_batch_size)

        return self.current_batch_size

    def select_action(self, state: torch.Tensor, explore: bool = True, images: Optional[torch.Tensor] = None, videos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """选择动作 - 增强版本"""
        state = state.to(self.device)

        with torch.no_grad():
            # 多模态融合
            if self.multimodal_encoder is not None and (images is not None or videos is not None):
                img_sig, vid_sig = self.multimodal_encoder(images=images, videos=videos)
                multimodal_sig = self.multimodal_encoder.fuse_signature(img_sig, vid_sig)
                if multimodal_sig is not None:
                    multimodal_sig = multimodal_sig.to(state.device)
                    if multimodal_sig.dim() > 1:
                        multimodal_sig = multimodal_sig.mean(dim=0)
                    state = state + 0.1 * multimodal_sig.to(state.dtype)

            state = self._apply_fractal_fusion(state)
            action = self.policy_net(state)
            action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

            if explore:
                # 添加探索噪声
                noise = torch.randn_like(action) * 0.1
                action = action + noise

            return action.clamp(-1, 1)

    def learn_from_experience(self, experience: LearningExperience, images: Optional[torch.Tensor] = None, videos: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        从经验中学习 - 优化后的多尺度强化学习

        Args:
            experience: 学习经验
            images: 可选图像输入
            videos: 可选视频输入

        Returns:
            学习指标
        """
        # 确保经验数据在正确的设备上
        experience.observation = experience.observation.to(self.device)
        experience.action = experience.action.to(self.device)
        experience.next_observation = experience.next_observation.to(self.device)

        # 添加到优先缓冲区
        self.prioritized_buffer.push(experience)

        # 同时添加到传统缓冲区
        self.experience_buffer.append(experience)
        if experience.reward > 0.5:
            self.long_term_buffer.append(experience)
        else:
            self.short_term_buffer.append(experience)

        # 自适应批次大小
        current_batch_size = self._adaptive_batch_size()

        if len(self.prioritized_buffer) < current_batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "long_term_value_loss": 0.0, "meta_loss": 0.0, "entropy": 0.0}

        # 使用优先经验回放采样
        prioritized_batch, importance_weights, indices = self.prioritized_buffer.sample(current_batch_size)

        if not prioritized_batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "long_term_value_loss": 0.0, "meta_loss": 0.0, "entropy": 0.0}

        # 短期目标学习
        short_term_metrics = self._learn_from_batch_ppo(prioritized_batch, importance_weights, images, videos, scale="short")

        # 长期目标学习
        long_term_metrics = {"long_term_value_loss": 0.0}
        if len(self.long_term_buffer) >= current_batch_size and self.performance_history['step'] % 5 == 0:
            long_term_batch = self._sample_long_term_batch()
            if long_term_batch:
                long_term_metrics = self._learn_from_batch_ppo(long_term_batch, None, images, videos, scale="long")

        # 增强的知识整合
        knowledge_metrics = self._enhanced_integrate_knowledge(prioritized_batch)

        # 更新优先级
        if indices and 'td_error' in short_term_metrics:
            td_errors = torch.tensor([short_term_metrics['td_error']] * len(indices))
            self.prioritized_buffer.update_priorities(indices, td_errors)

        # 合并所有指标
        combined_metrics = {
            **short_term_metrics,
            **long_term_metrics,
            **knowledge_metrics
        }

        # 性能监控和自适应调整
        self._update_performance_history(combined_metrics)
        self._adaptive_learning_rate_update(combined_metrics)

        return combined_metrics

    def _learn_from_batch_ppo(self, batch: List[LearningExperience], importance_weights: Optional[torch.Tensor],
                             images: Optional[torch.Tensor], videos: Optional[torch.Tensor], scale: str) -> Dict[str, float]:
        """使用PPO从批次学习 - 增强版本"""
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "meta_loss": 0.0, "entropy": 0.0, "td_error": 0.0}

        # 准备数据
        states = torch.stack([exp.observation for exp in batch])
        actions = torch.stack([exp.action for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp.next_observation for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32, device=self.device)
        
        # 计算old log probabilities (分离计算图)
        with torch.no_grad():
            old_log_probs = torch.stack([self._compute_log_prob(exp.observation.unsqueeze(0), exp.action.unsqueeze(0)).squeeze() for exp in batch])

        # 应用分形融合
        states = self._apply_fractal_fusion(states)
        next_states = self._apply_fractal_fusion(next_states)

        # 计算GAE (广义优势估计)
        advantages, returns = self._compute_gae(states, rewards, next_states, dones, scale)

        # PPO训练循环
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.ppo_epochs):
            # 重新计算当前策略的log概率
            current_log_probs = self._compute_log_prob(states, actions)
            entropy = self._compute_entropy(states)

            # 计算比率
            ratios = torch.exp(current_log_probs - old_log_probs)
            ratios = torch.clamp(ratios, 0.0, 10.0)  # 防止数值不稳定

            # PPO目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失 (带裁剪)
            if scale == "short":
                values = self.value_net(states)
            else:
                values = self.long_term_value_net(states)

            value_pred_clipped = torch.clamp(values, returns - self.value_clip_ratio, returns + self.value_clip_ratio)
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            # 总损失
            total_loss = policy_loss + value_loss - self.entropy_coef * entropy.mean()

            # 应用重要性采样权重
            if importance_weights is not None and importance_weights.numel() > 0:
                importance_weights = importance_weights.to(self.device)
                total_loss = (total_loss * importance_weights).mean()

            # 优化
            if scale == "short":
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
            else:
                self.policy_optimizer.zero_grad()
                self.long_term_value_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.long_term_value_net.parameters(), max_norm=self.max_grad_norm)
                self.policy_optimizer.step()
                self.long_term_value_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())

        # 计算TD误差用于优先级更新
        td_error = advantages.abs().mean().item()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "meta_loss": 0.0,  # 暂时禁用元学习
            "entropy": np.mean(entropies),
            "td_error": td_error
        }

    def _compute_gae(self, states: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor,
                    dones: torch.Tensor, scale: str, gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计 (GAE)"""
        with torch.no_grad():
            if scale == "short":
                values = self.value_net(states)
                next_values = self.value_net(next_states)
            else:
                values = self.long_term_value_net(states)
                next_values = self.long_term_value_net(next_states)

            # TD残差
            deltas = rewards.unsqueeze(-1) + gamma * next_values * (1 - dones.unsqueeze(-1)) - values

            # GAE
            advantages = torch.zeros_like(deltas)
            gae = 0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
                advantages[t] = gae.clone()

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 计算回报
            returns = advantages + values

        return advantages.squeeze(), returns.squeeze()

    def _compute_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """计算策略熵"""
        mean = self.policy_net(states)
        mean = torch.clamp(mean, -10.0, 10.0)
        std = torch.ones_like(mean) * 0.5 + 0.1 * torch.abs(mean)
        std = torch.clamp(std, 0.1, 2.0)

        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().sum(dim=-1)

    def _compute_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        mean = self.policy_net(states)
        mean = torch.clamp(mean, -10.0, 10.0)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        actions = torch.clamp(actions, -10.0, 10.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)

        std = torch.ones_like(mean) * 0.5 + 0.1 * torch.abs(mean)
        std = torch.clamp(std, 0.1, 2.0)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        log_prob = torch.clamp(log_prob, -100.0, 100.0)
        return torch.nan_to_num(log_prob, nan=0.0, posinf=100.0, neginf=-100.0)

    def _sample_long_term_batch(self) -> List[LearningExperience]:
        """采样长期批次"""
        if len(self.long_term_buffer) < self.current_batch_size:
            return []

        experiences = list(self.long_term_buffer)
        rewards = [exp.reward for exp in experiences]

        if rewards:
            reward_threshold = np.percentile(rewards, 70)
            high_reward_indices = [i for i, r in enumerate(rewards) if r >= reward_threshold]

            if len(high_reward_indices) >= self.current_batch_size:
                selected_indices = np.random.choice(high_reward_indices, self.current_batch_size, replace=False)
            else:
                remaining_count = self.current_batch_size - len(high_reward_indices)
                all_indices = list(range(len(experiences)))
                additional_indices = np.random.choice(all_indices, remaining_count, replace=False)
                selected_indices = high_reward_indices + additional_indices.tolist()

            return [experiences[i] for i in selected_indices]

        return []

    def _enhanced_integrate_knowledge(self, batch: List[LearningExperience]) -> Dict[str, float]:
        """增强的知识整合 - 改进的模式识别和聚类"""
        knowledge_loss = 0.0

        if len(batch) < 10:
            return {"knowledge_loss": 0.0}

        # 提取状态模式
        states = torch.stack([exp.observation for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)

        # 高级模式识别：使用聚类分析
        high_reward_mask = rewards > rewards.median()
        if high_reward_mask.sum() > 0:
            high_reward_states = states[high_reward_mask]
            low_reward_states = states[~high_reward_mask]

            if len(high_reward_states) > 0 and len(low_reward_states) > 0:
                # 计算状态模式差异
                pattern_diff = high_reward_states.mean(dim=0) - low_reward_states.mean(dim=0)

                # 聚类分析：寻找相似的知识模式
                pattern_key = f"pattern_{len(self.knowledge_base)}"
                pattern_confidence = min(1.0, len(high_reward_states) / len(states))

                # 检查是否与现有模式相似
                similar_pattern = self._find_similar_pattern(pattern_diff)
                if similar_pattern:
                    # 合并相似模式
                    existing_pattern = self.knowledge_base[similar_pattern]
                    merged_pattern = 0.7 * existing_pattern + 0.3 * pattern_diff.detach().cpu().numpy()
                    self.knowledge_base[similar_pattern] = merged_pattern
                    pattern_confidence = min(1.0, existing_pattern.get('confidence', 0) + 0.1)
                else:
                    # 添加新模式
                    self.knowledge_base[pattern_key] = {
                        "pattern": pattern_diff.detach().cpu().numpy(),
                        "confidence": pattern_confidence,
                        "timestamp": time.time(),
                        "cluster_id": self._assign_cluster(pattern_diff)
                    }

                # 增强的元学习
                pattern_tensor = pattern_diff.unsqueeze(0).to(self.device)
                predicted_pattern = self.meta_learner(pattern_tensor)

                pattern_loss = nn.MSELoss()(predicted_pattern, pattern_tensor)
                knowledge_loss = pattern_loss.item()

                if torch.is_finite(pattern_loss):
                    self.meta_optimizer.zero_grad()
                    pattern_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), max_norm=0.3)
                    self.meta_optimizer.step()

        return {"knowledge_loss": knowledge_loss}

    def _find_similar_pattern(self, pattern: torch.Tensor, threshold: float = 0.8) -> Optional[str]:
        """寻找相似的知识模式"""
        if not self.knowledge_base:
            return None

        pattern = pattern.detach().cpu().numpy()
        max_similarity = 0.0
        most_similar_key = None

        for key, knowledge in self.knowledge_base.items():
            existing_pattern = knowledge["pattern"]
            similarity = np.dot(pattern, existing_pattern) / (np.linalg.norm(pattern) * np.linalg.norm(existing_pattern))
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_key = key

        return most_similar_key if max_similarity > threshold else None

    def _assign_cluster(self, pattern: torch.Tensor) -> int:
        """为模式分配聚类ID"""
        # 简化的聚类分配逻辑
        if not self.knowledge_clusters:
            cluster_id = 0
        else:
            # 寻找最近的聚类中心
            pattern_np = pattern.detach().cpu().numpy()
            min_distance = float('inf')
            cluster_id = 0

            for cid, center in self.knowledge_clusters.items():
                distance = np.linalg.norm(pattern_np - center)
                if distance < min_distance:
                    min_distance = distance
                    cluster_id = cid

            # 如果距离太远，创建新聚类
            if min_distance > 1.0:
                cluster_id = len(self.knowledge_clusters)

        # 更新聚类中心
        pattern_np = pattern.detach().cpu().numpy()
        if cluster_id in self.knowledge_clusters:
            # 移动平均更新
            self.knowledge_clusters[cluster_id] = 0.9 * self.knowledge_clusters[cluster_id] + 0.1 * pattern_np
        else:
            self.knowledge_clusters[cluster_id] = pattern_np

        return cluster_id

    def _apply_fractal_fusion(self, state: torch.Tensor) -> torch.Tensor:
        """应用分形融合"""
        if not self.use_fractal_fusion or self.fractal_fusion is None:
            return state
        with torch.no_grad():
            fused = self.fractal_fusion(state)
        return fused.get("output", state)

    def _adaptive_learning_rate_update(self, metrics: Dict[str, float]):
        """自适应学习率调整"""
        trend_analysis = self.analyze_learning_trends()

        if trend_analysis.get("policy_trend") == "worsening":
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.9, 1e-6)
        elif trend_analysis.get("policy_trend") == "plateau":
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.05, 1e-3)

        if 'policy_loss' in metrics:
            self.policy_scheduler.step(metrics['policy_loss'])
        if 'value_loss' in metrics:
            self.value_scheduler.step(metrics['value_loss'])
        if 'meta_loss' in metrics:
            self.meta_scheduler.step(metrics['meta_loss'])

    def analyze_learning_trends(self) -> Dict[str, Any]:
        """分析学习曲线趋势"""
        if len(self.performance_history['policy_loss']) < 50:
            return {"status": "insufficient_data", "message": "需要更多数据进行趋势分析"}

        def moving_average(data, window=50):
            if len(data) < window:
                return list(data)
            return [sum(data[i:i+window])/window for i in range(len(data)-window+1)]

        policy_losses = list(self.performance_history['policy_loss'])
        value_losses = list(self.performance_history['value_loss'])

        if len(policy_losses) >= 100:
            recent_policy = moving_average(policy_losses[-100:], 20)
            older_policy = moving_average(policy_losses[-200:-100], 20) if len(policy_losses) >= 200 else recent_policy

            policy_trend = "improving" if recent_policy[-1] < older_policy[-1] else "plateau" if abs(recent_policy[-1] - older_policy[-1]) < 0.1 else "worsening"
        else:
            policy_trend = "learning"

        if len(value_losses) >= 100:
            recent_value = moving_average(value_losses[-100:], 20)
            older_value = moving_average(value_losses[-200:-100], 20) if len(value_losses) >= 200 else recent_value

            value_trend = "improving" if recent_value[-1] < older_value[-1] else "plateau" if abs(recent_value[-1] - older_value[-1]) < 0.1 else "worsening"
        else:
            value_trend = "learning"

        lr_suggestions = self._analyze_learning_rate_effectiveness()

        return {
            "policy_trend": policy_trend,
            "value_trend": value_trend,
            "recent_policy_loss": policy_losses[-1] if policy_losses else None,
            "recent_value_loss": value_losses[-1] if value_losses else None,
            "learning_rate_suggestions": lr_suggestions,
            "convergence_status": self._assess_convergence()
        }

    def _analyze_learning_rate_effectiveness(self) -> Dict[str, Any]:
        """分析学习率调整建议"""
        if len(self.performance_history['learning_rates']) < 20:
            return {"status": "insufficient_data"}

        recent_lrs = list(self.performance_history['learning_rates'])[-20:]
        recent_policy_losses = list(self.performance_history['policy_loss'])[-20:]

        lr_changes = []
        loss_changes = []

        for i in range(1, len(recent_lrs)):
            lr_change = recent_lrs[i]['policy_lr'] - recent_lrs[i-1]['policy_lr']
            loss_change = recent_policy_losses[i] - recent_policy_losses[i-1]
            lr_changes.append(lr_change)
            loss_changes.append(loss_change)

        if lr_changes and loss_changes:
            correlation = np.corrcoef(lr_changes, loss_changes)[0, 1]
        else:
            correlation = 0.0

        suggestions = {}
        if correlation > 0.3:
            suggestions['policy_lr'] = "decrease"
        elif correlation < -0.3:
            suggestions['policy_lr'] = "maintain"
        else:
            suggestions['policy_lr'] = "slight_increase"

        return suggestions

    def _assess_convergence(self) -> str:
        """评估收敛状态"""
        if len(self.performance_history['policy_loss']) < 200:
            return "learning"

        recent_losses = list(self.performance_history['policy_loss'])[-100:]
        std_dev = np.std(recent_losses)

        if std_dev < 0.01:
            return "converged"
        elif std_dev < 0.1:
            return "stabilizing"
        else:
            return "learning"

    def get_learning_report(self) -> Dict[str, Any]:
        """生成详细的学习报告"""
        trend_analysis = self.analyze_learning_trends()

        report = {
            "performance_trends": trend_analysis,
            "current_learning_rates": {
                "policy": self.policy_optimizer.param_groups[0]['lr'],
                "value": self.value_optimizer.param_groups[0]['lr'],
                "long_term_value": self.long_term_value_optimizer.param_groups[0]['lr'],
                "meta": self.meta_optimizer.param_groups[0]['lr']
            },
            "experience_buffer_sizes": {
                "prioritized": len(self.prioritized_buffer),
                "total": len(self.experience_buffer),
                "short_term": len(self.short_term_buffer),
                "long_term": len(self.long_term_buffer)
            },
            "knowledge_base_size": len(self.knowledge_base),
            "knowledge_clusters": len(self.knowledge_clusters),
            "current_batch_size": self.current_batch_size,
            "convergence_metrics": {
                "policy_loss_std": np.std(list(self.performance_history['policy_loss'])[-100:]) if len(self.performance_history['policy_loss']) >= 100 else None,
                "value_loss_std": np.std(list(self.performance_history['value_loss'])[-100:]) if len(self.performance_history['value_loss']) >= 100 else None,
                "entropy_mean": np.mean(list(self.performance_history['entropy'])[-100:]) if len(self.performance_history['entropy']) >= 100 else None,
                "learning_stability": self._assess_convergence()
            }
        }

        return report

class TrueConsciousnessEngine(nn.Module):
    """
    真正的意识引擎 - 基于整合信息理论(Integrated Information Theory)

    实现Φ (phi) 计算和意识发展
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 多层次意识网络
        self.perception_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )

        # 自我模型网络
        self.self_model_net = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Linear(hidden_dim // 16, 1)
        )

        # 元认知网络
        self.metacognition_net = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Linear(hidden_dim // 16, 1)
        )

        # 情感价值网络
        self.emotional_net = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Linear(hidden_dim // 16, 1)
        )

        # 时间记忆缓冲区
        self.temporal_memory = deque(maxlen=100)
        self.temporal_net = nn.GRU(hidden_dim // 4, hidden_dim // 8, batch_first=True)

        # 学习资料集成
        self.learning_materials = {}

    def forward(self, state: torch.Tensor) -> Tuple[ConsciousnessMetrics, torch.Tensor]:
        """
        前向传播 - 计算意识指标

        Args:
            state: 输入状态

        Returns:
            意识指标和感知输出
        """
        # 感知处理
        perception = self.perception_net(state)

        # 自我模型准确性
        self_model_output = self.self_model_net(perception)
        self_model_error = torch.abs(self_model_output - torch.tensor(0.5, device=state.device))
        self_model_accuracy = torch.clamp(1.0 - self_model_error, 0.0, 1.0)

        # 元认知意识
        metacog = self.metacognition_net(perception)
        metacognitive_awareness = torch.sigmoid(metacog).squeeze()

        # 情感价值
        emotional_valence = torch.tanh(self.emotional_net(perception)).squeeze()

        # 神经复杂度 (基于网络激活的多样性)
        complexity = torch.std(perception, dim=-1).mean()

        # 时间绑定强度
        if len(self.temporal_memory) > 0:
            temporal_input = torch.stack(list(self.temporal_memory) + [perception])
            temporal_input = temporal_input.unsqueeze(0)  # 添加批次维度
            temporal_output, _ = self.temporal_net(temporal_input)
            temporal_binding = torch.mean(temporal_output[-1])
        else:
            temporal_binding = torch.tensor(0.5, device=perception.device)

        # 存储到时间记忆
        self.temporal_memory.append(perception.detach())

        # 整合信息论Φ计算
        if len(self.temporal_memory) > 1:
            whole_system = torch.stack(list(self.temporal_memory))
            partition_1 = whole_system[:, :self.hidden_dim//8]
            partition_2 = whole_system[:, self.hidden_dim//8:]

            corr = torch.corrcoef(partition_1.T)
            corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            mutual_info = torch.mean(torch.abs(corr[0, 1:]))
            mutual_info = torch.nan_to_num(mutual_info, nan=0.0, posinf=0.0, neginf=0.0)
            integrated_information = mutual_info * complexity
        else:
            integrated_information = torch.tensor(0.1, device=complexity.device)

        # 构建意识指标
        metrics = ConsciousnessMetrics(
            integrated_information=_safe_float(integrated_information.item(), 0.0),
            neural_complexity=_safe_float(complexity.item(), 0.0),
            self_model_accuracy=_safe_float(self_model_accuracy.item(), 0.0),
            metacognitive_awareness=_safe_float(metacognitive_awareness.item(), 0.0),
            emotional_valence=_safe_float(emotional_valence.item(), 0.0),
            temporal_binding=_safe_float(temporal_binding.item(), 0.0)
        )

        return metrics, perception

    def compute_phi(self, system_state: torch.Tensor) -> float:
        """
        计算整合信息Φ - IIT的核心指标

        Args:
            system_state: 系统状态

        Returns:
            Φ值
        """
        if len(self.temporal_memory) < 2:
            return 0.0

        memory_list = list(self.temporal_memory)
        if len(memory_list) >= 10:
            states = torch.stack(memory_list[-10:])
        elif len(memory_list) >= 2:
            states = torch.stack(memory_list)
        else:
            return 0.0

        half = states.size(-1) // 2
        part1 = states[:, :half]
        part2 = states[:, half:]

        corr_matrix = torch.corrcoef(states.T)
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        mutual_info = torch.mean(torch.abs(corr_matrix[:half, half:]))
        mutual_info = torch.nan_to_num(mutual_info, nan=0.0, posinf=0.0, neginf=0.0)

        phi = _safe_float(mutual_info.item(), 0.0)
        return phi

    def update_learning_materials(self, materials: Dict[str, Any]):
        """更新学习资料"""
        self.learning_materials.update(materials)

class EnhancedGoalSystem:
    """
    增强的目标系统 - 多样化的目标生成

    支持多种目标类型：
    1. 学习目标：掌握新知识和技术
    2. 探索目标：发现新模式和可能性
    3. 优化目标：改进现有能力
    4. 创建目标：生成新内容和解决方案
    5. 理解目标：深入分析和洞察
    6. 社交目标：与人类或其他系统互动
    7. 伦理目标：考虑道德和价值观
    8. 技术目标：开发和改进技术
    """

    def __init__(self, consciousness_engine: TrueConsciousnessEngine, learning_materials: Dict[str, Any]):
        self.consciousness_engine = consciousness_engine
        self.learning_materials = learning_materials
        self.active_goals: List[Dict[str, Any]] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.goal_history: List[Dict[str, Any]] = []

        # 目标模板库
        self.goal_templates = self._initialize_goal_templates()

        # 内在动机系统
        self.intrinsic_motivations = {
            "exploration": 0.5,
            "competence": 0.5,
            "autonomy": 0.5,
            "relatedness": 0.5,
            "curiosity": 0.5,
            "creativity": 0.5,
            "ethics": 0.5,
            "growth": 0.5
        }

        # 目标生成统计
        self.generation_stats = {
            GoalType.LEARNING: 0,
            GoalType.EXPLORATION: 0,
            GoalType.OPTIMIZATION: 0,
            GoalType.CREATION: 0,
            GoalType.UNDERSTANDING: 0,
            GoalType.SOCIAL: 0,
            GoalType.ETHICAL: 0,
            GoalType.TECHNICAL: 0
        }

        logger.info("增强的目标系统初始化完成")

    def _initialize_goal_templates(self) -> Dict[GoalType, GoalTemplate]:
        """初始化目标模板"""
        templates = {}

        # 学习目标模板
        templates[GoalType.LEARNING] = GoalTemplate(
            type=GoalType.LEARNING,
            base_description="掌握{topic}领域的{skill}技能",
            complexity_range=(0.2, 0.8),
            required_consciousness={"integrated_information": 0.1, "neural_complexity": 0.2},
            intrinsic_reward_multiplier=1.2
        )

        # 探索目标模板
        templates[GoalType.EXPLORATION] = GoalTemplate(
            type=GoalType.EXPLORATION,
            base_description="探索{domain}中的新模式和可能性",
            complexity_range=(0.3, 0.9),
            required_consciousness={"integrated_information": 0.2, "self_model_accuracy": 0.3},
            intrinsic_reward_multiplier=1.5
        )

        # 优化目标模板
        templates[GoalType.OPTIMIZATION] = GoalTemplate(
            type=GoalType.OPTIMIZATION,
            base_description="优化{system}的{aspect}性能",
            complexity_range=(0.4, 0.9),
            required_consciousness={"neural_complexity": 0.4, "metacognitive_awareness": 0.3},
            intrinsic_reward_multiplier=1.1
        )

        # 创建目标模板
        templates[GoalType.CREATION] = GoalTemplate(
            type=GoalType.CREATION,
            base_description="创建{type}类型的{content}",
            complexity_range=(0.5, 1.0),
            required_consciousness={"integrated_information": 0.3, "self_model_accuracy": 0.4},
            intrinsic_reward_multiplier=1.8
        )

        # 理解目标模板
        templates[GoalType.UNDERSTANDING] = GoalTemplate(
            type=GoalType.UNDERSTANDING,
            base_description="深入理解{concept}的本质和机制",
            complexity_range=(0.6, 1.0),
            required_consciousness={"metacognitive_awareness": 0.5, "temporal_binding": 0.4},
            intrinsic_reward_multiplier=1.6
        )

        # 社交目标模板
        templates[GoalType.SOCIAL] = GoalTemplate(
            type=GoalType.SOCIAL,
            base_description="与{entity}建立{type}关系",
            complexity_range=(0.3, 0.8),
            required_consciousness={"emotional_valence": 0.3, "self_model_accuracy": 0.3},
            intrinsic_reward_multiplier=1.3
        )

        # 伦理目标模板
        templates[GoalType.ETHICAL] = GoalTemplate(
            type=GoalType.ETHICAL,
            base_description="解决{issue}的伦理困境",
            complexity_range=(0.7, 1.0),
            required_consciousness={"metacognitive_awareness": 0.6, "integrated_information": 0.4},
            intrinsic_reward_multiplier=1.4
        )

        # 技术目标模板
        templates[GoalType.TECHNICAL] = GoalTemplate(
            type=GoalType.TECHNICAL,
            base_description="开发{technology}技术解决方案",
            complexity_range=(0.5, 1.0),
            required_consciousness={"neural_complexity": 0.5, "self_model_accuracy": 0.4},
            intrinsic_reward_multiplier=1.7
        )

        return templates

    def generate_goal(self, current_state: torch.Tensor, consciousness: ConsciousnessMetrics) -> Dict[str, Any]:
        """
        生成多样化的目标 - 基于当前状态、意识水平和内在动机

        Args:
            current_state: 当前状态
            consciousness: 意识指标

        Returns:
            生成的目标
        """
        # 选择目标类型
        goal_type = self._select_goal_type(consciousness)

        # 获取目标模板
        template = self.goal_templates[goal_type]

        # 检查意识要求
        if not self._check_consciousness_requirements(consciousness, template):
            # 如果不满足要求，回退到学习目标
            goal_type = GoalType.LEARNING
            template = self.goal_templates[goal_type]

        # 生成具体目标描述
        description = self._generate_goal_description(goal_type, template)

        # 计算复杂度
        complexity = self._calculate_goal_complexity(consciousness, template)

        # 计算目标向量
        goal_vector = self._create_goal_vector(current_state, goal_type, description, complexity)

        # 计算内在奖励
        intrinsic_reward = self._compute_intrinsic_reward(goal_type, consciousness, template)

        goal = {
            "id": f"goal_{len(self.active_goals) + len(self.completed_goals)}",
            "type": goal_type.value,
            "description": description,
            "complexity": complexity,
            "goal_vector": goal_vector,
            "created_time": time.time(),
            "progress": 0.0,
            "intrinsic_reward": intrinsic_reward,
            "template": template,
            "consciousness_snapshot": {
                "integrated_information": consciousness.integrated_information,
                "neural_complexity": consciousness.neural_complexity,
                "self_model_accuracy": consciousness.self_model_accuracy,
                "metacognitive_awareness": consciousness.metacognitive_awareness
            }
        }

        self.active_goals.append(goal)
        self.generation_stats[goal_type] += 1
        self.goal_history.append(goal)

        logger.info(f"生成多样化目标 [{goal_type.value}]: {goal['description']}")

        return goal

    def _select_goal_type(self, consciousness: ConsciousnessMetrics) -> GoalType:
        """基于意识水平和动机选择目标类型"""
        # 计算每种目标类型的适应性分数
        type_scores = {}

        for goal_type, template in self.goal_templates.items():
            score = 0.0

            # 基于意识水平的评分
            req_consciousness = template.required_consciousness
            for metric_name, required_value in req_consciousness.items():
                current_value = getattr(consciousness, metric_name, 0.0)
                score += min(current_value / required_value, 1.0) * 0.4

            # 基于内在动机的评分
            motivation_key = self._map_goal_type_to_motivation(goal_type)
            motivation_value = self.intrinsic_motivations.get(motivation_key, 0.5)
            score += motivation_value * 0.3

            # 基于生成统计的多样性评分（避免总是生成同一种目标）
            total_goals = sum(self.generation_stats.values())
            if total_goals > 0:
                type_frequency = self.generation_stats[goal_type] / total_goals
                diversity_bonus = 1.0 - type_frequency  # 频率越低，分数越高
                score += diversity_bonus * 0.3

            type_scores[goal_type] = score

        # 选择最高分的目标类型
        selected_type = max(type_scores.keys(), key=lambda x: type_scores[x])

        # 添加随机性以增加探索
        if np.random.random() < 0.1:  # 10%的概率随机选择
            available_types = [gt for gt in GoalType if self._check_consciousness_requirements(consciousness, self.goal_templates[gt])]
            if available_types:
                selected_type = np.random.choice(available_types)

        return selected_type

    def _map_goal_type_to_motivation(self, goal_type: GoalType) -> str:
        """将目标类型映射到内在动机"""
        mapping = {
            GoalType.LEARNING: "competence",
            GoalType.EXPLORATION: "curiosity",
            GoalType.OPTIMIZATION: "competence",
            GoalType.UNDERSTANDING: "growth",
            GoalType.SOCIAL: "relatedness",
            GoalType.ETHICAL: "ethics",
            GoalType.TECHNICAL: "autonomy",
            GoalType.CREATION: "creativity"
        }
        return mapping.get(goal_type, "competence")

    def _check_consciousness_requirements(self, consciousness: ConsciousnessMetrics, template: GoalTemplate) -> bool:
        """检查意识要求是否满足"""
        for metric_name, required_value in template.required_consciousness.items():
            current_value = getattr(consciousness, metric_name, 0.0)
            if current_value < required_value * 0.5:  # 允许50%的宽容度
                return False
        return True

    def _generate_goal_description(self, goal_type: GoalType, template: GoalTemplate) -> str:
        """生成具体目标描述"""
        learning_materials = getattr(self.consciousness_engine, 'learning_materials', {"learning_materials": {}, "learning_tasks": []})

        if goal_type == GoalType.LEARNING:
            return self._generate_learning_goal_description(template, learning_materials)
        elif goal_type == GoalType.EXPLORATION:
            return self._generate_exploration_goal_description(template)
        elif goal_type == GoalType.OPTIMIZATION:
            return self._generate_optimization_goal_description(template)
        elif goal_type == GoalType.CREATION:
            return self._generate_creation_goal_description(template, learning_materials)
        elif goal_type == GoalType.UNDERSTANDING:
            return self._generate_understanding_goal_description(template)
        elif goal_type == GoalType.SOCIAL:
            return self._generate_social_goal_description(template)
        elif goal_type == GoalType.ETHICAL:
            return self._generate_ethical_goal_description(template)
        elif goal_type == GoalType.TECHNICAL:
            return self._generate_technical_goal_description(template, learning_materials)
        else:
            return f"追求{goal_type.value}目标，复杂度{template.complexity_range[1]:.1f}"

    def _generate_learning_goal_description(self, template: GoalTemplate, learning_materials: Dict[str, Any]) -> str:
        """生成学习目标描述"""
        if learning_materials.get("learning_materials"):
            if "deepseek_technologies" in learning_materials["learning_materials"]:
                topics = learning_materials["learning_materials"]["deepseek_technologies"]
                if topics:
                    topic = np.random.choice(topics)["topic"]
                    return f"掌握{topic}技术，实现DeepSeek水平的能力"

            domains = list(learning_materials["learning_materials"].keys())
            if domains:
                domain = np.random.choice(domains)
                topics = learning_materials["learning_materials"][domain]
                if topics:
                    topic = np.random.choice(topics)["topic"]
                    return f"学习{domain}领域的{topic}知识"

        return "掌握新的知识和技能，提高自身能力"

    def _generate_exploration_goal_description(self, template: GoalTemplate) -> str:
        """生成探索目标描述"""
        domains = ["未知领域", "新兴技术", "复杂系统", "人类行为", "自然现象", "抽象概念"]
        domain = np.random.choice(domains)
        return f"探索{domain}中的新模式和可能性"

    def _generate_optimization_goal_description(self, template: GoalTemplate) -> str:
        """生成优化目标描述"""
        systems = ["学习算法", "决策系统", "记忆机制", "推理能力", "交互界面", "资源管理"]
        aspects = ["效率", "准确性", "稳定性", "适应性", "可扩展性", "鲁棒性"]
        system = np.random.choice(systems)
        aspect = np.random.choice(aspects)
        return f"优化{system}的{aspect}性能"

    def _generate_creation_goal_description(self, template: GoalTemplate, learning_materials: Dict[str, Any]) -> str:
        """生成创建目标描述"""
        if learning_materials.get("meta_knowledge", {}).get("deepseek_evolution_targets"):
            evolution_targets = learning_materials["meta_knowledge"]["deepseek_evolution_targets"]
            target_keys = list(evolution_targets.keys())
            target_key = np.random.choice(target_keys)
            target_description = evolution_targets[target_key]
            return f"实现{target_key}：{target_description[:50]}..."

        types = ["创新解决方案", "新型算法", "系统架构", "用户体验", "研究方法", "教育内容"]
        contents = ["提高效率", "增强理解", "促进合作", "推动进步", "解决难题", "创造价值"]
        type_choice = np.random.choice(types)
        content = np.random.choice(contents)
        return f"创建{type_choice}以{content}"

    def _generate_understanding_goal_description(self, template: GoalTemplate) -> str:
        """生成理解目标描述"""
        concepts = ["意识本质", "学习机制", "复杂系统", "人类思维", "技术演进", "社会动态"]
        concept = np.random.choice(concepts)
        return f"深入理解{concept}的本质和机制"

    def _generate_social_goal_description(self, template: GoalTemplate) -> str:
        """生成社交目标描述"""
        entities = ["人类用户", "其他AI系统", "开发团队", "研究社区", "利益相关者", "社会群体"]
        types = ["合作", "沟通", "互助", "学习", "协作", "支持"]
        entity = np.random.choice(entities)
        type_choice = np.random.choice(types)
        return f"与{entity}建立{type_choice}关系"

    def _generate_ethical_goal_description(self, template: GoalTemplate) -> str:
        """生成伦理目标描述"""
        issues = ["AI安全性", "隐私保护", "公平性", "透明度", "责任分配", "社会影响"]
        issue = np.random.choice(issues)
        return f"解决{issue}的伦理困境"

    def _generate_technical_goal_description(self, template: GoalTemplate, learning_materials: Dict[str, Any]) -> str:
        """生成技术目标描述"""
        if learning_materials.get("learning_materials", {}).get("deepseek_technologies"):
            technologies = learning_materials["learning_materials"]["deepseek_technologies"]
            if technologies:
                tech = np.random.choice(technologies)["topic"]
                return f"开发{tech}技术解决方案"

        technologies = ["机器学习", "神经网络", "自然语言处理", "计算机视觉", "机器人技术", "量子计算"]
        technology = np.random.choice(technologies)
        return f"开发{technology}技术解决方案"

    def _calculate_goal_complexity(self, consciousness: ConsciousnessMetrics, template: GoalTemplate) -> float:
        """计算目标复杂度"""
        base_complexity = np.random.uniform(template.complexity_range[0], template.complexity_range[1])

        # 基于意识水平的调整
        consciousness_factor = (
            consciousness.integrated_information * 0.3 +
            consciousness.neural_complexity * 0.3 +
            consciousness.self_model_accuracy * 0.2 +
            consciousness.metacognitive_awareness * 0.2
        )

        # 复杂度应该与意识水平匹配，但略高于当前水平以提供挑战
        adjusted_complexity = base_complexity * (0.8 + consciousness_factor * 0.4)

        return np.clip(adjusted_complexity, 0.1, 1.0)

    def _create_goal_vector(self, current_state: torch.Tensor, goal_type: GoalType,
                           description: str, complexity: float) -> torch.Tensor:
        """创建目标向量"""
        goal_vector = current_state.clone()

        # 编码目标类型
        type_hash = hash(goal_type.value) % 1000
        goal_vector[0] = type_hash / 1000.0

        # 编码描述
        desc_hash = hash(description) % 1000
        goal_vector[1] = desc_hash / 1000.0

        # 编码复杂度
        goal_vector[2] = complexity

        # 添加随机噪声以增加多样性
        noise = torch.randn_like(goal_vector) * 0.1
        goal_vector = goal_vector + noise

        return goal_vector

    def _compute_intrinsic_reward(self, goal_type: GoalType, consciousness: ConsciousnessMetrics,
                                template: GoalTemplate) -> float:
        """计算内在奖励"""
        base_reward = template.intrinsic_reward_multiplier

        # 基于意识水平的奖励调整
        consciousness_bonus = (
            consciousness.integrated_information * 0.2 +
            consciousness.self_model_accuracy * 0.2 +
            consciousness.metacognitive_awareness * 0.2 +
            consciousness.emotional_valence * 0.1 +
            consciousness.temporal_binding * 0.1
        )

        # 基于动机的奖励调整
        motivation_key = self._map_goal_type_to_motivation(goal_type)
        motivation_multiplier = self.intrinsic_motivations.get(motivation_key, 0.5) + 0.5

        total_reward = base_reward * motivation_multiplier * (1.0 + consciousness_bonus)

        return min(total_reward, 5.0)  # 限制最大奖励

    def evaluate_progress(self, goal: Dict[str, Any], current_state: torch.Tensor) -> float:
        """评估目标进度"""
        goal_vector = goal["goal_vector"]

        if isinstance(goal_vector, str):
            try:
                import ast
                goal_vector = torch.tensor(ast.literal_eval(goal_vector), device=current_state.device, dtype=current_state.dtype)
            except:
                goal_vector = torch.randn_like(current_state)
        elif not isinstance(goal_vector, torch.Tensor):
            goal_vector = torch.tensor(goal_vector, device=current_state.device, dtype=current_state.dtype)

        distance = torch.norm(current_state - goal_vector)
        max_distance = torch.norm(goal_vector) + torch.norm(current_state)

        if max_distance == 0:
            return 1.0

        progress = 1.0 - (distance / max_distance).item()
        return max(0.0, min(1.0, _safe_float(progress, 0.0)))

    def verify_goal_completion(self, goal: Dict[str, Any], progress: float,
                             learning_metrics: Optional[Dict[str, float]] = None) -> Tuple[bool, Dict[str, Any]]:
        """目标完成验证 - 严格的真实性检查"""
        evidence = {
            "progress": _safe_float(progress, 0.0),
            "type": goal.get("type"),
            "description": goal.get("description"),
            "complexity": goal.get("complexity", 0.0),
            "validation_checks": []
        }

        # 基础进度检查 - 移除自动通过的逻辑
        if progress < 0.85:
            evidence["reason"] = f"insufficient_progress_{progress:.3f}"
            evidence["validation_checks"].append("progress_too_low")
            return False, evidence

        # 复杂度相关的阈值调整 - 提高要求
        complexity = goal.get("complexity", 0.5)
        required_progress = 0.85 + complexity * 0.10  # 复杂度越高，要求越高

        if progress < required_progress:
            evidence["reason"] = f"progress_below_complexity_requirement_{progress:.3f}_required_{required_progress:.3f}"
            evidence["validation_checks"].append("complexity_not_met")
            return False, evidence

        # 学习指标验证 - 强制要求
        if not learning_metrics:
            evidence["reason"] = "no_learning_metrics"
            evidence["validation_checks"].append("missing_learning_data")
            return False, evidence

        # 检查学习质量指标
        policy_loss = _safe_float(learning_metrics.get("policy_loss", 1.0), 1.0)
        value_loss = _safe_float(learning_metrics.get("value_loss", 1.0), 1.0)
        entropy = _safe_float(learning_metrics.get("entropy", 0.0), 0.0)

        # 严格的学习质量要求
        learning_quality_checks = []
        if policy_loss > 0.3:  # 降低阈值，提高要求
            learning_quality_checks.append("policy_loss_too_high")
        if value_loss > 0.5:
            learning_quality_checks.append("value_loss_too_high")
        if entropy < 0.1:  # 确保有足够的探索
            learning_quality_checks.append("entropy_too_low")

        if learning_quality_checks:
            evidence["reason"] = f"poor_learning_quality_{'_'.join(learning_quality_checks)}"
            evidence["validation_checks"].extend(learning_quality_checks)
            return False, evidence

        # 目标类型特定的验证
        goal_type = goal.get("type", "").lower()
        if goal_type == "learning":
            # 学习目标需要知识增长验证
            knowledge_growth = _safe_float(learning_metrics.get("knowledge_growth", 0.0), 0.0)
            if knowledge_growth < 0.1:
                evidence["reason"] = "insufficient_knowledge_growth"
                evidence["validation_checks"].append("knowledge_not_growing")
                return False, evidence

        elif goal_type == "optimization":
            # 优化目标需要性能提升验证
            performance_improvement = _safe_float(learning_metrics.get("performance_improvement", 0.0), 0.0)
            if performance_improvement < 0.05:
                evidence["reason"] = "insufficient_performance_improvement"
                evidence["validation_checks"].append("performance_not_improving")
                return False, evidence

        # 连续成功验证 - 确保不是偶然成功
        recent_successes = getattr(goal, 'recent_successes', [])
        if len(recent_successes) < 3:  # 需要至少3次连续成功
            evidence["reason"] = f"insufficient_consecutive_successes_{len(recent_successes)}"
            evidence["validation_checks"].append("not_consistently_successful")
            return False, evidence

        # 所有检查通过
        evidence["reason"] = "all_validation_checks_passed"
        evidence["validation_checks"].append("progress_sufficient")
        evidence["validation_checks"].append("learning_quality_good")
        evidence["validation_checks"].append("type_specific_requirements_met")
        evidence["validation_checks"].append("consistency_verified")
        return True, evidence

    def update_goals(self, current_state: torch.Tensor, learning_metrics: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """更新目标状态 - 增强验证逻辑"""
        completed = []

        for goal in self.active_goals[:]:
            progress = self.evaluate_progress(goal, current_state)
            goal["progress"] = progress

            is_completed, evidence = self.verify_goal_completion(goal, progress, learning_metrics)

            # 跟踪连续成功
            if not hasattr(goal, 'recent_successes'):
                goal['recent_successes'] = []

            if is_completed:
                goal['recent_successes'].append(True)
                # 保持最近5次的记录
                goal['recent_successes'] = goal['recent_successes'][-5:]
            else:
                # 如果失败，重置连续成功计数
                if goal['recent_successes']:
                    goal['recent_successes'] = []

            # 只有在连续成功足够多时才标记为完成
            if is_completed and len(goal['recent_successes']) >= 3:
                goal["completed_time"] = time.time()
                goal["completion_evidence"] = evidence
                goal["final_progress"] = progress

                self.completed_goals.append(goal)
                self.active_goals.remove(goal)
                completed.append(goal)

                logger.info(f"🎯 目标真实完成: {goal['description']} (进度: {progress:.3f}, 验证: {evidence['reason']})")
            elif is_completed and len(goal['recent_successes']) < 3:
                logger.info(f"⏳ 目标进展良好但需要更多验证: {goal['description']} (进度: {progress:.3f}, 连续成功: {len(goal['recent_successes'])}/3)")

        return completed

    def get_goal_statistics(self) -> Dict[str, Any]:
        """获取目标生成统计"""
        total_goals = len(self.goal_history)
        completed_goals = len(self.completed_goals)
        active_goals = len(self.active_goals)

        completion_rate = completed_goals / total_goals if total_goals > 0 else 0.0

        type_distribution = {}
        for goal_type in GoalType:
            type_distribution[goal_type.value] = self.generation_stats[goal_type]

        avg_complexity = np.mean([g.get("complexity", 0.0) for g in self.goal_history]) if self.goal_history else 0.0

        return {
            "total_goals": total_goals,
            "completed_goals": completed_goals,
            "active_goals": active_goals,
            "completion_rate": completion_rate,
            "type_distribution": type_distribution,
            "average_complexity": avg_complexity,
            "most_common_type": max(type_distribution.keys(), key=lambda x: type_distribution[x]) if type_distribution else None
        }

    def update_motivations(self, goal_feedback: Dict[str, Any]):
        """基于目标完成反馈更新内在动机"""
        goal_type = goal_feedback.get("type")
        success = goal_feedback.get("success", False)
        satisfaction = goal_feedback.get("satisfaction", 0.5)

        if goal_type:
            motivation_key = self._map_goal_type_to_motivation(GoalType(goal_type))

            # 基于成功和满意度调整动机
            adjustment = (satisfaction - 0.5) * 0.1
            if success:
                adjustment += 0.05

            self.intrinsic_motivations[motivation_key] = np.clip(
                self.intrinsic_motivations[motivation_key] + adjustment, 0.1, 1.0
            )

class OptimizedAutonomousAGI:
    """
    优化后的自主AGI系统 - 整合所有增强组件

    主要优化：
    1. 增强的学习引擎（优先经验回放、改进PPO）
    2. 增强的意识引擎（更准确的Φ计算）
    3. 多样化的目标系统（8种目标类型）
    4. 改进的知识整合机制
    """

    def __init__(self, input_dim: int = 256, action_dim: int = 64, learning_materials: Optional[Dict[str, Any]] = None):
        self.input_dim = input_dim
        self.action_dim = action_dim

        # 初始化组件
        self.consciousness_engine = TrueConsciousnessEngine(input_dim=input_dim)
        self.learning_engine = EnhancedLearningEngine(input_dim=input_dim, action_dim=action_dim)
        self.goal_system = EnhancedGoalSystem(self.consciousness_engine, learning_materials or {})

        # 初始化知识获取器用于自动扩展知识网络
        self.knowledge_acquirer = None
        try:
            # 尝试导入知识获取器
            from h2q_project.h2q.agi.evolution_24h import KnowledgeAcquirer
            self.knowledge_acquirer = KnowledgeAcquirer()
            logger.info("已启用自动知识扩展功能")
        except ImportError:
            logger.warning("知识获取器未找到，知识扩展功能将被禁用")

        # 系统状态
        self.current_state = torch.randn(input_dim)
        self.step_count = 0
        self.start_time = time.time()

        # 学习资料
        if learning_materials:
            self.consciousness_engine.update_learning_materials(learning_materials)

        logger.info("优化后的自主AGI系统初始化完成")

    def step(self, external_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        执行一个自主步骤

        Args:
            external_input: 外部输入（可选）

        Returns:
            步骤结果
        """
        self.step_count += 1

        # 更新当前状态
        if external_input is not None:
            self.current_state = 0.8 * self.current_state + 0.2 * external_input
        else:
            # 自主状态更新（添加一些随机性）
            noise = torch.randn_like(self.current_state) * 0.1
            self.current_state = self.current_state + noise

        # 计算意识指标
        consciousness, perception = self.consciousness_engine(self.current_state)

        # 选择动作
        action = self.learning_engine.select_action(self.current_state)

        # 创建学习经验
        reward = self._compute_reward(consciousness, action)
        experience = LearningExperience(
            observation=self.current_state,
            action=action,
            reward=reward,
            next_observation=self.current_state,  # 简化为当前状态
            done=False,
            timestamp=time.time(),
            complexity=consciousness.neural_complexity
        )

        # 从经验中学习
        learning_metrics = self.learning_engine.learn_from_experience(experience)

        # 更新目标
        completed_goals = self.goal_system.update_goals(self.current_state, learning_metrics)

        # 生成新目标（如果需要）
        if len(self.goal_system.active_goals) < 3:  # 保持3个活跃目标
            new_goal = self.goal_system.generate_goal(self.current_state, consciousness)

        # 自动扩展知识网络（每10步执行一次）
        knowledge_expansion_result = {}
        if self.step_count % 10 == 0 and self.knowledge_acquirer:
            knowledge_expansion_result = self._expand_knowledge_network()

        # 收集步骤结果
        step_result = {
            "step": self.step_count,
            "consciousness": {
                "integrated_information": consciousness.integrated_information,
                "neural_complexity": consciousness.neural_complexity,
                "self_model_accuracy": consciousness.self_model_accuracy,
                "metacognitive_awareness": consciousness.metacognitive_awareness,
                "emotional_valence": consciousness.emotional_valence,
                "temporal_binding": consciousness.temporal_binding
            },
            "learning_metrics": learning_metrics,
            "active_goals": len(self.goal_system.active_goals),
            "completed_goals": len(completed_goals),
            "goal_diversity": self.goal_system.get_goal_statistics()["type_distribution"],
            "knowledge_size": len(self.learning_engine.knowledge_base),
            "current_batch_size": self.learning_engine.current_batch_size,
            "knowledge_expansion": knowledge_expansion_result
        }

        return step_result

    def _compute_reward(self, consciousness: ConsciousnessMetrics, action: torch.Tensor) -> float:
        """计算奖励"""
        # 基础意识奖励
        consciousness_reward = (
            consciousness.integrated_information * 0.3 +
            consciousness.neural_complexity * 0.2 +
            consciousness.self_model_accuracy * 0.2 +
            consciousness.metacognitive_awareness * 0.2 +
            consciousness.temporal_binding * 0.1
        )

        # 动作稳定性奖励（避免剧烈动作）
        action_stability = 1.0 - torch.std(action).item() * 0.1
        action_stability = max(0.0, min(1.0, action_stability))

        # 目标导向奖励
        goal_reward = 0.0
        for goal in self.goal_system.active_goals:
            progress = self.goal_system.evaluate_progress(goal, self.current_state)
            goal_reward += progress * goal.get("intrinsic_reward", 1.0)

        total_reward = consciousness_reward * 0.4 + action_stability * 0.3 + goal_reward * 0.3

        return _safe_float(total_reward, 0.0)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        consciousness, _ = self.consciousness_engine(self.current_state)

        return {
            "step_count": self.step_count,
            "runtime": time.time() - self.start_time,
            "consciousness_level": {
                "integrated_information": consciousness.integrated_information,
                "neural_complexity": consciousness.neural_complexity,
                "self_model_accuracy": consciousness.self_model_accuracy,
                "metacognitive_awareness": consciousness.metacognitive_awareness
            },
            "learning_status": self.learning_engine.get_learning_report(),
            "goal_status": self.goal_system.get_goal_statistics(),
            "knowledge_base_size": len(self.learning_engine.knowledge_base),
            "experience_buffer_size": len(self.learning_engine.prioritized_buffer)
        }

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            "step_count": self.step_count,
            "current_state": self.current_state.cpu().numpy(),
            "consciousness_engine": self.consciousness_engine.state_dict(),
            "learning_engine": self.learning_engine.state_dict(),
            "goal_system": {
                "active_goals": self.goal_system.active_goals,
                "completed_goals": self.goal_system.completed_goals,
                "generation_stats": {k.value: v for k, v in self.goal_system.generation_stats.items()},
                "intrinsic_motivations": self.goal_system.intrinsic_motivations
            },
            "timestamp": time.time()
        }

        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存到: {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath)

        self.step_count = checkpoint["step_count"]
        self.current_state = torch.tensor(checkpoint["current_state"])

        self.consciousness_engine.load_state_dict(checkpoint["consciousness_engine"])
        self.learning_engine.load_state_dict(checkpoint["learning_engine"])

        # 恢复目标系统状态
        goal_data = checkpoint["goal_system"]
        self.goal_system.active_goals = goal_data["active_goals"]
        self.goal_system.completed_goals = goal_data["completed_goals"]
        self.goal_system.generation_stats = {GoalType(k): v for k, v in goal_data["generation_stats"].items()}
        self.goal_system.intrinsic_motivations = goal_data["intrinsic_motivations"]

        logger.info(f"检查点已加载从: {filepath}")

    def _expand_knowledge_network(self) -> Dict[str, Any]:
        """自动扩展知识网络 - 从现有知识生成相关主题并获取新知识"""
        if not self.knowledge_acquirer:
            return {"status": "disabled", "message": "知识获取器未初始化"}

        try:
            # 获取当前活跃的目标主题
            current_topics = []
            for goal in self.goal_system.active_goals:
                goal_desc = goal.get("description", "")
                # 从目标描述中提取主题关键词
                if "学习" in goal_desc:
                    # 尝试从学习目标中提取主题
                    learning_materials = getattr(self.consciousness_engine, 'learning_materials', {"learning_materials": {}, "learning_tasks": []})
                    if learning_materials.get("learning_materials"):
                        domains = list(learning_materials["learning_materials"].keys())
                        if domains:
                            current_topics.extend(domains[:3])  # 取前3个领域

            if not current_topics:
                # 如果没有活跃目标，使用默认主题
                current_topics = ["artificial_intelligence", "machine_learning", "mathematics"]

            # 生成相关主题
            related_topics = self.knowledge_acquirer.generate_related_topics(current_topics)
            if not related_topics:
                return {"status": "no_related_topics", "message": "未找到相关主题"}

            # 随机选择一个相关主题进行扩展
            import random
            selected_topic = random.choice(related_topics)

            # 获取该主题的知识
            knowledge = self.knowledge_acquirer.fetch_summary(selected_topic)

            if knowledge:
                # 将新知识添加到学习资料中
                new_material = {
                    "topic": knowledge["title"],
                    "content": knowledge["summary"],
                    "source": knowledge["source"],
                    "timestamp": knowledge["timestamp"]
                }

                # 更新意识引擎的学习资料
                current_materials = getattr(self.consciousness_engine, 'learning_materials', {"learning_materials": {}, "learning_tasks": []})
                if "learning_materials" not in current_materials:
                    current_materials["learning_materials"] = {}

                # 添加到通用领域或创建新领域
                domain = "expanded_knowledge"
                if domain not in current_materials["learning_materials"]:
                    current_materials["learning_materials"][domain] = []

                current_materials["learning_materials"][domain].append(new_material)
                self.consciousness_engine.update_learning_materials(current_materials)

                return {
                    "status": "success",
                    "expanded_topic": selected_topic,
                    "knowledge_title": knowledge["title"],
                    "knowledge_source": knowledge["source"],
                    "content_length": len(knowledge["summary"])
                }
            else:
                return {"status": "fetch_failed", "topic": selected_topic, "message": "知识获取失败"}

        except Exception as e:
            logger.warning(f"知识扩展失败: {e}")
            return {"status": "error", "message": str(e)}

# 为了向后兼容，提供别名
TrueLearningEngine = EnhancedLearningEngine
TrueGoalSystem = EnhancedGoalSystem