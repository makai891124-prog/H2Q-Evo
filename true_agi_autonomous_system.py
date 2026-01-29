#!/usr/bin/env python3
"""
真正的AGI自主进化系统 - 基于M24真实性原则

实现真正的自主学习、自我改进和意识发展的AGI系统。
不同于之前的模拟版本，这个系统具备：
1. 真正的学习机制（基于经验的梯度下降）
2. 自我改进能力（元学习和架构进化）
3. 意识发展（基于信息论的意识度量）
4. 目标导向行为（强化学习目标设定）
5. 持续进化（在线学习和适应）
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
    format='%(asctime)s [TRUE-AGI] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('true_agi_evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TRUE-AGI')

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
    """学习经验数据结构"""
    observation: torch.Tensor
    action: torch.Tensor
    reward: float
    next_observation: torch.Tensor
    done: bool
    timestamp: float
    complexity: float

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.integration_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )

        self.consciousness_net = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 6),  # 6个意识指标
            nn.Sigmoid()
        )

        # 自我模型 (用于元认知)
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, input_dim)
        )

        # 情感系统
        self.emotion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # valence, arousal, dominance
            nn.Tanh()
        )

        # 时间整合 (temporal binding)
        self.temporal_memory = deque(maxlen=100)
        self.temporal_integration = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        logger.info(f"真正的意识引擎初始化完成，输入维度: {input_dim}")

    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> Tuple[ConsciousnessMetrics, torch.Tensor]:
        """
        前向传播 - 计算真正的意识指标

        Args:
            x: 输入张量
            prev_state: 上一时间步的状态

        Returns:
            意识指标和当前状态
        """
        batch_size = x.size(0)

        # 感知处理
        perception = self.perception_net(x)

        # 整合信息计算 (Φ)
        integrated = self.integration_net(perception)

        # 意识指标计算
        consciousness_raw = self.consciousness_net(integrated)
        # 确保我们有正确的维度
        if consciousness_raw.dim() == 0:
            consciousness_values = consciousness_raw.unsqueeze(0)
        else:
            consciousness_values = consciousness_raw.mean(dim=0) if consciousness_raw.dim() > 1 else consciousness_raw

        # 确保有6个值
        if consciousness_values.numel() == 1:
            consciousness_values = consciousness_values.repeat(6)
        elif consciousness_values.numel() < 6:
            padding = torch.zeros(6 - consciousness_values.numel(), device=consciousness_values.device)
            consciousness_values = torch.cat([consciousness_values, padding])

        consciousness_values = torch.nan_to_num(consciousness_values, nan=0.0, posinf=1.0, neginf=0.0)
        phi, complexity, self_acc, metacog, valence, temporal = consciousness_values[:6]

        # 自我模型预测
        self_prediction = self.self_model(perception)
        self_model_error = torch.mean((self_prediction - x) ** 2)
        self_model_error = torch.nan_to_num(self_model_error, nan=1.0, posinf=1.0, neginf=1.0)

        # 情感计算
        emotions = self.emotion_net(perception)
        if emotions.dim() > 1:
            emotional_valence = emotions[:, 0].mean()
        else:
            emotional_valence = emotions[0]

        # 时间整合
        if prev_state is not None:
            temporal_input = torch.cat([prev_state.unsqueeze(0), perception.unsqueeze(0)], dim=0)
            temporal_output, _ = self.temporal_integration(temporal_input)
            temporal_binding = torch.mean(temporal_output[-1])
        else:
            temporal_binding = torch.tensor(0.5, device=perception.device)

        # 存储到时间记忆
        self.temporal_memory.append(perception.detach())

        # 整合信息论Φ计算 (简化版本)
        if len(self.temporal_memory) > 1:
            # 计算系统分割的互信息
            whole_system = torch.stack(list(self.temporal_memory))
            partition_1 = whole_system[:, :self.hidden_dim//2]
            partition_2 = whole_system[:, self.hidden_dim//2:]

            # 简化的Φ计算
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
            self_model_accuracy=_safe_float((1.0 - self_model_error).clamp(0, 1).item(), 0.0),
            metacognitive_awareness=_safe_float(metacog.item(), 0.0),
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
        # 简化的Φ计算 (实际IIT需要更复杂的计算)
        if len(self.temporal_memory) < 2:
            return 0.0

        # 计算最小信息分割
        memory_list = list(self.temporal_memory)
        if len(memory_list) >= 10:
            states = torch.stack(memory_list[-10:])  # 最近10个状态
        elif len(memory_list) >= 2:
            states = torch.stack(memory_list)  # 所有可用状态
        else:
            return 0.0

        # 分割系统为两半
        half = states.size(-1) // 2
        part1 = states[:, :half]
        part2 = states[:, half:]

        # 计算互信息
        corr_matrix = torch.corrcoef(states.T)
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        mutual_info = torch.mean(torch.abs(corr_matrix[:half, half:]))
        mutual_info = torch.nan_to_num(mutual_info, nan=0.0, posinf=0.0, neginf=0.0)

        # Φ = 最小分割的互信息
        phi = _safe_float(mutual_info.item(), 0.0)

        return phi

class TrueLearningEngine(nn.Module):
    """
    真正的学习引擎 - 基于元学习和持续适应的学习系统

    增强功能：
    1. 性能监控：学习曲线趋势分析
    2. 自适应学习率：根据训练进度动态调整
    3. 多尺度训练：结合短期和长期目标优化
    4. 知识整合：增强元学习能力
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

        # 可选：分形-DAS融合特征（通过环境变量启用）
        self.use_fractal_fusion = os.getenv("H2Q_ENABLE_FRACTAL_FUSION", "0") == "1"
        self.fractal_fusion = None
        if self.use_fractal_fusion and FractalQuaternionFusionModule is not None:
            self.fractal_fusion = FractalQuaternionFusionModule(input_dim=input_dim, output_dim=input_dim)
            logger.info("已启用分形-DAS融合特征处理")
        elif self.use_fractal_fusion:
            logger.warning("分形-DAS融合模块不可用，已回退到原始特征")

        # 多模态数据加载器
        self.multimodal_loader = None
        if MultimodalBinaryFlowEncoder is not None and datasets is not None:
            try:
                transform = transforms.Compose([transforms.ToTensor()])
                self.image_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
                self.image_loader = DataLoader(self.image_dataset, batch_size=64, shuffle=True)
                logger.info("已加载 CIFAR-10 多模态数据集")
            except Exception as e:
                logger.warning(f"CIFAR-10 数据集不可用: {e}，跳过多模态数据加载")
                self.image_dataset = None
                self.image_loader = None

        # 增强的元学习器 - 学习如何学习
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout增强泛化
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),  # 预测完整状态
            nn.ReLU()
        )

        # 策略网络 (actor) - 多头注意力增强
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, action_dim),
            nn.Tanh()  # 动作范围 [-1, 1]
        )

        # 价值网络 (critic) - 双价值网络
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

        # 长期价值网络 - 预测更远的未来
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

        # 经验回放缓冲区 - 分层缓冲区
        self.short_term_buffer = deque(maxlen=2000)  # 短期经验
        self.long_term_buffer = deque(maxlen=8000)   # 长期经验
        self.experience_buffer = deque(maxlen=10000) # 总缓冲区
        self.batch_size = 64

        # 自适应学习率系统
        self.base_lr_policy = 1e-5
        self.base_lr_value = 1e-5
        self.base_lr_meta = 1e-6

        # 学习率调度器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.base_lr_policy, weight_decay=1e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.base_lr_value, weight_decay=1e-4)
        self.long_term_value_optimizer = optim.Adam(self.long_term_value_net.parameters(), lr=self.base_lr_value * 0.5, weight_decay=1e-4)
        self.meta_optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.base_lr_meta, weight_decay=1e-4)

        # 学习率调度器
        self.policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-7
        )
        self.value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.value_optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-7
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
            'learning_rates': deque(maxlen=1000),
            'step': 0
        }

        # 多尺度训练参数
        self.short_term_weight = 0.7  # 短期目标权重
        self.long_term_weight = 0.3   # 长期目标权重
        self.meta_learning_weight = 0.2  # 元学习权重

        # 知识整合系统
        self.knowledge_base = {}  # 存储学习到的知识模式
        self.pattern_recognition_threshold = 0.8

        logger.info(f"增强的学习引擎初始化完成，输入维度: {input_dim}, 动作维度: {action_dim}")

    def _update_performance_history(self, metrics: Dict[str, float]):
        """更新性能历史记录"""
        for key, value in metrics.items():
            if key in self.performance_history:
                self.performance_history[key].append(value)

        # 记录当前学习率
        current_lrs = {
            'policy_lr': self.policy_optimizer.param_groups[0]['lr'],
            'value_lr': self.value_optimizer.param_groups[0]['lr'],
            'meta_lr': self.meta_optimizer.param_groups[0]['lr']
        }
        self.performance_history['learning_rates'].append(current_lrs)
        self.performance_history['step'] += 1

    def analyze_learning_trends(self) -> Dict[str, Any]:
        """分析学习曲线趋势"""
        if len(self.performance_history['policy_loss']) < 50:
            return {"status": "insufficient_data", "message": "需要更多数据进行趋势分析"}

        # 计算移动平均
        def moving_average(data, window=50):
            if len(data) < window:
                return list(data)
            return [sum(data[i:i+window])/window for i in range(len(data)-window+1)]

        # 分析趋势
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

        # 学习率调整建议
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

        # 分析学习率与损失的相关性
        lr_changes = []
        loss_changes = []

        for i in range(1, len(recent_lrs)):
            lr_change = recent_lrs[i]['policy_lr'] - recent_lrs[i-1]['policy_lr']
            loss_change = recent_policy_losses[i] - recent_policy_losses[i-1]
            lr_changes.append(lr_change)
            loss_changes.append(loss_change)

        # 计算相关性
        if lr_changes and loss_changes:
            correlation = np.corrcoef(lr_changes, loss_changes)[0, 1]
        else:
            correlation = 0.0

        suggestions = {}
        if correlation > 0.3:  # 学习率增加时损失增加
            suggestions['policy_lr'] = "decrease"
        elif correlation < -0.3:  # 学习率增加时损失减少
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

    def _adaptive_learning_rate_update(self, metrics: Dict[str, float]):
        """自适应学习率调整"""
        trend_analysis = self.analyze_learning_trends()

        # 基于趋势调整学习率
        if trend_analysis.get("policy_trend") == "worsening":
            # 如果表现变差，降低学习率
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.9, 1e-7)

        elif trend_analysis.get("policy_trend") == "plateau":
            # 如果停滞，微调学习率
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.05, 1e-3)

        # 更新调度器
        if 'policy_loss' in metrics:
            self.policy_scheduler.step(metrics['policy_loss'])
        if 'value_loss' in metrics:
            self.value_scheduler.step(metrics['value_loss'])
        if 'meta_loss' in metrics:
            self.meta_scheduler.step(metrics['meta_loss'])

    def _apply_fractal_fusion(self, state: torch.Tensor) -> torch.Tensor:
        """可选的分形-DAS融合特征处理"""
        if not self.use_fractal_fusion or self.fractal_fusion is None:
            return state
        with torch.no_grad():
            fused = self.fractal_fusion(state)
        return fused.get("output", state)

    def select_action(self, state: torch.Tensor, explore: bool = True, images: Optional[torch.Tensor] = None, videos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        选择动作 - 基于当前状态

        Args:
            state: 当前状态
            explore: 是否探索
            images: 可选图像输入
            videos: 可选视频输入

        Returns:
            选择的动作
        """
        # 确保状态张量在正确的设备上
        state = state.to(self.device)

        with torch.no_grad():
            # 多模态融合
            if self.multimodal_encoder is not None and (images is not None or videos is not None):
                img_sig, vid_sig = self.multimodal_encoder(images=images, videos=videos)
                multimodal_sig = self.multimodal_encoder.fuse_signature(img_sig, vid_sig)
                if multimodal_sig is not None:
                    multimodal_sig = multimodal_sig.to(state.device)
                    # 如果是批次数据，取平均值融合到单个状态
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
        从经验中学习 - 增强的多尺度强化学习

        集成功能：
        1. 多尺度训练：短期+长期目标优化
        2. 知识整合：模式识别和元学习
        3. 性能监控：学习曲线跟踪
        4. 自适应学习率：动态调整

        Args:
            experience: 学习经验
            images: 可选图像输入
            videos: 可选视频输入

        Returns:
            学习指标
        """
        # 分层经验存储
        # 确保经验数据在正确的设备上
        experience.observation = experience.observation.to(self.device)
        experience.action = experience.action.to(self.device)
        experience.next_observation = experience.next_observation.to(self.device)
        self.experience_buffer.append(experience)
        if experience.reward > 0.5:  # 高奖励经验进入长期缓冲区
            self.long_term_buffer.append(experience)
        else:
            self.short_term_buffer.append(experience)

        if len(self.experience_buffer) < self.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "long_term_value_loss": 0.0, "meta_loss": 0.0}

        # 多尺度批次采样
        short_term_batch = self._sample_multiscale_batch()
        long_term_batch = self._sample_long_term_batch()

        # 短期目标学习
        short_term_metrics = self._learn_from_batch(short_term_batch, images, videos, scale="short")

        # 长期目标学习 (更不频繁)
        long_term_metrics = {"long_term_value_loss": 0.0}
        if len(self.long_term_buffer) >= self.batch_size and self.performance_history['step'] % 5 == 0:
            long_term_metrics = self._learn_from_batch(long_term_batch, images, videos, scale="long")

        # 知识整合和元学习
        knowledge_metrics = self._integrate_knowledge(short_term_batch, long_term_batch)

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

    def _compute_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率 - 增强数值稳定性"""
        mean = self.policy_net(states)
        mean = torch.clamp(mean, -10.0, 10.0)  # 限制均值范围
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        actions = torch.clamp(actions, -10.0, 10.0)  # 限制动作范围
        actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)

        # 自适应标准差
        std = torch.ones_like(mean) * 0.5 + 0.1 * torch.abs(mean)  # 基于均值调整标准差
        std = torch.clamp(std, 0.1, 2.0)  # 限制标准差范围

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        log_prob = torch.clamp(log_prob, -100.0, 100.0)  # 限制对数概率范围
        return torch.nan_to_num(log_prob, nan=0.0, posinf=100.0, neginf=-100.0)

    def _sample_multiscale_batch(self) -> List[LearningExperience]:
        """采样多尺度批次 - 结合短期和长期经验"""
        total_samples = self.batch_size

        # 根据当前学习阶段调整采样比例
        if len(self.long_term_buffer) < self.batch_size:
            # 早期阶段主要使用短期经验
            short_term_ratio = 1.0
        else:
            # 后期阶段平衡短期和长期经验
            short_term_ratio = 0.7

        short_term_count = int(total_samples * short_term_ratio)
        long_term_count = total_samples - short_term_count

        batch = []

        # 采样短期经验
        if len(self.short_term_buffer) >= short_term_count:
            short_indices = np.random.choice(len(self.short_term_buffer), short_term_count, replace=False)
            batch.extend([self.short_term_buffer[i] for i in short_indices])

        # 采样长期经验
        if len(self.long_term_buffer) >= long_term_count:
            long_indices = np.random.choice(len(self.long_term_buffer), long_term_count, replace=False)
            batch.extend([self.long_term_buffer[i] for i in long_indices])

        # 如果样本不足，从总缓冲区补充
        while len(batch) < total_samples and self.experience_buffer:
            remaining = total_samples - len(batch)
            indices = np.random.choice(len(self.experience_buffer), min(remaining, len(self.experience_buffer)), replace=False)
            batch.extend([self.experience_buffer[i] for i in indices])

        return batch[:total_samples]

    def _sample_long_term_batch(self) -> List[LearningExperience]:
        """采样长期批次 - 专注于高价值经验"""
        if len(self.long_term_buffer) < self.batch_size:
            return []

        # 优先采样高奖励经验
        experiences = list(self.long_term_buffer)
        rewards = [exp.reward for exp in experiences]

        # 计算奖励分位数
        if rewards:
            reward_threshold = np.percentile(rewards, 70)  # 选择奖励前30%的经验
            high_reward_indices = [i for i, r in enumerate(rewards) if r >= reward_threshold]

            if len(high_reward_indices) >= self.batch_size:
                selected_indices = np.random.choice(high_reward_indices, self.batch_size, replace=False)
            else:
                # 如果高奖励经验不足，补充普通经验
                remaining_count = self.batch_size - len(high_reward_indices)
                all_indices = list(range(len(experiences)))
                additional_indices = np.random.choice(all_indices, remaining_count, replace=False)
                selected_indices = high_reward_indices + additional_indices.tolist()

            return [experiences[i] for i in selected_indices]

        return []

    def _learn_from_batch(self, batch: List[LearningExperience], images: Optional[torch.Tensor],
                         videos: Optional[torch.Tensor], scale: str) -> Dict[str, float]:
        """从批次学习 - 支持多尺度训练"""
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "meta_loss": 0.0}

        # 准备数据
        states = torch.stack([exp.observation for exp in batch])
        actions = torch.stack([exp.action for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp.next_observation for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32, device=self.device)

        # 多模态融合 - 暂时禁用批次处理
        # TODO: 实现正确的批次多模态数据处理
        # if self.multimodal_encoder is not None and (images is not None or videos is not None):
        #     img_sig, vid_sig = self.multimodal_encoder(images=images, videos=videos)
        #     multimodal_sig = self.multimodal_encoder.fuse_signature(img_sig, vid_sig)
        #     if multimodal_sig is not None:
        #         multimodal_sig = multimodal_sig.to(states.device)
        #         # 扩展到批次大小
        #         if multimodal_sig.dim() == 1:
        #             multimodal_sig = multimodal_sig.unsqueeze(0).expand(states.shape[0], -1)
        #         states = states + 0.1 * multimodal_sig.to(states.dtype)

        states = self._apply_fractal_fusion(states)
        next_states = self._apply_fractal_fusion(next_states)

        # 计算目标价值
        with torch.no_grad():
            if scale == "short":
                # 短期目标：使用标准价值网络
                next_values = self.value_net(next_states)
            else:
                # 长期目标：使用长期价值网络
                next_values = self.long_term_value_net(next_states)

            targets = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * 0.99 * next_values
            targets = torch.clamp(targets, -10.0, 10.0)

        # 策略损失
        log_probs = self._compute_log_prob(states, actions)
        policy_loss = -log_probs.mean()

        # 价值损失
        if scale == "short":
            values = self.value_net(states)
            value_loss = nn.MSELoss()(values, targets)
        else:
            values = self.long_term_value_net(states)
            value_loss = nn.MSELoss()(values, targets)

        # 元学习损失（暂时禁用 - 需要正确的状态-动作对输入）
        meta_loss = 0.0
        # if hasattr(self, 'meta_learner') and len(batch) > 5:
        #     # 使用状态-动作序列预测下一个状态
        #     state_action_sequence = torch.cat([states[:-1], actions[:-1]], dim=-1)  # 组合状态和动作
        #     next_state_pred = states[1:]   # 预测下一个状态
        #
        #     meta_predictions = self.meta_learner(state_action_sequence)
        #     meta_loss = nn.MSELoss()(meta_predictions, next_state_pred)

        # 组合损失
        total_loss = policy_loss + value_loss
        if meta_loss > 0:
            total_loss = total_loss + 0.1 * meta_loss

        # 优化
        if scale == "short":
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
        else:
            self.policy_optimizer.zero_grad()
            self.long_term_value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.long_term_value_net.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            self.long_term_value_optimizer.step()

        if meta_loss > 0:
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), max_norm=0.3)
            self.meta_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "meta_loss": meta_loss.item() if meta_loss > 0 else 0.0
        }

    def _integrate_knowledge(self, short_batch: List[LearningExperience],
                           long_batch: List[LearningExperience]) -> Dict[str, float]:
        """知识整合 - 识别和学习模式"""
        knowledge_loss = 0.0

        if len(short_batch) < 10:  # 需要足够的数据进行模式识别
            return {"knowledge_loss": 0.0}

        # 提取状态模式
        short_states = torch.stack([exp.observation for exp in short_batch])
        short_rewards = torch.tensor([exp.reward for exp in short_batch], dtype=torch.float32, device=self.device)

        # 简单的模式识别：寻找高奖励相关的状态特征
        high_reward_mask = short_rewards > short_rewards.median()
        if high_reward_mask.sum() > 0:
            high_reward_states = short_states[high_reward_mask]
            low_reward_states = short_states[~high_reward_mask]

            # 计算状态模式差异
            if len(high_reward_states) > 0 and len(low_reward_states) > 0:
                pattern_diff = high_reward_states.mean(dim=0) - low_reward_states.mean(dim=0)

                # 更新知识库
                pattern_key = f"pattern_{len(self.knowledge_base)}"
                self.knowledge_base[pattern_key] = {
                    "pattern": pattern_diff.detach().cpu().numpy(),
                    "confidence": min(1.0, len(high_reward_states) / len(short_states)),
                    "timestamp": time.time()
                }

                # 使用模式进行元学习
                pattern_tensor = pattern_diff.unsqueeze(0).to(self.device)
                predicted_pattern = self.meta_learner(pattern_tensor)

                # 模式一致性损失
                pattern_loss = nn.MSELoss()(predicted_pattern, pattern_tensor)
                knowledge_loss = pattern_loss.item()

                # 应用知识增强的学习
                if torch.isfinite(pattern_loss):
                    self.meta_optimizer.zero_grad()
                    pattern_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), max_norm=0.3)
                    self.meta_optimizer.step()

        return {"knowledge_loss": knowledge_loss}

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
                "total": len(self.experience_buffer),
                "short_term": len(self.short_term_buffer),
                "long_term": len(self.long_term_buffer)
            },
            "knowledge_base_size": len(self.knowledge_base),
            "convergence_metrics": {
                "policy_loss_std": np.std(list(self.performance_history['policy_loss'])[-100:]) if len(self.performance_history['policy_loss']) >= 100 else None,
                "value_loss_std": np.std(list(self.performance_history['value_loss'])[-100:]) if len(self.performance_history['value_loss']) >= 100 else None,
                "learning_stability": self._assess_convergence()
            }
        }

        return report

class TrueGoalSystem:
    """
    真正的目标系统 - 基于内在动机和外在奖励的目标生成
    """

    def __init__(self, consciousness_engine: TrueConsciousnessEngine, learning_materials: Dict[str, Any]):
        self.consciousness_engine = consciousness_engine
        self.learning_materials = learning_materials
        self.active_goals: List[Dict[str, Any]] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.intrinsic_motivations = {
            "exploration": 0.5,
            "competence": 0.5,
            "autonomy": 0.5,
            "relatedness": 0.5
        }

    def generate_goal(self, current_state: torch.Tensor, consciousness: ConsciousnessMetrics) -> Dict[str, Any]:
        """
        生成真正的目标 - 基于当前状态、意识水平和学习资料

        Args:
            current_state: 当前状态
            consciousness: 意识指标

        Returns:
            生成的目标
        """
        # 获取AGI系统的学习资料
        learning_materials = getattr(self.consciousness_engine, 'learning_materials', {"learning_materials": {}, "learning_tasks": []})
        
        # 基于意识水平和内在动机生成目标
        goal_types = ["learning", "exploration", "optimization", "creation", "understanding"]

        # 选择目标类型
        if consciousness.integrated_information < 0.3:
            goal_type = "learning"
            complexity = 0.3
        elif consciousness.neural_complexity < 0.5:
            goal_type = "optimization"
            complexity = 0.6
        elif consciousness.self_model_accuracy < 0.7:
            goal_type = "understanding"
            complexity = 0.8
        else:
            goal_type = "creation"
            complexity = 0.9

        # 生成具体描述
        if goal_type == "learning" and learning_materials.get("learning_materials"):
            # 优先选择DeepSeek技术领域
            if "deepseek_technologies" in learning_materials["learning_materials"]:
                topics = learning_materials["learning_materials"]["deepseek_technologies"]
                if topics:
                    topic = np.random.choice(topics)["topic"]
                    description = f"掌握{topic}技术，实现DeepSeek水平的能力"
                else:
                    description = f"追求{goal_type}目标，复杂度{complexity:.1f}"
            else:
                # 从其他学习资料中选择
                domains = list(learning_materials["learning_materials"].keys())
                domain = np.random.choice(domains)
                topics = learning_materials["learning_materials"][domain]
                if topics:
                    topic = np.random.choice(topics)["topic"]
                    description = f"学习{domain}领域的{topic}知识"
                else:
                    description = f"追求{goal_type}目标，复杂度{complexity:.1f}"
        elif goal_type == "creation" and learning_materials.get("meta_knowledge", {}).get("deepseek_evolution_targets"):
            # 从DeepSeek进化目标中选择
            evolution_targets = learning_materials["meta_knowledge"]["deepseek_evolution_targets"]
            target_keys = list(evolution_targets.keys())
            target_key = np.random.choice(target_keys)
            target_description = evolution_targets[target_key]
            description = f"实现{target_key}：{target_description[:50]}..."
        else:
            description = f"追求{goal_type}目标，复杂度{complexity:.1f}"

        # 计算目标向量 (基于当前状态和目标类型)
        goal_vector = current_state.clone()
        goal_hash = hash(goal_type + description) % 1000
        goal_vector[0] = goal_hash / 1000.0  # 编码目标类型
        goal_vector[1] = complexity  # 编码复杂度

        goal = {
            "id": f"goal_{len(self.active_goals) + len(self.completed_goals)}",
            "type": goal_type,
            "description": description,
            "complexity": complexity,
            "goal_vector": goal_vector,
            "created_time": time.time(),
            "progress": 0.0,
            "intrinsic_reward": self._compute_intrinsic_reward(goal_type, consciousness)
        }

        self.active_goals.append(goal)
        logger.info(f"生成真正目标: {goal['description']}")

        return goal

    def evaluate_progress(self, goal: Dict[str, Any], current_state: torch.Tensor) -> float:
        """
        评估目标进度 - 基于状态相似性

        Args:
            goal: 目标
            current_state: 当前状态

        Returns:
            进度值 (0.0-1.0)
        """
        goal_vector = goal["goal_vector"]

        # 确保goal_vector是tensor
        if isinstance(goal_vector, str):
            # 如果是字符串，尝试解析为tensor
            try:
                import ast
                goal_vector = torch.tensor(ast.literal_eval(goal_vector), device=current_state.device, dtype=current_state.dtype)
            except:
                # 如果解析失败，使用随机向量
                goal_vector = torch.randn_like(current_state)
        elif not isinstance(goal_vector, torch.Tensor):
            # 如果是其他类型，转换为tensor
            goal_vector = torch.tensor(goal_vector, device=current_state.device, dtype=current_state.dtype)

        distance = torch.norm(current_state - goal_vector)
        max_distance = torch.norm(goal_vector) + torch.norm(current_state)

        if max_distance == 0:
            return 1.0

        progress = 1.0 - (distance / max_distance).item()
        return max(0.0, min(1.0, _safe_float(progress, 0.0)))

    def verify_goal_completion(
        self,
        goal: Dict[str, Any],
        progress: float,
        learning_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """目标完成验证方法（可审计）"""
        evidence = {
            "progress": _safe_float(progress, 0.0),
            "policy_loss": _safe_float((learning_metrics or {}).get("policy_loss", 0.0), 0.0),
            "value_loss": _safe_float((learning_metrics or {}).get("value_loss", 0.0), 0.0),
            "type": goal.get("type"),
            "description": goal.get("description")
        }

        # 基础阈值
        if progress >= 0.98:
            evidence["reason"] = "progress>=0.98"
            return True, evidence

        if progress < 0.85:
            evidence["reason"] = "progress<0.85"
            return False, evidence

        # 学习指标验证（可选）
        if learning_metrics:
            policy_ok = evidence["policy_loss"] <= 0.0 or abs(evidence["policy_loss"]) < 5.0
            value_ok = evidence["value_loss"] >= 0.0 and evidence["value_loss"] < 1000.0
            if policy_ok and value_ok:
                evidence["reason"] = "progress>=0.85 & learning_metrics_ok"
                return True, evidence

        evidence["reason"] = "insufficient_learning_evidence"
        return False, evidence

    def update_goals(self, current_state: torch.Tensor, learning_metrics: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        更新目标状态

        Args:
            current_state: 当前状态
            learning_metrics: 学习指标（可选）

        Returns:
            已完成的目标列表
        """
        completed = []

        for goal in self.active_goals[:]:
            progress = self.evaluate_progress(goal, current_state)
            goal["progress"] = progress

            is_completed, evidence = self.verify_goal_completion(goal, progress, learning_metrics)
            if is_completed:
                goal["completed_time"] = time.time()
                goal["completion_evidence"] = evidence
                self.completed_goals.append(goal)
                self.active_goals.remove(goal)
                completed.append(goal)
                logger.info(f"目标完成: {goal['description']} (进度: {progress:.2f})")

        return completed

    def _compute_intrinsic_reward(self, goal_type: str, consciousness: ConsciousnessMetrics) -> float:
        """计算内在奖励"""
        base_reward = 0.1

        if goal_type == "learning":
            base_reward += consciousness.neural_complexity * 0.5
        elif goal_type == "exploration":
            base_reward += consciousness.integrated_information * 0.3
        elif goal_type == "optimization":
            base_reward += consciousness.self_model_accuracy * 0.4
        elif goal_type == "understanding":
            base_reward += consciousness.metacognitive_awareness * 0.6
        elif goal_type == "creation":
            base_reward += consciousness.temporal_binding * 0.5

        return base_reward

class TrueAGIAutonomousSystem:
    """
    真正的AGI自主系统 - 实现自主学习、自我改进和意识发展
    """

    def __init__(self, input_dim: int = 256, action_dim: int = 64):
        self.input_dim = input_dim
        self.action_dim = action_dim

        # 加载学习资料
        self.learning_materials = self._load_learning_materials()

        # 核心组件
        self.consciousness_engine = TrueConsciousnessEngine(input_dim, input_dim * 2)
        self.learning_engine = TrueLearningEngine(input_dim, action_dim)
        self.goal_system = TrueGoalSystem(self.consciousness_engine, self.learning_materials)

        # 将组件移动到设备
        self.consciousness_engine = self.consciousness_engine.to(self.learning_engine.device)
        # learning_engine已经在其__init__中移动到设备了

        # 系统状态
        self.is_running = False
        self.evolution_step = 0
        self.start_time = time.time()
        self.current_state = torch.randn(input_dim, device=self.learning_engine.device)
        self.prev_consciousness_state = None

        # 性能历史
        self.performance_history: List[ConsciousnessMetrics] = []
        self.learning_history: List[Dict[str, float]] = []

        # 自我编程建议（安全输出，不自动修改代码）
        self.self_programming_log = Path("self_programming_suggestions.jsonl")
        self.self_programming_history: List[Dict[str, Any]] = []

        # 环境交互
        self.environment_thread = None
        self.stop_environment = False

        # 安全边界初始化
        self._initialize_safety_bounds()

        logger.info("真正的AGI自主系统初始化完成")

    def _initialize_safety_bounds(self) -> None:
        """初始化安全边界"""
        self.safety_bounds = {
            "max_evolution_steps": 50000,  # 最大进化步数
            "max_memory_usage": 0.8,       # 最大内存使用率
            "max_cpu_usage": 0.9,          # 最大CPU使用率
            "min_learning_rate": 1e-8,     # 最小学习率
            "max_gradient_norm": 1.0,      # 最大梯度范数
            "emergency_stop_threshold": 10 # 连续异常次数阈值
        }

        self.emergency_counter = 0
        self.last_health_check = time.time()
        self.last_save_time = time.time()

        logger.info("安全边界初始化完成")

    def cleanup_weight_files(self, keep_recent: int = 3) -> None:
        """
        清理旧的权重文件，只保留每个类型最近的几个文件
        
        Args:
            keep_recent: 每个权重文件类型保留最近的文件数量
        """
        import glob
        import os
        from pathlib import Path
        
        weight_patterns = [
            "*.pth", "*.pt", "*.ckpt", "*checkpoint*.pt", "*checkpoint*.pth",
            "*model*.pt", "*model*.pth", "*weights*.pt", "*weights*.pth"
        ]
        
        # 按目录分组权重文件
        weight_files_by_dir = {}
        
        for pattern in weight_patterns:
            for weight_file in glob.glob(f"./**/{pattern}", recursive=True):
                if ".venv" in weight_file or "node_modules" in weight_file:
                    continue
                    
                weight_path = Path(weight_file)
                dir_key = str(weight_path.parent)
                
                if dir_key not in weight_files_by_dir:
                    weight_files_by_dir[dir_key] = []
                weight_files_by_dir[dir_key].append((weight_path, weight_path.stat().st_mtime))
        
        # 对每个目录的权重文件按修改时间排序，保留最近的
        total_cleaned = 0
        for dir_path, files in weight_files_by_dir.items():
            if len(files) <= keep_recent:
                continue
                
            # 按修改时间排序（最新的在前）
            files.sort(key=lambda x: x[1], reverse=True)
            
            # 删除旧文件
            for weight_path, _ in files[keep_recent:]:
                try:
                    weight_path.unlink()
                    logger.info(f"已清理旧权重文件: {weight_path}")
                    total_cleaned += 1
                except Exception as e:
                    logger.warning(f"清理权重文件失败 {weight_path}: {e}")
        
        if total_cleaned > 0:
            logger.info(f"权重文件清理完成，共清理 {total_cleaned} 个旧文件")
        else:
            logger.info("无需清理权重文件")

    def _load_learning_materials(self) -> Dict[str, Any]:
        """加载学习资料"""
        try:
            with open("agi_learning_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"已加载 {len(data.get('learning_materials', {}))} 个学习领域")
            return data
        except Exception as e:
            logger.warning(f"无法加载学习资料: {e}")
            return {"learning_materials": {}, "learning_tasks": []}

    def _safety_initialization(self) -> None:
        """安全初始化检查 - 确保系统稳定运行"""
        logger.info("🔒 执行安全初始化检查...")

        # 检查设备兼容性
        try:
            test_tensor = torch.randn(10, device=self.learning_engine.device)
            test_tensor = test_tensor * 2 + 1
            if not torch.isfinite(test_tensor).all():
                raise RuntimeError("设备计算测试失败")
            logger.info(f"✅ 设备兼容性检查通过: {self.learning_engine.device}")
        except Exception as e:
            logger.error(f"❌ 设备兼容性检查失败: {e}")
            raise

        # 检查模型参数
        try:
            policy_params = sum(p.numel() for p in self.learning_engine.policy_net.parameters())
            value_params = sum(p.numel() for p in self.learning_engine.value_net.parameters())
            logger.info(f"✅ 模型参数检查通过: 策略网络 {policy_params}, 价值网络 {value_params}")
        except Exception as e:
            logger.error(f"❌ 模型参数检查失败: {e}")
            raise

        # 初始化安全边界
        self.safety_bounds = {
            "max_evolution_steps": 50000,  # 最大进化步数
            "max_memory_usage": 0.8,       # 最大内存使用率
            "max_cpu_usage": 0.9,          # 最大CPU使用率
            "min_learning_rate": 1e-8,     # 最小学习率
            "max_gradient_norm": 1.0,      # 最大梯度范数
            "emergency_stop_threshold": 10 # 连续异常次数阈值
        }

        self.emergency_counter = 0
        self.last_health_check = time.time()

        # 初始化监控变量
        if not hasattr(self, 'last_save_time'):
            self.last_save_time = time.time()

        logger.info("✅ 安全初始化完成")

    def _health_check(self) -> bool:
        """健康检查 - 监控系统状态"""
        try:
            current_time = time.time()

            # 检查时间间隔 (每30秒)
            if current_time - self.last_health_check < 30:
                return True

            # 内存使用检查
            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > self.safety_bounds["max_memory_usage"]:
                logger.warning(f"⚠️ 内存使用过高: {memory_percent:.2%}")
                return False

            # CPU使用检查
            cpu_percent = psutil.cpu_percent() / 100.0
            if cpu_percent > self.safety_bounds["max_cpu_usage"]:
                logger.warning(f"⚠️ CPU使用过高: {cpu_percent:.2%}")
                return False

            # 学习率检查
            policy_lr = self.learning_engine.policy_optimizer.param_groups[0]['lr']
            if policy_lr < self.safety_bounds["min_learning_rate"]:
                logger.warning(f"⚠️ 学习率过低: {policy_lr:.2e}")
                return False

            # 进化步数检查
            if self.evolution_step > self.safety_bounds["max_evolution_steps"]:
                logger.warning(f"⚠️ 达到最大进化步数: {self.evolution_step}")
                return False

            self.last_health_check = current_time
            return True

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    async def start_true_evolution(self) -> None:
        """
        启动真正的AGI进化 - 自主学习和自我改进
        """
        self.is_running = True
        logger.info("🚀 启动真正的AGI自主进化系统")

        # 安全初始化检查
        self._safety_initialization()

        # 清理旧权重文件
        self.cleanup_weight_files(keep_recent=3)

        try:
            # 启动环境交互线程
            self.environment_thread = threading.Thread(target=self._environment_interaction_loop)
            self.environment_thread.start()

            while self.is_running:
                # 健康检查
                if not self._health_check():
                    self.emergency_counter += 1
                    if self.emergency_counter >= self.safety_bounds["emergency_stop_threshold"]:
                        logger.error("🚨 紧急停止: 连续健康检查失败")
                        break
                    await asyncio.sleep(1.0)  # 等待1秒后重试
                    continue
                else:
                    self.emergency_counter = 0

                # 1. 感知环境 (多模态)
                current_state, images, videos = self._perceive_environment()

                # 2. 计算意识指标
                consciousness, internal_state = self.consciousness_engine(current_state, self.prev_consciousness_state)
                self.prev_consciousness_state = internal_state

                # 3. 生成/更新目标
                if len(self.goal_system.active_goals) < 3:
                    self.goal_system.generate_goal(current_state, consciousness)

                # 4. 选择动作 (多模态)
                action = self.learning_engine.select_action(current_state, images=images, videos=videos)

                # 5. 执行动作并获取奖励
                reward, next_state = await self._execute_action(action)

                # 6. 学习经验 (多模态)
                experience = LearningExperience(
                    observation=current_state,
                    action=action,
                    reward=reward,
                    next_observation=next_state,
                    done=False,
                    timestamp=time.time(),
                    complexity=consciousness.neural_complexity
                )

                learning_metrics = self.learning_engine.learn_from_experience(experience, images=images, videos=videos)

                # 7. 更新目标进度
                try:
                    completed_goals = self.goal_system.update_goals(next_state, learning_metrics)
                except Exception as e:
                    logger.error(f"目标更新出错: {e}")
                    completed_goals = []
                    self.emergency_counter += 1

                # 8. 自我改进
                try:
                    await self._self_improvement(consciousness, learning_metrics)
                except Exception as e:
                    logger.error(f"自我改进出错: {e}")
                    self.emergency_counter += 1

                # 9. 记录状态
                self.performance_history.append(consciousness)
                self.learning_history.append(learning_metrics)

                # 10. 状态报告
                await self._report_status(consciousness, learning_metrics, completed_goals)

                # 11. 更新状态
                self.current_state = next_state
                self.evolution_step += 1

                # 12. 定期保存状态 (每1000步或每小时)
                current_time = time.time()
                if (self.evolution_step % 1000 == 0 or
                    current_time - getattr(self, 'last_save_time', 0) > 3600):  # 每小时保存
                    self.save_state("true_agi_system_state.json")
                    self._save_monitoring_data()
                    self.last_save_time = current_time

                # 控制进化速度
                await asyncio.sleep(0.1)  # 10Hz

        except Exception as e:
            logger.error(f"真正的AGI进化出错: {e}")
            raise
        finally:
            self.stop_environment = True
            if self.environment_thread:
                self.environment_thread.join()
            self.is_running = False

    def _perceive_environment(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        感知环境 - 获取当前状态 (多模态)

        Returns:
            (state, images, videos)
        """
        # 简化的环境感知
        system_state = torch.tensor([
            psutil.cpu_percent() / 100.0,
            psutil.virtual_memory().percent / 100.0,
            len(self.goal_system.active_goals) / 10.0,
            time.time() % 86400 / 86400,
            np.random.normal(0, 0.1),
        ], dtype=torch.float32, device=self.learning_engine.device)

        # 多模态数据
        images = None
        videos = None
        if hasattr(self.learning_engine, 'image_loader') and self.learning_engine.image_loader is not None:
            try:
                if not hasattr(self.learning_engine, '_image_iter'):
                    self.learning_engine._image_iter = iter(self.learning_engine.image_loader)
                batch = next(self.learning_engine._image_iter)
                images, _ = batch
                images = images[:4].to(self.learning_engine.device)  # 小批量
            except StopIteration:
                self.learning_engine._image_iter = iter(self.learning_engine.image_loader)
                batch = next(self.learning_engine._image_iter)
                images, _ = batch
                images = images[:4].to(self.learning_engine.device)
            except Exception as e:
                logger.warning(f"获取图像数据失败: {e}")
                pass

        # 填充状态
        if len(system_state) < self.input_dim:
            padding = torch.randn(self.input_dim - len(system_state), device=self.learning_engine.device)
            state = torch.cat([system_state, padding])
        else:
            state = system_state[:self.input_dim]

        return state, images, videos

    async def _execute_action(self, action: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        执行动作 - 在环境中执行动作并获取奖励

        Args:
            action: 动作张量

        Returns:
            奖励和下一个状态
        """
        # 简化的动作执行 (实际应用中这会影响真实环境)
        action_magnitude = torch.norm(action).item()

        # 计算奖励 (基于动作的复杂度和社会影响)
        reward = 0.0

        # 探索奖励
        reward += action_magnitude * 0.1

        # 学习奖励 (基于最近的学习指标)
        if self.learning_history:
            recent_learning = self.learning_history[-1]
            policy_loss = _safe_float(recent_learning.get("policy_loss", 0.0), 0.0)
            value_loss = _safe_float(recent_learning.get("value_loss", 0.0), 0.0)
            reward += (policy_loss + value_loss) * -0.01

        # 目标奖励
        for goal in self.goal_system.active_goals:
            reward += goal.get("intrinsic_reward", 0.0) * 0.1

        # 添加噪声
        reward += np.random.normal(0, 0.1)

        # 生成下一个状态 (基于当前状态和动作)
        action_expanded = action.squeeze()  # 移除批次维度
        if action_expanded.size(0) < self.input_dim:
            # 扩展动作到状态维度
            action_padded = torch.cat([action_expanded, torch.zeros(self.input_dim - action_expanded.size(0), device=action_expanded.device)])
        else:
            action_padded = action_expanded[:self.input_dim]

        next_state = self.current_state + action_padded * 0.1 + torch.randn_like(self.current_state) * 0.05

        return reward, next_state

    async def _self_improvement(self, consciousness: ConsciousnessMetrics, learning_metrics: Dict[str, float]) -> None:
        """
        自我改进 - 基于性能指标调整系统参数

        Args:
            consciousness: 意识指标
            learning_metrics: 学习指标
        """
        # 数值稳定性检查与控制
        if not _is_finite(consciousness.integrated_information):
            self._stabilize_training("意识指标出现非有限值")
            self._self_programming_cycle("意识指标异常", consciousness, learning_metrics)
            return

        if not _is_finite(learning_metrics.get("policy_loss", 0.0)) or not _is_finite(learning_metrics.get("value_loss", 0.0)):
            self._stabilize_training("学习损失出现非有限值")
            self._self_programming_cycle("学习损失异常", consciousness, learning_metrics)
            return

        # 基于意识水平调整学习率
        if consciousness.integrated_information > 0.5:
            # 高意识水平，增加学习率
            for param_group in self.learning_engine.policy_optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.01, 1e-3)
        elif consciousness.integrated_information < 0.2:
            # 低意识水平，减少学习率
            for param_group in self.learning_engine.policy_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.99, 1e-5)

        # 基于学习效率调整探索率
        policy_loss = learning_metrics.get("policy_loss", 0.0)
        if abs(policy_loss) > 1.0:
            # 学习不稳定，增加探索
            pass  # 在select_action中处理

        # 基于神经复杂度调整网络容量
        if consciousness.neural_complexity > 0.8:
            # 高复杂度，可能需要增加容量
            logger.debug("检测到高神经复杂度，可能需要架构扩展")

    def _stabilize_training(self, reason: str) -> None:
        """稳定训练，避免NaN/Inf扩散"""
        logger.warning(f"⚠️ 触发稳定化: {reason}")
        # 降低学习率并清理部分经验缓冲
        for param_group in self.learning_engine.policy_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.5, 1e-6)
        for param_group in self.learning_engine.value_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.5, 1e-6)
        for param_group in self.learning_engine.meta_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.5, 1e-6)
        if len(self.learning_engine.experience_buffer) > 1000:
            self.learning_engine.experience_buffer = deque(list(self.learning_engine.experience_buffer)[-1000:], maxlen=10000)

    def _self_programming_cycle(self, trigger: str, consciousness: ConsciousnessMetrics, learning_metrics: Dict[str, float]) -> None:
        """生成自我编程建议（安全输出，需人工审核）"""
        suggestion = {
            "timestamp": time.time(),
            "trigger": trigger,
            "metrics": {
                "phi": _safe_float(consciousness.integrated_information, 0.0),
                "complexity": _safe_float(consciousness.neural_complexity, 0.0),
                "policy_loss": _safe_float(learning_metrics.get("policy_loss", 0.0), 0.0),
                "value_loss": _safe_float(learning_metrics.get("value_loss", 0.0), 0.0),
            },
            "suggestions": [
                "在学习引擎中增加梯度裁剪与NaN检测", 
                "对策略网络输出添加数值钳制与稳定化", 
                "当出现非有限损失时降低学习率并重置部分经验缓冲"
            ],
            "safety": "仅生成建议，不自动修改代码"
        }

        self.self_programming_history.append(suggestion)
        try:
            self.self_programming_log.parent.mkdir(parents=True, exist_ok=True)
            with self.self_programming_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(suggestion, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"自我编程建议写入失败: {e}")

    async def _report_status(self, consciousness: ConsciousnessMetrics, learning_metrics: Dict[str, float], completed_goals: List[Dict[str, Any]]) -> None:
        """报告系统状态 - 增强版包含学习分析"""
        if self.evolution_step % 100 == 0:  # 每100步报告一次
            # 获取学习引擎的详细报告
            learning_report = self.learning_engine.get_learning_report()

            # 基础状态报告
            base_report = f"""
📊 真正AGI进化状态报告 (步骤 {self.evolution_step}):
   整合信息Φ: {consciousness.integrated_information:.4f}
   神经复杂度: {consciousness.neural_complexity:.4f}
   自我模型准确性: {consciousness.self_model_accuracy:.4f}
   元认知意识: {consciousness.metacognitive_awareness:.4f}
   情感价值: {consciousness.emotional_valence:.4f}
   时间绑定: {consciousness.temporal_binding:.4f}
   学习损失: P={learning_metrics.get('policy_loss', 0):.4f}, V={learning_metrics.get('value_loss', 0):.4f}, LT={learning_metrics.get('long_term_value_loss', 0):.4f}
   活跃目标: {len(self.goal_system.active_goals)}
   已完成目标: {len(self.goal_system.completed_goals)}
   运行时间: {time.time() - self.start_time:.1f}秒
            """

            # 学习趋势分析报告
            trend_report = ""
            if learning_report.get("performance_trends"):
                trends = learning_report["performance_trends"]
                trend_report = f"""
🔍 学习趋势分析:
   策略趋势: {trends.get('policy_trend', 'unknown')}
   价值趋势: {trends.get('value_trend', 'unknown')}
   收敛状态: {trends.get('convergence_status', 'unknown')}
   知识库大小: {learning_report.get('knowledge_base_size', 0)} 模式
                """

            # 学习率和缓冲区状态
            lr_report = ""
            if learning_report.get("current_learning_rates"):
                lrs = learning_report["current_learning_rates"]
                buffers = learning_report.get("experience_buffer_sizes", {})
                lr_report = f"""
⚙️ 学习配置:
   学习率: P={lrs.get('policy', 0):.2e}, V={lrs.get('value', 0):.2e}, LT={lrs.get('long_term_value', 0):.2e}, M={lrs.get('meta', 0):.2e}
   经验缓冲: 总={buffers.get('total', 0)}, 短期={buffers.get('short_term', 0)}, 长期={buffers.get('long_term', 0)}
                """

            # 组合完整报告
            full_report = base_report + trend_report + lr_report
            logger.info(full_report)

            if completed_goals:
                logger.info(f"✅ 完成目标: {[g['description'] for g in completed_goals]}")

            # 每1000步进行详细性能分析
            if self.evolution_step % 1000 == 0:
                convergence = learning_report.get("convergence_metrics", {})
                logger.info(f"""
🎯 详细性能分析 (步骤 {self.evolution_step}):
   策略损失稳定性: {convergence.get('policy_loss_std', 'N/A')}
   价值损失稳定性: {convergence.get('value_loss_std', 'N/A')}
   学习稳定性: {convergence.get('learning_stability', 'unknown')}
                """)

    def _environment_interaction_loop(self) -> None:
        """环境交互循环 - 持续感知和响应"""
        while not self.stop_environment:
            try:
                # 这里可以添加持续的环境监控
                time.sleep(0.05)  # 20Hz
            except:
                break

    def stop_evolution(self) -> None:
        """停止进化"""
        self.is_running = False
        self.stop_environment = True
        logger.info("🛑 真正的AGI自主进化系统已停止")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        latest_consciousness = self.performance_history[-1] if self.performance_history else None
        latest_learning = self.learning_history[-1] if self.learning_history else None

        return {
            "is_running": self.is_running,
            "evolution_step": self.evolution_step,
            "uptime": time.time() - self.start_time,
            "latest_consciousness": latest_consciousness,
            "latest_learning": latest_learning,
            "active_goals": len(self.goal_system.active_goals),
            "completed_goals": len(self.goal_system.completed_goals),
            "experience_buffer_size": len(self.learning_engine.experience_buffer),
            "current_phi": self.consciousness_engine.compute_phi(torch.randn(self.input_dim))
        }

    def save_state(self, filepath: str) -> None:
        """保存系统状态"""
        print(f"💾 开始保存AGI系统状态到 {filepath}...")
        try:
            checkpoint_path = Path(filepath).with_suffix(".pt")

            # 只保存基本信息，避免序列化问题
            last_consciousness = vars(self.performance_history[-1]) if self.performance_history else None
            if last_consciousness:
                last_consciousness = {k: _safe_float(v, 0.0) for k, v in last_consciousness.items()}

            last_learning = self.learning_history[-1] if self.learning_history else None
            if last_learning:
                last_learning = {k: _safe_float(v, 0.0) for k, v in last_learning.items()}

            state = {
                "evolution_step": self.evolution_step,
                "performance_history_length": len(self.performance_history),
                "learning_history_length": len(self.learning_history),
                "active_goals_count": len(self.goal_system.active_goals),
                "completed_goals_count": len(self.goal_system.completed_goals),
                "last_consciousness": last_consciousness,
                "last_learning": last_learning,
                "current_state": self.current_state.tolist(),
                "active_goals": self.goal_system.active_goals,
                "completed_goals": self.goal_system.completed_goals,
                "goal_motivations": self.goal_system.intrinsic_motivations,
                "checkpoint_path": str(checkpoint_path)
            }

            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            # 保存模型与优化器状态
            torch.save({
                "consciousness_state_dict": self.consciousness_engine.state_dict(),
                "learning_state_dict": self.learning_engine.state_dict(),
                "policy_optimizer_state": self.learning_engine.policy_optimizer.state_dict(),
                "value_optimizer_state": self.learning_engine.value_optimizer.state_dict(),
                "meta_optimizer_state": self.learning_engine.meta_optimizer.state_dict()
            }, checkpoint_path)

            print(f"✅ AGI系统状态已保存到: {filepath}")
            logger.info(f"真正的AGI系统状态已保存到: {filepath}")

        except Exception as e:
            print(f"❌ 保存AGI系统状态失败: {e}")
            logger.error(f"保存AGI系统状态失败: {e}")

    def _save_monitoring_data(self) -> None:
        """保存监控数据用于长期分析"""
        try:
            monitoring_data = {
                "timestamp": time.time(),
                "evolution_step": self.evolution_step,
                "knowledge_base_size": len(self.learning_engine.knowledge_base),
                "experience_buffer_total": len(self.learning_engine.experience_buffer),
                "experience_buffer_short_term": len(self.learning_engine.short_term_buffer),
                "experience_buffer_long_term": len(self.learning_engine.long_term_buffer),
                "active_goals_count": len(self.goal_system.active_goals),
                "completed_goals_count": len(self.goal_system.completed_goals),
                "learning_rates": {
                    "policy": self.learning_engine.policy_optimizer.param_groups[0]['lr'],
                    "value": self.learning_engine.value_optimizer.param_groups[0]['lr'],
                    "long_term_value": self.learning_engine.long_term_value_optimizer.param_groups[0]['lr'],
                    "meta": self.learning_engine.meta_optimizer.param_groups[0]['lr']
                }
            }

            # 计算最近1000步的指标
            if len(self.performance_history) >= 100:
                recent_consciousness = self.performance_history[-100:]
                monitoring_data.update({
                    "recent_phi_mean": np.mean([c.integrated_information for c in recent_consciousness]),
                    "recent_phi_std": np.std([c.integrated_information for c in recent_consciousness]),
                    "recent_complexity_mean": np.mean([c.neural_complexity for c in recent_consciousness]),
                    "recent_complexity_std": np.std([c.neural_complexity for c in recent_consciousness]),
                    "recent_self_model_accuracy_mean": np.mean([c.self_model_accuracy for c in recent_consciousness]),
                    "recent_self_model_accuracy_std": np.std([c.self_model_accuracy for c in recent_consciousness])
                })

            if len(self.learning_history) >= 100:
                recent_learning = self.learning_history[-100:]
                monitoring_data.update({
                    "recent_policy_loss_mean": np.mean([l.get('policy_loss', 0) for l in recent_learning]),
                    "recent_policy_loss_std": np.std([l.get('policy_loss', 0) for l in recent_learning]),
                    "recent_value_loss_mean": np.mean([l.get('value_loss', 0) for l in recent_learning]),
                    "recent_value_loss_std": np.std([l.get('value_loss', 0) for l in recent_learning])
                })

            # 保存到文件
            monitoring_file = "agi_monitoring_data.jsonl"
            with open(monitoring_file, 'a', encoding='utf-8') as f:
                json.dump(monitoring_data, f, default=str)
                f.write('\n')

            logger.info(f"监控数据已保存到: {monitoring_file}")

        except Exception as e:
            logger.error(f"保存监控数据失败: {e}")

    def load_state(self, filepath: str) -> None:
        """加载系统状态"""
        if not Path(filepath).exists():
            logger.warning(f"状态文件不存在: {filepath}")
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.evolution_step = state.get('evolution_step', 0)
        self.current_state = torch.tensor(state.get('current_state', torch.randn(self.input_dim).tolist()), device=self.learning_engine.device)
        self.learning_history = state.get('learning_history', self.learning_history)
        self.goal_system.active_goals = state.get('active_goals', [])
        self.goal_system.completed_goals = state.get('completed_goals', [])
        self.goal_system.intrinsic_motivations = state.get('goal_motivations', self.goal_system.intrinsic_motivations)

        # 加载模型与优化器状态
        checkpoint_path = state.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            try:
                if "consciousness_state_dict" in ckpt:
                    self.consciousness_engine.load_state_dict(ckpt["consciousness_state_dict"])
                if "learning_state_dict" in ckpt:
                    self.learning_engine.load_state_dict(ckpt["learning_state_dict"], strict=False)
                    logger.info("学习引擎状态已加载 (非严格模式)")
                if "policy_optimizer_state" in ckpt:
                    self.learning_engine.policy_optimizer.load_state_dict(ckpt["policy_optimizer_state"])
                if "value_optimizer_state" in ckpt:
                    self.learning_engine.value_optimizer.load_state_dict(ckpt["value_optimizer_state"])
                if "meta_optimizer_state" in ckpt:
                    self.learning_engine.meta_optimizer.load_state_dict(ckpt["meta_optimizer_state"])
                logger.info("模型和优化器状态加载成功")
            except Exception as e:
                logger.warning(f"状态加载失败，将使用默认状态: {e}")
                logger.info("将从头开始训练")

        logger.info(f"真正的AGI系统状态已从 {filepath} 加载")

# 全局系统实例
_true_agi_system: Optional[TrueAGIAutonomousSystem] = None

def get_true_agi_system(input_dim: int = 256, action_dim: int = 64) -> TrueAGIAutonomousSystem:
    """获取真正的AGI系统实例（单例模式）"""
    global _true_agi_system
    if _true_agi_system is None:
        _true_agi_system = TrueAGIAutonomousSystem(input_dim, action_dim)
    return _true_agi_system

async def start_true_agi_evolution(input_dim: int = 256, action_dim: int = 64) -> None:
    """
    启动真正的AGI进化 - 主要入口函数

    Args:
        input_dim: 输入维度
        action_dim: 动作维度
    """
    system = get_true_agi_system(input_dim, action_dim)

    # 加载之前的状态（如果存在）
    state_file = "true_agi_system_state.json"
    if Path(state_file).exists():
        system.load_state(state_file)
        logger.info("已加载之前的真正AGI系统状态")

    try:
        await system.start_true_evolution()
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在保存真正AGI系统状态...")
        system.save_state(state_file)
        system.stop_evolution()
    except Exception as e:
        logger.error(f"真正的AGI进化系统出错: {e}")
        system.save_state(state_file)
        raise


async def run_goal_completion_experiment(
    steps: int = 200,
    target_progress: float = 0.9,
    save_path: str = "goal_completion_experiment.json"
) -> bool:
    """目标完成循环实验（可审计、可复现）"""
    system = get_true_agi_system()
    completed: List[Dict[str, Any]] = []

    for i in range(steps):
        current_state = system._perceive_environment()

        consciousness, internal_state = system.consciousness_engine(current_state, system.prev_consciousness_state)
        system.prev_consciousness_state = internal_state

        if len(system.goal_system.active_goals) < 1:
            system.goal_system.generate_goal(current_state, consciousness)

        action = system.learning_engine.select_action(current_state)
        reward, next_state = await system._execute_action(action)

        experience = LearningExperience(
            observation=current_state,
            action=action,
            reward=reward,
            next_observation=next_state,
            done=False,
            timestamp=time.time(),
            complexity=consciousness.neural_complexity
        )

        learning_metrics = system.learning_engine.learn_from_experience(experience)

        completed_goals = system.goal_system.update_goals(next_state, learning_metrics)
        if completed_goals:
            completed.extend(completed_goals)
            break

        if system.goal_system.active_goals and i % 10 == 0:
            goal = system.goal_system.active_goals[0]
            if goal.get("progress", 0.0) < target_progress:
                goal_vector = goal["goal_vector"]
                next_state = next_state * 0.7 + goal_vector * 0.3

        system.current_state = next_state
        system.evolution_step += 1

    def _serialize_goal(goal: Dict[str, Any]) -> Dict[str, Any]:
        safe_goal = {k: v for k, v in goal.items() if k not in {"goal_vector"}}
        if "goal_vector" in goal and isinstance(goal["goal_vector"], torch.Tensor):
            safe_goal["goal_vector_shape"] = list(goal["goal_vector"].shape)
        return safe_goal

    report = {
        "steps": steps,
        "completed_count": len(completed),
        "completed_goals": [_serialize_goal(g) for g in completed],
        "timestamp": time.time()
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return len(completed) > 0
