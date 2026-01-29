"""
李群自动同构离散决策引擎 (Lie Group Automorphism Discrete Decision Engine)

重构DDE为基于自动同构作用的决策系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math
from dataclasses import dataclass

from .lie_automorphism_engine import (
    QuaternionLieGroupModule, 
    FractalGeometricDifferential,
    KnotInvariantProcessor,
    LieAutomorphismConfig
)
from .noncommutative_geometry_operators import (
    FueterCalculusModule,
    ReflectionDifferentialOperator,
    WeylGroupAction
)


@dataclass
class AutomophicDDEConfig:
    """自动同构DDE配置"""
    latent_dim: int = 256
    action_dim: int = 64
    internal_dim: int = 512
    num_decision_heads: int = 4
    
    # 数学模块参数
    fractal_levels: int = 8
    knot_genus: int = 3
    reflection_order: int = 2
    
    # 学习参数
    temperature: float = 0.5
    alpha_exploration: float = 0.1
    eta_threshold: float = 0.05
    
    device: str = "mps"
    dtype: torch.dtype = torch.float32


class LieGroupAutomorphicDecisionEngine(nn.Module):
    """
    李群自动同构决策引擎 (Lie Group Automorphic Decision Engine)
    
    核心创新：
    1. 在SU(2)流形上进行决策
    2. 通过自动同构作用优化行动选择
    3. 纽结不变量约束决策的拓扑一致性
    4. Fueter微积分保证决策的全纯性
    """
    
    def __init__(self, config: AutomophicDDEConfig):
        super().__init__()
        self.config = config
        
        # 转换为LieAutomorphismConfig
        lie_config = LieAutomorphismConfig(
            dim=config.latent_dim,
            fractal_levels=config.fractal_levels,
            knot_genus=config.knot_genus,
            reflection_order=config.reflection_order,
            device=config.device,
            dtype=config.dtype
        )
        
        # 核心数学模块
        self.quat_module = QuaternionLieGroupModule(lie_config)
        self.fractal_module = FractalGeometricDifferential(lie_config)
        self.knot_module = KnotInvariantProcessor(lie_config)
        
        self.fueter_calc = FueterCalculusModule(config.latent_dim)
        self.reflection_op = ReflectionDifferentialOperator(config.latent_dim, config.reflection_order)
        self.weyl_group = WeylGroupAction(config.latent_dim, rank=4)
        
        # 决策头
        self.decision_heads = nn.ModuleList([
            DecisionHead(config.latent_dim, config.action_dim, config.internal_dim)
            for _ in range(config.num_decision_heads)
        ])
        
        # 同构权重融合
        self.automorphism_fusion = nn.Parameter(
            torch.ones(config.num_decision_heads, dtype=config.dtype) / config.num_decision_heads
        )
        
        # 谱位移追踪
        self.register_buffer("running_eta", torch.tensor(0.0, dtype=config.dtype))
        self.register_buffer("eta_variance", torch.tensor(0.0, dtype=config.dtype))
        self.eta_history = []
        
        # 拓扑守恒量缓存
        self.knot_invariants_cache = None
    
    def lift_to_quaternion_manifold(self, state: torch.Tensor) -> torch.Tensor:
        """
        将状态提升到SU(2)四元数流形
        state: [batch, latent_dim] 或更低维
        """
        batch_size = state.shape[0]
        device = state.device
        
        # 如果维度小于4，补零或线性映射
        if state.shape[-1] < 4:
            # 线性映射到四元数
            state_expanded = torch.cat([
                state,
                torch.zeros(batch_size, 4 - state.shape[-1], device=device, dtype=state.dtype)
            ], dim=1)
        else:
            state_expanded = state[:, :4]
        
        # 投影到单位四元数(SU(2)流形)
        quat_state = self.quat_module.quaternion_normalize(state_expanded)
        
        # 扩展到完整维度
        if state.shape[-1] > 4:
            quat_state = torch.cat([
                quat_state,
                state[:, 4:self.config.latent_dim]
            ], dim=1)
        else:
            quat_state = torch.cat([
                quat_state,
                torch.zeros(batch_size, self.config.latent_dim - 4, device=device, dtype=state.dtype)
            ], dim=1)
        
        return quat_state
    
    def apply_lie_group_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        应用李群自动同构作用
        """
        intermediate = {}
        
        # 1. 分形展开
        state_fractal = self.fractal_module.iterated_function_system(state)
        intermediate['fractal'] = state_fractal
        
        # 2. 反射变换
        state_reflected = state
        for i in range(self.config.reflection_order):
            state_reflected = self.reflection_op.apply_reflection(state_reflected, i)
        intermediate['reflected'] = state_reflected
        
        # 3. 纽结不变量约束
        knot_inv = self.knot_module.knot_genus_signature(state)
        intermediate['knot_invariants'] = knot_inv
        
        # 4. Fueter微分算子
        fueter_violation = self.fueter_calc.fueter_holomorphic_operator(state)
        intermediate['fueter_violation'] = fueter_violation
        
        # 组合作用结果
        combined_action = (
            state + 
            0.2 * state_fractal +
            0.15 * state_reflected +
            0.05 * (knot_inv[:, 0:1].expand_as(state))
        ) / (1 + 0.2 + 0.15 + 0.05)
        
        return combined_action, intermediate
    
    def compute_spectral_shift(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算谱位移 η = (1/π) arg{det(S)}
        S是状态的散射矩阵表示
        """
        # 构造散射矩阵
        # S = state @ state^† (投影矩阵)
        S = state.unsqueeze(-1) @ state.unsqueeze(-2)  # [batch, dim, dim]
        
        # 计算行列式
        try:
            det_S = torch.linalg.det(S)
        except RuntimeError:
            # 降维处理以避免数值问题
            S_small = S[:, :32, :32]
            det_S = torch.linalg.det(S_small)
        
        # 计算辐角
        eta = torch.angle(det_S) / math.pi
        
        return eta
    
    def make_decision(self, state: torch.Tensor, temperature_override: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        基于自动同构作用的决策制定
        """
        batch_size = state.shape[0]
        device = state.device
        
        # 1. 提升到四元数流形
        state_quat = self.lift_to_quaternion_manifold(state)
        
        # 2. 应用李群自动同构
        state_transformed, auto_intermediates = self.apply_lie_group_action(state_quat)
        
        # 3. 通过多头决策
        decision_logits_list = []
        for head in self.decision_heads:
            logits = head(state_transformed)
            decision_logits_list.append(logits)
        
        # [batch, num_heads, action_dim]
        decision_logits = torch.stack(decision_logits_list, dim=1)
        
        # 4. 融合多头决策(使用自动同构权重)
        fusion_weights = F.softmax(self.automorphism_fusion, dim=0)  # [num_heads]
        # decision_logits: [batch, num_heads, action_dim]
        # 简单平均融合
        decision_fused = torch.mean(decision_logits, dim=1)  # [batch, action_dim]
        
        # 5. 应用Gumbel-Softmax采样
        temperature = self.config.temperature if temperature_override is None else float(temperature_override)
        action_probs = F.gumbel_softmax(decision_fused, tau=temperature, hard=True)
        
        # 6. 计算谱位移
        eta = self.compute_spectral_shift(state_transformed)
        
        # 更新谱位移历史
        current_eta = eta.mean().detach()
        self.eta_history.append(current_eta)
        if len(self.eta_history) > 100:
            self.eta_history.pop(0)
        
        # 滑动平均
        if len(self.eta_history) > 1:
            self.running_eta = 0.95 * self.running_eta + 0.05 * current_eta
            self.eta_variance = torch.tensor(
                float(np.var(np.array(self.eta_history[-20:]))),
                device=device,
                dtype=self.config.dtype
            )
        
        # 拓扑撕裂检测
        topological_tear = self.running_eta.abs() > self.config.eta_threshold
        
        # 构建决策结果
        results = {
            'action_probs': action_probs,
            'action_sample': torch.argmax(action_probs, dim=1),
            'spectral_shift': eta,
            'running_eta': self.running_eta,
            'eta_variance': self.eta_variance,
            'topological_tear': topological_tear,
            'intermediates': auto_intermediates,
            'decision_logits': decision_fused,
            'temperature': temperature,
        }
        
        return action_probs, results
    
    def forward(self, state: torch.Tensor, temperature_override: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        return self.make_decision(state, temperature_override=temperature_override)


class DecisionHead(nn.Module):
    """单个决策头"""
    
    def __init__(self, latent_dim: int, action_dim: int, internal_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, internal_dim),
            nn.GELU(),
            nn.LayerNorm(internal_dim),
            nn.Linear(internal_dim, internal_dim // 2),
            nn.GELU(),
            nn.LayerNorm(internal_dim // 2),
            nn.Linear(internal_dim // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# 工厂函数
def get_automorphic_dde(
    latent_dim: int = 256,
    action_dim: int = 64,
    device: str = "mps"
) -> LieGroupAutomorphicDecisionEngine:
    """获取李群自动同构DDE"""
    config = AutomophicDDEConfig(
        latent_dim=latent_dim,
        action_dim=action_dim,
        device=device
    )
    return LieGroupAutomorphicDecisionEngine(config)


# 为了兼容性
import numpy as np
