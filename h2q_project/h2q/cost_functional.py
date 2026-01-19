# h2q/cost_functional.py

import torch
import torch.nn as nn
from typing import Dict, Callable

class AutonomyCost(nn.Module):
    """
    实现: Cost_auto = -1/2 * t * log(t) + sum(η) + ∫μ(E)dE
    
    所有张量操作都应支持批量处理。
    """
    
    def __init__(self, epsilon=1e-9):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, 
                # 离散决策点的ID，假设是一个整数列表或张量
                decision_indices: torch.Tensor,      # Shape: [B, num_decisions]
                # 包含每个决策点ID及其对应谱位移值的字典
                eta_map: Dict[int, float],
                # 一个可调用对象，输入能量E（张量），返回μ(E)（张量）
                mu_func: Callable[[torch.Tensor], torch.Tensor],
                # 当前时间t，一个浮点数
                t: float) -> torch.Tensor:
        
        # 获取当前批次的设备
        batch_size = decision_indices.shape[0]
        device = decision_indices.device
        
        # 1. 散射成本 (Scattering Cost): -1/2 * t * log(t)
        # 这是一个标量，但我们将其扩展为批次大小，以便后续相加
        t_tensor = torch.tensor(t, device=device)
        scattering_cost = -0.5 * t_tensor * torch.log(t_tensor + self.epsilon)
        scattering_cost = scattering_cost.expand(batch_size) # Shape: [B]

        # 2. 学习迹 (Learning Trace): sum of spectral shifts at decision points
        # 我们需要为批次中的每个样本计算其迹
        trace_cost = torch.zeros(batch_size, device=device) # Shape: [B]
        for i in range(batch_size):
            sample_trace = 0.0
            for decision_idx in decision_indices[i]:
                # .item() 将0维张量转为python数字
                idx = decision_idx.item()
                if idx in eta_map:
                    sample_trace += eta_map[idx]
            trace_cost[i] = sample_trace

        # 3. 环境拖拽 (Environmental Drag): integrated measure
        # 使用梯形法则近似计算积分 ∫μ(E)dE
        # 注意：这是一个简化。在真实场景中，积分的上限可能与t或能量有关
        E_vals = torch.linspace(0, t, 100, device=device) # 积分采样点
        mu_vals = mu_func(E_vals) # 计算每个采样点的μ值
        # trapz的第一个参数是y，第二个是x
        drag_cost_scalar = torch.trapz(mu_vals, E_vals)
        drag_cost = drag_cost_scalar.expand(batch_size) # Shape: [B]
        
        total_cost = scattering_cost + trace_cost + drag_cost
        return total_cost # Shape: [B]
    # 在 h2q/cost_functional.py 中添加

class SpectralShiftFunction(nn.Module):
    """
    实现 η(λ) = (1/π) * arg{det(S(λ))}
    
    我们将 S(λ) 建模为一个将上下文映射到散射矩阵的神经网络。
    """
    
    def __init__(self, context_dim: int, matrix_dim: int):
        super().__init__()
        self.matrix_dim = matrix_dim
        # 这个网络将上下文向量映射成一个方阵
        self.context_to_matrix = nn.Linear(context_dim, matrix_dim * matrix_dim)
        # 我们可以加上一些非线性层
        # nn.init.orthogonal_(self.scattering_matrix.weight) # 初始化很重要

    def forward(self, decision_context: torch.Tensor) -> torch.Tensor:
        """
        计算给定上下文的谱位移。
        
        Args:
            decision_context: Shape [B, context_dim]
        
        Returns:
            eta: Shape [B]
        """
        # 1. 从上下文生成散射矩阵 S
        flat_matrix = self.context_to_matrix(decision_context) # Shape: [B, matrix_dim*matrix_dim]
        S = flat_matrix.view(-1, self.matrix_dim, self.matrix_dim) # Shape: [B, D, D]
        
        # 2. 计算 det(S) 的相位
        # torch.linalg.slogdet 返回 (sign, logabsdet)
        # 我们需要复数行列式，所以先确保S是复数
        S_complex = S.to(torch.complex64)
        det_S = torch.linalg.det(S_complex) # Shape: [B]
        
        # 3. 计算相位角 (arg)
        phase = torch.angle(det_S) # Shape: [B]
        
        # 4. 根据定义返回谱位移
        eta = (1.0 / torch.pi) * phase
        return eta