# h2q/cem.py

import torch
import torch.nn as nn

class ContinuousEnvironmentModel(nn.Module):
    """
    连续环境模型 (CEM)。
    
    通过一个神经网络学习并表示环境的压力函数 μ(E)。
    """
    
    def __init__(self, energy_dim: int = 1, hidden_dim: int = 64):
        """
        Args:
            energy_dim (int): 输入能量的维度（通常为1）。
            hidden_dim (int): 神经网络隐藏层的维度。
        """
        super().__init__()
        
        # 将 μ(E) 参数化为一个小型神经网络
        self.mu_net = nn.Sequential(
            nn.Linear(energy_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保输出 μ(E) 永远是正数，因为压力/成本不能为负
        )
        
    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        """
        对于给定的能量E，预测环境压力μ(E)。
        
        Args:
            energy (torch.Tensor): Shape [B, 1] or [B]
        
        Returns:
            torch.Tensor: Shape [B, 1]
        """
        # 确保输入是二维的 [B, 1]
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)
            
        return self.mu_net(energy)