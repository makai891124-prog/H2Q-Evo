# h2q/spacetime_3d_kernel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class H2Q_Spacetime3D_Kernel(nn.Module):
    """
    H2Q L0 视觉核 (特征提取器版)
    功能：4维 YCbCr -> 256维 视觉拓扑特征
    """
    def __init__(self, in_dim=4, hidden_dim=256, depth=6): # [核心修正] hidden_dim 默认为 256
        super().__init__()
        
        # 1. 初始投影 (升维)
        self.project_in = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        
        # 2. 深度时空演化
        self.layers = nn.ModuleList([
            ConvBlock(hidden_dim) for _ in range(depth)
        ])
        
        # [核心修正] 移除 project_out。内核的职责是输出高维特征。

    def forward(self, q_img):
        # q_img: [B, H, W, 4]
        
        # 转换维度: [B, 4, H, W]
        x = q_img.permute(0, 3, 1, 2)
        
        # 投影到 256 维特征空间
        x = self.project_in(x)
        
        # 演化
        for layer in self.layers:
            x = x + layer(x) # 残差连接
            
        # 输出高维特征图: [B, 256, H, W]
        return x