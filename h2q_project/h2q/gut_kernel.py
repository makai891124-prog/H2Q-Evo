# h2q/gut_kernel.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fractal_embedding import FractalEmbedding # 导入分形模块

class ProjectiveNorm(nn.Module):
    """射影归一化：只保留方向"""
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

class AxiomaticLayer(nn.Module):
    """公理化层：因果卷积 + 门控"""
    def __init__(self, dim, dilation):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=0, dilation=dilation)
        self.pad_size = (3 - 1) * dilation
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)
        self.trans = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x_t = x.transpose(1, 2) 
        x_t = F.pad(x_t, (self.pad_size, 0))
        x_t = self.conv(x_t)
        x_t = x_t.transpose(1, 2)
        
        t = self.trans(x)
        g = torch.sigmoid(self.gate(x + x_t))
        return self.norm(residual + t * g)

class H2Q_Geometric_Kernel(nn.Module):
    """
    GUT 内核 v3.0 (Fractal Edition)
    
    特性：
    1. 输入端：使用 FractalEmbedding (2->256 动态展开)
    2. 中间层：12层 AxiomaticLayer (循环扩张)
    3. 输出端：射影归一化 + 权重解绑 (因为输入是分形的，输出头需要独立)
    """
    def __init__(self, dim=256, vocab_size=257, depth=12):
        super().__init__()
        self.dim = dim
        
        # [升级] 使用分形嵌入
        # 这将强制模型学习数据的层次化结构
        self.emb = FractalEmbedding(vocab_size=vocab_size, target_dim=dim)
        
        # 构造性层次
        dilations = [2**(i % 6) for i in range(depth)]
        self.layers = nn.ModuleList([
            AxiomaticLayer(dim, dilation=d) for d in dilations
        ])
        
        self.projector = ProjectiveNorm()
        
        # 输出头
        # 注意：由于 Embedding 是分形构造的，不再是一个简单的矩阵，
        # 所以这里我们解绑权重，让 Head 独立学习如何将 256 维几何投影回 257 个符号
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        # 1. 分形展开 (2 -> 256)
        h = self.emb(x)
        
        # 2. 几何流转
        for layer in self.layers:
            h_new = layer(h)
            h = h + h_new
            
        # 3. 射影预测
        h_projected = self.projector(h)
        logits = self.head(h_projected)
        
        return logits, h_projected