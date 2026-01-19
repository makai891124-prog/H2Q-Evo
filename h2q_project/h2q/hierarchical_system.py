# h2q/hierarchical_system.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .knot_kernel import H2Q_Knot_Kernel
from .spacetime_3d_kernel import H2Q_Spacetime3D_Kernel

class H2Q_Hierarchical_System(nn.Module):
    def __init__(self, vocab_size=257, dim=256):
        super().__init__()
        # L0 文本核
        self.text_l0 = H2Q_Knot_Kernel(max_dim=dim, vocab_size=vocab_size, depth=6)
        # L0 视觉核
        self.vision_l0 = H2Q_Spacetime3D_Kernel(hidden_dim=dim, depth=6)
        
        # 8:1 压缩器
        self.seq_pool = nn.Conv1d(dim, dim, kernel_size=8, stride=8, groups=dim//4)
        self.img_pool = nn.Conv2d(dim, dim, kernel_size=8, stride=8, groups=dim//4)
        
        # L1 概念层
        self.concept_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x, is_vision=False):
        if is_vision:
            # x: [B, H, W, 4]
            # feat: [B, 256, H, W]
            feat = self.vision_l0(x)
            # c: [B, (H/8*W/8), 256]
            c = self.img_pool(feat).flatten(2).transpose(1, 2).contiguous()
        else:
            # x: [B, S]
            # feat: [B, S, 256]
            feat, _ = self.text_l0(x, return_features=True)
            # c: [B, S/8, 256]
            c = self.seq_pool(feat.transpose(1, 2)).transpose(1, 2).contiguous()
        
        c = F.normalize(c, p=2, dim=-1)
        
        h = c
        for layer in self.concept_layers:
            h = F.normalize(layer(h), p=2, dim=-1) + h
            h = F.normalize(h, p=2, dim=-1)
            
        return self.head(h), c