# h2q/hierarchical_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuaternionLinear(nn.Module):
    """四元数线性层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.r_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.i_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.j_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.k_weight = nn.Parameter(torch.randn(out_features, in_features))
        
        scale = (in_features) ** -0.5
        for w in [self.r_weight, self.i_weight, self.j_weight, self.k_weight]:
            nn.init.uniform_(w, -scale, scale)
        
    def forward(self, q_in):
        B, S, D = q_in.shape
        # 兼容非四元数倍数的维度 (虽然理论上不应该发生)
        if D % 4 != 0:
            return F.linear(q_in, self.r_weight) # 降级为普通线性
            
        n_quaternions = D // 4
        q = q_in.view(B, S, n_quaternions, 4)
        r, i, j, k = torch.unbind(q, dim=-1)
        
        out_r = F.linear(r, self.r_weight) - F.linear(i, self.i_weight) - F.linear(j, self.j_weight) - F.linear(k, self.k_weight)
        out_i = F.linear(r, self.i_weight) + F.linear(i, self.r_weight) + F.linear(j, self.k_weight) - F.linear(k, self.j_weight)
        out_j = F.linear(r, self.j_weight) - F.linear(i, self.k_weight) + F.linear(j, self.r_weight) + F.linear(k, self.i_weight)
        out_k = F.linear(r, self.k_weight) + F.linear(i, self.j_weight) - F.linear(j, self.i_weight) + F.linear(k, self.r_weight)
        
        out = torch.stack([out_r, out_i, out_j, out_k], dim=-1)
        return out.view(B, S, -1)

class KnotRefiner(nn.Module):
    """纽结精炼块"""
    def __init__(self, dim):
        super().__init__()
        self.q_dim = dim // 4
        self.linear = QuaternionLinear(self.q_dim, self.q_dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = self.act(x)
        return self.norm(residual + x)

class FractalStage(nn.Module):
    """分形展开阶段：1 -> 2"""
    def __init__(self, dim):
        super().__init__()
        self.upsampler = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.refiner = KnotRefiner(dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.upsampler(x)
        x = x.transpose(1, 2)
        x = self.refiner(x)
        return x

class ConceptDecoder(nn.Module):
    """
    H2Q 分形纽结解码器 (支持 Stride=1 直通模式)
    """
    def __init__(self, dim=256, vocab_size=257, stride=8):
        super().__init__()
        self.dim = dim
        self.stride = stride
        
        # [修正] 支持 Stride=1 (用于 L0 预训练) 和 Stride=8 (用于 L1 分形)
        if stride == 8:
            self.stage1 = FractalStage(dim)
            self.stage2 = FractalStage(dim)
            self.stage3 = FractalStage(dim)
        elif stride == 1:
            # 直通模式：只加一个精炼层，不进行上采样
            self.identity_refiner = KnotRefiner(dim)
        else:
            raise ValueError(f"不支持的 Stride: {stride}. 仅支持 1 或 8.")
        
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, concept_vector):
        h = concept_vector
        
        if self.stride == 8:
            # 分形展开 1->8
            h = self.stage1(h)
            h = self.stage2(h)
            h = self.stage3(h)
        else:
            # 直通模式 1->1
            h = self.identity_refiner(h)
            
        logits = self.head(h)
        return logits