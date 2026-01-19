# h2q/fractal_embedding.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalEmbedding(nn.Module):
    """
    H2Q 分形嵌入 (Fractal Embedding)
    
    原理：
    不直接学习 256 维的向量，而是从 2 维核心开始，通过递归的“对称性破缺”展开。
    
    数学过程：
    1. 初始态 (Dim=2): [存在, 反存在]
    2. 展开算子 T(v): v -> [v + δ, v - δ]
       其中 δ (Delta) 是该层级生成的“差异信息”。
       如果 δ=0，则信息完全复用（对称性保持）；
       如果 δ!=0，则产生分化（对称性破缺）。
    
    层级：2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
    """
    def __init__(self, vocab_size=257, target_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.target_dim = target_dim
        
        # 1. 核心种子 (The Seed): 2维
        # 代表最基本的二元对立 (阴/阳, 0/1)
        self.seed_emb = nn.Embedding(vocab_size, 2)
        
        # 2. 差异生成器 (Innovations)
        # 每一层负责生成“差异”，推动维度翻倍
        self.expanders = nn.ModuleList()
        
        current_dim = 2
        while current_dim < target_dim:
            # 这是一个非线性变换，用于从当前状态生成“变化量”
            # 使用 Tanh 限制差异的幅度，保证数值稳定性
            layer = nn.Sequential(
                nn.Linear(current_dim, current_dim),
                nn.Tanh() 
            )
            self.expanders.append(layer)
            current_dim *= 2
            
        print(f"🌌 [Fractal] 分形树构建完成: 2 -> ... -> {target_dim} (共 {len(self.expanders)} 次分裂)")

    def forward(self, x):
        # x: [Batch, Seq]
        
        # 1. 种子萌发
        h = self.seed_emb(x) # [B, S, 2]
        
        # 2. 递归展开 (Recursive Expansion)
        for expander in self.expanders:
            # 计算差异项 delta
            delta = expander(h)
            
            # 分裂与复用：
            # 左支：继承 + 变异
            # 右支：继承 - 变异
            # 这种结构强制模型保留上一层级的“中心特征”
            h = torch.cat([h + delta, h - delta], dim=-1)
            
        return h