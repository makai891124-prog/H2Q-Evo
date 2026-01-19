# h2q/spacetime_kernel.py

import torch
import torch.nn as nn
from .group_ops import group_action, q_normalize

class SpacetimeCell(nn.Module):
    """
    时空分形单元 (Spacetime Fractal Cell)
    
    功能：
    1. 接收上一层级的四元数流。
    2. 通过可学习的“群算子”进行时空旋转 (Group Exchange)。
    3. 分裂为 [正相, 反相] 两个波包，实现维度翻倍。
    """
    def __init__(self, num_quaternions):
        super().__init__()
        # 每一个四元数通道都有一个独立的旋转算子
        # 形状: [1, 1, num_quaternions, 4]
        self.operator = nn.Parameter(torch.randn(1, 1, num_quaternions, 4))
        
        # 扰动生成器 (用于制造分形差异)
        self.diff_gen = nn.Parameter(torch.randn(1, 1, num_quaternions, 4))

    def forward(self, q_stream):
        # q_stream: [Batch, Seq, N_Q, 4]
        
        # 1. 群交换 (Group Exchange) - 信息的“旋转”
        # 这模拟了粒子在时空中的演化
        q_rotated = group_action(q_stream, self.operator)
        
        # 2. 生成相位差异 (Phase Difference)
        # 这是一个微小的扰动，用于推动分形生长
        diff = group_action(q_rotated, self.diff_gen)
        
        # 3. 分形展开 (Fractal Expansion)
        # 类似于波的干涉：
        # 左支: 波峰叠加 (Constructive)
        # 右支: 波谷叠加 (Destructive)
        left = q_rotated + diff
        right = q_rotated - diff
        
        # 维度翻倍: [B, S, N_Q * 2, 4]
        q_next = torch.cat([left, right], dim=2)
        
        # 归一化 (保持能量守恒)
        return q_normalize(q_next)

class H2Q_Spacetime_Kernel(nn.Module):
    """
    H2Q 四维时空波形内核
    
    结构：
    L0 (1Q): 基础时空点 (4维)
    L1 (2Q): 对偶 (8维)
    L2 (4Q): 相互作用 (16维)
    ...
    L6 (64Q): 宏观概念 (256维)
    """
    def __init__(self, max_dim=256, vocab_size=257):
        super().__init__()
        assert max_dim % 4 == 0
        self.target_quaternions = max_dim // 4
        
        # 1. 奇点 (Singularity): 1个四元数
        self.emb = nn.Embedding(vocab_size, 4)
        
        # 2. 分形时空树 (The Spacetime Tree)
        self.cells = nn.ModuleList()
        current_q = 1
        while current_q < self.target_quaternions:
            self.cells.append(SpacetimeCell(current_q))
            current_q *= 2
            
        # 3. 观测头 (Observer)
        # 将最终的波函数坍缩为概率分布
        self.head = nn.Linear(max_dim, vocab_size, bias=False)

    def forward(self, x):
        # x: [B, S]
        
        # 1. 注入时空
        q = self.emb(x) # [B, S, 4]
        q = q.unsqueeze(2) # [B, S, 1, 4]
        
        # 2. 分形演化
        # 这是一个动态的、递归的过程
        for cell in self.cells:
            q = cell(q)
            
        # q 现在是 [B, S, 64, 4] (256维)
        
        # 3. 观测 (展平)
        h_flat = q.view(x.shape[0], x.shape[1], -1)
        logits = self.head(h_flat)
        
        # 4. 计算波形稳定性 (Waveform Stability)
        # 我们希望波形是“纯粹”的，即实部 w (时间分量) 趋于稳定
        # 或者我们可以计算所有四元数的“共线性”，代表波的相干性
        w_components = q[..., 0]
        stability_loss = torch.std(w_components) # 波动越小越稳定
        
        return logits, stability_loss