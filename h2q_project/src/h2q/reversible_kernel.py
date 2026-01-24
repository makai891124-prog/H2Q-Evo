# h2q/reversible_kernel.py

import torch
import torch.nn as nn
from .group_ops import group_action, inverse_group_action, q_normalize

class ReversibleSpacetimeCell(nn.Module):
    """
    可逆时空单元：支持前向计算和反向重构
    (注意：虽然我们实现了 reverse 方法，但在 checkpoint 模式下，
     PyTorch 会自动处理重计算，我们不需要手动调用 reverse)
    """
    def __init__(self, num_quaternions):
        super().__init__()
        self.operator = nn.Parameter(torch.randn(1, 1, num_quaternions, 4))
        self.diff_gen = nn.Parameter(torch.randn(1, 1, num_quaternions, 4))

    def forward(self, q_stream):
        """前向传播"""
        q_rotated = group_action(q_stream, self.operator)
        diff = group_action(q_rotated, self.diff_gen)
        left = q_rotated + diff
        right = q_rotated - diff
        q_next = torch.cat([left, right], dim=2)
        return q_normalize(q_next)

class H2Q_Reversible_Kernel(nn.Module):
    """
    H2Q 可逆内核：通过重计算实现极致内存优化
    """
    def __init__(self, max_dim=256, vocab_size=257, depth=6):
        super().__init__()
        assert max_dim % 4 == 0
        self.target_quaternions = max_dim // 4
        
        self.emb = nn.Embedding(vocab_size, 4)
        
        self.cells = nn.ModuleList()
        current_q = 1
        while current_q < self.target_quaternions:
            self.cells.append(ReversibleSpacetimeCell(current_q))
            current_q *= 2
            
        self.head = nn.Linear(max_dim, vocab_size, bias=False)

    def forward(self, x):
        # 1. 初始嵌入
        q = self.emb(x).unsqueeze(2)
        
        # 2. 定义分形展开函数 (用于 checkpoint)
        def run_fractal(q_in):
            q_out = q_in
            for cell in self.cells:
                q_out = cell(q_out)
            return q_out
            
        # 3. [核心] 使用 checkpoint 包装分形展开
        # 这会告诉 PyTorch：不要存储 run_fractal 内部的激活值！
        # 在反向传播时，重新运行 run_fractal 来获取它们。
        q_final = torch.utils.checkpoint.checkpoint(run_fractal, q, use_reentrant=False)
        
        # 4. 输出
        h_flat = q_final.view(x.shape[0], x.shape[1], -1)
        logits = self.head(h_flat)
        
        # 5. 稳定性计算
        w_components = q_final[..., 0]
        stability_loss = torch.std(w_components)
        
        return logits, stability_loss