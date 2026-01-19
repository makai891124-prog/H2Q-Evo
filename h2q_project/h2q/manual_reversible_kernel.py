# h2q/manual_reversible_kernel.py

import torch
import torch.nn as nn
from .group_ops import group_action, inverse_group_action, q_normalize

class ManualReversibleCellFunction(torch.autograd.Function):
    """
    手动实现可逆层的 autograd 功能
    """
    @staticmethod
    def forward(ctx, q_stream, operator, diff_gen):
        # ctx: 上下文对象，用于存储反向传播需要的信息
        
        # [关键] 在 no_grad 环境下执行，不保存中间激活
        with torch.no_grad():
            q_rotated = group_action(q_stream, operator)
            diff = group_action(q_rotated, diff_gen)
            left = q_rotated + diff
            right = q_rotated - diff
            q_next = torch.cat([left, right], dim=2)
            q_next = q_normalize(q_next)
        
        # 我们只保存输入和权重，用于反向重计算
        ctx.save_for_backward(q_stream, operator, diff_gen)
        
        return q_next

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: 从上一层传来的梯度
        
        # 1. 恢复输入和权重
        q_stream, operator, diff_gen = ctx.saved_tensors
        
        # 2. [核心] 重新计算前向传播，但这次是在 autograd 环境下
        # 这样 PyTorch 就能追踪操作，以便我们计算对权重的梯度
        with torch.enable_grad():
            # 确保输入也需要梯度，以便链式法则回传
            q_stream_clone = q_stream.detach().requires_grad_(True)
            operator_clone = operator.detach().requires_grad_(True)
            diff_gen_clone = diff_gen.detach().requires_grad_(True)
            
            q_rotated = group_action(q_stream_clone, operator_clone)
            diff = group_action(q_rotated, diff_gen_clone)
            left = q_rotated + diff
            right = q_rotated - diff
            q_next = torch.cat([left, right], dim=2)
            q_next = q_normalize(q_next)

        # 3. 使用重计算的结果进行反向传播
        # 这会计算出 q_next 相对于 q_stream_clone, operator_clone, diff_gen_clone 的梯度
        q_next.backward(grad_output)
        
        # 返回的梯度必须与 forward 的输入一一对应
        return q_stream_clone.grad, operator_clone.grad, diff_gen_clone.grad

class ManualReversibleCell(nn.Module):
    def __init__(self, num_quaternions):
        super().__init__()
        self.operator = nn.Parameter(torch.randn(1, 1, num_quaternions, 4))
        self.diff_gen = nn.Parameter(torch.randn(1, 1, num_quaternions, 4))

    def forward(self, q_stream):
        # 调用我们自定义的 autograd 函数
        return ManualReversibleCellFunction.apply(q_stream, self.operator, self.diff_gen)

class H2Q_Manual_Kernel(nn.Module):
    """
    H2Q 手动可逆内核
    """
    def __init__(self, max_dim=256, vocab_size=257, depth=6):
        super().__init__()
        assert max_dim % 4 == 0
        self.target_quaternions = max_dim // 4
        
        self.emb = nn.Embedding(vocab_size, 4)
        
        self.cells = nn.ModuleList()
        current_q = 1
        while current_q < self.target_quaternions:
            self.cells.append(ManualReversibleCell(current_q))
            current_q *= 2
            
        self.head = nn.Linear(max_dim, vocab_size, bias=False)

    def forward(self, x):
        q = self.emb(x).unsqueeze(2)
        
        for cell in self.cells:
            q = cell(q)
            
        h_flat = q.view(x.shape[0], x.shape[1], -1)
        logits = self.head(h_flat)
        
        w_components = q[..., 0]
        stability_loss = torch.std(w_components)
        
        return logits, stability_loss