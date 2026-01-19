# h2q/knot_kernel.py

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
        
        # 初始化优化
        scale = (in_features) ** -0.5
        for w in [self.r_weight, self.i_weight, self.j_weight, self.k_weight]:
            nn.init.uniform_(w, -scale, scale)
        
    def forward(self, q_in):
        r, i, j, k = torch.unbind(q_in, dim=-1)
        out_r = F.linear(r, self.r_weight) - F.linear(i, self.i_weight) - F.linear(j, self.j_weight) - F.linear(k, self.k_weight)
        out_i = F.linear(r, self.i_weight) + F.linear(i, self.r_weight) + F.linear(j, self.k_weight) - F.linear(k, self.j_weight)
        out_j = F.linear(r, self.j_weight) - F.linear(i, self.k_weight) + F.linear(j, self.r_weight) + F.linear(k, self.i_weight)
        out_k = F.linear(r, self.k_weight) + F.linear(i, self.j_weight) - F.linear(j, self.i_weight) + F.linear(k, self.r_weight)
        return torch.stack([out_r, out_i, out_j, out_k], dim=-1)

def quaternion_normalize(q):
    norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    return q / (norm + 1e-8)

class H2Q_Knot_Kernel(nn.Module):
    """
    H2Q 纽结内核 (底层拼写核) - 修正版
    """
    def __init__(self, max_dim=256, vocab_size=257, depth=6):
        super().__init__()
        assert max_dim % 4 == 0
        self.q_dim = max_dim // 4
        
        self.emb = nn.Embedding(vocab_size, 4) 
        
        self.expanders = nn.ModuleList()
        current_q = 1
        while current_q < self.q_dim:
            self.expanders.append(QuaternionLinear(current_q, current_q))
            current_q *= 2 
            
        self.layers = nn.ModuleList([
            QuaternionLinear(self.q_dim, self.q_dim) for _ in range(depth)
        ])
        
        self.head = nn.Linear(max_dim, vocab_size, bias=False)

    def forward(self, x, return_features=False):
        # x: [B, S]
        q = self.emb(x) 
        q = torch.unsqueeze(q, 2) 
        
        # 分形展开
        for expander in self.expanders:
            delta = torch.tanh(expander(q))
            q = torch.cat([q + delta, q - delta], dim=2)
            
        # 深度演化
        for layer in self.layers:
            q = layer(q)
            q = quaternion_normalize(q)
            
        # [关键修正] 始终计算稳定性损失
        # w 分量代表实部，我们希望它趋近于 0 (纯虚四元数/纯空间纽结)
        w_components = q[..., 0]
        stability_loss = torch.mean(torch.abs(w_components))
            
        # 展平: [B, S, max_dim]
        h_flat = q.view(x.shape[0], x.shape[1], -1)
        
        if return_features:
            # 现在返回的是 Tensor 类型的 loss，可以进行反向传播了
            return h_flat, stability_loss

        logits = self.head(h_flat)
        
        return logits, stability_loss