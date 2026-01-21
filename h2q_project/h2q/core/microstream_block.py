"""Optional microstream blocks inspired by H2Q-MicroStream.

This module is *opt-in* and non-disruptive to existing pipelines. It provides:
- BalancedHamiltonLayer: energy-symmetric linear transform with low-rank factors.
- QuaternionAttention: lightweight quaternion-aware attention.
- MicroStreamBlock: reversible-style block that composes the above with LayerNorm.

All components stay CPU/MPS/CUDA compatible and rely only on torch + existing
h2q.quaternion_ops utilities.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from h2q.quaternion_ops import quaternion_mul, quaternion_normalize

__all__ = [
    "BalancedHamiltonLayer",
    "QuaternionAttention",
    "MicroStreamBlock",
]


class BalancedHamiltonLayer(nn.Module):
    """Low-rank Hamiltonian-style mixer.

    Args:
        dim: feature dimension (should be divisible by 4 for quaternion blocks).
        rank: number of low-rank factors (defaults to 8 per MicroStream design).
        dropout: optional dropout after mixing.
    """

    def __init__(self, dim: int, rank: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # factors: [rank, dim, dim] orthogonal init scaled by (r+1)^-0.5
        self.factors = nn.Parameter(torch.empty(rank, dim, dim))
        with torch.no_grad():
            for r in range(rank):
                nn.init.orthogonal_(self.factors[r])
                self.factors[r] *= (r + 1) ** -0.5

        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, d = x.shape
        assert d == self.dim, "dim mismatch"
        x_flat = x.view(b * t, d)
        mixed = 0
        for f in self.factors:
            mixed = mixed + torch.matmul(x_flat, f)
        mixed = mixed / math.sqrt(self.rank)
        mixed = mixed + self.bias
        mixed = mixed.view(b, t, d)
        return self.dropout(mixed)

    def ortho_loss(self) -> torch.Tensor:
        # Encourage factors to remain orthogonal
        losses = []
        for f in self.factors:
            gram = torch.matmul(f, f.transpose(-1, -2))
            eye = torch.eye(f.shape[-1], device=f.device, dtype=f.dtype)
            losses.append(F.mse_loss(gram, eye))
        return torch.stack(losses).mean()


class QuaternionAttention(nn.Module):
    """Simplified quaternion attention.

    Interprets the last dimension as stacked quaternions (chunks of 4) and
    applies dot-product attention in quaternion space. The attention scores use
    the real part of the Hamilton product to stay numerically stable.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % 4 == 0, "dim should be a multiple of 4 for quaternion blocks"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, "head_dim must align to quaternion packing"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim)

        # reshape to quaternion blocks
        def to_quat(tensor):
            return tensor.view(b, t, self.num_heads, self.head_dim // 4, 4)

        q_q = quaternion_normalize(to_quat(q))
        k_q = quaternion_normalize(to_quat(k))
        v_q = to_quat(v)

        # attention logits from real part of q * k_conj
        k_conj = k_q.clone()
        k_conj[..., 1:] = -k_conj[..., 1:]
        logits = quaternion_mul(q_q, k_conj)[..., 0]  # real component
        scale = math.sqrt(self.head_dim // 4)
        attn = torch.softmax(logits / scale, dim=1)  # over time dimension
        attn = self.dropout(attn)

        # weighted sum in quaternion space, then flatten
        v_weighted = (attn.unsqueeze(-1) * v_q).sum(dim=1)
        v_weighted = v_weighted.view(b, self.num_heads, self.head_dim)
        out = self.out_proj(v_weighted.view(b, self.dim))
        return out.view(b, 1, self.dim).expand(b, t, self.dim)

    def ortho_loss(self) -> torch.Tensor:
        # lightly encourage projection matrices to stay near-orthogonal
        loss = 0
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            w = proj.weight
            gram = torch.matmul(w, w.transpose(0, 1))
            eye = torch.eye(gram.shape[0], device=w.device, dtype=w.dtype)
            loss = loss + F.mse_loss(gram, eye)
        return loss / 4


class MicroStreamBlock(nn.Module):
    """Reversible-style block: LN -> QuaternionAttention -> LN -> Hamilton mix."""

    def __init__(
        self,
        dim: int,
        *,
        rank: int = 8,
        attn_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.half = dim // 2
        self.n1 = nn.LayerNorm(self.half)
        self.attn = QuaternionAttention(self.half, num_heads=attn_heads, dropout=dropout)
        self.n2 = nn.LayerNorm(self.half)
        self.mix1 = BalancedHamiltonLayer(self.half, rank=rank, dropout=dropout)
        self.act = getattr(F, activation) if hasattr(F, activation) else F.gelu
        self.mix2 = BalancedHamiltonLayer(self.half, rank=rank, dropout=dropout)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.n1(x))

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix2(self.act(self.mix1(self.n2(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reversible pattern: (x1, x2) -> (x1 + f(x2), x2 + g(x1))
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(x1)
        return torch.cat([y1, y2], dim=-1)

    def aux_losses(self) -> torch.Tensor:
        return 0.5 * (self.attn.ortho_loss() + self.mix1.ortho_loss() + self.mix2.ortho_loss())
