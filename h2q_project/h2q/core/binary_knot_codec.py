#!/usr/bin/env python3
"""
二进制流核心纽结再编码器

目标：
- 直接将token ids映射为二进制位向量
- 通过低秩投影形成“纽结式”特征
- 保持自然解码：位向量可逆还原为token ids

遵循M24真实性原则：
- 编码/解码路径为确定性计算
- 不对外宣称超出可验证的能力
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn


class BinaryKnotReEncoder(nn.Module):
    """
    二进制编码流核心纽结再编码器

    token -> bits -> low-rank knot -> hidden
    """

    def __init__(
        self,
        vocab_size: int,
        bit_width: int = 16,
        knot_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bit_width = bit_width

        bit_weights = 2 ** torch.arange(bit_width, dtype=torch.long)
        self.register_buffer("bit_weights", bit_weights)

        self.bit_project = nn.Linear(bit_width, knot_dim)
        self.knot_project = nn.Linear(knot_dim, hidden_dim)

        # 门控默认关闭，保证与既有编码兼容
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def encode_bits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token ids -> bit向量 (B,S,bit_width)"""
        ids = input_ids.to(torch.long)
        bits = (ids.unsqueeze(-1) >> torch.arange(self.bit_width, device=ids.device)) & 1
        return bits.to(torch.float32)

    def decode_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """bit向量 -> token ids（自然解码）"""
        if bits.dtype != torch.long:
            bits = (bits > 0.5).to(torch.long)
        weights = self.bit_weights.to(bits.device)
        ids = torch.sum(bits * weights, dim=-1)
        return ids.clamp_min(0).clamp_max(self.vocab_size - 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """输出二进制纽结特征 (B,S,hidden_dim)"""
        gate = torch.tanh(self.fusion_gate)
        if torch.abs(gate).item() < 1e-6:
            return torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                self.knot_project.out_features,
                device=input_ids.device,
                dtype=torch.float32,
            )

        bits = self.encode_bits(input_ids)
        knot = torch.relu(self.bit_project(bits))
        hidden = torch.relu(self.knot_project(knot))
        return hidden * gate


def binary_knot_enabled() -> bool:
    return os.getenv("H2Q_ENABLE_BINARY_KNOT", "0") == "1"
