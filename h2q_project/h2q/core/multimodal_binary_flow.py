#!/usr/bin/env python3
"""
Multimodal Binary Flow Encoder

将图像/视频的二维位置 + 亮度(depth) + 颜色(流形坐标)映射为二进制流式签名，
用于统一架构中的闭环控制与多模态表征。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultimodalBinaryFlowEncoder(nn.Module):
    def __init__(self, dim: int = 256, bit_width: int = 8, flow_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.bit_width = bit_width
        self.flow_dim = flow_dim

        # 低秩投影到统一签名空间
        self.flow_conv = nn.Conv2d(5 * bit_width, flow_dim, kernel_size=1)
        self.image_project = nn.Linear(flow_dim, dim)
        self.video_project = nn.Linear(flow_dim, dim)
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def _make_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=0)  # [2, H, W]

    def _rgb_to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.cat([y, cb, cr], dim=1)

    def _quantize_bits(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]
        xq = torch.clamp(x, 0, 1)
        scaled = (xq * (2 ** self.bit_width - 1)).round().to(torch.long)
        bits = (scaled.unsqueeze(-1) >> torch.arange(self.bit_width, device=x.device)) & 1
        return bits.float()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,3,H,W]
        b, _, h, w = images.shape
        device = images.device
        grid = self._make_grid(h, w, device)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # [B,2,H,W]

        ycbcr = self._rgb_to_ycbcr(images)
        luma = ycbcr[:, 0:1]  # depth z
        chroma = ycbcr[:, 1:3]

        # 组合：位置(2) + 亮度(1) + 颜色(2) => 5通道
        flow = torch.cat([grid, luma, chroma], dim=1)
        flow = F.avg_pool2d(flow, 4)  # 下采样减少开销

        bits = self._quantize_bits((flow + 1) / 2)  # [B,5,H,W,bit]
        bits = bits.permute(0, 1, 4, 2, 3).contiguous()
        bits = bits.view(b, 5 * self.bit_width, bits.shape[-2], bits.shape[-1])
        # 使用卷积映射到固定通道，再全局池化
        feats = self.flow_conv(bits)
        pooled = feats.mean(dim=[2, 3])
        return pooled

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        # videos: [B,T,3,H,W]
        b, t, c, h, w = videos.shape
        device = videos.device
        vids = videos.view(b * t, c, h, w)
        feats = self.encode_image(vids)  # [B*T, flow_dim]
        feats = feats.view(b, t, -1)
        # 时间一致性：取均值 + 位移差
        mean_feat = feats.mean(dim=1)
        if t > 1:
            delta = (feats[:, 1:] - feats[:, :-1]).mean(dim=1)
        else:
            delta = torch.zeros_like(mean_feat)
        return mean_feat + 0.5 * delta

    def forward(self, images: Optional[torch.Tensor] = None, videos: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        img_sig = None
        vid_sig = None
        if images is not None:
            img_feat = self.encode_image(images)
            img_sig = self.image_project(img_feat)
        if videos is not None:
            vid_feat = self.encode_video(videos)
            vid_sig = self.video_project(vid_feat)
        return img_sig, vid_sig

    def fuse_signature(self, img_sig: Optional[torch.Tensor], vid_sig: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        gate = torch.tanh(self.fusion_gate)
        if img_sig is None and vid_sig is None:
            return None
        if img_sig is None:
            return gate * vid_sig
        if vid_sig is None:
            return gate * img_sig
        return gate * (img_sig + vid_sig) * 0.5
