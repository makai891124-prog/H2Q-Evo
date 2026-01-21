"""Rolling stream loader (optional, low-intrusion).

Derived from H2Q-MicroStream's RollingWheelLoader concept:
- Stream raw Unicode/byte data without a tokenizer.
- Supports chunked validation->train loops (rolling horizon).
- Minimal dependencies (built-in urllib); no new install required.

Usage pattern (example):
    loader = RollingStreamLoader(
        file_path="data_tinystories/TinyStories-train.txt",
        auto_download_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        chunk_size_mb=10,
        batch_size=24,
        seq_len=128,
        device="mps"
    )
    next_chunk = loader.load_next_chunk_tensor()

This module is opt-in and does not alter existing pipelines.
"""

from __future__ import annotations

import os
import sys
import math
import urllib.request
from typing import Optional

import torch

__all__ = ["RollingStreamLoader"]


class RollingStreamLoader:
    """Byte-level streaming loader with rolling horizon support."""

    def __init__(
        self,
        file_path: str,
        *,
        auto_download_url: Optional[str] = None,
        chunk_size_mb: int = 10,
        batch_size: int = 24,
        seq_len: int = 128,
        device: Optional[str] = None,
        resume_offset: int = 0,
        encoding: str = "utf-8",
    ) -> None:
        self.file_path = file_path
        self.auto_download_url = auto_download_url
        self.chunk_size = chunk_size_mb * 1024 * 1024
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device(device) if device else torch.device("cpu")
        self.encoding = encoding

        self._prepare_data()
        self.file = open(self.file_path, "r", encoding=self.encoding, errors="ignore")
        self.current_offset = resume_offset
        if resume_offset > 0:
            self.file.seek(resume_offset)

    # ------------------------------------------------------------------
    def _prepare_data(self) -> None:
        directory = os.path.dirname(self.file_path) or "."
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.file_path):
            return
        if not self.auto_download_url:
            print(f"❌ 数据文件缺失且未提供 auto_download_url: {self.file_path}")
            sys.exit(1)
        try:
            print(f"📦 正在下载数据集 -> {self.file_path}")
            with urllib.request.urlopen(self.auto_download_url) as resp, open(
                self.file_path, "wb"
            ) as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            print("✅ 下载完成")
        except Exception as exc:  # pragma: no cover
            print(f"❌ 下载失败: {exc}")
            sys.exit(1)

    # ------------------------------------------------------------------
    def load_next_chunk_tensor(self) -> Optional[torch.Tensor]:
        try:
            text = self.file.read(self.chunk_size)
        except Exception:
            return None

        if not text:
            # rewind for cyclic streaming
            self.file.seek(0)
            self.current_offset = 0
            text = self.file.read(self.chunk_size)

        if not text:
            return None

        self.current_offset += len(text)
        data_list = [ord(ch) if ord(ch) < 256 else 32 for ch in text]
        data = torch.tensor(data_list, dtype=torch.long, device=self.device)

        # reshape into batch-first stream blocks
        num_batches = len(data) // (self.seq_len * self.batch_size)
        valid_len = num_batches * self.seq_len * self.batch_size
        if valid_len == 0:
            return None

        trimmed = data[:valid_len].view(self.batch_size, num_batches, self.seq_len)
        # return contiguous block [batch, steps, seq_len]
        return trimmed.contiguous()

    # ------------------------------------------------------------------
    def iter_rolling(self):
        """Yield successive chunks indefinitely (useful for rolling horizon)."""
        while True:
            chunk = self.load_next_chunk_tensor()
            if chunk is None:
                break
            yield chunk

    # ------------------------------------------------------------------
    def get_bookmark(self) -> int:
        return self.current_offset

    # ------------------------------------------------------------------
    def decode(self, tensor: torch.Tensor) -> str:
        flat = tensor.view(-1).tolist()
        return "".join(chr(i) if i > 0 else "" for i in flat)

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_tokens(chunk_size_mb: int, seq_len: int, batch_size: int) -> int:
        bytes_total = chunk_size_mb * 1024 * 1024
        return math.floor(bytes_total / seq_len) * seq_len * batch_size
