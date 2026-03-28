#!/usr/bin/env python3
"""
Readable loader for DAS token structure package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class DASRotorProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, num_rotors: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_rotors = num_rotors

        self.basis_in = nn.Parameter(torch.empty(rank, in_dim))
        self.basis_out = nn.Parameter(torch.empty(rank, out_dim))
        self.path_logits = nn.Parameter(torch.zeros(rank))
        self.rotor_angles = nn.Parameter(torch.zeros(num_rotors))

        self.register_buffer("rotor_i", torch.arange(num_rotors) % in_dim, persistent=False)
        self.register_buffer("rotor_j", (torch.arange(num_rotors) * 3 + 1) % in_dim, persistent=False)

    def _apply_rotors(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for t in range(self.num_rotors):
            i = int(self.rotor_i[t].item())
            j = int(self.rotor_j[t].item())
            th = self.rotor_angles[t]
            c = torch.cos(th)
            s = torch.sin(th)
            xi = h[..., i]
            xj = h[..., j]
            yi = c * xi - s * xj
            yj = s * xi + c * xj
            h = h.clone()
            h[..., i] = yi
            h[..., j] = yj
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._apply_rotors(x)
        coeff = h @ self.basis_in.T
        coeff = coeff * torch.sigmoid(self.path_logits).unsqueeze(0)
        return coeff @ self.basis_out


class DASQKVHead(nn.Module):
    def __init__(self, in_dim: int, head_dim: int, rank: int, num_rotors: int) -> None:
        super().__init__()
        self.q = DASRotorProjector(in_dim, head_dim, rank, num_rotors)
        self.k = DASRotorProjector(in_dim, head_dim, rank, num_rotors)
        self.v = DASRotorProjector(in_dim, head_dim, rank, num_rotors)


class DASTokenMapper(nn.Module):
    def __init__(self, hidden_dim: int, token_table_size: int, rank: int, num_rotors: int) -> None:
        super().__init__()
        self.projector = DASRotorProjector(hidden_dim, token_table_size, rank, num_rotors)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden)


class DASTokenStructure:
    def __init__(self, package: dict[str, Any], device: torch.device) -> None:
        self.package = package
        self.device = device

        qcfg = package["qkv_config"]
        tcfg = package["token_mapper_config"]

        self.qkv = DASQKVHead(
            in_dim=int(qcfg["in_dim"]),
            head_dim=int(qcfg["head_dim"]),
            rank=int(qcfg["rank"]),
            num_rotors=int(qcfg["num_rotors"]),
        ).to(device)
        self.token_mapper = DASTokenMapper(
            hidden_dim=int(tcfg["hidden_dim"]),
            token_table_size=int(tcfg["token_table_size"]),
            rank=int(tcfg["rank"]),
            num_rotors=int(tcfg["num_rotors"]),
        ).to(device)

        self.qkv.load_state_dict(package["qkv_state_dict"])
        self.token_mapper.load_state_dict(package["token_mapper_state_dict"])
        self.token_ids = package["token_ids"].to(device)

        self.qkv.eval()
        self.token_mapper.eval()

    @torch.inference_mode()
    def map_token_logits_subset(self, hidden_final: torch.Tensor) -> torch.Tensor:
        return self.token_mapper(hidden_final)

    @torch.inference_mode()
    def map_token_logits_full(self, hidden_final: torch.Tensor, vocab_size: int) -> torch.Tensor:
        subset = self.map_token_logits_subset(hidden_final)
        out = torch.full((hidden_final.shape[0], vocab_size), -1e9, dtype=subset.dtype, device=subset.device)
        out[:, self.token_ids] = subset
        return out



def load_das_token_structure(path: Path, device: torch.device) -> DASTokenStructure:
    pkg = torch.load(path, map_location=device)
    return DASTokenStructure(pkg, device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load and inspect DAS token structure")
    p.add_argument("--structure", type=str, required=True)
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    structure_path = Path(args.structure)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    das = load_das_token_structure(structure_path, device)

    summary = {
        "structure": str(structure_path),
        "device": str(device),
        "model_id": das.package.get("model_id", "unknown"),
        "layer_idx": das.package.get("layer_idx"),
        "head_idx": das.package.get("head_idx"),
        "token_table_size": int(das.token_ids.numel()),
        "qkv_config": das.package.get("qkv_config", {}),
        "token_mapper_config": das.package.get("token_mapper_config", {}),
        "format_version": das.package.get("format_version"),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
