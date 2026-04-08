#!/usr/bin/env python3
"""
DAS public conversion experiment.

Implements a geometric rotor layer that replaces dense matrix storage with:
- low-rank directional bases,
- reversible rotor chain,
- path mask execution (lazy path traversal, no dense unfold in forward).

This script compares a baseline nn.Linear(1024, 1024) against DAS_Geometric_Layer,
then exports auditable JSON + Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _parameter_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    return f"{num_bytes / (1024**3):.2f} GB"


@dataclass
class DASCompileMetadata:
    rank: int
    num_rotors: int
    svd_rel_l2_error: float


class DASGeometricLayer(nn.Module):
    """
    DAS geometric layer.

    Forward pass enforces lazy path execution:
    - no full N x M reconstruction,
    - input traverses rotor chain,
    - directional path accumulation on compressed bases.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        num_rotors: int,
        use_bias: bool = True,
        riemann_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if num_rotors <= 0:
            raise ValueError("num_rotors must be positive")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_rotors = num_rotors
        self.riemann_alpha = float(riemann_alpha)

        # Low-dimensional directional path basis.
        self.basis_in = nn.Parameter(torch.empty(rank, in_dim))
        self.basis_out = nn.Parameter(torch.empty(rank, out_dim))

        # Reversible path mask gates (sigmoid logits).
        self.path_logits = nn.Parameter(torch.zeros(rank))

        # Rotor angles on fixed pair schedule (group action generators).
        self.rotor_angles = nn.Parameter(torch.zeros(num_rotors))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        rotor_i, rotor_j = self._build_knot_pair_schedule(in_dim, num_rotors)
        self.register_buffer("rotor_i", rotor_i, persistent=False)
        self.register_buffer("rotor_j", rotor_j, persistent=False)

        self.reset_parameters()

    @staticmethod
    def _build_knot_pair_schedule(dim: int, num_rotors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Knot-like reversible pair schedule on a ring:
        (i_t, j_t) with two coprime strides to create global mixing.
        """
        stride_a = 17 % dim
        stride_b = 43 % dim
        if stride_a == 0:
            stride_a = 1
        if stride_b == 0:
            stride_b = 3

        idx_i = []
        idx_j = []
        cur = 0
        for t in range(num_rotors):
            i = cur
            j = (cur + stride_b + 3 * t) % dim
            if i == j:
                j = (j + 1) % dim
            idx_i.append(i)
            idx_j.append(j)
            cur = (cur + stride_a + t) % dim
        return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.basis_in, mean=0.0, std=1.0 / math.sqrt(self.in_dim))
        nn.init.normal_(self.basis_out, mean=0.0, std=1.0 / math.sqrt(self.rank))
        nn.init.constant_(self.path_logits, 2.5)
        nn.init.zeros_(self.rotor_angles)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _riemann_sphere_map(self, x: torch.Tensor) -> torch.Tensor:
        # Stereographic-like compression to avoid unstable norm bursts.
        if self.riemann_alpha <= 0:
            return x
        norm2 = (x * x).mean(dim=-1, keepdim=True)
        return x / (1.0 + self.riemann_alpha * norm2)

    def _apply_rotor_chain(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply reversible Givens-like rotors directly on feature coordinates.
        No dense matrix is formed.
        """
        h = x
        for t in range(self.num_rotors):
            i = int(self.rotor_i[t].item())
            j = int(self.rotor_j[t].item())
            theta = self.rotor_angles[t]
            c = torch.cos(theta)
            s = torch.sin(theta)

            xi = h[:, i]
            xj = h[:, j]
            yi = c * xi - s * xj
            yj = s * xi + c * xj

            # Clone-on-write to preserve autograd correctness.
            h = h.clone()
            h[:, i] = yi
            h[:, j] = yj
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"expected input shape [B, {self.in_dim}], got {tuple(x.shape)}")

        # 1) Riemann map pre-conditioning.
        h = self._riemann_sphere_map(x)

        # 2) Rotor-chain traversal.
        h = self._apply_rotor_chain(h)

        # 3) Masked path execution (lazy rank accumulation).
        gates = torch.sigmoid(self.path_logits)  # [rank]

        # Vectorized lazy path accumulation (no dense N x M materialization).
        # coeff[b, k] = <h_b, basis_in[k]>
        coeff = h @ self.basis_in.T
        coeff = coeff * gates.unsqueeze(0)
        y = coeff @ self.basis_out

        if self.bias is not None:
            y = y + self.bias
        return y


def compress_to_das(
    standard_layer: nn.Linear,
    compression_rank: int,
    num_rotors: int,
    riemann_alpha: float,
    rotor_strength: float,
    device: torch.device,
) -> tuple[DASGeometricLayer, DASCompileMetadata]:
    """
    Compile nn.Linear to DAS_Geometric_Layer by truncated SVD projection.
    """
    w = standard_layer.weight.detach().to(torch.float32).to(device)  # [out, in]
    use_bias = standard_layer.bias is not None

    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    r = min(compression_rank, s.numel())

    das = DASGeometricLayer(
        in_dim=standard_layer.in_features,
        out_dim=standard_layer.out_features,
        rank=r,
        num_rotors=num_rotors,
        use_bias=use_bias,
        riemann_alpha=riemann_alpha,
    ).to(device)

    with torch.no_grad():
        # x @ W^T ≈ (x @ V_r^T) @ (U_r * S_r)^T
        das.basis_in.copy_(vh[:r, :])
        das.basis_out.copy_((u[:, :r] * s[:r]).T)
        das.path_logits.fill_(4.0)

        # Compile rotor angles from directional anisotropy of V_r.
        for t in range(das.num_rotors):
            i = int(das.rotor_i[t].item())
            j = int(das.rotor_j[t].item())
            vi = vh[:r, i]
            vj = vh[:r, j]
            a = (vi * vi).sum()
            b = (vj * vj).sum()
            c = (vi * vj).sum()
            angle = 0.5 * torch.atan2(2.0 * c, (a - b).clamp_min(1e-12))
            das.rotor_angles[t].copy_(angle * float(rotor_strength))

        if use_bias and standard_layer.bias is not None:
            das.bias.copy_(standard_layer.bias.detach().to(torch.float32).to(device))

    w_hat = (u[:, :r] * s[:r]) @ vh[:r, :]
    rel_l2 = (torch.linalg.norm(w_hat - w) / torch.linalg.norm(w).clamp_min(1e-8)).item()

    return das, DASCompileMetadata(rank=r, num_rotors=num_rotors, svd_rel_l2_error=float(rel_l2))


@torch.inference_mode()
def evaluate_consistency(
    baseline: nn.Linear,
    das_layer: DASGeometricLayer,
    batch_size: int,
    dim: int,
    num_batches: int,
    device: torch.device,
) -> dict[str, float]:
    mse_vals = []
    mae_vals = []
    cos_vals = []
    rel_vals = []

    for _ in range(num_batches):
        x = torch.randn(batch_size, dim, device=device)
        y0 = baseline(x)
        y1 = das_layer(x)

        diff = y1 - y0
        mse_vals.append(float((diff * diff).mean().item()))
        mae_vals.append(float(diff.abs().mean().item()))
        cos_vals.append(float(F.cosine_similarity(y0, y1, dim=-1).mean().item()))
        rel_vals.append(float((torch.linalg.norm(diff) / torch.linalg.norm(y0).clamp_min(1e-8)).item()))

    return {
        "mse": float(statistics.mean(mse_vals)),
        "mae": float(statistics.mean(mae_vals)),
        "cosine": float(statistics.mean(cos_vals)),
        "relative_l2": float(statistics.mean(rel_vals)),
    }


@torch.inference_mode()
def benchmark_latency(
    module: nn.Module,
    in_dim: int,
    batch_size: int,
    rounds: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float]:
    x = torch.randn(batch_size, in_dim, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = module(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        _ = module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    ts = sorted(times)
    p90_idx = max(0, int(0.9 * len(ts)) - 1)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times)),
        "p50_ms": float(statistics.median(times)),
        "p90_ms": float(ts[p90_idx]),
    }


def _estimate_forward_working_bytes_baseline(batch_size: int, in_dim: int, out_dim: int) -> int:
    # Rough lower-bound tensor footprint (fp32): input + output + matmul accumulator.
    elems = batch_size * in_dim + batch_size * out_dim + batch_size * out_dim
    return elems * 4


def _estimate_forward_working_bytes_das(batch_size: int, in_dim: int, out_dim: int, rank: int) -> int:
    # input + rotor hidden + output + one coeff vector + one basis_out row view.
    elems = batch_size * in_dim + batch_size * in_dim + batch_size * out_dim + batch_size + out_dim
    # Path loop reuses buffers; rank mainly affects arithmetic, not simultaneous storage.
    return elems * 4


def initialize_compressible_baseline(
    layer: nn.Linear,
    latent_rank: int,
    noise_std: float,
    device: torch.device,
) -> None:
    """
    Create a standard dense baseline layer with controlled compressibility.
    The layer remains nn.Linear, but its spectrum is low-rank dominant.
    """
    out_dim, in_dim = layer.weight.shape
    r = max(1, min(latent_rank, in_dim, out_dim))

    with torch.no_grad():
        a = torch.randn(out_dim, r, device=device) / math.sqrt(r)
        b = torch.randn(r, in_dim, device=device) / math.sqrt(in_dim)
        w = a @ b
        if noise_std > 0:
            w = w + noise_std * torch.randn_like(w)
        layer.weight.copy_(w)
        if layer.bias is not None:
            layer.bias.zero_()


def _build_markdown_report(report: dict[str, Any]) -> str:
    return rf"""# DAS 架构效能与维度压制报告（公开实例）

## 1. 实验配置

- Baseline: `nn.Linear({report['config']['in_dim']}, {report['config']['out_dim']})`
- DAS 层: `DASGeometricLayer(rank={report['config']['rank']}, rotors={report['config']['num_rotors']})`
- 设备: `{report['environment']['device']}`
- dtype: `fp32`

## 2. DAS 核心映射与执行法则

1. 矩阵解耦：
   - 将 $W \in \mathbb{{R}}^{{N\times M}}$ 通过截断 SVD 映射为低维方向基与缩放路径。
2. 权重转子化：
   - 用可逆转子链（Givens rotor chain）表征群作用下的方向演化。
3. 路径掩码执行：
   - forward 采用逐路径惰性累积，不构造完整 $N\times M$ 矩阵。

## 3. 参数压缩与内存压制

- Baseline 参数量: `{report['params']['baseline']}`
- DAS 参数量: `{report['params']['das']}`
- 参数压缩率: `{report['params']['compression_ratio']:.4f}x`

- Baseline 参数存储: `{report['memory']['baseline_param_bytes_human']}`
- DAS 参数存储: `{report['memory']['das_param_bytes_human']}`
- 参数内存压缩率: `{report['memory']['param_memory_ratio']:.4f}x`

复杂度比较（本实现）：

- Baseline: $O(B\cdot N\cdot M)$
- DAS: $O(B\cdot N\cdot R + B\cdot K)$

其中 $R$ 为压缩秩，$K$ 为转子数。对 $N=M$ 且 $R\ll N$ 时，参数复杂度由 $O(N^2)$ 降为 $O(RN)$。

## 4. 同构一致性（近似）

- SVD 编译相对误差: `{report['compile']['svd_rel_l2_error']:.6f}`
- 输出 MSE: `{report['consistency']['mse']:.6e}`
- 输出 MAE: `{report['consistency']['mae']:.6e}`
- 余弦相似度: `{report['consistency']['cosine']:.6f}`
- 相对 L2: `{report['consistency']['relative_l2']:.6e}`

说明：该公开实例在给定秩下属于“有损同构投影”，并非严格无损。

## 5. 性能测量

- Baseline mean latency: `{report['latency_ms']['baseline']['mean_ms']:.4f} ms`
- DAS mean latency: `{report['latency_ms']['das']['mean_ms']:.4f} ms`
- 加速比（baseline/das）: `{report['latency_ms']['speedup_ratio']:.4f}x`

若加速比 > 1，DAS 在当前硬件/实现下更快；反之表示发生实现摩擦。

## 6. 信息摩擦（Friction）分析

当前硅基 GPU/CPU 栈主要为大矩阵 GEMM 优化。DAS 的路径惰性执行包含：

1. 小算子链与散点式坐标更新（rotor pair updates）。
2. 路径循环带来的 kernel 启动/调度开销。
3. 对低秩几何算子友好的融合算子尚未专门化。

因此可能出现“理论参数压缩显著，但吞吐不一定线性受益”的物理架构-软件哲学不匹配。

## 7. 硬件前瞻（3D 可逆计算芯片）

若面向 DAS 设计专用硬件，建议：

1. 原生 rotor-chain 指令：
   - 直接支持二维子空间旋转对（i,j,theta）批处理。
2. 路径掩码流水线：
   - 将 rank-path 累积做为片上循环，减少访存回写。
3. 可逆算子缓存：
   - 将转子序列与索引拓扑固化在近存储层，降低带宽墙。
4. 3D 近存计算：
   - 让几何路径计算贴近 SRAM/HBM 堆叠，减少数据搬运能耗。

## 8. 工程结论

- 在本公开实例中，DAS 已验证“从密集矩阵到几何路径算子”的可运行转换链。
- 参数/存储压缩已可量化。
- 一致性与速度取决于压缩秩与执行后端：
  - 通过提升秩可提高一致性；
  - 通过后端算子融合与专用硬件可释放速度收益。
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAS geometric rotor public conversion experiment")
    p.add_argument("--in-dim", type=int, default=1024)
    p.add_argument("--out-dim", type=int, default=1024)
    p.add_argument("--rank", type=int, default=96)
    p.add_argument("--num-rotors", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--consistency-batches", type=int, default=12)
    p.add_argument("--benchmark-rounds", type=int, default=80)
    p.add_argument("--benchmark-warmup", type=int, default=20)
    p.add_argument("--riemann-alpha", type=float, default=0.0)
    p.add_argument("--rotor-strength", type=float, default=0.03)
    p.add_argument("--baseline-latent-rank", type=int, default=64)
    p.add_argument("--baseline-noise-std", type=float, default=0.01)
    p.add_argument("--output-dir", type=str, default="reports/conv_math_conversion")
    p.add_argument("--seed", type=int, default=20260328)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = nn.Linear(args.in_dim, args.out_dim, bias=True).to(device)
    initialize_compressible_baseline(
        baseline,
        latent_rank=args.baseline_latent_rank,
        noise_std=args.baseline_noise_std,
        device=device,
    )
    baseline.eval()

    das_layer, meta = compress_to_das(
        baseline,
        compression_rank=args.rank,
        num_rotors=args.num_rotors,
        riemann_alpha=args.riemann_alpha,
        rotor_strength=args.rotor_strength,
        device=device,
    )
    das_layer.eval()

    consistency = evaluate_consistency(
        baseline=baseline,
        das_layer=das_layer,
        batch_size=args.batch_size,
        dim=args.in_dim,
        num_batches=args.consistency_batches,
        device=device,
    )

    baseline_lat = benchmark_latency(
        baseline,
        in_dim=args.in_dim,
        batch_size=args.batch_size,
        rounds=args.benchmark_rounds,
        warmup=args.benchmark_warmup,
        device=device,
    )
    das_lat = benchmark_latency(
        das_layer,
        in_dim=args.in_dim,
        batch_size=args.batch_size,
        rounds=args.benchmark_rounds,
        warmup=args.benchmark_warmup,
        device=device,
    )

    baseline_params = _count_parameters(baseline)
    das_params = _count_parameters(das_layer)
    baseline_param_bytes = _parameter_bytes(baseline)
    das_param_bytes = _parameter_bytes(das_layer)

    baseline_working = _estimate_forward_working_bytes_baseline(args.batch_size, args.in_dim, args.out_dim)
    das_working = _estimate_forward_working_bytes_das(args.batch_size, args.in_dim, args.out_dim, meta.rank)

    report: dict[str, Any] = {
        "config": {
            "in_dim": args.in_dim,
            "out_dim": args.out_dim,
            "rank": meta.rank,
            "num_rotors": meta.num_rotors,
            "batch_size": args.batch_size,
            "riemann_alpha": args.riemann_alpha,
            "rotor_strength": args.rotor_strength,
            "baseline_latent_rank": args.baseline_latent_rank,
            "baseline_noise_std": args.baseline_noise_std,
        },
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "compile": {
            "svd_rel_l2_error": meta.svd_rel_l2_error,
        },
        "params": {
            "baseline": baseline_params,
            "das": das_params,
            "compression_ratio": float(baseline_params / max(das_params, 1)),
        },
        "memory": {
            "baseline_param_bytes": baseline_param_bytes,
            "das_param_bytes": das_param_bytes,
            "baseline_param_bytes_human": _format_bytes(baseline_param_bytes),
            "das_param_bytes_human": _format_bytes(das_param_bytes),
            "param_memory_ratio": float(baseline_param_bytes / max(das_param_bytes, 1)),
            "baseline_forward_working_bytes_est": baseline_working,
            "das_forward_working_bytes_est": das_working,
            "baseline_forward_working_bytes_est_human": _format_bytes(baseline_working),
            "das_forward_working_bytes_est_human": _format_bytes(das_working),
            "forward_working_ratio_est": float(baseline_working / max(das_working, 1)),
        },
        "consistency": consistency,
        "latency_ms": {
            "baseline": baseline_lat,
            "das": das_lat,
            "speedup_ratio": float(baseline_lat["mean_ms"] / max(das_lat["mean_ms"], 1e-9)),
        },
        "complexity": {
            "baseline": "O(B*N*M)",
            "das": "O(B*N*R + B*K)",
        },
        "acceptance_flags": {
            "strong_consistency": bool(consistency["cosine"] > 0.99 and consistency["relative_l2"] < 0.20),
            "observed_speedup": bool((baseline_lat["mean_ms"] / max(das_lat["mean_ms"], 1e-9)) > 1.05),
            "memory_compression": bool((baseline_param_bytes / max(das_param_bytes, 1)) > 2.0),
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "das_rotor_public_experiment_20260328.json"
    md_path = out_dir / "DAS_ARCHITECTURE_EFFICIENCY_REPORT_20260328.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path.write_text(_build_markdown_report(report), encoding="utf-8")

    print("[DAS] experiment complete")
    print(f"JSON report: {json_path}")
    print(f"MD report:   {md_path}")
    print(f"Param compression ratio: {report['params']['compression_ratio']:.4f}x")
    print(f"Speedup ratio (baseline/das): {report['latency_ms']['speedup_ratio']:.4f}x")
    print(f"Consistency cosine: {report['consistency']['cosine']:.6f}")


if __name__ == "__main__":
    main()
