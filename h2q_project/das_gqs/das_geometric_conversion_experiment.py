from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autocompute import (
    apply_rotor_scalar,
    apply_rotor_staged,
    apply_runtime_knobs,
    autotune_callable,
    build_nonoverlap_stages,
    cache_lookup,
    cache_store,
    choose_rotor_kernel,
    default_thread_candidates,
    detect_hardware_profile,
    load_compute_plan_cache,
    make_cache_key,
    pack_stage_steps,
    parse_thread_candidates,
    profile_as_dict,
    resolve_device,
    save_compute_plan_cache,
    try_compile_module,
)


@dataclass
class LayerStats:
    params: int
    param_bytes: int
    param_mebibytes: float


@dataclass
class ForwardMetrics:
    device: str
    batch_size: int
    repeat: int
    baseline_mean_ms: float
    das_mean_ms: float
    speedup_x: float
    baseline_peak_mem_bytes: int | None
    das_peak_mem_bytes: int | None


@dataclass
class QualityMetrics:
    rank: int
    mse: float
    mae: float
    max_abs: float
    rel_fro_error: float


@dataclass
class RiemannSphereMetrics:
    mean_geodesic_rad: float
    p95_geodesic_rad: float
    interaction_rel_fro_error: float


@dataclass
class ComputeMetrics:
    device_request: str
    resolved_device: str
    rotor_kernel: str
    compiled: bool
    matmul_precision: str
    selected_torch_threads: int
    cache_hit: bool
    cache_key: str
    autotune_timings_ms: dict[str, float] | None


class DASGeometricLayer(nn.Module):
    """
    Rotorized low-memory layer:

    y = ((x @ in_paths^T) * scales) --rotor_chain--> latent @ out_paths + bias

    The rotor chain is executed directly on the latent channels and never unfolds
    to a full in_features x out_features dense matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        rotor_steps: int,
        bias: bool,
        rotor_kernel: str = "scalar",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.rotor_kernel = rotor_kernel

        self.in_paths = nn.Parameter(torch.empty(rank, in_features, dtype=dtype))
        self.out_paths = nn.Parameter(torch.empty(rank, out_features, dtype=dtype))
        self.scales = nn.Parameter(torch.ones(rank, dtype=dtype))
        self.rotor_angles = nn.Parameter(torch.zeros(rotor_steps, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("rotor_pairs", self._build_rotor_pairs(rank, rotor_steps), persistent=False)
        rotor_stages = build_nonoverlap_stages(self.rotor_pairs)
        stage_steps, stage_lengths = pack_stage_steps(rotor_stages, device=self.rotor_pairs.device)
        self.register_buffer("rotor_stage_steps", stage_steps, persistent=False)
        self.register_buffer("rotor_stage_lengths", stage_lengths, persistent=False)

    @staticmethod
    def _build_rotor_pairs(rank: int, rotor_steps: int) -> torch.Tensor:
        if rotor_steps <= 0 or rank < 2:
            return torch.empty((0, 2), dtype=torch.long)

        pairs: list[tuple[int, int]] = []
        i = 0
        j = 1
        for _ in range(rotor_steps):
            pairs.append((i, j))
            i = (i + 1) % rank
            j = (j + 2) % rank
            if i == j:
                j = (j + 1) % rank
        return torch.tensor(pairs, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # O(batch * in * rank)
        latent = x.matmul(self.in_paths.t())
        latent = latent * self.scales

        # Lazy rotor execution: rotate only the selected latent channel pairs.
        if self.rotor_kernel == "staged":
            latent = apply_rotor_staged(
                latent,
                self.rotor_pairs,
                self.rotor_angles,
                self.rotor_stage_steps,
                self.rotor_stage_lengths,
            )
        else:
            latent = apply_rotor_scalar(latent, self.rotor_pairs, self.rotor_angles)

        # O(batch * rank * out)
        out = latent.matmul(self.out_paths)
        if self.bias is not None:
            out = out + self.bias
        return out


def compress_to_das(
    standard_layer: nn.Linear,
    compression_rank: int,
    rotor_steps: int = 8,
    refine_steps: int = 120,
    refine_batch: int = 256,
    refine_lr: float = 2e-2,
    structure_lambda: float = 0.15,
    interaction_lambda: float = 0.05,
    rotor_kernel: str = "scalar",
    seed: int = 7,
) -> DASGeometricLayer:
    """
    Compile a trained dense layer into DAS path form using SVD.

    W ~= U_r * diag(S_r) * V_r^T

    Then map to DAS parameters:
    in_paths <- V_r^T
    scales   <- S_r
    out_paths<- U_r^T

    Optional small refinement tunes rotor angles and path coefficients on sampled
    inputs without ever constructing the full dense matrix in forward.
    """
    W = standard_layer.weight.detach().to(torch.float32)
    out_features, in_features = W.shape

    rank = max(1, min(compression_rank, in_features, out_features))
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    das = DASGeometricLayer(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        rotor_steps=max(0, rotor_steps),
        bias=standard_layer.bias is not None,
        rotor_kernel=rotor_kernel,
        dtype=W.dtype,
    )

    with torch.no_grad():
        das.in_paths.data.copy_(Vh[:rank, :])
        das.scales.data.copy_(S[:rank])
        das.out_paths.data.copy_(U[:, :rank].t())
        if standard_layer.bias is not None and das.bias is not None:
            das.bias.data.copy_(standard_layer.bias.detach().to(torch.float32))

    if refine_steps > 0:
        _refine_das(
            das,
            standard_layer,
            steps=refine_steps,
            batch_size=refine_batch,
            lr=refine_lr,
            structure_lambda=structure_lambda,
            interaction_lambda=interaction_lambda,
            seed=seed,
        )

    return das


def _refine_das(
    das: DASGeometricLayer,
    baseline: nn.Linear,
    steps: int,
    batch_size: int,
    lr: float,
    structure_lambda: float,
    interaction_lambda: float,
    seed: int,
) -> None:
    baseline = baseline.eval()
    das.train()

    torch.manual_seed(seed)
    opt = torch.optim.AdamW(das.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))

    for _ in range(steps):
        x = torch.randn(batch_size, baseline.in_features, dtype=torch.float32)
        # A mixed input distribution improves low-rank fit robustness on long-tailed activations.
        if (_ % 3) == 0:
            x = x + 0.3 * torch.randn_like(x) * torch.sign(torch.randn_like(x))
        with torch.no_grad():
            y_ref = baseline(x)
        y_hat = das(x)
        loss_mse = F.mse_loss(y_hat, y_ref)

        loss_geo = y_hat.new_tensor(0.0)
        loss_inter = y_hat.new_tensor(0.0)
        if structure_lambda > 0.0 or interaction_lambda > 0.0:
            s_ref = _euclidean_to_riemann_sphere(y_ref)
            s_hat = _euclidean_to_riemann_sphere(y_hat)
            dots = torch.sum(s_ref * s_hat, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            loss_geo = torch.acos(dots).mean()

            if interaction_lambda > 0.0:
                m = min(64, s_ref.shape[0])
                ref_g = s_ref[:m].matmul(s_ref[:m].t())
                hat_g = s_hat[:m].matmul(s_hat[:m].t())
                loss_inter = F.mse_loss(hat_g, ref_g)

        loss = loss_mse + structure_lambda * loss_geo + interaction_lambda * loss_inter
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

    das.eval()


@torch.no_grad()
def _layer_stats(layer: nn.Module) -> LayerStats:
    params = sum(p.numel() for p in layer.parameters())
    bytes_total = sum(p.numel() * p.element_size() for p in layer.parameters())
    return LayerStats(
        params=int(params),
        param_bytes=int(bytes_total),
        param_mebibytes=float(bytes_total / (1024.0 * 1024.0)),
    )


@torch.no_grad()
def _quality_metrics(baseline: nn.Linear, das: DASGeometricLayer, test_batch: int, seed: int) -> QualityMetrics:
    torch.manual_seed(seed)
    x = torch.randn(test_batch, baseline.in_features, dtype=torch.float32)
    y_ref = baseline(x)
    y_hat = das(x)

    diff = y_hat - y_ref
    mse = float(torch.mean(diff * diff).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    max_abs = float(torch.max(torch.abs(diff)).item())

    W_ref = baseline.weight.detach().to(torch.float32)
    W_das = _materialize_effective_weight(das)
    rel_fro_error = float(torch.norm(W_ref - W_das).item() / (torch.norm(W_ref).item() + 1e-12))

    return QualityMetrics(
        rank=das.rank,
        mse=mse,
        mae=mae,
        max_abs=max_abs,
        rel_fro_error=rel_fro_error,
    )


@torch.no_grad()
def _euclidean_to_riemann_sphere(x: torch.Tensor) -> torch.Tensor:
    """
    Stereographic lift from R^d to unit sphere S^d embedded in R^(d+1).
    This maps Euclidean Hilbert activations to a compact curved interaction manifold.
    """
    norm2 = torch.sum(x * x, dim=-1, keepdim=True)
    denom = (norm2 + 1.0).clamp_min(1e-9)
    head = (norm2 - 1.0) / denom
    tail = (2.0 * x) / denom
    out = torch.cat([head, tail], dim=-1)
    return F.normalize(out, dim=-1)


@torch.no_grad()
def _riemann_sphere_metrics(baseline: nn.Linear, das: DASGeometricLayer, test_batch: int, seed: int) -> RiemannSphereMetrics:
    torch.manual_seed(seed)
    x = torch.randn(test_batch, baseline.in_features, dtype=torch.float32)
    y_ref = baseline(x)
    y_hat = das(x)

    s_ref = _euclidean_to_riemann_sphere(y_ref)
    s_hat = _euclidean_to_riemann_sphere(y_hat)

    dots = torch.sum(s_ref * s_hat, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    geodesic = torch.acos(dots)

    m = min(256, s_ref.shape[0])
    g_ref = s_ref[:m].matmul(s_ref[:m].t())
    g_hat = s_hat[:m].matmul(s_hat[:m].t())
    rel_fro = float(torch.norm(g_ref - g_hat).item() / (torch.norm(g_ref).item() + 1e-12))

    return RiemannSphereMetrics(
        mean_geodesic_rad=float(geodesic.mean().item()),
        p95_geodesic_rad=float(torch.quantile(geodesic, 0.95).item()),
        interaction_rel_fro_error=rel_fro,
    )


@torch.no_grad()
def _materialize_effective_weight(das: DASGeometricLayer) -> torch.Tensor:
    """
    For reporting only. Not used in DAS forward.

    W_das = out_paths^T * R * diag(scales) * in_paths
    """
    r = das.rank
    R = torch.eye(r, dtype=das.in_paths.dtype)
    for step in range(das.rotor_pairs.shape[0]):
        i = int(das.rotor_pairs[step, 0].item())
        j = int(das.rotor_pairs[step, 1].item())
        a = float(das.rotor_angles[step].item())
        c = math.cos(a)
        s = math.sin(a)
        G = torch.eye(r, dtype=R.dtype)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        R = G.matmul(R)

    D = torch.diag(das.scales)
    return das.out_paths.t().matmul(R).matmul(D).matmul(das.in_paths)


@torch.no_grad()
def _benchmark_forward(
    baseline: nn.Linear,
    das: DASGeometricLayer,
    batch_size: int,
    repeat: int,
    warmup: int,
    device: torch.device,
    seed: int,
) -> ForwardMetrics:
    baseline = baseline.to(device).eval()
    das = das.to(device).eval()

    torch.manual_seed(seed)
    x = torch.randn(batch_size, baseline.in_features, device=device, dtype=torch.float32)

    for _ in range(warmup):
        _ = baseline(x)
        _ = das(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = baseline(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(repeat):
        _ = das(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t3 = time.perf_counter()

    baseline_ms = (t1 - t0) * 1000.0 / repeat
    das_ms = (t3 - t2) * 1000.0 / repeat

    baseline_peak_mem = None
    das_peak_mem = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _ = baseline(x)
        torch.cuda.synchronize(device)
        baseline_peak_mem = int(torch.cuda.max_memory_allocated(device))

        torch.cuda.reset_peak_memory_stats(device)
        _ = das(x)
        torch.cuda.synchronize(device)
        das_peak_mem = int(torch.cuda.max_memory_allocated(device))

    return ForwardMetrics(
        device=str(device),
        batch_size=batch_size,
        repeat=repeat,
        baseline_mean_ms=float(baseline_ms),
        das_mean_ms=float(das_ms),
        speedup_x=float(baseline_ms / max(das_ms, 1e-12)),
        baseline_peak_mem_bytes=baseline_peak_mem,
        das_peak_mem_bytes=das_peak_mem,
    )


def _fmt_bytes(num: int | None) -> str:
    if num is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(num)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def _build_report(
    in_features: int,
    out_features: int,
    rank: int,
    rotor_steps: int,
    baseline_stats: LayerStats,
    das_stats: LayerStats,
    quality: QualityMetrics,
    riemann: RiemannSphereMetrics,
    compute: ComputeMetrics,
    hardware: dict[str, Any],
    perf: ForwardMetrics,
) -> dict[str, Any]:
    baseline_complexity = f"O({in_features}*{out_features})"
    das_complexity = f"O(rank*({in_features}+{out_features}) + rotor_steps)"

    compression_ratio = baseline_stats.params / max(das_stats.params, 1)
    param_reduction_pct = (1.0 - das_stats.params / max(baseline_stats.params, 1)) * 100.0

    report: dict[str, Any] = {
        "config": {
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "rotor_steps": rotor_steps,
        },
        "baseline": asdict(baseline_stats),
        "das": asdict(das_stats),
        "compression": {
            "compression_ratio": compression_ratio,
            "param_reduction_percent": param_reduction_pct,
            "baseline_param_complexity": baseline_complexity,
            "das_param_complexity": das_complexity,
        },
        "quality": asdict(quality),
        "riemann_sphere_alignment": asdict(riemann),
        "hardware_profile": hardware,
        "compute_plan": asdict(compute),
        "performance": asdict(perf),
        "memory_wall_interpretation": {
            "why_lazy_path_helps": (
                "DAS forward executes latent path contractions and rotor channel rotations "
                "without materializing a dense N x N matrix at inference time."
            ),
            "bandwidth_effect": (
                "Weight fetch changes from contiguous dense matrix loads to rank-path coefficient loads, "
                "reducing model-weight traffic from O(N^2) to O(rank*N)."
            ),
            "kv_cache_note": (
                "For transformer adaptation, the same path-lazy principle can be applied to projection layers "
                "to reduce projection parameter footprint and memory bandwidth pressure."
            ),
        },
        "friction_analysis": {
            "current_gpu_mismatch": (
                "Modern CUDA kernels are deeply optimized for dense GEMM/TensorCore schedules. "
                "Rotor/path operations are memory-light but involve smaller contractions and channel-wise updates, "
                "which can under-utilize matrix-specialized hardware."
            ),
            "future_chip_outlook": (
                "A dedicated 3D reversible compute fabric should co-design: "
                "(1) native rotor-pair SIMD primitives, "
                "(2) on-chip path-core SRAM routing, "
                "(3) fused contraction+rotation pipelines with reversible state checkpoints."
            ),
            "riemann_structure_note": (
                "Hilbert-space Euclidean activations are structurally unconstrained in open dimensions. "
                "This experiment additionally projects outputs onto a compact Riemann sphere and measures "
                "geodesic + interaction-gram consistency, making curved topology the target benchmark."
            ),
        },
    }
    return report


def _report_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    b = report["baseline"]
    d = report["das"]
    c = report["compression"]
    q = report["quality"]
    rs = report["riemann_sphere_alignment"]
    cp = report["compute_plan"]
    p = report["performance"]

    lines: list[str] = []
    lines.append("# DAS Architecture Efficiency and Dimensional Suppression Report")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append(f"- baseline layer: nn.Linear({cfg['in_features']}, {cfg['out_features']})")
    lines.append(f"- baseline profile: {cfg.get('baseline_profile', 'unknown')}")
    lines.append(f"- DAS rank: {cfg['rank']}")
    lines.append(f"- rotor steps: {cfg['rotor_steps']}")
    lines.append("")
    lines.append("## Parameter and Memory Suppression")
    lines.append("| item | baseline | DAS |")
    lines.append("|---|---:|---:|")
    lines.append(f"| params | {b['params']} | {d['params']} |")
    lines.append(f"| param bytes | {b['param_bytes']} | {d['param_bytes']} |")
    lines.append(f"| param size | {b['param_mebibytes']:.6f} MiB | {d['param_mebibytes']:.6f} MiB |")
    lines.append(f"| complexity | {c['baseline_param_complexity']} | {c['das_param_complexity']} |")
    lines.append("")
    lines.append(f"- compression ratio: {c['compression_ratio']:.4f}x")
    lines.append(f"- parameter reduction: {c['param_reduction_percent']:.4f}%")
    lines.append("")
    lines.append("## Reconstruction Quality")
    lines.append(f"- MSE: {q['mse']:.6e}")
    lines.append(f"- MAE: {q['mae']:.6e}")
    lines.append(f"- max abs error: {q['max_abs']:.6e}")
    lines.append(f"- relative Frobenius error: {q['rel_fro_error']:.6e}")
    lines.append("")
    lines.append("## Riemann Sphere Structural Alignment")
    lines.append("- target manifold: stereographic lift from Euclidean output space to unit Riemann sphere")
    lines.append(f"- mean geodesic error: {rs['mean_geodesic_rad']:.6e} rad")
    lines.append(f"- p95 geodesic error: {rs['p95_geodesic_rad']:.6e} rad")
    lines.append(f"- interaction Gram relative Frobenius error: {rs['interaction_rel_fro_error']:.6e}")
    lines.append("")
    lines.append("## Inference Runtime")
    lines.append(f"- compute device (request/resolved): {cp['device_request']} -> {cp['resolved_device']}")
    lines.append(f"- rotor kernel: {cp['rotor_kernel']}")
    lines.append(f"- torch.compile enabled: {cp['compiled']}")
    lines.append(f"- matmul precision: {cp['matmul_precision']}")
    lines.append(f"- selected torch threads: {cp['selected_torch_threads']}")
    lines.append(f"- compute-plan cache hit: {cp['cache_hit']}")
    if cp["autotune_timings_ms"] is not None:
        lines.append(f"- autotune timings (ms): {cp['autotune_timings_ms']}")
    lines.append(f"- device: {p['device']}")
    lines.append(f"- batch size: {p['batch_size']}")
    lines.append(f"- baseline mean latency: {p['baseline_mean_ms']:.6f} ms")
    lines.append(f"- DAS mean latency: {p['das_mean_ms']:.6f} ms")
    lines.append(f"- speedup (baseline / DAS): {p['speedup_x']:.6f}x")
    lines.append(f"- baseline peak memory: {_fmt_bytes(p['baseline_peak_mem_bytes'])}")
    lines.append(f"- DAS peak memory: {_fmt_bytes(p['das_peak_mem_bytes'])}")
    lines.append("")
    lines.append("## Memory-Wall and Friction Analysis")
    lines.append("- DAS lazy-path inference avoids dense matrix unfolding and lowers weight traffic from O(N^2) to O(rank*N).")
    lines.append("- On current GPUs, rotor/path kernels can face architectural friction because hardware is tuned for dense GEMM throughput.")
    lines.append("- A future dedicated 3D reversible chip should fuse path contraction and rotor primitives in one on-chip dataflow.")
    lines.append("")
    return "\n".join(lines)


def run_experiment(
    in_features: int,
    out_features: int,
    rank: int,
    rotor_steps: int,
    batch_size: int,
    repeat: int,
    warmup: int,
    refine_steps: int,
    structure_lambda: float,
    interaction_lambda: float,
    compute_device: str,
    rotor_kernel: str,
    compile_enabled: bool,
    autotune_threads: bool,
    thread_candidates: str | None,
    autotune_kernel: bool,
    compute_plan_cache: Path,
    refresh_compute_plan_cache: bool,
    baseline_profile: str,
    seed: int,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    hw_profile = detect_hardware_profile()
    original_threads = torch.get_num_threads()
    selected_threads = original_threads

    if autotune_threads:
        cands = parse_thread_candidates(thread_candidates, cpu_count=max(1, hw_profile.cpu_count))
    else:
        cands = [original_threads]

    if len(cands) > 0:
        selected_threads = cands[-1] if not autotune_threads else cands[0]
        torch.set_num_threads(selected_threads)
    device = resolve_device(compute_device)
    matmul_precision = apply_runtime_knobs(device)
    selected_kernel = choose_rotor_kernel(rotor_kernel, device=device, rank=rank, rotor_steps=rotor_steps)

    baseline = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
    _init_baseline_weight(baseline, profile=baseline_profile, seed=seed)
    baseline.eval()

    das = compress_to_das(
        baseline,
        compression_rank=rank,
        rotor_steps=rotor_steps,
        refine_steps=refine_steps,
        structure_lambda=structure_lambda,
        interaction_lambda=interaction_lambda,
        rotor_kernel=selected_kernel,
        seed=seed,
    )

    plan_probe: dict[str, float] | None = None
    cache_hit = False
    cache_key = make_cache_key(
        namespace="das_geometric_conversion",
        hardware=hw_profile,
        params={
            "compute_device": compute_device,
            "rotor_kernel": rotor_kernel,
            "rank": rank,
            "rotor_steps": rotor_steps,
            "batch_size": batch_size,
            "baseline_profile": baseline_profile,
        },
    )
    cache = load_compute_plan_cache(compute_plan_cache)
    cached_plan = None if refresh_compute_plan_cache else cache_lookup(cache, cache_key)

    if compute_device == "auto":
        if cached_plan is not None:
            cache_hit = True
            cached_device = str(cached_plan.get("resolved_device", "cpu"))
            cached_kernel = str(cached_plan.get("rotor_kernel", "scalar"))
            cached_threads = int(cached_plan.get("selected_torch_threads", selected_threads))
            device = resolve_device(cached_device)
            selected_kernel = cached_kernel
            selected_threads = max(1, cached_threads)
            torch.set_num_threads(selected_threads)
            das.rotor_kernel = selected_kernel
            matmul_precision = apply_runtime_knobs(device)
        else:
            tuned_device, tuned_kernel, tuned_threads, plan_probe = _autotune_device_and_kernel(
                baseline=baseline,
                das=das,
                batch_size=batch_size,
                seed=seed + 5,
                thread_candidates=cands,
            )
            device = tuned_device
            selected_kernel = tuned_kernel
            selected_threads = tuned_threads
            torch.set_num_threads(selected_threads)
            das.rotor_kernel = tuned_kernel
            matmul_precision = apply_runtime_knobs(device)
            cache_store(
                cache,
                cache_key,
                {
                    "resolved_device": str(device),
                    "rotor_kernel": selected_kernel,
                    "selected_torch_threads": int(selected_threads),
                },
            )
            save_compute_plan_cache(compute_plan_cache, cache)

    autotune_timings: dict[str, float] | None = None
    if autotune_kernel and das.rotor_pairs.shape[0] > 0:
        x_probe = torch.randn(batch_size, in_features, dtype=torch.float32)
        old_kernel = das.rotor_kernel
        candidates = {
            "scalar": lambda: _forward_with_kernel(das, x_probe, "scalar"),
            "staged": lambda: _forward_with_kernel(das, x_probe, "staged"),
        }
        best_kernel, timings = autotune_callable(candidates, warmup=6, repeat=20, device=None)
        das.rotor_kernel = best_kernel
        selected_kernel = best_kernel
        autotune_timings = {k: float(v) for k, v in timings.items()}
        if old_kernel != best_kernel:
            pass

    if plan_probe is not None:
        if autotune_timings is None:
            autotune_timings = {}
        for k, v in plan_probe.items():
            autotune_timings[f"plan::{k}"] = float(v)

    compile_flag = compile_enabled and device.type != "mps"
    baseline, baseline_compiled = try_compile_module(baseline, enabled=compile_flag)
    das, das_compiled = try_compile_module(das, enabled=compile_flag)
    compile_used = bool(baseline_compiled and das_compiled)

    baseline_stats = _layer_stats(baseline)
    das_stats = _layer_stats(das)
    quality = _quality_metrics(baseline, das, test_batch=max(512, batch_size), seed=seed + 11)
    riemann = _riemann_sphere_metrics(baseline, das, test_batch=max(512, batch_size), seed=seed + 17)

    perf = _benchmark_forward(
        baseline=baseline,
        das=das,
        batch_size=batch_size,
        repeat=repeat,
        warmup=warmup,
        device=device,
        seed=seed + 23,
    )

    compute = ComputeMetrics(
        device_request=compute_device,
        resolved_device=str(device),
        rotor_kernel=selected_kernel,
        compiled=compile_used,
        matmul_precision=matmul_precision,
        selected_torch_threads=torch.get_num_threads(),
        cache_hit=cache_hit,
        cache_key=cache_key,
        autotune_timings_ms=autotune_timings,
    )

    report = _build_report(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        rotor_steps=rotor_steps,
        baseline_stats=baseline_stats,
        das_stats=das_stats,
        quality=quality,
        riemann=riemann,
        compute=compute,
        hardware=profile_as_dict(hw_profile),
        perf=perf,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    report["config"]["baseline_profile"] = baseline_profile
    report["config"]["target_structure"] = "riemann_sphere_interaction"
    report["config"]["structure_lambda"] = structure_lambda
    report["config"]["interaction_lambda"] = interaction_lambda
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(_report_md(report), encoding="utf-8")
    return report


@torch.no_grad()
def _init_baseline_weight(layer: nn.Linear, profile: str, seed: int) -> None:
    torch.manual_seed(seed)
    out_features, in_features = layer.weight.shape
    k = min(in_features, out_features)

    if profile == "random":
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5.0))
        if layer.bias is not None:
            fan_in = in_features
            bound = 1.0 / math.sqrt(float(fan_in))
            nn.init.uniform_(layer.bias, -bound, bound)
        return

    # Spectral-decay profile: synthetic trained-like matrix with smooth singular decay.
    A = torch.randn(out_features, out_features, dtype=layer.weight.dtype)
    B = torch.randn(in_features, in_features, dtype=layer.weight.dtype)
    U_full, _ = torch.linalg.qr(A)
    V_full, _ = torch.linalg.qr(B)
    U = U_full[:, :k]
    V = V_full[:, :k]
    sigma = torch.exp(-torch.linspace(0.0, 16.0, k, dtype=layer.weight.dtype))
    W = U.matmul(torch.diag(sigma)).matmul(V.t())
    layer.weight.copy_(W)
    if layer.bias is not None:
        layer.bias.zero_()


@torch.no_grad()
def _forward_with_kernel(model: DASGeometricLayer, x: torch.Tensor, kernel: str) -> None:
    old = model.rotor_kernel
    model.rotor_kernel = kernel
    _ = model(x)
    model.rotor_kernel = old


@torch.no_grad()
def _probe_das_latency(
    das: DASGeometricLayer,
    batch_size: int,
    in_features: int,
    device: torch.device,
    kernel: str,
    seed: int,
    warmup: int = 8,
    repeat: int = 30,
) -> float:
    if device.type == "cuda" and not torch.cuda.is_available():
        return float("inf")
    if device.type == "mps" and not bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return float("inf")

    model = copy.deepcopy(das).to(device).eval()
    model.rotor_kernel = kernel
    torch.manual_seed(seed)
    x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeat


@torch.no_grad()
def _autotune_device_and_kernel(
    baseline: nn.Linear,
    das: DASGeometricLayer,
    batch_size: int,
    seed: int,
    thread_candidates: list[int],
) -> tuple[torch.device, str, int, dict[str, float]]:
    device_candidates: list[torch.device] = [torch.device("cpu")]
    if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        device_candidates.append(torch.device("mps"))
    if torch.cuda.is_available():
        device_candidates.append(torch.device("cuda"))

    kernel_candidates = ["scalar", "staged"] if das.rotor_pairs.shape[0] > 0 else ["scalar"]

    probe: dict[str, float] = {}
    best_dev = device_candidates[0]
    best_kernel = "scalar"
    best_threads = torch.get_num_threads()
    best_ms = float("inf")
    original_threads = torch.get_num_threads()

    try:
        for dev in device_candidates:
            apply_runtime_knobs(dev)
            if dev.type == "cpu":
                thread_sweep = thread_candidates
            else:
                thread_sweep = [original_threads]

            for th in thread_sweep:
                torch.set_num_threads(th)
                for ker in kernel_candidates:
                    ms = _probe_das_latency(
                        das=das,
                        batch_size=batch_size,
                        in_features=baseline.in_features,
                        device=dev,
                        kernel=ker,
                        seed=seed,
                    )
                    key = f"{dev.type}:t{th}:{ker}"
                    probe[key] = float(ms)
                    if ms < best_ms:
                        best_ms = ms
                        best_dev = dev
                        best_kernel = ker
                        best_threads = th
    finally:
        torch.set_num_threads(original_threads)

    return best_dev, best_kernel, best_threads, probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAS geometric conversion experiment for dense nn.Linear.")
    p.add_argument("--in-features", type=int, default=1024)
    p.add_argument("--out-features", type=int, default=1024)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--rotor-steps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--repeat", type=int, default=120)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--refine-steps", type=int, default=120)
    p.add_argument(
        "--structure-lambda",
        type=float,
        default=0.15,
        help="Weight for geodesic structure loss on Riemann sphere.",
    )
    p.add_argument(
        "--interaction-lambda",
        type=float,
        default=0.05,
        help="Weight for sphere interaction-gram alignment loss.",
    )
    p.add_argument(
        "--compute-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Execution device selection policy.",
    )
    p.add_argument(
        "--rotor-kernel",
        type=str,
        default="auto",
        choices=["auto", "scalar", "staged"],
        help="Rotor update kernel policy.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for baseline and DAS modules when available.",
    )
    p.add_argument(
        "--autotune-threads",
        action="store_true",
        help="Autotune torch CPU thread count on local hardware.",
    )
    p.add_argument(
        "--thread-candidates",
        type=str,
        default="",
        help="Comma-separated thread candidates for autotune, e.g. 1,2,4,6,8,10",
    )
    p.add_argument(
        "--autotune-kernel",
        action="store_true",
        help="Autotune scalar/staged rotor kernels on a probe batch.",
    )
    p.add_argument(
        "--compute-plan-cache",
        type=Path,
        default=Path("reports/compute_plan_cache.json"),
        help="Offline cache file for (device, kernel, threads) compute plan.",
    )
    p.add_argument(
        "--refresh-compute-plan-cache",
        action="store_true",
        help="Ignore cache and re-run probing for this invocation.",
    )
    p.add_argument(
        "--baseline-profile",
        type=str,
        default="spectral_decay",
        choices=["spectral_decay", "random"],
        help="Baseline weight profile: spectral_decay is a trained-like public conversion target.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/das_geometric_conversion_report.json"),
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/das_geometric_conversion_report.md"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(
        in_features=args.in_features,
        out_features=args.out_features,
        rank=args.rank,
        rotor_steps=args.rotor_steps,
        batch_size=args.batch_size,
        repeat=args.repeat,
        warmup=args.warmup,
        refine_steps=args.refine_steps,
        structure_lambda=args.structure_lambda,
        interaction_lambda=args.interaction_lambda,
        compute_device=args.compute_device,
        rotor_kernel=args.rotor_kernel,
        compile_enabled=args.compile,
        autotune_threads=args.autotune_threads,
        thread_candidates=args.thread_candidates,
        autotune_kernel=args.autotune_kernel,
        compute_plan_cache=args.compute_plan_cache,
        refresh_compute_plan_cache=args.refresh_compute_plan_cache,
        baseline_profile=args.baseline_profile,
        seed=args.seed,
        output_json=args.output_json,
        output_md=args.output_md,
    )

    print("[DAS-CONVERSION] done")
    print(f"baseline params: {report['baseline']['params']}")
    print(f"das params: {report['das']['params']}")
    print(f"compression ratio: {report['compression']['compression_ratio']:.4f}x")
    print(f"param reduction: {report['compression']['param_reduction_percent']:.4f}%")
    print(f"quality MAE: {report['quality']['mae']:.6e}")
    print(
        "riemann mean geodesic(rad): "
        f"{report['riemann_sphere_alignment']['mean_geodesic_rad']:.6e}"
    )
    print(
        "compute plan: "
        f"device={report['compute_plan']['resolved_device']}, "
        f"kernel={report['compute_plan']['rotor_kernel']}, "
        f"threads={report['compute_plan']['selected_torch_threads']}, "
        f"compiled={report['compute_plan']['compiled']}"
    )
    print(f"speedup baseline/das: {report['performance']['speedup_x']:.6f}x")


if __name__ == "__main__":
    main()
