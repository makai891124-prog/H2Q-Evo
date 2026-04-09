#!/usr/bin/env python3
"""
Harsh-environment benchmark for DAS math core on local Apple Silicon style setup.

Goals:
1) Measure real local performance under CPU and optional MPS profiles.
2) Stress memory-aware execution (micro-batch slicing).
3) Compare local verified values with publicly disclosed challenge scales
   (if same-scale benchmark is not locally feasible).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psutil
import torch
import torch.nn as nn

from h2q_project.tools.das_geometric_rotor_experiment import initialize_compressible_baseline, compress_to_das


@dataclass
class ProfileResult:
    profile: str
    device: str
    batch_size: int
    micro_batch: int
    mean_ms_baseline: float
    mean_ms_das: float
    p90_ms_baseline: float
    p90_ms_das: float
    speedup_ratio: float
    throughput_baseline_items_s: float
    throughput_das_items_s: float
    rss_before_mb: float
    rss_after_baseline_mb: float
    rss_after_das_mb: float
    rss_peak_delta_baseline_mb: float
    rss_peak_delta_das_mb: float


def _sysctl_value(key: str) -> str | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        return out
    except Exception:
        return None


def _hardware_snapshot() -> dict[str, Any]:
    perf_cores = _sysctl_value("hw.perflevel0.physicalcpu")
    eff_cores = _sysctl_value("hw.perflevel1.physicalcpu")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": _sysctl_value("machdep.cpu.brand_string"),
        "logical_cpu": os.cpu_count(),
        "perf_cores": int(perf_cores) if perf_cores and perf_cores.isdigit() else None,
        "eff_cores": int(eff_cores) if eff_cores and eff_cores.isdigit() else None,
        "torch_version": torch.__version__,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }


def _parameter_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


@torch.inference_mode()
def _run_timed(
    module: nn.Module,
    x: torch.Tensor,
    rounds: int,
    warmup: int,
    micro_batch: int,
    sync: Callable[[], None],
) -> tuple[list[float], float]:
    for _ in range(warmup):
        if micro_batch >= x.shape[0]:
            _ = module(x)
        else:
            for s in range(0, x.shape[0], micro_batch):
                _ = module(x[s : s + micro_batch])
        sync()

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        if micro_batch >= x.shape[0]:
            _ = module(x)
        else:
            for s in range(0, x.shape[0], micro_batch):
                _ = module(x[s : s + micro_batch])
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(statistics.mean(times)) if times else float("nan")
    return times, mean_ms


def _sync_for(device: torch.device) -> Callable[[], None]:
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            return torch.mps.synchronize
    return lambda: None


def _to_device(module: nn.Module, device: torch.device) -> nn.Module:
    return module.to(device)


def _p90(values: list[float]) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    idx = max(0, int(0.9 * len(v)) - 1)
    return float(v[idx])


def _profile_run(
    name: str,
    device: torch.device,
    in_dim: int,
    out_dim: int,
    rank: int,
    rotors: int,
    batch_size: int,
    micro_batch: int,
    rounds: int,
    warmup: int,
    seed: int,
) -> ProfileResult:
    torch.manual_seed(seed)

    baseline = nn.Linear(in_dim, out_dim, bias=True)
    initialize_compressible_baseline(baseline, latent_rank=max(8, rank // 2), noise_std=0.01, device=torch.device("cpu"))

    das_layer, _meta = compress_to_das(
        standard_layer=baseline,
        compression_rank=rank,
        num_rotors=rotors,
        riemann_alpha=0.0,
        rotor_strength=0.03,
        device=torch.device("cpu"),
    )

    baseline = _to_device(baseline.eval(), device)
    das_layer = _to_device(das_layer.eval(), device)

    x = torch.randn(batch_size, in_dim, device=device)
    sync = _sync_for(device)

    rss_before = _rss_mb()
    times_base, mean_base = _run_timed(baseline, x, rounds=rounds, warmup=warmup, micro_batch=micro_batch, sync=sync)
    rss_after_base = _rss_mb()

    times_das, mean_das = _run_timed(das_layer, x, rounds=rounds, warmup=warmup, micro_batch=micro_batch, sync=sync)
    rss_after_das = _rss_mb()

    base_peak = max(rss_after_base, rss_before)
    das_peak = max(rss_after_das, rss_after_base)

    items = float(batch_size)
    thr_base = 1000.0 * items / max(mean_base, 1e-9)
    thr_das = 1000.0 * items / max(mean_das, 1e-9)

    return ProfileResult(
        profile=name,
        device=str(device),
        batch_size=batch_size,
        micro_batch=micro_batch,
        mean_ms_baseline=mean_base,
        mean_ms_das=mean_das,
        p90_ms_baseline=_p90(times_base),
        p90_ms_das=_p90(times_das),
        speedup_ratio=float(mean_base / max(mean_das, 1e-9)),
        throughput_baseline_items_s=thr_base,
        throughput_das_items_s=thr_das,
        rss_before_mb=rss_before,
        rss_after_baseline_mb=rss_after_base,
        rss_after_das_mb=rss_after_das,
        rss_peak_delta_baseline_mb=float(base_peak - rss_before),
        rss_peak_delta_das_mb=float(das_peak - rss_after_base),
    )


def _public_reference() -> dict[str, Any]:
    # Use repository-tracked public references to avoid fabricated external numbers.
    ref = {
        "public_challenge": None,
        "unified_rcs_xeb": None,
    }
    p1 = Path("reports/das_gqs_public_challenge_gap_report.json")
    p2 = Path("reports/das_gqs_public_rcs_xeb_unified_report.json")
    if p1.exists():
        ref["public_challenge"] = json.loads(p1.read_text(encoding="utf-8"))
    if p2.exists():
        ref["unified_rcs_xeb"] = json.loads(p2.read_text(encoding="utf-8"))
    return ref


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAS harsh-environment benchmark for Apple Silicon/local CPU")
    p.add_argument("--in-dim", type=int, default=1024)
    p.add_argument("--out-dim", type=int, default=1024)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--rotors", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260328)
    p.add_argument("--output-dir", type=str, default="reports/conv_math_conversion")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hw = _hardware_snapshot()

    # Interop threads can only be configured once before heavy parallel work starts.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    perf_cores = hw.get("perf_cores") or max(1, (os.cpu_count() or 4) // 2)
    logical = hw.get("logical_cpu") or (os.cpu_count() or 4)

    # CPU baseline profile.
    torch.set_num_threads(max(1, logical))
    torch.set_float32_matmul_precision("high")
    cpu_base = _profile_run(
        name="cpu_full_threads",
        device=torch.device("cpu"),
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        rank=args.rank,
        rotors=args.rotors,
        batch_size=args.batch_size,
        micro_batch=args.batch_size,
        rounds=args.rounds,
        warmup=args.warmup,
        seed=args.seed,
    )

    # CPU perf-core focused profile with memory-guarded micro-batching.
    torch.set_num_threads(max(1, int(perf_cores)))
    cpu_guard = _profile_run(
        name="cpu_perfcore_memguard",
        device=torch.device("cpu"),
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        rank=args.rank,
        rotors=args.rotors,
        batch_size=args.batch_size,
        micro_batch=max(1, min(args.micro_batch, args.batch_size)),
        rounds=args.rounds,
        warmup=args.warmup,
        seed=args.seed,
    )

    profiles: list[ProfileResult] = [cpu_base, cpu_guard]

    if hw.get("mps_available"):
        # MPS profile, reusing memory-guarded micro-batching.
        torch.set_float32_matmul_precision("high")
        mps_guard = _profile_run(
            name="mps_memguard",
            device=torch.device("mps"),
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            rank=args.rank,
            rotors=args.rotors,
            batch_size=args.batch_size,
            micro_batch=max(1, min(args.micro_batch, args.batch_size)),
            rounds=args.rounds,
            warmup=args.warmup,
            seed=args.seed,
        )
        profiles.append(mps_guard)

    # Parameter memory comparison is backend independent.
    base_tmp = nn.Linear(args.in_dim, args.out_dim, bias=True)
    initialize_compressible_baseline(base_tmp, latent_rank=max(8, args.rank // 2), noise_std=0.01, device=torch.device("cpu"))
    das_tmp, _ = compress_to_das(
        standard_layer=base_tmp,
        compression_rank=args.rank,
        num_rotors=args.rotors,
        riemann_alpha=0.0,
        rotor_strength=0.03,
        device=torch.device("cpu"),
    )

    baseline_param_bytes = _parameter_bytes(base_tmp)
    das_param_bytes = _parameter_bytes(das_tmp)

    public = _public_reference()

    best_speed = max(profiles, key=lambda r: r.speedup_ratio)
    best_throughput = max(profiles, key=lambda r: r.throughput_das_items_s)
    best_mem = min(profiles, key=lambda r: r.rss_peak_delta_das_mb)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw,
        "config": {
            "in_dim": args.in_dim,
            "out_dim": args.out_dim,
            "rank": args.rank,
            "rotors": args.rotors,
            "batch_size": args.batch_size,
            "micro_batch": args.micro_batch,
            "rounds": args.rounds,
            "warmup": args.warmup,
        },
        "profiles": [asdict(p) for p in profiles],
        "param_memory": {
            "baseline_param_bytes": baseline_param_bytes,
            "das_param_bytes": das_param_bytes,
            "compression_ratio": float(baseline_param_bytes / max(das_param_bytes, 1)),
        },
        "local_summary": {
            "best_speed_profile": best_speed.profile,
            "best_speedup_ratio": best_speed.speedup_ratio,
            "best_throughput_profile": best_throughput.profile,
            "best_throughput_items_s": best_throughput.throughput_das_items_s,
            "best_mem_profile": best_mem.profile,
            "lowest_das_rss_peak_delta_mb": best_mem.rss_peak_delta_das_mb,
        },
        "public_comparison": {
            "note": "If same-scale public benchmark cannot be fully reproduced locally, compare verified local metrics against publicly disclosed challenge scales and boundaries.",
            "source_reports": {
                "challenge_gap": "reports/das_gqs_public_challenge_gap_report.json",
                "rcs_xeb_unified": "reports/das_gqs_public_rcs_xeb_unified_report.json",
            },
            "public_reference_loaded": {
                "challenge_gap": public.get("public_challenge") is not None,
                "rcs_xeb_unified": public.get("unified_rcs_xeb") is not None,
            },
            "challenge_snapshot": {
                "public_max_qubits": (
                    public.get("unified_rcs_xeb", {}).get("gap_summary", {}).get("public_max_qubits")
                    if public.get("unified_rcs_xeb")
                    else None
                ),
                "local_max_qubits": (
                    public.get("unified_rcs_xeb", {}).get("gap_summary", {}).get("local_max_qubits")
                    if public.get("unified_rcs_xeb")
                    else None
                ),
                "qubit_gap": (
                    public.get("unified_rcs_xeb", {}).get("gap_summary", {}).get("qubit_gap")
                    if public.get("unified_rcs_xeb")
                    else None
                ),
            },
        },
        "harsh_environment_interpretation": {
            "statement": "This local machine is treated as a constrained/negative-optimization environment. Verified gains under this setup are considered robustness evidence rather than ideal-hardware upper bounds.",
            "accepted_boundary": "Claim local verified acceleration and memory suppression; do not claim same-scale hardware-level quantum advantage reproduction.",
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "das_m4_harshenv_benchmark_20260328.json"
    md_path = out_dir / "DAS_M4_HARSHENV_BENCHMARK_20260328.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# DAS Harsh-Environment Benchmark (Local Apple Silicon / CPU-Memory Focus)")
    lines.append("")
    lines.append(f"- timestamp: {report['timestamp']}")
    lines.append(f"- cpu: {hw.get('cpu_brand')}")
    lines.append(f"- mps available: {hw.get('mps_available')}")
    lines.append("")
    lines.append("## Local Profile Results")
    lines.append("")
    for p in report["profiles"]:
        lines.append(
            "- {profile} ({device}): speedup={speedup_ratio:.4f}x, baseline={mean_ms_baseline:.4f}ms, das={mean_ms_das:.4f}ms, das_throughput={throughput_das_items_s:.2f} items/s, das_rss_delta={rss_peak_delta_das_mb:.2f}MB".format(
                **p
            )
        )

    lines.append("")
    lines.append("## Memory Compression")
    lines.append("")
    lines.append(
        f"- parameter compression ratio: {report['param_memory']['compression_ratio']:.4f}x"
    )
    lines.append("")
    lines.append("## Public Challenge Scale Context")
    lines.append("")
    gap = report["public_comparison"]["challenge_snapshot"]
    lines.append(f"- public max qubits: {gap.get('public_max_qubits')}")
    lines.append(f"- local max qubits: {gap.get('local_max_qubits')}")
    lines.append(f"- qubit gap: {gap.get('qubit_gap')}")
    lines.append("")
    lines.append("## Boundary Statement")
    lines.append("")
    lines.append(f"- {report['harsh_environment_interpretation']['statement']}")
    lines.append(f"- {report['harsh_environment_interpretation']['accepted_boundary']}")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("[DAS] harsh-env benchmark complete")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    print(f"Best speedup: {report['local_summary']['best_speedup_ratio']:.4f}x ({report['local_summary']['best_speed_profile']})")


if __name__ == "__main__":
    main()
