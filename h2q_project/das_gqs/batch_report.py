from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from h2q_project.das_gqs.backends import available_backends
from h2q_project.das_gqs.batch_sampling import (
    estimate_chsh_batch,
    noise_robustness_report,
    resolve_batch_sampling_compute_plan,
)
from h2q_project.das_gqs.core import Bivector, Vector, generate_rotor, sandwich_rotate


def backend_consistency_probe(samples: int = 24, seed: int = 123) -> dict:
    backends = available_backends()
    if "clifford" not in backends:
        return {
            "available_backends": backends,
            "ran_consistency": False,
            "reason": "clifford backend not installed",
        }

    rng = np.random.default_rng(seed)
    max_l2 = 0.0
    mean_l2 = 0.0
    for _ in range(samples):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(-2.0 * np.pi, 2.0 * np.pi))
        vector = rng.normal(size=3)

        rotor = generate_rotor(Bivector(axis), angle=angle)
        v = Vector(vector)
        out_np = sandwich_rotate(v, rotor, backend="numpy").value
        out_cf = sandwich_rotate(v, rotor, backend="clifford").value
        l2 = float(np.linalg.norm(out_np - out_cf))
        max_l2 = max(max_l2, l2)
        mean_l2 += l2

    mean_l2 /= samples
    return {
        "available_backends": backends,
        "ran_consistency": True,
        "samples": samples,
        "max_l2": max_l2,
        "mean_l2": mean_l2,
    }


def render_markdown(backend: dict, batch: dict, robustness: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# DAS-GQS Batch Statistical Report")
    lines.append("")
    lines.append("## 1. Backend Consistency")
    if backend.get("ran_consistency"):
        lines.append(f"- backends: {backend['available_backends']}")
        lines.append(f"- samples: {backend['samples']}")
        lines.append(f"- max L2: {backend['max_l2']:.6e}")
        lines.append(f"- mean L2: {backend['mean_l2']:.6e}")
    else:
        lines.append(f"- backends: {backend.get('available_backends', [])}")
        lines.append(f"- skipped: {backend.get('reason', 'unknown')}")

    lines.append("")
    lines.append("## 2. CHSH Batch Estimate (95% CI)")
    lines.append(f"- S mean: {batch['S']['mean']:.6f}")
    lines.append(f"- S 95% CI: [{batch['S']['ci_low']:.6f}, {batch['S']['ci_high']:.6f}]")
    lines.append(f"- |S|: {batch['abs_S']:.6f}")
    lines.append(f"- Tsirelson target: {2 * math.sqrt(2):.6f}")
    if batch.get("compute_plan"):
        cp = batch["compute_plan"]
        lines.append(f"- selected threads: {cp.get('selected_torch_threads')}")
        lines.append(f"- compute plan cache hit: {cp.get('cache_hit')}")

    lines.append("")
    lines.append("## 3. Noise Robustness")
    lines.append("| scenario | jitter(deg) | flip_p | S mean | 95% CI low | 95% CI high | |S| | CI excludes 2? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in robustness:
        lines.append(
            f"| {row['scenario']} | {row['axis_jitter_deg']:.2f} | {row['outcome_flip_prob']:.3f} | "
            f"{row['S_mean']:.6f} | {row['S_ci_low']:.6f} | {row['S_ci_high']:.6f} | "
            f"{row['abs_S']:.6f} | {row['violates_classical_limit_with_95ci']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAS-GQS batch statistical report with autocompute integration")
    p.add_argument("--n-pairs", type=int, default=30000)
    p.add_argument("--batch-seed", type=int, default=9)
    p.add_argument("--noise-seed", type=int, default=19)
    p.add_argument(
        "--autotune-threads",
        action="store_true",
        help="Autotune CPU threads and cache selected plan.",
    )
    p.add_argument(
        "--thread-candidates",
        type=str,
        default="",
        help="Comma-separated thread candidates, e.g. 1,2,4,6,8",
    )
    p.add_argument(
        "--compute-plan-cache",
        type=Path,
        default=Path("reports/compute_plan_cache.json"),
        help="Offline compute plan cache path shared across DAS scripts.",
    )
    p.add_argument(
        "--refresh-compute-plan-cache",
        action="store_true",
        help="Ignore existing cache and re-probe thread plan.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/das_gqs_noise_robustness_report.json"),
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/das_gqs_noise_robustness_report.md"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hardware, compute_plan = resolve_batch_sampling_compute_plan(
        n_pairs=args.n_pairs,
        compute_plan_cache=args.compute_plan_cache,
        refresh_compute_plan_cache=args.refresh_compute_plan_cache,
        autotune_threads=args.autotune_threads,
        thread_candidates=args.thread_candidates,
    )

    backend = backend_consistency_probe(samples=32, seed=321)
    batch = asdict(
        estimate_chsh_batch(
            n_pairs=args.n_pairs,
            seed=args.batch_seed,
            axis_jitter_deg=0.0,
            outcome_flip_prob=0.0,
            hardware_profile=hardware,
            compute_plan=compute_plan,
        )
    )
    robustness = [
        asdict(r)
        for r in noise_robustness_report(
            n_pairs=args.n_pairs,
            seed=args.noise_seed,
            hardware_profile=hardware,
            compute_plan=compute_plan,
        )
    ]

    payload = {
        "backend_consistency": backend,
        "batch_chsh": batch,
        "noise_robustness": robustness,
        "hardware_profile": hardware,
        "compute_plan": compute_plan,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(backend, batch, robustness), encoding="utf-8")

    print("=== DAS-GQS batch report generated ===")
    print(f"json: {args.output_json}")
    print(f"md:   {args.output_md}")
    print(f"S mean: {batch['S']['mean']:.6f}, CI=[{batch['S']['ci_low']:.6f}, {batch['S']['ci_high']:.6f}]")
    print(
        "compute plan: "
        f"threads={compute_plan.get('selected_torch_threads')}, "
        f"cache_hit={compute_plan.get('cache_hit')}"
    )


if __name__ == "__main__":
    main()
