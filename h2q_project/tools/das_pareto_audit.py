#!/usr/bin/env python3
"""
Multi-seed + multi-rank Pareto audit for DAS token distillation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrialResult:
    rank: int
    seed: int
    cosine: float
    top5: float
    speedup: float
    compression: float


def _ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    m = statistics.mean(values)
    if len(values) == 1:
        return (m, m)
    s = statistics.pstdev(values)
    delta = 1.96 * s / math.sqrt(len(values))
    return (m - delta, m + delta)


def _run_trial(
    python_exe: str,
    script: Path,
    out_dir: Path,
    model_id: str,
    rank: int,
    seed: int,
    token_table_size: int,
    steps: int,
    temperature: float,
    temperature_end: float,
    topk: int,
    ranking_weight: float,
    mse_weight: float,
    ranking_margin: float,
    hard_neg_k: int,
    hard_neg_weight: float,
    stage_split: float,
    stage1_rank_scale: float,
) -> TrialResult:
    trial_dir = out_dir / f"rank_{rank}_seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(script),
        "--model-id",
        model_id,
        "--qkv-rank",
        str(max(16, rank // 2)),
        "--token-rank",
        str(rank),
        "--token-table-size",
        str(token_table_size),
        "--qkv-steps",
        str(max(40, steps // 2)),
        "--token-steps",
        str(steps),
        "--temperature",
        str(temperature),
        "--temperature-end",
        str(temperature_end),
        "--topk",
        str(topk),
        "--ranking-weight",
        str(ranking_weight),
        "--mse-weight",
        str(mse_weight),
        "--ranking-margin",
        str(ranking_margin),
        "--hard-neg-k",
        str(hard_neg_k),
        "--hard-neg-weight",
        str(hard_neg_weight),
        "--stage-split",
        str(stage_split),
        "--stage1-rank-scale",
        str(stage1_rank_scale),
        "--seed",
        str(seed),
        "--output-dir",
        str(trial_dir),
    ]

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    _ = proc.stdout

    model_name = model_id.replace("/", "__")
    report_path = trial_dir / f"das_qkv_token_distill_{model_name}_20260328.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    return TrialResult(
        rank=rank,
        seed=seed,
        cosine=float(report["token_distillation"]["cosine"]),
        top5=float(report["token_distillation"]["top5_overlap"]),
        speedup=float(report["latency_ms"]["speedup_ratio"]),
        compression=float(report["memory"]["param_compression_ratio"]),
    )


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Maximize cosine, speedup, compression.
    front = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                b["mean_cosine"] >= a["mean_cosine"]
                and b["mean_speedup"] >= a["mean_speedup"]
                and b["mean_compression"] >= a["mean_compression"]
            )
            strictly_better = (
                b["mean_cosine"] > a["mean_cosine"]
                or b["mean_speedup"] > a["mean_speedup"]
                or b["mean_compression"] > a["mean_compression"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda x: (x["rank"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAS multi-seed multi-rank Pareto audit")
    p.add_argument("--model-id", type=str, default="distilgpt2")
    p.add_argument("--ranks", type=str, default="32,48,64")
    p.add_argument("--seeds", type=str, default="11,17,23")
    p.add_argument("--token-table-size", type=int, default=2048)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-end", type=float, default=0.55)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--ranking-weight", type=float, default=0.35)
    p.add_argument("--mse-weight", type=float, default=0.15)
    p.add_argument("--ranking-margin", type=float, default=0.20)
    p.add_argument("--hard-neg-k", type=int, default=6)
    p.add_argument("--hard-neg-weight", type=float, default=0.15)
    p.add_argument("--stage-split", type=float, default=0.45)
    p.add_argument("--stage1-rank-scale", type=float, default=0.35)
    p.add_argument("--output-dir", type=str, default="reports/conv_math_conversion/das_pareto_audit")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("das_qkv_token_distill_experiment.py")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    trials: list[TrialResult] = []
    for rank in ranks:
        for seed in seeds:
            t = _run_trial(
                python_exe=args.python_exe,
                script=script,
                out_dir=out_dir,
                model_id=args.model_id,
                rank=rank,
                seed=seed,
                token_table_size=args.token_table_size,
                steps=args.steps,
                temperature=args.temperature,
                temperature_end=args.temperature_end,
                topk=args.topk,
                ranking_weight=args.ranking_weight,
                mse_weight=args.mse_weight,
                ranking_margin=args.ranking_margin,
                hard_neg_k=args.hard_neg_k,
                hard_neg_weight=args.hard_neg_weight,
                stage_split=args.stage_split,
                stage1_rank_scale=args.stage1_rank_scale,
            )
            trials.append(t)

    grouped: dict[int, list[TrialResult]] = {}
    for t in trials:
        grouped.setdefault(t.rank, []).append(t)

    rows = []
    for rank in sorted(grouped.keys()):
        g = grouped[rank]
        cos = [x.cosine for x in g]
        top5 = [x.top5 for x in g]
        spd = [x.speedup for x in g]
        cmp_ = [x.compression for x in g]
        cos_ci = _ci95(cos)
        top5_ci = _ci95(top5)
        spd_ci = _ci95(spd)
        cmp_ci = _ci95(cmp_)

        row = {
            "rank": rank,
            "mean_cosine": float(statistics.mean(cos)),
            "ci95_cosine": [float(cos_ci[0]), float(cos_ci[1])],
            "mean_top5": float(statistics.mean(top5)),
            "ci95_top5": [float(top5_ci[0]), float(top5_ci[1])],
            "mean_speedup": float(statistics.mean(spd)),
            "ci95_speedup": [float(spd_ci[0]), float(spd_ci[1])],
            "mean_compression": float(statistics.mean(cmp_)),
            "ci95_compression": [float(cmp_ci[0]), float(cmp_ci[1])],
            "acceptance": {
                "consistency": bool(statistics.mean(cos) >= 0.97 and statistics.mean(top5) >= 0.55),
                "speedup": bool(statistics.mean(spd) >= 1.05),
                "compression": bool(statistics.mean(cmp_) >= 2.0),
            },
        }
        rows.append(row)

    pareto = _pareto_front(rows)

    report = {
        "model_id": args.model_id,
        "ranks": ranks,
        "seeds": seeds,
        "token_table_size": args.token_table_size,
        "steps": args.steps,
        "distill_hparams": {
            "temperature": args.temperature,
            "temperature_end": args.temperature_end,
            "topk": args.topk,
            "ranking_weight": args.ranking_weight,
            "mse_weight": args.mse_weight,
            "ranking_margin": args.ranking_margin,
            "hard_neg_k": args.hard_neg_k,
            "hard_neg_weight": args.hard_neg_weight,
            "stage_split": args.stage_split,
            "stage1_rank_scale": args.stage1_rank_scale,
        },
        "rows": rows,
        "pareto_front": pareto,
        "thresholds": {
            "cosine_min": 0.97,
            "top5_min": 0.55,
            "speedup_min": 1.05,
            "compression_min": 2.0,
        },
    }

    json_path = out_dir / "das_pareto_audit_20260328.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / "DAS_PARETO_AUDIT_20260328.md"
    lines = []
    lines.append("# DAS Pareto Audit (Multi-seed + Multi-rank)")
    lines.append("")
    lines.append(f"- model: `{args.model_id}`")
    lines.append(f"- ranks: `{ranks}`")
    lines.append(f"- seeds: `{seeds}`")
    lines.append(
        "- distill hparams: `temp={:.2f}->{:.2f}, topk={}, rank_w={:.2f}, mse_w={:.2f}, margin={:.2f}, hard_k={}, hard_w={:.2f}, split={:.2f}, stage1={:.2f}`".format(
            args.temperature,
            args.temperature_end,
            args.topk,
            args.ranking_weight,
            args.mse_weight,
            args.ranking_margin,
            args.hard_neg_k,
            args.hard_neg_weight,
            args.stage_split,
            args.stage1_rank_scale,
        )
    )
    lines.append("")
    lines.append("## Aggregated Metrics")
    lines.append("")
    for row in rows:
        lines.append(
            "- rank {rank}: cosine={mean_cosine:.4f}, top5={mean_top5:.4f}, speedup={mean_speedup:.4f}x, compression={mean_compression:.4f}x".format(**row)
        )
    lines.append("")
    lines.append("## Pareto Front")
    lines.append("")
    for row in pareto:
        lines.append(
            "- rank {rank}: cosine={mean_cosine:.4f}, speedup={mean_speedup:.4f}x, compression={mean_compression:.4f}x".format(**row)
        )
    lines.append("")
    lines.append(f"JSON report: `{json_path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Pareto JSON: {json_path}")
    print(f"Pareto MD:   {md_path}")


if __name__ == "__main__":
    main()
