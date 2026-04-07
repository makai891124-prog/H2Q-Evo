#!/usr/bin/env python3
"""End-to-end trusted AGI uplift pipeline.

Runs:
1) nano seed generation from open-source crystal
2) distillation pipeline
3) trusted nano LM weight training
4) bootstrap execute cycle with axiom consistency
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"


def _run(cmd: List[str], name: str) -> Dict[str, Any]:
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": int(p.returncode),
        "stdout_tail": "\n".join((p.stdout or "").splitlines()[-25:]),
        "stderr_tail": "\n".join((p.stderr or "").splitlines()[-25:]),
    }


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run trusted AGI uplift end-to-end pipeline")
    parser.add_argument("--train-method", choices=["full", "lora"], default="lora")
    parser.add_argument("--train-model", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--synthetic-min-size", type=int, default=120)
    parser.add_argument("--include-valid-prompts", action="store_true")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--bootstrap-iters", type=int, default=4)
    parser.add_argument("--benchmark-gain-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    train_script = "tools/train_trusted_nano_lora.py" if args.train_method == "lora" else "tools/train_trusted_nano_lm.py"
    output_prefix = "trusted_nano_lora" if args.train_method == "lora" else "trusted_nano_lm"

    steps: List[Dict[str, Any]] = []
    steps.append(
        _run(
            [
                str(PY),
                "tools/run_local_incremental_benchmark.py",
                "--seed",
                "42",
                "--output-prefix",
                "local_incremental_benchmark_uplift_before",
            ],
            "incremental_benchmark_before",
        )
    )

    steps.append(
        _run(
            [
                str(PY),
                "tools/train_nano_core_seed_from_crystal.py",
                "--crystal",
                "h2q_qwen_crystal.pt",
                "--dataset",
                "reports/self_eval_distill_dataset_latest.json",
                "--seed-count",
                "128",
                "--seed-dim",
                "64",
            ],
            "train_nano_seed",
        )
    )

    steps.append(
        _run(
            [
                str(PY),
                "tools/run_self_eval_distillation_pipeline.py",
                "--sessions",
                str(max(1, int(args.sessions))),
                "--teacher-provider",
                "heuristic",
                "--max-samples",
                str(max(1, int(args.max_samples))),
                "--synthetic-min-size",
                str(max(0, int(args.synthetic_min_size))),
                *( ["--include-valid-prompts"] if args.include_valid_prompts else [] ),
            ],
            "distillation_pipeline",
        )
    )

    steps.append(
        _run(
            [
                str(PY),
                train_script,
                "--model-name",
                args.train_model,
                "--dataset",
                "reports/self_eval_distill_dataset_latest.json",
                "--epochs",
                str(max(1, int(args.epochs))),
                "--lr",
                str(float(args.lr)),
                "--batch-size",
                str(max(1, int(args.batch_size))),
                "--max-samples",
                str(max(1, int(args.max_samples)) * 6),
                "--output-prefix",
                output_prefix,
            ],
            "trusted_weight_training",
        )
    )

    steps.append(
        _run(
            [
                str(PY),
                "tools/run_local_incremental_benchmark.py",
                "--seed",
                "42",
                "--output-prefix",
                "local_incremental_benchmark_uplift_after",
            ],
            "incremental_benchmark_after",
        )
    )

    steps.append(
        _run(
            [
                str(PY),
                "tools/run_autoresearch_h2q_bootstrap.py",
                "--execute",
                "--max-iterations",
                str(max(1, int(args.bootstrap_iters))),
                "--timeout-sec",
                "900",
                "--axiom-contract",
                "axiom_contract.json",
                "--distill-sessions",
                str(max(1, int(args.sessions))),
                "--distill-teacher-provider",
                "heuristic",
                "--distill-max-samples",
                str(max(1, int(args.max_samples))),
                "--distill-synthetic-min-size",
                str(max(0, int(args.synthetic_min_size))),
                "--hard-gate-benchmark-gain",
                str(float(args.benchmark_gain_threshold)),
                *( ["--distill-include-valid-prompts"] if args.include_valid_prompts else [] ),
            ],
            "bootstrap_with_axiom",
        )
    )

    bench_before = _load(REPORTS / "local_incremental_benchmark_uplift_before_latest.json")
    bench_after = _load(REPORTS / "local_incremental_benchmark_uplift_after_latest.json")
    gain_before = float(bench_before.get("gain", 0.0) or 0.0)
    gain_after = float(bench_after.get("gain", 0.0) or 0.0)
    benchmark_gate_pass = bool(gain_after >= float(args.benchmark_gain_threshold))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "config": {
            "train_method": args.train_method,
            "train_model": args.train_model,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "max_samples": int(args.max_samples),
            "synthetic_min_size": int(args.synthetic_min_size),
            "include_valid_prompts": bool(args.include_valid_prompts),
            "sessions": int(args.sessions),
            "bootstrap_iters": int(args.bootstrap_iters),
        },
        "seed": _load(REPORTS / "nano_core_seed_latest.json"),
        "benchmark_before": bench_before,
        "benchmark_after": bench_after,
        "benchmark_gate": {
            "threshold": float(args.benchmark_gain_threshold),
            "gain_before": gain_before,
            "gain_after": gain_after,
            "pass": benchmark_gate_pass,
        },
        "distillation": _load(REPORTS / "self_eval_distillation_pipeline_latest.json"),
        "trusted_weight_training": _load(REPORTS / f"{output_prefix}_training_latest.json"),
        "bootstrap": _load(REPORTS / "autoresearch_h2q_bootstrap_fusion_latest.json"),
    }

    ts = int(time.time())
    out_json = REPORTS / f"trusted_agi_uplift_pipeline_{ts}.json"
    latest_json = REPORTS / "trusted_agi_uplift_pipeline_latest.json"
    out_md = REPORTS / f"trusted_agi_uplift_pipeline_{ts}.md"
    latest_md = REPORTS / "trusted_agi_uplift_pipeline_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")

    dist_m = (payload.get("distillation") or {}).get("metrics") or {}
    tw = payload.get("trusted_weight_training") or {}
    bsum = (payload.get("bootstrap") or {}).get("summary") or {}
    lines = [
        "# Trusted AGI Uplift Pipeline",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
        "## Distillation",
        f"- delta_schema_valid_rate: `{float(dist_m.get('delta_schema_valid_rate', 0.0)):+.6f}`",
        "",
        "## Incremental Benchmark Gate",
        f"- threshold: `{float(args.benchmark_gain_threshold):.6f}`",
        f"- gain_before: `{gain_before:+.6f}`",
        f"- gain_after: `{gain_after:+.6f}`",
        f"- pass: `{benchmark_gate_pass}`",
        "",
        "## Weight Training",
        f"- model: `{tw.get('model_name', '')}`",
        f"- loss_initial: `{tw.get('loss_initial', None)}`",
        f"- loss_final: `{tw.get('loss_final', None)}`",
        f"- weights_latest_dir: `{tw.get('weights_latest_dir', '')}`",
        "",
        "## Bootstrap",
        f"- keep/discard/crash: `{int(bsum.get('keep', 0))}/{int(bsum.get('discard', 0))}/{int(bsum.get('crash', 0))}`",
        "",
        "## Step Status",
    ]
    for s in steps:
        lines.append(f"- {s['name']}: returncode={s['returncode']}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
