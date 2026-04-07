#!/usr/bin/env python3
"""Orchestrate nano seed generation + distillation pipeline and emit a combined report."""

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
        "stdout_tail": "\n".join((p.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((p.stderr or "").splitlines()[-20:]),
    }


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nano-seeded distillation pipeline")
    parser.add_argument("--crystal", default="h2q_qwen_crystal.pt")
    parser.add_argument("--dataset", default="reports/self_eval_distill_dataset_latest.json")
    parser.add_argument("--seed-count", type=int, default=128)
    parser.add_argument("--seed-dim", type=int, default=64)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--teacher-provider", choices=["deepseek", "heuristic"], default="heuristic")
    parser.add_argument("--output-prefix", default="nano_seeded_distillation_pipeline")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    steps: List[Dict[str, Any]] = []
    steps.append(
        _run(
            [
                str(PY),
                "tools/train_nano_core_seed_from_crystal.py",
                "--crystal",
                args.crystal,
                "--dataset",
                args.dataset,
                "--seed-count",
                str(args.seed_count),
                "--seed-dim",
                str(args.seed_dim),
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
                str(args.sessions),
                "--teacher-provider",
                args.teacher_provider,
            ],
            "run_distillation_pipeline",
        )
    )

    seed_meta = _load(REPORTS / "nano_core_seed_latest.json")
    distill = _load(REPORTS / "self_eval_distillation_pipeline_latest.json")
    metrics = distill.get("metrics") or {}

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "seed": {
            "path": str(REPORTS / "nano_core_seed_latest.pt"),
            "meta": seed_meta,
        },
        "distillation": {
            "path": str(REPORTS / "self_eval_distillation_pipeline_latest.json"),
            "metrics": metrics,
        },
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "# Nano Seeded Distillation Pipeline",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- nano_seed_path: `{payload['seed']['path']}`",
        f"- distill_path: `{payload['distillation']['path']}`",
        "",
        "## Distillation Metrics",
        f"- baseline_schema_valid_rate: `{float(metrics.get('baseline_schema_valid_rate', 0.0)):.6f}`",
        f"- distilled_schema_valid_rate: `{float(metrics.get('distilled_schema_valid_rate', 0.0)):.6f}`",
        f"- delta_schema_valid_rate: `{float(metrics.get('delta_schema_valid_rate', 0.0)):+.6f}`",
        f"- schema_valid_rate_positive: `{bool(metrics.get('schema_valid_rate_positive', False))}`",
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
