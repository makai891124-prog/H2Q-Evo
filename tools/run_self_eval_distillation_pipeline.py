#!/usr/bin/env python3
"""Run self-eval distillation pipeline: collect -> train -> benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"


def _run(cmd: List[str], name: str, timeout_sec: int) -> Dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=max(60, int(timeout_sec)),
        )
        return {
            "name": name,
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-25:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-25:]),
            "elapsed_sec": time.time() - start,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "cmd": cmd,
            "returncode": 124,
            "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-25:]),
            "stderr_tail": "\n".join(((exc.stderr or "") + "\nTIMEOUT").splitlines()[-25:]),
            "elapsed_sec": time.time() - start,
        }


def _latest_metric(path: Path, key: str, default: float = 0.0) -> float:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(((payload.get("metrics") or {}).get(key, default)))
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Run self-eval distillation enhancement pipeline")
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--teacher-provider", choices=["deepseek", "heuristic"], default="deepseek")
    parser.add_argument("--teacher-key-file", default="secrets/deepseek_api_key.txt")
    parser.add_argument("--include-valid-prompts", action="store_true")
    parser.add_argument("--synthetic-min-size", type=int, default=120)
    parser.add_argument("--dataset", default="reports/self_eval_distill_dataset_latest.json")
    parser.add_argument(
        "--execution-mode",
        choices=["full", "compressed"],
        default="full",
        help="full: collect+train+benchmark; compressed: skip collect when dataset already exists.",
    )
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--collect-timeout-sec", type=int, default=2400)
    parser.add_argument("--train-timeout-sec", type=int, default=600)
    parser.add_argument("--benchmark-timeout-sec", type=int, default=900)
    parser.add_argument("--output-prefix", default="self_eval_distillation_pipeline")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    steps = []

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    do_collect = bool(args.execution_mode == "full")
    if args.execution_mode == "compressed":
        do_collect = not dataset_path.exists()
    if args.skip_collect:
        do_collect = False

    if do_collect:
        steps.append(
            _run(
                [
                    str(PY),
                    "tools/collect_self_eval_distill_samples.py",
                    "--teacher-provider",
                    args.teacher_provider,
                    "--teacher-key-file",
                    args.teacher_key_file,
                    "--max-samples",
                    str(max(1, args.max_samples)),
                    "--synthetic-min-size",
                    str(max(0, args.synthetic_min_size)),
                    *(["--include-valid-prompts"] if args.include_valid_prompts else []),
                ],
                "collect_samples",
                timeout_sec=int(args.collect_timeout_sec),
            )
        )
        dataset_for_train = "reports/self_eval_distill_dataset_latest.json"
    else:
        steps.append(
            {
                "name": "collect_samples",
                "cmd": [],
                "returncode": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "elapsed_sec": 0.0,
                "skipped": True,
                "skip_reason": "execution-mode compressed or --skip-collect",
            }
        )
        dataset_for_train = str(dataset_path)

    steps.append(
        _run(
            [str(PY), "tools/train_self_eval_distillation_adapter.py", "--dataset", dataset_for_train],
            "train_adapter",
            timeout_sec=int(args.train_timeout_sec),
        )
    )

    steps.append(
        _run(
            [
                str(PY),
                "tools/run_self_model_consistency_benchmark.py",
                "--sessions",
                str(max(1, args.sessions)),
                "--schema-retries",
                "2",
                "--output-prefix",
                "self_model_consistency_distilled",
                "--self-eval-distill-model",
                "reports/self_eval_distill_model_latest.json",
            ],
            "distilled_benchmark",
            timeout_sec=int(args.benchmark_timeout_sec),
        )
    )

    baseline_rate = _latest_metric(REPORTS / "self_model_consistency_baseline_latest.json", "schema_valid_rate", 0.0)
    distilled_rate = _latest_metric(REPORTS / "self_model_consistency_distilled_latest.json", "schema_valid_rate", 0.0)
    delta = distilled_rate - baseline_rate

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "execution_mode": args.execution_mode,
            "skip_collect": bool(args.skip_collect),
            "dataset": str(dataset_path),
            "collect_timeout_sec": int(args.collect_timeout_sec),
            "train_timeout_sec": int(args.train_timeout_sec),
            "benchmark_timeout_sec": int(args.benchmark_timeout_sec),
        },
        "steps": steps,
        "metrics": {
            "baseline_schema_valid_rate": baseline_rate,
            "distilled_schema_valid_rate": distilled_rate,
            "delta_schema_valid_rate": delta,
            "schema_valid_rate_positive": distilled_rate > 0.0,
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Self-Eval Distillation Pipeline",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- baseline_schema_valid_rate: `{baseline_rate:.6f}`",
        f"- distilled_schema_valid_rate: `{distilled_rate:.6f}`",
        f"- delta_schema_valid_rate: `{delta:+.6f}`",
        f"- schema_valid_rate_positive: `{distilled_rate > 0.0}`",
        "",
        "## Steps",
    ]
    for s in steps:
        lines.append(f"- {s['name']}: returncode={s['returncode']}")

    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    latest_md.write_text(text, encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
