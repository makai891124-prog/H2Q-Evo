#!/usr/bin/env python3
"""Execute self-improvement plan as tasks and verify weakness reduction."""

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
PY = ROOT / ".venv" / "bin" / "python"
REPORTS = ROOT / "reports"


def run_cmd(cmd: List[str], log_path: Path) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
        "log": str(log_path),
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_metric(path: Path, key: str) -> float:
    data = load_json(path)
    return float(((data.get("metrics") or {}).get(key, 0.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run self-improvement closed loop validation")
    parser.add_argument("--output-prefix", default="self_improvement_closed_loop")
    parser.add_argument("--sessions", type=int, default=3)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    baseline_prefix = "self_model_consistency_baseline"
    post_prefix = "self_model_consistency_post"

    task_runs: List[Dict[str, Any]] = []

    baseline_log = REPORTS / f"{args.output_prefix}_{ts}_baseline.log"
    baseline_cmd = [
        str(PY),
        "tools/run_self_model_consistency_benchmark.py",
        "--sessions",
        str(max(1, args.sessions)),
        "--schema-retries",
        "0",
        "--output-prefix",
        baseline_prefix,
    ]
    task_runs.append(
        {
            "task": "baseline_self_model_consistency",
            "run": run_cmd(baseline_cmd, baseline_log),
            "predicted_effect": "Establish baseline weakness for structured self-evaluation output.",
        }
    )

    probe_log = REPORTS / f"{args.output_prefix}_{ts}_schema_probe.log"
    probe_cmd = [
        str(PY),
        "tools/trusted_local_agi_chat.py",
        "--profile",
        "quick",
        "--skip-rsa",
        "--self-eval-max-retries",
        "2",
        "--no-auto-start-server",
    ]
    probe_input = (
        "请仅输出JSON，字段必须包含 capability_boundaries, failure_risks, improvement_plan, confidence。"
        "禁止使用.../TBD/占位词；每个action和metric都必须具体可执行。\n"
        "/exit\n"
    )
    proc = subprocess.run(
        probe_cmd,
        cwd=str(ROOT),
        text=True,
        input=probe_input,
        capture_output=True,
    )
    probe_log.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    task_runs.append(
        {
            "task": "schema_repair_probe",
            "run": {
                "cmd": probe_cmd,
                "returncode": proc.returncode,
                "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
                "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
                "log": str(probe_log),
            },
            "predicted_effect": "Activate strict schema retry path on introspection dialogue.",
        }
    )

    post_log = REPORTS / f"{args.output_prefix}_{ts}_post.log"
    post_cmd = [
        str(PY),
        "tools/run_self_model_consistency_benchmark.py",
        "--sessions",
        str(max(1, args.sessions)),
        "--schema-retries",
        "2",
        "--output-prefix",
        post_prefix,
    ]
    task_runs.append(
        {
            "task": "post_self_model_consistency",
            "run": run_cmd(post_cmd, post_log),
            "predicted_effect": "Increase schema validity and overall consistency score.",
        }
    )

    baseline_json = REPORTS / f"{baseline_prefix}_latest.json"
    post_json = REPORTS / f"{post_prefix}_latest.json"

    baseline_schema = latest_metric(baseline_json, "schema_valid_rate")
    post_schema = latest_metric(post_json, "schema_valid_rate")
    baseline_score = latest_metric(baseline_json, "overall_score")
    post_score = latest_metric(post_json, "overall_score")

    delta_schema = post_schema - baseline_schema
    delta_score = post_score - baseline_score

    weakness_reduced = (delta_schema >= 0.0) and (delta_score >= 0.0)

    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_weakness": "structured_self_evaluation_and_cross_session_consistency",
        "baseline": {
            "schema_valid_rate": baseline_schema,
            "overall_score": baseline_score,
            "report": str(baseline_json),
        },
        "post": {
            "schema_valid_rate": post_schema,
            "overall_score": post_score,
            "report": str(post_json),
        },
        "delta": {
            "schema_valid_rate": delta_schema,
            "overall_score": delta_score,
        },
        "weakness_reduced": weakness_reduced,
        "task_runs": task_runs,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Self-Improvement Closed Loop",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- target_weakness: `{payload['target_weakness']}`",
        f"- baseline.schema_valid_rate: `{baseline_schema:.6f}`",
        f"- post.schema_valid_rate: `{post_schema:.6f}`",
        f"- delta.schema_valid_rate: `{delta_schema:+.6f}`",
        f"- baseline.overall_score: `{baseline_score:.6f}`",
        f"- post.overall_score: `{post_score:.6f}`",
        f"- delta.overall_score: `{delta_score:+.6f}`",
        f"- weakness_reduced: `{weakness_reduced}`",
        "",
        "## Logs",
        f"- `{baseline_log}`",
        f"- `{probe_log}`",
        f"- `{post_log}`",
    ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")
    print(f"Latest JSON: {latest_json}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
