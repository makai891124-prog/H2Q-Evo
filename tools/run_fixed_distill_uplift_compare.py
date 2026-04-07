#!/usr/bin/env python3
"""Run fixed distillation_uplift rounds and generate keep-only comparison report."""

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


def _run(cmd: List[str], timeout_sec: int) -> int:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=max(60, int(timeout_sec)),
        )
        return int(p.returncode)
    except subprocess.TimeoutExpired:
        return 124


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _resolve_timeout_sec(timeout_sec: int, max_samples: int, teacher_provider: str) -> int:
    if int(timeout_sec) > 0:
        return int(timeout_sec)
    provider_factor = 24 if str(teacher_provider) == "deepseek" else 10
    # Local resource-aware auto timeout: linear to sample size, tuned for macOS/API latency.
    return max(900, 240 + int(max_samples) * provider_factor)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed distillation uplift compare")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--synthetic-min-size", type=int, default=120)
    parser.add_argument("--teacher-provider", choices=["deepseek", "heuristic"], default="deepseek")
    parser.add_argument("--teacher-key-file", default="secrets/deepseek_api_key.txt")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument(
        "--execution-mode",
        choices=["full", "compressed"],
        default="compressed",
        help="compressed: first round full collect, later rounds reuse dataset and skip recollection.",
    )
    parser.add_argument("--output-prefix", default="fixed_distillation_uplift_compare")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    timeout_sec = _resolve_timeout_sec(
        timeout_sec=int(args.timeout_sec),
        max_samples=int(args.max_samples),
        teacher_provider=args.teacher_provider,
    )

    rounds: List[Dict[str, Any]] = []

    for i in range(1, max(1, args.rounds) + 1):
        distill_cmd = [
                str(PY),
                "tools/run_self_eval_distillation_pipeline.py",
                "--sessions",
                str(max(1, int(args.sessions))),
                "--teacher-provider",
                args.teacher_provider,
                "--teacher-key-file",
                args.teacher_key_file,
                "--max-samples",
                str(max(1, int(args.max_samples))),
                "--synthetic-min-size",
                str(max(0, int(args.synthetic_min_size))),
                "--include-valid-prompts",
            ]

        # Structural compression: keep round-1 full, then reuse distilled dataset in later rounds.
        if args.execution_mode == "compressed" and i > 1:
            distill_cmd.extend(
                [
                    "--execution-mode",
                    "compressed",
                    "--skip-collect",
                    "--dataset",
                    "reports/self_eval_distill_dataset_latest.json",
                ]
            )
        rc_distill = _run(distill_cmd, timeout_sec=timeout_sec)

        rc_research = _run([str(PY), "tools/run_research_aggregation_cross_validation.py"], timeout_sec=timeout_sec)
        rc_systemic = _run(
            [
                str(PY),
                "tools/run_systemic_platform_joint_capability_assessment.py",
                "--ci-safe",
                "--blueprint-cycles",
                "1",
                "--longrun-cycles",
                "1",
            ],
            timeout_sec=timeout_sec,
        )

        dist = _read_json(REPORTS / "self_eval_distillation_pipeline_latest.json")
        research = _read_json(REPORTS / "research_aggregation_cross_validation_latest.json")
        systemic = _read_json(REPORTS / "systemic_platform_joint_capability_latest.json")

        delta = _f(((dist.get("metrics") or {}).get("delta_schema_valid_rate", 0.0)))
        research_score = _f(((research.get("aggregate_effectiveness") or {}).get("score", 0.0)))
        systemic_score = _f(((systemic.get("aggregate_effectiveness") or {}).get("score", 0.0)))

        keep = (rc_distill == 0 and rc_research == 0 and rc_systemic == 0 and delta >= 0.0)

        rounds.append(
            {
                "round": i,
                "delta_schema_valid_rate": delta,
                "research_aggregate": research_score,
                "systemic_score": systemic_score,
                "keep": keep,
                "returncodes": {
                    "distillation_uplift": rc_distill,
                    "research_cross_validation": rc_research,
                    "systemic_joint_capability": rc_systemic,
                },
            }
        )

    keep_only = [r for r in rounds if r["keep"]]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rounds": int(args.rounds),
            "sessions": int(args.sessions),
            "max_samples": int(args.max_samples),
            "synthetic_min_size": int(args.synthetic_min_size),
            "teacher_provider": args.teacher_provider,
            "execution_mode": args.execution_mode,
            "timeout_sec": int(timeout_sec),
        },
        "criteria": "keep when all return codes are 0 and delta_schema_valid_rate >= 0",
        "rounds": rounds,
        "keep_only": keep_only,
        "summary": {
            "keep_count": len(keep_only),
            "total_rounds": len(rounds),
            "best_keep_research_aggregate": max((r["research_aggregate"] for r in keep_only), default=0.0),
            "best_keep_systemic_score": max((r["systemic_score"] for r in keep_only), default=0.0),
        },
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Fixed Distillation Uplift Compare",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- keep_count: `{payload['summary']['keep_count']}/{payload['summary']['total_rounds']}`",
        "",
        "## Keep Only",
    ]
    for row in keep_only:
        lines.append(
            f"- round={row['round']}, delta={row['delta_schema_valid_rate']:+.6f}, "
            f"research={row['research_aggregate']:.6f}, systemic={row['systemic_score']:.6f}"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
