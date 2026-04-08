#!/usr/bin/env python3
"""Generate a benchmark before/after delta report with root-cause hints."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get_score(payload: Dict[str, Any], key: str) -> float:
    val = payload.get(key)
    try:
        return float(val)
    except Exception:
        return 0.0


def _collect_task_scores(payload: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for group in ("glue", "superglue"):
        tasks = payload.get(group) or {}
        if isinstance(tasks, dict):
            for task, task_payload in tasks.items():
                if isinstance(task_payload, dict) and "accuracy" in task_payload:
                    try:
                        out[f"{group}.{task}"] = float(task_payload["accuracy"])
                    except Exception:
                        pass
    mmlu = payload.get("mmlu")
    if isinstance(mmlu, dict) and "accuracy" in mmlu:
        try:
            out["mmlu.accuracy"] = float(mmlu["accuracy"])
        except Exception:
            pass
    math_payload = payload.get("math")
    if isinstance(math_payload, dict) and "accuracy" in math_payload:
        try:
            out["math.accuracy"] = float(math_payload["accuracy"])
        except Exception:
            pass
    return out


def _task_deltas(before: Dict[str, float], after: Dict[str, float]) -> List[Dict[str, Any]]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    rows: List[Dict[str, Any]] = []
    for k in keys:
        b = float(before.get(k, 0.0))
        a = float(after.get(k, 0.0))
        rows.append({"task": k, "before": b, "after": a, "delta": a - b})
    rows.sort(key=lambda x: x["delta"])
    return rows


def _collect_failures(after_payload: Dict[str, Any], public_payload: Dict[str, Any]) -> List[str]:
    failures: List[str] = []

    for group in ("glue", "superglue"):
        tasks = after_payload.get(group) or {}
        if isinstance(tasks, dict):
            for task, task_payload in tasks.items():
                if isinstance(task_payload, dict) and task_payload.get("error"):
                    failures.append(f"{group}.{task}: {task_payload.get('error')}")

    infra = public_payload.get("infra_probe") or {}
    if infra.get("infra_invalid"):
        failures.append(f"infra_invalid: {infra.get('summary', 'probe failed')}")
    if infra.get("json_probe_error"):
        failures.append(f"json_probe_error: {infra.get('json_probe_error')}")

    guard = ((public_payload.get("solver_runs") or {}).get("model-json-guard") or {}).get("metrics") or {}
    parse_errors = int(guard.get("parse_errors", 0) or 0)
    if parse_errors > 0:
        failures.append(f"model-json-guard.parse_errors: {parse_errors}")

    return failures


def _derive_causes(
    overall_delta: float,
    task_rows: List[Dict[str, Any]],
    public_payload: Dict[str, Any],
    research_payload: Dict[str, Any],
) -> List[str]:
    causes: List[str] = []
    infra = public_payload.get("infra_probe") or {}

    if infra.get("infra_invalid"):
        causes.append("Serving protocol instability: /generate JSON contract is not consistently satisfied.")

    guard = ((public_payload.get("solver_runs") or {}).get("model-json-guard") or {}).get("metrics") or {}
    if int(guard.get("parse_errors", 0) or 0) > 0:
        causes.append("High JSON parse-error debt suggests response formatting drift and contract leakage.")

    worst = [r for r in task_rows if r["delta"] < -0.05][:3]
    if worst:
        regressed = ", ".join(r["task"] for r in worst)
        causes.append(f"Task-specific regressions concentrated in: {regressed}.")

    if overall_delta < 0:
        causes.append("Global score dropped, indicating current decision mapping may overfit some tasks and harm others.")

    cv = (research_payload.get("cross_validation") or {}).get("std_score")
    try:
        if float(cv) > 0.0015:
            causes.append("Cross-validation variance increased; robustness sensitivity may be under-controlled.")
    except Exception:
        pass

    if not causes:
        causes.append("No dominant failure mode detected from current artifacts.")

    return causes


def _build_try_plan(causes: List[str]) -> List[str]:
    plan: List[str] = []
    if any("protocol" in c.lower() or "json" in c.lower() for c in causes):
        plan.append("Force /generate contract mode on probe traffic and verify with a dedicated contract smoke test in CI.")
    plan.append("Add per-task calibration constants (NLI/Bool/MCQ) and tune on a held-out split before full benchmark runs.")
    plan.append("Track per-task confidence histograms and reject low-confidence predictions with deterministic fallback labels.")
    plan.append("Gate release on three checks: infra_valid=true, parse_errors<=target, and non-negative overall_score delta.")
    return plan


def _md_table(rows: List[Dict[str, Any]], limit: int) -> str:
    head = "| Task | Before | After | Delta |\n|---|---:|---:|---:|"
    lines = [head]
    for r in rows[:limit]:
        lines.append(
            f"| {r['task']} | {r['before']:.4f} | {r['after']:.4f} | {r['delta']:+.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark before/after delta report")
    parser.add_argument("--before", default="", help="Path to baseline benchmark JSON")
    parser.add_argument("--after", default="agi_benchmark_results.json", help="Path to current benchmark JSON")
    parser.add_argument(
        "--public-report",
        default="reports/public_benchmark_algorithm_effect_latest.json",
        help="Path to public benchmark effect report",
    )
    parser.add_argument(
        "--research-report",
        default="reports/research_aggregation_cross_validation_latest.json",
        help="Path to research aggregation report",
    )
    parser.add_argument("--output-prefix", default="benchmark_delta_analysis")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    after_path = (ROOT / args.after).resolve()
    public_path = (ROOT / args.public_report).resolve()
    research_path = (ROOT / args.research_report).resolve()

    after_payload = _read_json(after_path)
    public_payload = _read_json(public_path) if public_path.exists() else {}
    research_payload = _read_json(research_path) if research_path.exists() else {}

    if args.before:
        before_path = (ROOT / args.before).resolve()
        before_payload = _read_json(before_path)
        before_available = True
    else:
        before_path = Path("")
        before_payload = {}
        before_available = False

    before_overall = _safe_get_score(before_payload, "overall_score") if before_available else 0.0
    after_overall = _safe_get_score(after_payload, "overall_score")
    overall_delta = after_overall - before_overall if before_available else 0.0

    before_tasks = _collect_task_scores(before_payload) if before_available else {}
    after_tasks = _collect_task_scores(after_payload)
    task_rows = _task_deltas(before_tasks, after_tasks) if before_available else []

    failures = _collect_failures(after_payload, public_payload)
    causes = _derive_causes(overall_delta, task_rows, public_payload, research_payload)
    try_plan = _build_try_plan(causes)

    top_regressions = [r for r in task_rows if r["delta"] < 0][:10] if before_available else []
    top_improvements = sorted([r for r in task_rows if r["delta"] > 0], key=lambda x: x["delta"], reverse=True)[:10] if before_available else []

    payload = {
        "generated_at_utc": _now_utc(),
        "before": str(before_path) if before_available else "",
        "after": str(after_path),
        "before_available": before_available,
        "summary": {
            "overall_before": before_overall if before_available else None,
            "overall_after": after_overall,
            "overall_delta": overall_delta if before_available else None,
        },
        "task_deltas": task_rows,
        "top_regressions": top_regressions,
        "top_improvements": top_improvements,
        "failures": failures,
        "potential_causes": causes,
        "recommended_attempts": try_plan,
        "linked_reports": {
            "public_report": str(public_path),
            "research_report": str(research_path),
        },
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Benchmark Delta Analysis")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append(f"- after: `{after_path}`")
    if before_available:
        lines.append(f"- before: `{before_path}`")
        lines.append(f"- overall_before: {before_overall:.6f}")
        lines.append(f"- overall_after: {after_overall:.6f}")
        lines.append(f"- overall_delta: {overall_delta:+.6f}")
    else:
        lines.append("- before: (not provided)")
        lines.append(f"- overall_after: {after_overall:.6f}")

    if before_available:
        lines.append("")
        lines.append("## Top Regressions")
        lines.append(_md_table(top_regressions, limit=min(10, len(top_regressions))))
        lines.append("")
        lines.append("## Top Improvements")
        lines.append(_md_table(top_improvements, limit=min(10, len(top_improvements))))

    lines.append("")
    lines.append("## Failure Signals")
    if failures:
        for item in failures:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Potential Causes")
    for item in causes:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Recommended Attempts")
    for item in try_plan:
        lines.append(f"- {item}")

    md_text = "\n".join(lines) + "\n"
    out_md.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    print(f"Delta JSON: {out_json}")
    print(f"Delta MD: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
