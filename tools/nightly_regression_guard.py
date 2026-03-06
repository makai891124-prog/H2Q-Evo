#!/usr/bin/env python3
"""Nightly day-over-day regression guard for AGI alignment and robustness.

Compares latest and previous report snapshots and optionally fails on large drops.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_two(glob_pat: str) -> List[Path]:
    files = sorted(REPORTS.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    if len(files) >= 2:
        return [files[-2], files[-1]]
    return files


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly day-over-day regression guard")
    parser.add_argument("--warn-drop", type=float, default=0.02, help="Warn when score drop >= this value")
    parser.add_argument("--fail-drop", type=float, default=0.05, help="Fail when score drop >= this value")
    parser.add_argument("--output-prefix", default="nightly_regression_guard")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    alignment_files = _latest_two("public_alignment_report_*.json")
    capability_files = _latest_two("capability_registry_*.json")

    alignment_prev = _load_json(alignment_files[0]) if len(alignment_files) >= 2 else {}
    alignment_curr = _load_json(alignment_files[-1]) if alignment_files else {}
    capability_prev = _load_json(capability_files[0]) if len(capability_files) >= 2 else {}
    capability_curr = _load_json(capability_files[-1]) if capability_files else {}

    prev_overall = _to_float(alignment_prev.get("alignment", {}).get("overall", 0.0))
    curr_overall = _to_float(alignment_curr.get("alignment", {}).get("overall", 0.0))
    prev_robust = _to_float(capability_prev.get("capabilities", {}).get("robustness", 0.0))
    curr_robust = _to_float(capability_curr.get("capabilities", {}).get("robustness", 0.0))

    overall_drop = max(0.0, prev_overall - curr_overall)
    robustness_drop = max(0.0, prev_robust - curr_robust)

    can_compare_alignment = len(alignment_files) >= 2
    can_compare_robustness = len(capability_files) >= 2

    warn_triggered = (
        (can_compare_alignment and overall_drop >= max(0.0, args.warn_drop))
        or (can_compare_robustness and robustness_drop >= max(0.0, args.warn_drop))
    )
    fail_triggered = (
        (can_compare_alignment and overall_drop >= max(0.0, args.fail_drop))
        or (can_compare_robustness and robustness_drop >= max(0.0, args.fail_drop))
    )

    payload = {
        "meta": {
            "created_at_utc": _now_utc(),
            "warn_drop": max(0.0, args.warn_drop),
            "fail_drop": max(0.0, args.fail_drop),
            "sources": {
                "alignment_previous": str(alignment_files[0]) if len(alignment_files) >= 2 else "",
                "alignment_current": str(alignment_files[-1]) if alignment_files else "",
                "capability_previous": str(capability_files[0]) if len(capability_files) >= 2 else "",
                "capability_current": str(capability_files[-1]) if capability_files else "",
            },
        },
        "comparison": {
            "alignment_overall": {
                "previous": prev_overall,
                "current": curr_overall,
                "drop": overall_drop,
                "comparable": can_compare_alignment,
            },
            "robustness": {
                "previous": prev_robust,
                "current": curr_robust,
                "drop": robustness_drop,
                "comparable": can_compare_robustness,
            },
        },
        "status": {
            "warn": warn_triggered,
            "fail": fail_triggered,
            "ok": not fail_triggered,
        },
        "notes": [
            "If comparable is false, historical artifacts are insufficient for day-over-day checks.",
            "Warn does not fail the pipeline, fail exits with non-zero.",
        ],
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    out_latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    lines = [
        "# Nightly Regression Guard",
        "",
        f"- created_at_utc: `{payload['meta']['created_at_utc']}`",
        f"- warn_drop: `{payload['meta']['warn_drop']:.3f}`",
        f"- fail_drop: `{payload['meta']['fail_drop']:.3f}`",
        f"- alignment_overall: `{curr_overall:.3f}` (prev `{prev_overall:.3f}`, drop `{overall_drop:.3f}`, comparable `{can_compare_alignment}`)",
        f"- robustness: `{curr_robust:.3f}` (prev `{prev_robust:.3f}`, drop `{robustness_drop:.3f}`, comparable `{can_compare_robustness}`)",
        f"- warn: `{warn_triggered}`",
        f"- fail: `{fail_triggered}`",
        "",
        "## Sources",
        "",
        f"- alignment_previous: `{payload['meta']['sources']['alignment_previous']}`",
        f"- alignment_current: `{payload['meta']['sources']['alignment_current']}`",
        f"- capability_previous: `{payload['meta']['sources']['capability_previous']}`",
        f"- capability_current: `{payload['meta']['sources']['capability_current']}`",
        "",
    ]

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_latest_md.write_text("\n".join(lines), encoding="utf-8")

    print("Nightly regression guard completed")
    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")
    if warn_triggered:
        print("Warning: detected day-over-day decline above warn threshold")
    if fail_triggered:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
