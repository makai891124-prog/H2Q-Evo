#!/usr/bin/env python3
"""Generate daily public alignment report against AGI-facing benchmark dimensions.

Alignment dimensions:
- ARC-AGI style interactive reasoning breadth/generalization.
- SWE-bench style engineering problem-solving closure.
- METR style horizon/autonomy robustness under sustained tasks.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest(glob_pat: str) -> Optional[Path]:
    files = sorted(REPORTS.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Public benchmark alignment report")
    parser.add_argument("--output-prefix", default="public_alignment_report")
    parser.add_argument("--arc-target", type=float, default=0.65)
    parser.add_argument("--swe-target", type=float, default=0.55)
    parser.add_argument("--metr-target", type=float, default=0.60)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    cap_path = _latest("capability_registry_latest.json")
    gate_path = _latest("release_gate_latest.json")
    interactive_path = _latest("interactive_reasoning_benchmark_latest.json")
    math_ablation_path = _latest("math_ablation_latest.json")

    cap = _load_json(cap_path)
    gate = _load_json(gate_path)
    interactive = _load_json(interactive_path)
    math_ablation = _load_json(math_ablation_path)

    caps = cap.get("capabilities", {}) if isinstance(cap, dict) else {}
    gate_signals = gate.get("signals", {}) if isinstance(gate, dict) else {}
    interactive_metrics = interactive.get("metrics", {}) if isinstance(interactive, dict) else {}

    breadth = float(caps.get("breadth", 0.0) or 0.0)
    depth = float(caps.get("depth", 0.0) or 0.0)
    autonomy = float(caps.get("autonomy", 0.0) or 0.0)
    horizon = float(caps.get("horizon", 0.0) or 0.0)
    robustness = float(caps.get("robustness", 0.0) or 0.0)

    interactive_success = float(interactive_metrics.get("success_rate", 0.0) or 0.0)
    assist_success = float(gate_signals.get("assist_success_rate", 0.0) or 0.0)
    framework_score = float(gate_signals.get("framework_score", 0.0) or 0.0)

    arc_alignment = _clamp(0.35 * breadth + 0.25 * depth + 0.40 * interactive_success)
    swe_alignment = _clamp(0.35 * autonomy + 0.35 * assist_success + 0.30 * framework_score)
    metr_alignment = _clamp(0.45 * horizon + 0.35 * robustness + 0.20 * autonomy)

    overall_alignment = _clamp((arc_alignment + swe_alignment + metr_alignment) / 3.0)

    payload = {
        "meta": {
            "created_at_utc": _now_utc(),
            "sources": {
                "capability_registry": str(cap_path) if cap_path else "",
                "release_gate": str(gate_path) if gate_path else "",
                "interactive_benchmark": str(interactive_path) if interactive_path else "",
                "math_ablation": str(math_ablation_path) if math_ablation_path else "",
            },
        },
        "alignment": {
            "arc_agi": {
                "score": arc_alignment,
                "target": max(0.0, args.arc_target),
                "ok": arc_alignment >= max(0.0, args.arc_target),
            },
            "swe_bench": {
                "score": swe_alignment,
                "target": max(0.0, args.swe_target),
                "ok": swe_alignment >= max(0.0, args.swe_target),
            },
            "metr_horizon": {
                "score": metr_alignment,
                "target": max(0.0, args.metr_target),
                "ok": metr_alignment >= max(0.0, args.metr_target),
            },
            "overall": overall_alignment,
        },
        "notes": [
            "ARC-AGI alignment is approximated by breadth/depth plus interactive task success.",
            "SWE-bench alignment is approximated by autonomy, assist reliability, and framework score.",
            "METR alignment is approximated by sustained horizon, robustness, and autonomy.",
        ],
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_latest = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    out_latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    lines = [
        "# Public Alignment Report",
        "",
        f"- created_at_utc: `{payload['meta']['created_at_utc']}`",
        f"- overall: `{overall_alignment:.3f}`",
        f"- arc_agi: `{arc_alignment:.3f}` (target `{max(0.0, args.arc_target):.3f}`)",
        f"- swe_bench: `{swe_alignment:.3f}` (target `{max(0.0, args.swe_target):.3f}`)",
        f"- metr_horizon: `{metr_alignment:.3f}` (target `{max(0.0, args.metr_target):.3f}`)",
        "",
        "## Sources",
        "",
        f"- capability_registry: `{payload['meta']['sources']['capability_registry']}`",
        f"- release_gate: `{payload['meta']['sources']['release_gate']}`",
        f"- interactive_benchmark: `{payload['meta']['sources']['interactive_benchmark']}`",
        f"- math_ablation: `{payload['meta']['sources']['math_ablation']}`",
        "",
    ]

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_latest_md.write_text("\n".join(lines), encoding="utf-8")

    print("Public alignment report generated")
    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")


if __name__ == "__main__":
    main()
