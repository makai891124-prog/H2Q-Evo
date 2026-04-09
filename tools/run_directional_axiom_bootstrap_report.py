#!/usr/bin/env python3
"""Generate directional axiom bootstrap report from autonomous evolution state."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / max(1, len(values)))


def _phase_counts(history: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"simulation": 0, "shadow": 0, "gate_enforced": 0, "disabled": 0}
    for row in history:
        phase = str(row.get("phase", "disabled"))
        out[phase] = out.get(phase, 0) + 1
    return out


def _recommendations(summary: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    if not summary.get("enabled", False):
        recs.append("Set ENABLE_DIRECTIONAL_AXIOM=true to start collecting directional manifold signals.")
        return recs

    if int(summary.get("samples", 0)) < 3:
        recs.append("Collect more generations before promoting from simulation to shadow.")
    if float(summary.get("rolling_horizon_pass_rate", 0.0)) < 0.8:
        recs.append("Tune thresholds or improve latent quality before gate-enforced rollout.")
    if float(summary.get("avg_projection_error", 1.0)) > 0.3:
        recs.append("Projection error is high; adjust rank constraint or improve embedding manifold quality.")
    if float(summary.get("avg_direction_stability", 0.0)) < 0.8:
        recs.append("Direction stability is low; keep shadow mode and monitor longer horizon windows.")
    if str(summary.get("latest_phase", "")) == "gate_enforced":
        recs.append("Keep rollback guard active: fallback to shadow if stability drops under gate threshold.")

    if not recs:
        recs.append("Directional axiom prototype looks healthy for current rollout depth.")
    return recs


def main() -> int:
    parser = argparse.ArgumentParser(description="Directional axiom bootstrap report")
    parser.add_argument("--state-file", default="autonomous_evolution_state.json")
    parser.add_argument("--output-prefix", default="directional_axiom_bootstrap")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = ROOT / state_path

    state = _load_json(state_path)
    history = state.get("directional_axiom_metrics_history", []) if isinstance(state, dict) else []
    history = history if isinstance(history, list) else []

    pass_values = [1.0 if bool(row.get("rolling_horizon_pass", False)) else 0.0 for row in history]
    proj_values = [float(row.get("projection_error", 0.0) or 0.0) for row in history]
    stab_values = [float(row.get("direction_stability", 0.0) or 0.0) for row in history]

    summary = {
        "enabled": bool(state.get("directional_axiom_enabled", False)),
        "latest_phase": str(state.get("directional_axiom_phase", "disabled")),
        "samples": len(history),
        "rolling_horizon_pass_rate": _mean(pass_values),
        "avg_projection_error": _mean(proj_values),
        "avg_direction_stability": _mean(stab_values),
        "phase_counts": _phase_counts(history),
    }

    payload = {
        "generated_at_utc": _now_utc(),
        "sources": {
            "state_file": str(state_path),
        },
        "summary": summary,
        "latest_metrics": history[-1] if history else {},
        "recommendations": _recommendations(summary),
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Directional Axiom Bootstrap Report",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- enabled: `{summary['enabled']}`",
        f"- latest_phase: `{summary['latest_phase']}`",
        f"- samples: `{summary['samples']}`",
        f"- rolling_horizon_pass_rate: `{summary['rolling_horizon_pass_rate']:.4f}`",
        f"- avg_projection_error: `{summary['avg_projection_error']:.4f}`",
        f"- avg_direction_stability: `{summary['avg_direction_stability']:.4f}`",
        "",
        "## Phase Counts",
        "",
    ]
    for name, value in summary["phase_counts"].items():
        lines.append(f"- {name}: `{value}`")

    lines.extend(["", "## Recommendations", ""])
    for rec in payload["recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
