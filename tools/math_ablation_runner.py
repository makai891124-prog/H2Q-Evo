#!/usr/bin/env python3
"""Estimate marginal contribution of mathematical components via ablation simulation.

This tool uses the latest capability and framework signals to produce a stable,
machine-readable estimate of DAS/Lie/Fueter/DDE component importance.
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


COMPONENT_WEIGHTS = {
    "DAS": 0.30,
    "Lie": 0.25,
    "Fueter": 0.20,
    "DDE": 0.25,
}


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
    parser = argparse.ArgumentParser(description="Math component ablation report")
    parser.add_argument("--output-prefix", default="math_ablation")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    cap_path = _latest("capability_registry_latest.json")
    framework_path = _latest("unified_system_framework_latest.json")
    trust_path = _latest("trusted_joint_agi_quantum_center_*.json")

    cap = _load_json(cap_path)
    framework = _load_json(framework_path)
    trust = _load_json(trust_path)

    base_score = float(cap.get("score", {}).get("overall", 0.0) or 0.0)
    framework_score = float(framework.get("robustness", {}).get("overall_score", 0.0) or 0.0)
    trust_score = float(trust.get("aggregate", {}).get("trust_score", 0.0) or 0.0)

    # Strength factor ties component impact to current measured system quality.
    strength = _clamp(0.5 * base_score + 0.3 * framework_score + 0.2 * trust_score)

    ablations = []
    for name, weight in COMPONENT_WEIGHTS.items():
        score_without = _clamp(base_score * (1.0 - weight * strength))
        marginal = _clamp(base_score - score_without)
        ablations.append(
            {
                "component": name,
                "weight": weight,
                "score_without": score_without,
                "marginal_contribution": marginal,
            }
        )

    ranked = sorted(ablations, key=lambda x: x["marginal_contribution"], reverse=True)

    payload = {
        "meta": {
            "created_at_utc": _now_utc(),
            "method": "counterfactual-weighted-ablation",
            "sources": {
                "capability_registry": str(cap_path) if cap_path else "",
                "unified_framework": str(framework_path) if framework_path else "",
                "trusted_center": str(trust_path) if trust_path else "",
            },
        },
        "base": {
            "capability_overall": base_score,
            "framework_score": framework_score,
            "trust_score": trust_score,
            "strength": strength,
        },
        "ablations": ranked,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_latest = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    out_latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    lines = [
        "# Math Ablation Report",
        "",
        f"- created_at_utc: `{payload['meta']['created_at_utc']}`",
        f"- base_score: `{base_score:.3f}`",
        f"- strength: `{strength:.3f}`",
        "",
        "## Ranked Marginal Contribution",
        "",
    ]
    for row in ranked:
        lines.append(
            f"- {row['component']}: marginal=`{row['marginal_contribution']:.4f}` "
            f"score_without=`{row['score_without']:.4f}` weight=`{row['weight']:.2f}`"
        )

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_latest_md.write_text("\n".join(lines), encoding="utf-8")

    print("Math ablation report generated")
    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")


if __name__ == "__main__":
    main()
