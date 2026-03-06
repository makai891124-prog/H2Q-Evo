#!/usr/bin/env python3
"""Auto-generated blueprint candidate module."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest(glob_pat: str):
    files = sorted(REPORTS.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_json(path):
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generated blueprint candidate")
    parser.add_argument("--output-prefix", default="generated_blueprint_maintenance_patch")
    parser.add_argument("--min-objective", type=float, default=0.0)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    cap = load_json(latest("capability_registry_latest.json"))
    align = load_json(latest("public_alignment_report_latest.json"))

    caps = cap.get("capabilities", {}) if isinstance(cap, dict) else {}
    alignment = align.get("alignment", {}) if isinstance(align, dict) else {}

    robustness = float(caps.get("robustness", 0.0) or 0.0)
    horizon = float(caps.get("horizon", 0.0) or 0.0)
    breadth = float(caps.get("breadth", 0.0) or 0.0)
    overall = float(alignment.get("overall", 0.0) or 0.0)

    objective = max(0.0, min(1.0, 0.35 * overall + 0.25 * robustness + 0.20 * horizon + 0.20 * breadth))

    payload = {
        "meta": {
            "created_at_utc": now_utc(),
            "module_id": "maintenance_patch",
            "title": "Generated maintenance candidate",
            "focus": "maintenance",
        },
        "result": {
            "objective": objective,
            "meets_min_objective": objective >= max(0.0, args.min_objective),
        },
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_latest = REPORTS / f"{args.output_prefix}_latest.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated blueprint module completed")
    print(f"JSON: {out_json}")

    if not payload["result"]["meets_min_objective"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
