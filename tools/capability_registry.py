#!/usr/bin/env python3
"""Build a capability registry snapshot from existing AGI reports.

This script provides a machine-readable baseline for future AGI gates.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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


@dataclass
class CapabilitySignals:
    breadth: float
    depth: float
    autonomy: float
    horizon: float
    robustness: float


def _derive_signals(
    round_obj: Dict[str, Any],
    monitor_obj: Dict[str, Any],
    gate_obj: Dict[str, Any],
    interactive_obj: Dict[str, Any],
) -> CapabilitySignals:
    round_section = round_obj.get("round", {})
    entries = round_section.get("entries", []) if isinstance(round_section, dict) else []
    acceptance = round_section.get("acceptance", {}) if isinstance(round_section, dict) else {}
    assist_summary = round_section.get("assist_summary", {}) if isinstance(round_section, dict) else {}

    prompt_count = int(round_section.get("prompt_count", 0) or 0)
    overall_ratio = float(acceptance.get("overall_ratio", 0.0) or 0.0)
    core_ratio = float(acceptance.get("core_ratio", 0.0) or 0.0)

    # Breadth proxy: combine cycle prompt coverage with interactive benchmark signal.
    prompt_breadth = min(1.0, float(prompt_count) / 20.0)
    active_prompt_breadth = min(1.0, float(prompt_count) / 6.0)
    interactive_success = float(
        interactive_obj.get("metrics", {}).get("success_rate", 0.0) or 0.0
    )
    if interactive_success > 0.0:
        breadth = max(prompt_breadth, min(1.0, 0.7 * interactive_success + 0.3 * active_prompt_breadth))
    else:
        breadth = max(prompt_breadth, 0.5 * active_prompt_breadth)

    # Depth proxy: acceptance quality over core and overall requirements.
    depth = max(0.0, min(1.0, 0.55 * overall_ratio + 0.45 * core_ratio))

    # Autonomy proxy: non-fallback route share.
    fallback_count = 0
    assist_enabled_count = 0
    for entry in entries:
        runtime = entry.get("runtime", {}) if isinstance(entry, dict) else {}
        route = str(runtime.get("route", ""))
        assist = runtime.get("assist", {}) if isinstance(runtime, dict) else {}
        if "fallback" in route:
            fallback_count += 1
        if bool(assist.get("enabled", False)):
            assist_enabled_count += 1
    non_fallback_ratio = 1.0
    if entries:
        non_fallback_ratio = max(0.0, 1.0 - (float(fallback_count) / float(len(entries))))

    assist_success_rate = float(assist_summary.get("success_rate", 0.0) or 0.0)
    if assist_enabled_count > 0 and assist_summary == {}:
        ok_count = 0
        for entry in entries:
            runtime = entry.get("runtime", {}) if isinstance(entry, dict) else {}
            assist = runtime.get("assist", {}) if isinstance(runtime, dict) else {}
            if bool(assist.get("enabled", False)) and bool(assist.get("ok", False)):
                ok_count += 1
        assist_success_rate = float(ok_count) / float(max(1, assist_enabled_count))

    autonomy = max(0.0, min(1.0, 0.7 * non_fallback_ratio + 0.3 * assist_success_rate))

    # Horizon proxy: sustained operation coverage over lookback window.
    lookback_rounds = int(monitor_obj.get("lookback_rounds", 0) or 0)
    round_count = int(monitor_obj.get("metrics", {}).get("round_count", 0) or 0)
    if lookback_rounds > 0:
        horizon = max(0.0, min(1.0, float(round_count) / float(lookback_rounds)))
    else:
        horizon = 0.0

    # Robustness proxy: combine monitor stability with previous gate component quality.
    gate_signals = gate_obj.get("signals", {}) if isinstance(gate_obj, dict) else {}
    quality_bits = [
        1.0 if bool(gate_signals.get("trust_ok", False)) else 0.0,
        1.0 if bool(gate_signals.get("acceptance_ok", False)) else 0.0,
        1.0 if bool(gate_signals.get("docker_ok", False)) else 0.0,
        1.0 if bool(gate_signals.get("monitor_ok", False)) else 0.0,
        1.0 if bool(gate_signals.get("assist_gate_ok", False)) else 0.0,
    ]
    gate_quality = sum(quality_bits) / float(len(quality_bits)) if quality_bits else 0.5
    monitor_metrics = monitor_obj.get("metrics", {}) if isinstance(monitor_obj, dict) else {}
    success_rate = float(monitor_metrics.get("success_rate", 0.0) or 0.0)
    assist_hit_rate = float(monitor_metrics.get("assist_hit_rate", 0.0) or 0.0)
    monitor_signal = max(0.0, min(1.0, 0.8 * success_rate + 0.2 * assist_hit_rate))
    robustness = max(0.0, min(1.0, 0.6 * gate_quality + 0.4 * monitor_signal))

    return CapabilitySignals(
        breadth=breadth,
        depth=depth,
        autonomy=autonomy,
        horizon=horizon,
        robustness=robustness,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build capability registry snapshot")
    parser.add_argument("--output-prefix", default="capability_registry")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    round_path = _latest("agi_self_evolution_round_*.json")
    monitor_path = _latest("agi_realtime_monitor_latest.json")
    gate_path = _latest("release_gate_latest.json")
    interactive_path = _latest("interactive_reasoning_benchmark_latest.json")

    round_obj = _load_json(round_path)
    monitor_obj = _load_json(monitor_path)
    gate_obj = _load_json(gate_path)
    interactive_obj = _load_json(interactive_path)

    signals = _derive_signals(
        round_obj=round_obj,
        monitor_obj=monitor_obj,
        gate_obj=gate_obj,
        interactive_obj=interactive_obj,
    )

    payload = {
        "meta": {
            "created_at_utc": _now_utc(),
            "sources": {
                "round": str(round_path) if round_path else "",
                "monitor": str(monitor_path) if monitor_path else "",
                "release_gate": str(gate_path) if gate_path else "",
                "interactive_reasoning": str(interactive_path) if interactive_path else "",
            },
        },
        "capabilities": {
            "breadth": signals.breadth,
            "depth": signals.depth,
            "autonomy": signals.autonomy,
            "horizon": signals.horizon,
            "robustness": signals.robustness,
        },
        "score": {
            "overall": max(
                0.0,
                min(
                    1.0,
                    0.25 * signals.breadth
                    + 0.20 * signals.depth
                    + 0.20 * signals.autonomy
                    + 0.20 * signals.horizon
                    + 0.15 * signals.robustness,
                ),
            )
        },
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_latest = REPORTS / f"{args.output_prefix}_latest.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Capability registry generated")
    print(f"JSON: {out_json}")
    print(f"Latest: {out_latest}")


if __name__ == "__main__":
    main()
