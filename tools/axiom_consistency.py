#!/usr/bin/env python3
"""Axiom consistency scoring helpers for bootstrap decisions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "axiom_contract.json"


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_axiom_contract(path: Path | None = None) -> Dict[str, Any]:
    p = path or DEFAULT_CONTRACT
    obj = _load_json(p)
    if obj and isinstance(obj.get("axioms"), list):
        return obj
    return {"version": "empty", "axioms": []}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _metric_value(metric: str, snapshot: Dict[str, float], signals: Dict[str, float], meaning_score: float, experiments: List[Dict[str, Any]]) -> float:
    if metric == "abs(ap_gp_boundary_ratio - phi)":
        d = abs(_f(signals.get("ap_gp_boundary_ratio", 0.0)) - _f(signals.get("phi", (1 + math.sqrt(5)) / 2)))
        return _clip01(1.0 - d / max(_f(signals.get("phi", 1.618), 1.618), 1e-9))
    if metric == "nonnegative_snapshot":
        vals = [
            _f(snapshot.get("distill_delta", 0.0)),
            _f(snapshot.get("research_aggregate", 0.0)),
            _f(snapshot.get("systemic_score", 0.0)),
        ]
        return 1.0 if all(v >= 0.0 for v in vals) else 0.0
    if metric == "1-crash_rate":
        total = max(1, len(experiments))
        crash = sum(1 for e in experiments if str(e.get("status", "")).lower() == "crash")
        return _clip01(1.0 - crash / total)
    if metric == "z1_flatness":
        # Flat background proxy: lower |projection_acceleration| is better.
        accel = abs(_f(signals.get("projection_acceleration", 0.0)))
        return _clip01(1.0 - min(1.0, accel / 0.05))
    if metric == "z2_accel_quality":
        accel = _f(signals.get("projection_acceleration", 0.0))
        return _clip01(0.5 + 0.5 * math.tanh(8.0 * accel))
    if metric == "meaning_score":
        return _clip01(meaning_score)
    if metric == "systemic_score":
        return _clip01(_f(snapshot.get("systemic_score", 0.0)))
    return 0.0


def evaluate_axiom_consistency(
    snapshot: Dict[str, float],
    signals: Dict[str, float],
    meaning_score: float,
    experiments: List[Dict[str, Any]],
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    axioms = contract.get("axioms") or []
    if not axioms:
        return {
            "score": 0.0,
            "pass_rate": 0.0,
            "passed": 0,
            "total": 0,
            "components": [],
            "violations": [],
        }

    comps: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0
    passed = 0

    for ax in axioms:
        aid = str(ax.get("id", ""))
        metric = str(ax.get("metric", ""))
        threshold = _f(ax.get("threshold", 0.5), 0.5)
        weight = _f(ax.get("weight", 1.0), 1.0)
        direction = str(ax.get("direction", "max")).lower()

        value = _metric_value(metric, snapshot, signals, meaning_score, experiments)
        ok = value >= threshold if direction == "max" else value <= threshold
        if ok:
            passed += 1

        comps.append(
            {
                "id": aid,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "weight": weight,
                "pass": ok,
            }
        )
        weighted_sum += weight * _clip01(value)
        weight_total += weight

    score = weighted_sum / max(weight_total, 1e-9)
    total = len(comps)
    violations = [c["id"] for c in comps if not c["pass"]]

    return {
        "score": _clip01(score),
        "pass_rate": 0.0 if total == 0 else passed / total,
        "passed": passed,
        "total": total,
        "components": comps,
        "violations": violations,
    }
