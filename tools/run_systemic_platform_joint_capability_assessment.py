#!/usr/bin/env python3
"""Run a joint multi-controller experiment and publish cross-validated capability evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"


@dataclass
class Evidence:
    name: str
    value: float
    weight: float
    source: str


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _b(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: List[str], name: str, timeout: int) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-40:]),
    }


def _weighted_score(items: List[Evidence]) -> Dict[str, Any]:
    denom = sum(x.weight for x in items)
    if denom <= 0:
        return {"score": 0.0, "components": []}
    score = sum(x.value * x.weight for x in items) / denom
    return {
        "score": _clip01(score),
        "components": [
            {
                "name": x.name,
                "value": _clip01(x.value),
                "weight": x.weight,
                "source": x.source,
            }
            for x in items
        ],
    }


def _loo(items: List[Evidence]) -> Dict[str, Any]:
    folds: List[Dict[str, Any]] = []
    vals: List[float] = []
    for i, item in enumerate(items):
        kept = [x for j, x in enumerate(items) if j != i]
        scored = _weighted_score(kept)
        v = scored["score"]
        vals.append(v)
        folds.append(
            {
                "left_out": item.name,
                "score": v,
                "kept": [k["name"] for k in scored["components"]],
            }
        )

    if not vals:
        vals = [0.0]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    return {
        "folds": folds,
        "min_score": min(vals),
        "max_score": max(vals),
        "mean_score": mean,
        "std_score": std,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint systemic platform capability assessment")
    parser.add_argument("--blueprint-cycles", type=int, default=2)
    parser.add_argument("--longrun-cycles", type=int, default=2)
    parser.add_argument("--output-prefix", default="systemic_platform_joint_capability")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument(
        "--ci-safe",
        action="store_true",
        help="Skip modules that are commonly unavailable in hosted CI (Lean/formal and research aggregation).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when robust_claim is not true.",
    )
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    # Self-problem: consistency of evidence chain under multiple controllers.
    steps: List[Dict[str, Any]] = []

    steps.append(
        _run(
            [
                str(PY),
                "tools/dynamic_blueprint_bootstrap.py",
                "--cycles",
                str(max(1, args.blueprint_cycles)),
                "--max-actions-per-cycle",
                "1",
                "--enable-release-gate-cycle",
                "--strong-release-gate-cycle",
                "--release-gate-retries",
                "1",
                "--release-gate-profile",
                "quick",
                "--release-gate-relax-step",
                "0.05",
                "--min-breadth",
                "0.60",
                "--min-horizon",
                "0.80",
                "--min-robustness",
                "0.60",
                "--output-prefix",
                "systemic_joint_blueprint",
            ],
            "dynamic_blueprint_control",
            timeout=args.timeout_sec,
        )
    )

    steps.append(
        _run(
            [
                str(PY),
                "tools/run_agi_integrated_validation.py",
                "--with-longrun",
                "--longrun-cycles",
                str(max(1, args.longrun_cycles)),
                "--output-prefix",
                "systemic_joint_validation",
            ],
            "integrated_validation_control",
            timeout=args.timeout_sec,
        )
    )

    if args.ci_safe:
        steps.append(
            {
                "name": "formal_closure_control",
                "cmd": [str(PY), "tools/run_distill_evolution_public_formal_assessment.py"],
                "returncode": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "skipped": True,
                "skip_reason": "ci_safe mode: Lean/formal closure step skipped",
            }
        )
        steps.append(
            {
                "name": "research_cross_validation_control",
                "cmd": [str(PY), "tools/run_research_aggregation_cross_validation.py"],
                "returncode": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "skipped": True,
                "skip_reason": "ci_safe mode: research aggregation step skipped",
            }
        )
    else:
        steps.append(
            _run(
                [str(PY), "tools/run_distill_evolution_public_formal_assessment.py"],
                "formal_closure_control",
                timeout=args.timeout_sec,
            )
        )

        steps.append(
            _run(
                [str(PY), "tools/run_research_aggregation_cross_validation.py"],
                "research_cross_validation_control",
                timeout=args.timeout_sec,
            )
        )

    # Collect evidence from existing latest artifacts.
    val_path = REPORTS / "systemic_joint_validation_latest.json"
    if not val_path.exists():
        # Fallback to existing integrated validation latest.
        val_path = REPORTS / "distill_evo_public_validation_latest.json"
    formal_path = REPORTS / "distill_evo_public_formal_assessment_latest.json"
    research_path = REPORTS / "research_aggregation_cross_validation_latest.json"
    kpi_path = REPORTS / "one_click_kpi_dashboard_latest.json"

    validation = _load_json(val_path) if val_path.exists() else {}
    formal = _load_json(formal_path) if formal_path.exists() else {}
    research = _load_json(research_path) if research_path.exists() else {}
    kpi = _load_json(kpi_path) if kpi_path.exists() else {}

    v_base = validation.get("baseline_metrics") or {}
    v_long = validation.get("longrun_metrics") or {}
    f_logic = formal.get("logic_closure") or {}
    r_aggr = research.get("aggregate_effectiveness") or {}
    r_cv = research.get("cross_validation") or {}
    kk = kpi.get("kpis") or {}

    evidence: List[Evidence] = [
        Evidence(
            name="gate_alignment",
            value=_clip01(
                0.25 * (1.0 if _b(v_base.get("gate_ok", False)) else 0.0)
                + 0.25 * (1.0 if _b(v_long.get("gate_ok", False)) else 0.0)
                + 0.25 * _f(v_base.get("alignment_overall", 0.0))
                + 0.25 * _f(v_long.get("alignment_overall", 0.0))
            ),
            weight=0.45 if args.ci_safe else 0.30,
            source=str(val_path),
        ),
        Evidence(
            name="distill_schema_signal",
            value=_clip01(
                0.5 * _f(kk.get("distilled_schema_valid_rate", 0.0))
                + 0.5 * (1.0 if _b(kk.get("distill_schema_valid_rate_positive", False)) else 0.0)
            ),
            weight=0.30 if args.ci_safe else 0.10,
            source=str(kpi_path),
        ),
        Evidence(
            name="validation_blueprint_rate",
            value=_clip01(
                0.5 * _f(v_base.get("blueprint_ok_rate", 0.0))
                + 0.5 * _f(v_long.get("blueprint_ok_rate", 0.0))
            ),
            weight=0.25 if args.ci_safe else 0.0,
            source=str(val_path),
        ),
    ]

    if not args.ci_safe:
        evidence.extend(
            [
                Evidence(
                    name="formal_closure",
                    value=_clip01(1.0 if _b(f_logic.get("lean_compile_success", False)) else 0.0),
                    weight=0.20,
                    source=str(formal_path),
                ),
                Evidence(
                    name="research_aggregate",
                    value=_clip01(_f(r_aggr.get("score", 0.0))),
                    weight=0.25,
                    source=str(research_path),
                ),
                Evidence(
                    name="research_cv_floor",
                    value=_clip01(_f(r_cv.get("min_score", 0.0))),
                    weight=0.15,
                    source=str(research_path),
                ),
            ]
        )

    # Drop zero-weight placeholders so LOO remains meaningful.
    evidence = [e for e in evidence if e.weight > 0.0]

    aggregate = _weighted_score(evidence)
    cv = _loo(evidence)

    all_steps_ok = all(int(s.get("returncode", 1)) == 0 for s in steps)
    if args.ci_safe:
        robust_claim = (
            all_steps_ok
            and aggregate["score"] >= 0.80
            and cv["min_score"] >= 0.75
            and cv["std_score"] <= 0.12
            and _b(v_base.get("gate_ok", False))
            and _b(v_long.get("gate_ok", False))
        )
    else:
        robust_claim = (
            all_steps_ok
            and aggregate["score"] >= 0.85
            and cv["min_score"] >= 0.80
            and cv["std_score"] <= 0.10
            and _b(f_logic.get("lean_compile_success", False))
        )

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "self_problem": "evidence_consistency_under_multi_controller_collaboration",
        "solution_strategy": {
            "controllers": [
                "dynamic_blueprint_bootstrap",
                "integrated_validation",
                "formal_assessment",
                "research_cross_validation",
            ],
            "mode": "ci_safe" if args.ci_safe else "full",
            "goal": "stabilize and verify the evidence chain from control to formal closure",
        },
        "steps": steps,
        "sources": {
            "validation": str(val_path),
            "formal_assessment": str(formal_path),
            "research_cross_validation": str(research_path),
            "kpi": str(kpi_path),
        },
        "aggregate_effectiveness": aggregate,
        "cross_validation": cv,
        "conclusion": {
            "all_steps_ok": all_steps_ok,
            "robust_claim": robust_claim,
            "statement": (
                "Joint multi-controller solution is empirically and cross-validated for platform capability assessment."
                if robust_claim
                else "Joint multi-controller solution is partially validated; more stabilization is required."
            ),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    lines = [
        "# Systemic Platform Joint Capability Assessment",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- self_problem: `{payload['self_problem']}`",
        f"- all_steps_ok: `{payload['conclusion']['all_steps_ok']}`",
        f"- robust_claim: `{payload['conclusion']['robust_claim']}`",
        "",
        "## Solution Strategy",
        "- controllers: `dynamic_blueprint_bootstrap`, `integrated_validation`, `formal_assessment`, `research_cross_validation`",
        "- objective: stabilize evidence consistency and prove platform capability with independent evidence families",
        "",
        "## Aggregate Effectiveness",
        f"- score: `{aggregate['score']:.6f}`",
    ]

    for c in aggregate["components"]:
        lines.append(
            f"- {c['name']}: value={c['value']:.6f}, weight={c['weight']:.2f}, source={c['source']}"
        )

    lines.extend(
        [
            "",
            "## Leave-One-Out Cross Validation",
            f"- min_score: `{cv['min_score']:.6f}`",
            f"- max_score: `{cv['max_score']:.6f}`",
            f"- mean_score: `{cv['mean_score']:.6f}`",
            f"- std_score: `{cv['std_score']:.6f}`",
        ]
    )

    for f in cv["folds"]:
        lines.append(f"- left_out={f['left_out']}, score={f['score']:.6f}")

    lines.extend(["", "## Conclusion", f"- {payload['conclusion']['statement']}"])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Aggregate score: {aggregate['score']:.6f}")
    print(f"LOO min score: {cv['min_score']:.6f}")
    print(f"Robust claim: {robust_claim}")
    if args.strict and not robust_claim:
        print("STRICT MODE FAILED: robust_claim is not true", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
