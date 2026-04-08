#!/usr/bin/env python3
"""Assess distilled evolution capability on public validation artifacts and verify logic closure in Lean4."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _all_steps_ok(steps: List[Dict[str, Any]]) -> bool:
    if not steps:
        return False
    def _step_ok(step: Dict[str, Any]) -> bool:
        raw = step.get("returncode", 1)
        try:
            rc = int(raw)
        except Exception:
            rc = 1
        return rc == 0

    return all(_step_ok(s) for s in steps)


def _latest_distilled_by_sessions(sessions: int) -> Dict[str, Any]:
    candidates = []
    for p in REPORTS.glob("self_model_consistency_distilled_*.json"):
        if p.name.endswith("_latest.json"):
            continue
        try:
            payload = _load_json(p)
            meta = payload.get("meta") or {}
            if int(meta.get("sessions", 0) or 0) != sessions:
                continue
            candidates.append((p.stat().st_mtime, p, payload))
        except Exception:
            continue

    if not candidates:
        return {}
    _, path, payload = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    metrics = payload.get("metrics") or {}
    meta = payload.get("meta") or {}
    return {
        "path": str(path),
        "sessions": int(meta.get("sessions", 0) or 0),
        "total_runs": int(meta.get("total_runs", 0) or 0),
        "schema_valid_rate": _safe_float(metrics.get("schema_valid_rate", 0.0)),
        "overall_score": _safe_float(metrics.get("overall_score", 0.0)),
        "grade": str(metrics.get("grade", "")),
    }


def _build_lean_content(facts: Dict[str, bool]) -> str:
    return "\n".join(
        [
            "/-",
            "  Auto-generated Lean4 proof for distillation + evolution logic closure.",
            "  Generated from latest reports in /reports.",
            "-/",
            "",
            "namespace H2Q.DistillEvolutionClosure",
            "",
            f"def distillPipelineAllStepsOk : Bool := {str(facts['distill_pipeline_all_steps_ok']).lower()}",
            f"def publicValidationAllStepsOk : Bool := {str(facts['public_validation_all_steps_ok']).lower()}",
            f"def distilledSchemaPositive : Bool := {str(facts['distilled_schema_positive']).lower()}",
            f"def baselineGateOk : Bool := {str(facts['baseline_gate_ok']).lower()}",
            f"def longrunGateOk : Bool := {str(facts['longrun_gate_ok']).lower()}",
            "",
            "def logicalClosure : Prop :=",
            "  distillPipelineAllStepsOk = true /\\",
            "  publicValidationAllStepsOk = true /\\",
            "  distilledSchemaPositive = true /\\",
            "  baselineGateOk = true /\\",
            "  longrunGateOk = true",
            "",
            "theorem logical_closure_verified : logicalClosure := by",
            "  simp [logicalClosure, distillPipelineAllStepsOk, publicValidationAllStepsOk, distilledSchemaPositive, baselineGateOk, longrunGateOk]",
            "",
            "end H2Q.DistillEvolutionClosure",
            "",
        ]
    )


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    distill_pipeline_path = REPORTS / "self_eval_distillation_pipeline_latest.json"
    distill_bench_path = REPORTS / "self_model_consistency_distilled_latest.json"
    public_validation_path = REPORTS / "distill_evo_public_validation_latest.json"

    if not distill_pipeline_path.exists():
        raise SystemExit(f"Missing file: {distill_pipeline_path}")
    if not distill_bench_path.exists():
        raise SystemExit(f"Missing file: {distill_bench_path}")
    if not public_validation_path.exists():
        raise SystemExit(f"Missing file: {public_validation_path}")

    distill_pipeline = _load_json(distill_pipeline_path)
    distill_bench = _load_json(distill_bench_path)
    public_validation = _load_json(public_validation_path)

    dp_metrics = distill_pipeline.get("metrics") or {}
    db_metrics = distill_bench.get("metrics") or {}
    db_meta = distill_bench.get("meta") or {}
    pv_base = public_validation.get("baseline_metrics") or {}
    pv_long = public_validation.get("longrun_metrics") or {}

    distill_30 = _latest_distilled_by_sessions(30)
    distill_50 = _latest_distilled_by_sessions(50)

    facts = {
        "distill_pipeline_all_steps_ok": _all_steps_ok(distill_pipeline.get("steps") or []),
        "public_validation_all_steps_ok": _all_steps_ok(public_validation.get("steps") or []),
        "distilled_schema_positive": _safe_bool(dp_metrics.get("schema_valid_rate_positive", False)),
        "baseline_gate_ok": _safe_bool(pv_base.get("gate_ok", False)),
        "longrun_gate_ok": _safe_bool(pv_long.get("gate_ok", False)),
    }

    ts = int(time.time())
    lean_path = REPORTS / f"distill_evolution_logic_closure_{ts}.lean"
    lean_latest = REPORTS / "distill_evolution_logic_closure_latest.lean"
    lean_path.write_text(_build_lean_content(facts), encoding="utf-8")
    shutil.copy2(lean_path, lean_latest)

    lean_proc = subprocess.run(
        ["lean", str(lean_path)],
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        timeout=120,
    )
    lean_ok = lean_proc.returncode == 0

    out_json = REPORTS / f"distill_evo_public_formal_assessment_{ts}.json"
    latest_json = REPORTS / "distill_evo_public_formal_assessment_latest.json"
    out_md = REPORTS / f"distill_evo_public_formal_assessment_{ts}.md"
    latest_md = REPORTS / "distill_evo_public_formal_assessment_latest.md"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "distill_pipeline": str(distill_pipeline_path),
            "distill_benchmark": str(distill_bench_path),
            "public_validation": str(public_validation_path),
            "lean_file": str(lean_path),
        },
        "distillation": {
            "sessions": int(db_meta.get("sessions", 0) or 0),
            "total_runs": int(db_meta.get("total_runs", 0) or 0),
            "schema_valid_rate": _safe_float(db_metrics.get("schema_valid_rate", 0.0)),
            "overall_score": _safe_float(db_metrics.get("overall_score", 0.0)),
            "grade": str(db_metrics.get("grade", "")),
            "pipeline_delta_schema_valid_rate": _safe_float(dp_metrics.get("delta_schema_valid_rate", 0.0)),
        },
        "robustness_compare": {
            "sessions_30": distill_30,
            "sessions_50": distill_50,
            "delta_schema_valid_rate_50_minus_30": _safe_float(distill_50.get("schema_valid_rate", 0.0))
            - _safe_float(distill_30.get("schema_valid_rate", 0.0)),
            "delta_overall_score_50_minus_30": _safe_float(distill_50.get("overall_score", 0.0))
            - _safe_float(distill_30.get("overall_score", 0.0)),
        },
        "public_validation": {
            "baseline_gate_ok": _safe_bool(pv_base.get("gate_ok", False)),
            "longrun_gate_ok": _safe_bool(pv_long.get("gate_ok", False)),
            "baseline_alignment_overall": _safe_float(pv_base.get("alignment_overall", 0.0)),
            "longrun_alignment_overall": _safe_float(pv_long.get("alignment_overall", 0.0)),
            "baseline_blueprint_ok_rate": _safe_float(pv_base.get("blueprint_ok_rate", 0.0)),
            "longrun_blueprint_ok_rate": _safe_float(pv_long.get("blueprint_ok_rate", 0.0)),
        },
        "logic_closure": {
            "facts": facts,
            "lean_compile_success": lean_ok,
            "lean_returncode": int(lean_proc.returncode),
            "lean_stdout_tail": "\n".join((lean_proc.stdout or "").splitlines()[-40:]),
            "lean_stderr_tail": "\n".join((lean_proc.stderr or "").splitlines()[-40:]),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    lines = [
        "# Distill-Evolution Public Formal Assessment",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
        "## Distillation Capability",
        f"- sessions: `{payload['distillation']['sessions']}`",
        f"- total_runs: `{payload['distillation']['total_runs']}`",
        f"- schema_valid_rate: `{payload['distillation']['schema_valid_rate']:.6f}`",
        f"- overall_score: `{payload['distillation']['overall_score']:.6f}`",
        f"- grade: `{payload['distillation']['grade']}`",
        f"- pipeline_delta_schema_valid_rate: `{payload['distillation']['pipeline_delta_schema_valid_rate']:+.6f}`",
        "",
        "## Robustness (30 vs 50)",
        f"- sessions=30 schema_valid_rate: `{_safe_float(distill_30.get('schema_valid_rate', 0.0)):.6f}`",
        f"- sessions=30 overall_score: `{_safe_float(distill_30.get('overall_score', 0.0)):.6f}`",
        f"- sessions=50 schema_valid_rate: `{_safe_float(distill_50.get('schema_valid_rate', 0.0)):.6f}`",
        f"- sessions=50 overall_score: `{_safe_float(distill_50.get('overall_score', 0.0)):.6f}`",
        f"- delta_schema_valid_rate(50-30): `{payload['robustness_compare']['delta_schema_valid_rate_50_minus_30']:+.6f}`",
        f"- delta_overall_score(50-30): `{payload['robustness_compare']['delta_overall_score_50_minus_30']:+.6f}`",
        "",
        "## Public Validation (Open Experimental Set)",
        f"- baseline_gate_ok: `{payload['public_validation']['baseline_gate_ok']}`",
        f"- longrun_gate_ok: `{payload['public_validation']['longrun_gate_ok']}`",
        f"- baseline_alignment_overall: `{payload['public_validation']['baseline_alignment_overall']:.6f}`",
        f"- longrun_alignment_overall: `{payload['public_validation']['longrun_alignment_overall']:.6f}`",
        f"- baseline_blueprint_ok_rate: `{payload['public_validation']['baseline_blueprint_ok_rate']:.6f}`",
        f"- longrun_blueprint_ok_rate: `{payload['public_validation']['longrun_blueprint_ok_rate']:.6f}`",
        "",
        "## Lean4 Logical Closure",
        f"- lean_file: `{lean_path}`",
        f"- lean_compile_success: `{payload['logic_closure']['lean_compile_success']}`",
        f"- facts: `{facts}`",
    ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Lean: {lean_path}")
    print(f"Lean latest: {lean_latest}")
    print(f"Lean compile success: {lean_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
