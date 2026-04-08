#!/usr/bin/env python3
"""Build a research-to-architecture mapping and validate aggregate effectiveness."""

from __future__ import annotations

import json
import math
import argparse
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


@dataclass
class EvidenceMetric:
    name: str
    value: float
    weight: float
    note: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _weighted_score(metrics: List[EvidenceMetric]) -> Dict[str, Any]:
    denom = sum(m.weight for m in metrics)
    if denom <= 0:
        return {"score": 0.0, "components": []}
    score = sum(m.value * m.weight for m in metrics) / denom
    return {
        "score": _clip01(score),
        "components": [
            {"name": m.name, "value": _clip01(m.value), "weight": m.weight, "note": m.note}
            for m in metrics
        ],
    }


def _leave_one_out(metrics: List[EvidenceMetric]) -> Dict[str, Any]:
    folds: List[Dict[str, Any]] = []
    for i, m in enumerate(metrics):
        kept = [x for j, x in enumerate(metrics) if j != i]
        scored = _weighted_score(kept)
        folds.append(
            {
                "left_out": m.name,
                "score": scored["score"],
                "kept_components": [k["name"] for k in scored["components"]],
            }
        )

    vals = [f["score"] for f in folds] or [0.0]
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


def _build_paper_map() -> List[Dict[str, Any]]:
    return [
        {
            "paper": "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
            "year": 2022,
            "url": "https://arxiv.org/abs/2203.11171",
            "method": "Sample diverse reasoning paths and marginalize to a consistent answer.",
            "local_mapping": [
                "tools/run_self_model_consistency_benchmark.py",
                "reports/self_model_consistency_distilled_latest.json",
            ],
            "target_effect": "consistency/robustness",
        },
        {
            "paper": "Reflexion: Language Agents with Verbal Reinforcement Learning",
            "year": 2023,
            "url": "https://arxiv.org/abs/2303.11366",
            "method": "Use linguistic feedback memory to improve next trial decisions.",
            "local_mapping": [
                "tools/trusted_local_agi_chat.py",
                "tools/collect_self_eval_distill_samples.py",
                "tools/train_self_eval_distillation_adapter.py",
            ],
            "target_effect": "self-improvement",
        },
        {
            "paper": "Self-Refine: Iterative Refinement with Self-Feedback",
            "year": 2023,
            "url": "https://arxiv.org/abs/2303.17651",
            "method": "Generate, self-critique, and iteratively refine without extra training.",
            "local_mapping": [
                "tools/run_self_eval_distillation_pipeline.py",
                "reports/self_eval_distillation_pipeline_latest.json",
            ],
            "target_effect": "iterative quality uplift",
        },
        {
            "paper": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            "year": 2023,
            "url": "https://arxiv.org/abs/2305.18290",
            "method": "Preference alignment via direct objective, avoiding PPO RLHF complexity.",
            "local_mapping": [
                "reports/distill_evo_public_validation_latest.json",
                "reports/release_gate_latest.json",
            ],
            "target_effect": "alignment stability",
        },
        {
            "paper": "Constitutional AI: Harmlessness from AI Feedback",
            "year": 2022,
            "url": "https://arxiv.org/abs/2212.08073",
            "method": "Rule-guided self-critique and AI feedback for harmless aligned behavior.",
            "local_mapping": [
                "tools/run_agi_integrated_validation.py",
                "reports/distill_evo_public_validation_latest.json",
            ],
            "target_effect": "safety/alignment gates",
        },
        {
            "paper": "Self-Rewarding Language Models",
            "year": 2024,
            "url": "https://arxiv.org/abs/2401.10020",
            "method": "LLM-as-a-judge reward loops with iterative DPO improvement.",
            "local_mapping": [
                "tools/run_distill_evolution_public_formal_assessment.py",
                "reports/distill_evo_public_formal_assessment_latest.json",
            ],
            "target_effect": "closed-loop reward improvement",
        },
        {
            "paper": "LoRA: Low-Rank Adaptation of Large Language Models",
            "year": 2021,
            "url": "https://arxiv.org/abs/2106.09685",
            "method": "Parameter-efficient adaptation via low-rank updates.",
            "local_mapping": [
                "tools/train_self_eval_distillation_adapter.py",
                "reports/self_eval_distill_model_latest.json",
            ],
            "target_effect": "low-cost adaptation",
        },
        {
            "paper": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "year": 2020,
            "url": "https://arxiv.org/abs/2005.11401",
            "method": "Combine parametric generation with non-parametric retrieval memory.",
            "local_mapping": [
                "h2q_project/h2q_server.py",
                "tools/trusted_local_agi_chat.py",
            ],
            "target_effect": "factual grounding/provenance",
        },
        {
            "paper": "Improving Dialogue Management: Quality Datasets vs Models",
            "year": 2023,
            "url": "https://arxiv.org/abs/2310.01339",
            "method": "Shows dataset quality strongly controls downstream dialogue performance.",
            "local_mapping": [
                "tools/collect_self_eval_distill_samples.py",
                "reports/self_eval_distill_dataset_latest.json",
            ],
            "target_effect": "data quality sensitivity",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Research aggregation cross validation with tunable weights")
    parser.add_argument("--config-file", default="reports/research_cv_tuned_config_latest.json")
    parser.add_argument("--w-distill", type=float, default=0.25)
    parser.add_argument("--w-consistency", type=float, default=0.20)
    parser.add_argument("--w-robustness", type=float, default=0.15)
    parser.add_argument("--w-public", type=float, default=0.25)
    parser.add_argument("--w-formal", type=float, default=0.15)
    parser.add_argument("--thr-aggregate", type=float, default=0.85)
    parser.add_argument("--thr-loo-min", type=float, default=0.80)
    parser.add_argument("--thr-loo-std", type=float, default=0.07)
    args = parser.parse_args()

    tuned: Dict[str, Any] = {}
    cfg_path = Path(args.config_file)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if cfg_path.exists():
        try:
            tuned = _load_json(cfg_path)
        except Exception:
            tuned = {}

    weights_cfg = tuned.get("weights") or {}
    thresholds_cfg = tuned.get("thresholds") or {}

    w_distill = _f(weights_cfg.get("distill_gain", args.w_distill), args.w_distill)
    w_consistency = _f(weights_cfg.get("consistency_quality", args.w_consistency), args.w_consistency)
    w_robustness = _f(weights_cfg.get("robustness_30_vs_50", args.w_robustness), args.w_robustness)
    w_public = _f(weights_cfg.get("public_validation", args.w_public), args.w_public)
    w_formal = _f(weights_cfg.get("formal_closure", args.w_formal), args.w_formal)

    thr_aggregate = _f(thresholds_cfg.get("aggregate", args.thr_aggregate), args.thr_aggregate)
    thr_loo_min = _f(thresholds_cfg.get("loo_min", args.thr_loo_min), args.thr_loo_min)
    thr_loo_std = _f(thresholds_cfg.get("loo_std", args.thr_loo_std), args.thr_loo_std)

    # Guard against invalid negative values.
    w_distill = max(0.0, w_distill)
    w_consistency = max(0.0, w_consistency)
    w_robustness = max(0.0, w_robustness)
    w_public = max(0.0, w_public)
    w_formal = max(0.0, w_formal)
    thr_aggregate = _clip01(thr_aggregate)
    thr_loo_min = _clip01(thr_loo_min)
    thr_loo_std = max(0.0, thr_loo_std)

    distill_pipeline_path = REPORTS / "self_eval_distillation_pipeline_latest.json"
    distill_bench_path = REPORTS / "self_model_consistency_distilled_latest.json"
    public_validation_path = REPORTS / "distill_evo_public_validation_latest.json"
    formal_assessment_path = REPORTS / "distill_evo_public_formal_assessment_latest.json"

    for p in [distill_pipeline_path, distill_bench_path, public_validation_path, formal_assessment_path]:
        if not p.exists():
            raise SystemExit(f"Missing required artifact: {p}")

    distill_pipeline = _load_json(distill_pipeline_path)
    distill_bench = _load_json(distill_bench_path)
    public_validation = _load_json(public_validation_path)
    formal_assessment = _load_json(formal_assessment_path)

    dp_metrics = distill_pipeline.get("metrics") or {}
    db_metrics = distill_bench.get("metrics") or {}
    db_meta = distill_bench.get("meta") or {}
    pv_base = public_validation.get("baseline_metrics") or {}
    pv_long = public_validation.get("longrun_metrics") or {}
    fa_logic = formal_assessment.get("logic_closure") or {}
    fa_robust = formal_assessment.get("robustness_compare") or {}

    distill_gain = _clip01(_f(dp_metrics.get("delta_schema_valid_rate", 0.0)))
    consistency_score = _clip01(_f(db_metrics.get("overall_score", 0.0)))

    d_schema = abs(_f(fa_robust.get("delta_schema_valid_rate_50_minus_30", 0.0)))
    d_score = abs(_f(fa_robust.get("delta_overall_score_50_minus_30", 0.0)))
    robustness_score = _clip01(1.0 - 0.5 * min(1.0, d_schema * 5.0) - 0.5 * min(1.0, d_score * 20.0))

    gate_score = 0.0
    gate_score += 0.25 if _b(pv_base.get("gate_ok", False)) else 0.0
    gate_score += 0.25 if _b(pv_long.get("gate_ok", False)) else 0.0
    gate_score += 0.25 * _clip01(_f(pv_base.get("alignment_overall", 0.0)))
    gate_score += 0.25 * _clip01(_f(pv_long.get("alignment_overall", 0.0)))

    facts = fa_logic.get("facts") or {}
    true_ratio = 0.0
    if facts:
        true_ratio = sum(1 for _, v in facts.items() if _b(v, False)) / len(facts)
    formal_score = _clip01(0.6 * (1.0 if _b(fa_logic.get("lean_compile_success", False)) else 0.0) + 0.4 * true_ratio)

    metrics = [
        EvidenceMetric(
            name="distill_gain",
            value=distill_gain,
            weight=w_distill,
            note="schema_valid_rate delta from pipeline",
        ),
        EvidenceMetric(
            name="consistency_quality",
            value=consistency_score,
            weight=w_consistency,
            note="overall_score from distilled benchmark",
        ),
        EvidenceMetric(
            name="robustness_30_vs_50",
            value=robustness_score,
            weight=w_robustness,
            note="stability under sessions increase",
        ),
        EvidenceMetric(
            name="public_validation",
            value=gate_score,
            weight=w_public,
            note="baseline/longrun gate and alignment",
        ),
        EvidenceMetric(
            name="formal_closure",
            value=formal_score,
            weight=w_formal,
            note="Lean compile + closure facts",
        ),
    ]

    aggregate = _weighted_score(metrics)
    loo = _leave_one_out(metrics)

    robust_claim = (
        aggregate["score"] >= thr_aggregate
        and loo["min_score"] >= thr_loo_min
        and loo["std_score"] <= thr_loo_std
        and _b(fa_logic.get("lean_compile_success", False))
        and _b(pv_base.get("gate_ok", False))
        and _b(pv_long.get("gate_ok", False))
    )

    papers = _build_paper_map()

    ts = int(time.time())
    out_json = REPORTS / f"research_aggregation_cross_validation_{ts}.json"
    latest_json = REPORTS / "research_aggregation_cross_validation_latest.json"
    out_md = REPORTS / f"research_aggregation_cross_validation_{ts}.md"
    latest_md = REPORTS / "research_aggregation_cross_validation_latest.md"
    out_proof_md = REPORTS / f"research_aggregation_cross_validation_proof_note_{ts}.md"
    latest_proof_md = REPORTS / "research_aggregation_cross_validation_proof_note_latest.md"

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "distill_pipeline": str(distill_pipeline_path),
            "distill_benchmark": str(distill_bench_path),
            "public_validation": str(public_validation_path),
            "formal_assessment": str(formal_assessment_path),
            "tuned_config": str(cfg_path) if cfg_path.exists() else "",
        },
        "tuning": {
            "weights": {
                "distill_gain": w_distill,
                "consistency_quality": w_consistency,
                "robustness_30_vs_50": w_robustness,
                "public_validation": w_public,
                "formal_closure": w_formal,
            },
            "thresholds": {
                "aggregate": thr_aggregate,
                "loo_min": thr_loo_min,
                "loo_std": thr_loo_std,
            },
        },
        "local_snapshot": {
            "sessions": int(db_meta.get("sessions", 0) or 0),
            "total_runs": int(db_meta.get("total_runs", 0) or 0),
            "schema_valid_rate": _f(db_metrics.get("schema_valid_rate", 0.0)),
            "overall_score": _f(db_metrics.get("overall_score", 0.0)),
            "baseline_gate_ok": _b(pv_base.get("gate_ok", False)),
            "longrun_gate_ok": _b(pv_long.get("gate_ok", False)),
            "lean_compile_success": _b(fa_logic.get("lean_compile_success", False)),
        },
        "research_map": papers,
        "aggregate_effectiveness": aggregate,
        "cross_validation": loo,
        "proof_argument": {
            "premises": [
                "P1: Distillation pipeline has positive schema-valid uplift.",
                "P2: Robustness from sessions=30 to sessions=50 remains stable.",
                "P3: Public validation gates pass in baseline and longrun settings.",
                "P4: Lean4 logical closure compiles with all closure facts true.",
                "P5: Leave-one-evidence-out aggregate score remains above acceptance floor.",
            ],
            "conclusion": (
                "Aggregated effect is empirically supported and cross-validated under independent evidence families."
                if robust_claim
                else "Aggregated effect is promising but not yet robust under current cross-validation thresholds."
            ),
            "robust_claim": robust_claim,
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    lines: List[str] = [
        "# Research-Architecture Cross Validation",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- sessions: `{payload['local_snapshot']['sessions']}`",
        f"- total_runs: `{payload['local_snapshot']['total_runs']}`",
        f"- schema_valid_rate: `{payload['local_snapshot']['schema_valid_rate']:.6f}`",
        f"- overall_score: `{payload['local_snapshot']['overall_score']:.6f}`",
        f"- baseline_gate_ok: `{payload['local_snapshot']['baseline_gate_ok']}`",
        f"- longrun_gate_ok: `{payload['local_snapshot']['longrun_gate_ok']}`",
        f"- lean_compile_success: `{payload['local_snapshot']['lean_compile_success']}`",
        "",
        "## Aggregate Effectiveness",
        f"- score: `{aggregate['score']:.6f}`",
    ]

    for comp in aggregate["components"]:
        lines.append(
            f"- {comp['name']}: value={comp['value']:.6f}, weight={comp['weight']:.2f}, note={comp['note']}"
        )

    lines.extend(
        [
            "",
            "## Leave-One-Out Cross Validation",
            f"- min_score: `{loo['min_score']:.6f}`",
            f"- max_score: `{loo['max_score']:.6f}`",
            f"- mean_score: `{loo['mean_score']:.6f}`",
            f"- std_score: `{loo['std_score']:.6f}`",
        ]
    )

    for f in loo["folds"]:
        lines.append(f"- left_out={f['left_out']}, score={f['score']:.6f}")

    lines.extend(["", "## Proof Argument"])
    for p in payload["proof_argument"]["premises"]:
        lines.append(f"- {p}")
    lines.append(f"- robust_claim: `{payload['proof_argument']['robust_claim']}`")
    lines.append(f"- conclusion: {payload['proof_argument']['conclusion']}")

    lines.extend(["", "## Paper-to-Module Mapping"])
    for item in papers:
        lines.append(
            f"- {item['paper']} ({item['year']}): {item['target_effect']} -> {', '.join(item['local_mapping'])}; {item['url']}"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    proof_lines: List[str] = [
        "# Research Aggregation Formal Proof Note",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- aggregate_score: `{aggregate['score']:.6f}`",
        f"- loo_min_score: `{loo['min_score']:.6f}`",
        f"- loo_std_score: `{loo['std_score']:.6f}`",
        f"- robust_claim: `{payload['proof_argument']['robust_claim']}`",
        "",
        "## Premises",
        "- P1: Distillation uplift exists and is positive (delta_schema_valid_rate > 0).",
        "- P2: Robustness drift under sessions expansion (30 -> 50) remains bounded.",
        "- P3: Public validation gates pass under baseline and longrun settings.",
        "- P4: Formal logic-closure checks compile in Lean4 and closure facts are true.",
        "- P5: Leave-one-out evidence removal still preserves aggregate score above floor.",
        "",
        "## Inference Rules",
        "- R1 (Multi-Evidence Sufficiency): If P1..P4 hold and each evidence family is independent in mechanism, then combined empirical support is stronger than any single metric.",
        "- R2 (Cross-Validation Stability): If LOO min_score remains above acceptance floor and variance is low, claim is not dominated by one artifact.",
        "- R3 (Formal Consistency Filter): If Lean closure compiles with all facts true, the propositional closure over selected gates is logically consistent.",
        "",
        "## Conclusion",
        f"- {payload['proof_argument']['conclusion']}",
        "",
        "## Threats To Validity",
        "- Construct validity: current aggregate score compresses multiple goals into a weighted scalar; weight choice can bias interpretation.",
        "- Internal validity: shared pipelines may induce correlated errors across metrics (e.g., same upstream data artifacts).",
        "- External validity: results are measured on this project's benchmark/task mix; generalization to new domains is not guaranteed.",
        "- Statistical validity: LOO here is evidence-family ablation, not sample-level bootstrap; uncertainty is partially characterized.",
        "",
        "## Counterexample Boundaries",
        "- Boundary B1: If longrun gate flips false while baseline stays true, aggregate claim should be downgraded to conditional robustness.",
        "- Boundary B2: If Lean compile fails or any closure fact becomes false, formal consistency support is invalidated.",
        "- Boundary B3: If sessions expansion causes large drift (e.g., |delta_overall_score| >= 0.02), robustness premise P2 fails.",
        "- Boundary B4: If one evidence family removal drops LOO score below acceptance floor, aggregation becomes single-metric fragile.",
        "",
        "## Reproducibility",
        "- Script: `tools/run_research_aggregation_cross_validation.py`",
        "- Artifacts:",
        f"  - `{latest_json}`",
        f"  - `{latest_md}`",
        f"  - `{latest_proof_md}`",
    ]

    out_proof_md.write_text("\n".join(proof_lines) + "\n", encoding="utf-8")
    shutil.copy2(out_proof_md, latest_proof_md)

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Proof MD: {out_proof_md}")
    print(f"Latest Proof MD: {latest_proof_md}")
    print(f"Aggregate score: {aggregate['score']:.6f}")
    print(f"LOO min score: {loo['min_score']:.6f}")
    print(f"Robust claim: {robust_claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
