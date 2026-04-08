#!/usr/bin/env python3
"""Discover open-source LLMs, separate edge/core tiers, and emit a fusion pathway report.

This script is designed to reuse H2Q-Evo's existing architecture artifacts.
It does not require model downloads and works in scan-only mode by default.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


def _safe_run(cmd: List[str], cwd: Path | None = None, timeout: int = 12) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _infer_size_b(name: str) -> float:
    lower = name.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", lower)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # Common aliases.
    if "tinyllama" in lower:
        return 1.1
    if "qwen2.5-0.5b" in lower:
        return 0.5
    if "7b" in lower:
        return 7.0
    if "13b" in lower:
        return 13.0
    if "33b" in lower:
        return 33.0
    if "34b" in lower:
        return 34.0
    if "70b" in lower:
        return 70.0
    if "236b" in lower:
        return 236.0
    return 0.0


def _specialties(name: str) -> List[str]:
    lower = name.lower()
    tags: List[str] = []
    if any(x in lower for x in ["coder", "code", "codellama", "starcoder"]):
        tags.append("coding")
    if any(x in lower for x in ["math", "qwen-math", "wizardmath"]):
        tags.append("math")
    if any(x in lower for x in ["instruct", "chat"]):
        tags.append("instruction")
    if any(x in lower for x in ["deepseek", "llama", "mistral", "qwen", "gemma"]):
        tags.append("general_reasoning")
    if not tags:
        tags.append("general")
    return tags


def _score_edge_core(size_b: float, tags: List[str]) -> Tuple[float, float]:
    # Edge prefers small and focused models.
    if size_b <= 0:
        edge = 0.45
        core = 0.45
    elif size_b <= 2:
        edge = 0.92
        core = 0.28
    elif size_b <= 8:
        edge = 0.84
        core = 0.48
    elif size_b <= 16:
        edge = 0.65
        core = 0.62
    elif size_b <= 40:
        edge = 0.42
        core = 0.78
    else:
        edge = 0.22
        core = 0.93

    if "coding" in tags:
        edge += 0.05
    if "instruction" in tags:
        edge += 0.03
    if "general_reasoning" in tags:
        core += 0.04
    if "math" in tags:
        core += 0.04

    return max(0.0, min(1.0, edge)), max(0.0, min(1.0, core))


def _tier(edge_score: float, core_score: float) -> str:
    if core_score >= 0.78:
        return "core"
    if edge_score >= 0.75:
        return "edge"
    return "bridge"


def discover_models() -> Dict[str, Any]:
    discovered: Dict[str, Dict[str, Any]] = {}

    # Source 1: Ollama runtime if available.
    ollama = _safe_run(["ollama", "list"])  # format: NAME ID SIZE MODIFIED
    if ollama["ok"]:
        lines = [x.strip() for x in (ollama.get("stdout") or "").splitlines() if x.strip()]
        for line in lines[1:]:
            name = line.split()[0]
            discovered.setdefault(name, {"sources": []})
            discovered[name]["sources"].append("ollama")

    # Source 2: Internalized registry.
    registry_path = ROOT / "models" / "registry.json"
    registry_obj = _load_json(registry_path) if registry_path.exists() else {}
    if isinstance(registry_obj, dict):
        for name in registry_obj.keys():
            discovered.setdefault(name, {"sources": []})
            discovered[name]["sources"].append("internal_registry")

    # Source 3: Built-in seed candidates from existing architecture scripts.
    seeds = [
        "tinyllama-1.1b",
        "deepseek-coder-6.7b",
        "codellama-7b",
        "llama-2-7b-chat",
        "qwen2.5-0.5b",
        "deepseek-coder-v2-236b",
    ]
    for name in seeds:
        discovered.setdefault(name, {"sources": []})
        discovered[name]["sources"].append("seed_catalog")

    models: List[Dict[str, Any]] = []
    for name, item in discovered.items():
        size_b = _infer_size_b(name)
        tags = _specialties(name)
        edge_score, core_score = _score_edge_core(size_b, tags)
        models.append(
            {
                "name": name,
                "size_b": size_b,
                "specialties": tags,
                "edge_score": edge_score,
                "core_score": core_score,
                "tier": _tier(edge_score, core_score),
                "sources": sorted(set(item.get("sources") or [])),
            }
        )

    models.sort(key=lambda x: (x["tier"], -x["core_score"], x["name"]))
    return {
        "models": models,
        "scan_status": {
            "ollama_ok": bool(ollama.get("ok", False)),
            "ollama_stderr_tail": _tail(ollama.get("stderr", "")),
            "registry_path": str(registry_path),
            "registry_exists": registry_path.exists(),
        },
    }


def load_capability_snapshot() -> Dict[str, Any]:
    paths = {
        "distill_pipeline": REPORTS / "self_eval_distillation_pipeline_latest.json",
        "distilled_benchmark": REPORTS / "self_model_consistency_distilled_latest.json",
        "systemic_joint": REPORTS / "systemic_platform_joint_capability_latest.json",
        "formal_assessment": REPORTS / "distill_evo_public_formal_assessment_latest.json",
    }

    snap: Dict[str, Any] = {"sources": {k: str(v) for k, v in paths.items()}, "metrics": {}}

    dp = _load_json(paths["distill_pipeline"]) if paths["distill_pipeline"].exists() else {}
    db = _load_json(paths["distilled_benchmark"]) if paths["distilled_benchmark"].exists() else {}
    sj = _load_json(paths["systemic_joint"]) if paths["systemic_joint"].exists() else {}
    fa = _load_json(paths["formal_assessment"]) if paths["formal_assessment"].exists() else {}

    dp_m = dp.get("metrics") or {}
    db_m = db.get("metrics") or {}
    sj_aggr = (sj.get("aggregate") or {})
    sj_cv = (sj.get("cross_validation") or {})
    fa_logic = (fa.get("logic_closure") or {})

    snap["metrics"] = {
        "edge_schema_valid_rate": float(dp_m.get("distilled_schema_valid_rate", db_m.get("schema_valid_rate", 0.0)) or 0.0),
        "edge_delta_schema_valid_rate": float(dp_m.get("delta_schema_valid_rate", 0.0) or 0.0),
        "bridge_consistency_overall_score": float(db_m.get("overall_score", 0.0) or 0.0),
        "core_systemic_score": float(sj_aggr.get("score", 0.0) or 0.0),
        "core_cv_min_score": float(sj_cv.get("min_score", 0.0) or 0.0),
        "formal_logic_closed": bool(fa_logic.get("all_true", False)),
    }

    return snap


def build_pathway(models: List[Dict[str, Any]], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    edge_models = [m["name"] for m in models if m.get("tier") == "edge"]
    bridge_models = [m["name"] for m in models if m.get("tier") == "bridge"]
    core_models = [m["name"] for m in models if m.get("tier") == "core"]

    m = snapshot.get("metrics") or {}
    edge_rate = float(m.get("edge_schema_valid_rate", 0.0) or 0.0)
    bridge_score = float(m.get("bridge_consistency_overall_score", 0.0) or 0.0)
    core_score = float(m.get("core_systemic_score", 0.0) or 0.0)

    maturity = {
        "edge": "stable" if edge_rate >= 0.75 else "developing",
        "bridge": "stable" if bridge_score >= 0.80 else "developing",
        "core": "stable" if core_score >= 0.80 else "developing",
    }

    stages = [
        {
            "stage": "S1_edge_skill_hardening",
            "goal": "Harden fast, low-cost capabilities (schema fidelity, formatting, retrieval snippets)",
            "drivers": [
                "tools/run_self_eval_distillation_pipeline.py",
                "tools/run_self_model_consistency_benchmark.py",
            ],
            "entry_metric": "edge_schema_valid_rate",
            "entry_value": edge_rate,
            "target": ">= 0.80",
            "candidate_models": edge_models[:4],
            "exit_gate": "delta_schema_valid_rate > 0 and schema_valid_rate >= 0.80",
        },
        {
            "stage": "S2_bridge_transfer_alignment",
            "goal": "Transfer edge gains into robust multi-turn self-evaluation and adapter compatibility",
            "drivers": [
                "tools/openclaw_h2q_adapter.py",
                "tools/run_self_model_consistency_benchmark.py",
            ],
            "entry_metric": "bridge_consistency_overall_score",
            "entry_value": bridge_score,
            "target": ">= 0.82",
            "candidate_models": bridge_models[:4],
            "exit_gate": "overall_score >= 0.82 and semantic_consistency >= 0.80",
        },
        {
            "stage": "S3_core_reasoning_consolidation",
            "goal": "Consolidate long-horizon planning, cross-validation stability, and release-gate trust",
            "drivers": [
                "tools/dynamic_blueprint_bootstrap.py",
                "tools/run_systemic_platform_joint_capability_assessment.py",
            ],
            "entry_metric": "core_systemic_score",
            "entry_value": core_score,
            "target": ">= 0.85",
            "candidate_models": core_models[:4],
            "exit_gate": "aggregate.score >= 0.85 and cv.min_score >= 0.80",
        },
        {
            "stage": "S4_formalized_self_improvement",
            "goal": "Convert empirical gains into formal closure and repeatable governance checks",
            "drivers": [
                "tools/run_distill_evolution_public_formal_assessment.py",
                "tools/release_gate.py",
            ],
            "entry_metric": "formal_logic_closed",
            "entry_value": bool(m.get("formal_logic_closed", False)),
            "target": "true",
            "candidate_models": core_models[:2] + bridge_models[:2],
            "exit_gate": "logic_closure all_true and gate_ok true",
        },
    ]

    routing_policy = {
        "edge_route": {
            "when": "short prompts, low uncertainty, deterministic formatting tasks",
            "tier": "edge",
            "max_tokens": 256,
            "fallback": "bridge",
        },
        "bridge_route": {
            "when": "medium complexity, multi-turn consistency checks, adapter tasks",
            "tier": "bridge",
            "max_tokens": 512,
            "fallback": "core",
        },
        "core_route": {
            "when": "high uncertainty, long-horizon planning, release-gate critical tasks",
            "tier": "core",
            "max_tokens": 1024,
            "fallback": "bridge with decomposition",
        },
    }

    return {
        "tier_maturity": maturity,
        "tier_counts": {
            "edge": len(edge_models),
            "bridge": len(bridge_models),
            "core": len(core_models),
        },
        "routing_policy": routing_policy,
        "self_improvement_stages": stages,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Open-Source Model Fusion Pathway")
    lines.append("")
    lines.append(f"- generated_at_utc: `{payload['generated_at_utc']}`")
    lines.append(f"- discovered_models: `{len(payload['model_scan']['models'])}`")
    lines.append(f"- ollama_scan_ok: `{payload['model_scan']['scan_status']['ollama_ok']}`")
    lines.append("")

    tm = payload["fusion_plan"]["tier_maturity"]
    lines.append("## Tier Maturity")
    lines.append(f"- edge: `{tm['edge']}`")
    lines.append(f"- bridge: `{tm['bridge']}`")
    lines.append(f"- core: `{tm['core']}`")
    lines.append("")

    lines.append("## Dynamic Separation")
    lines.append("| model | size_b | tier | edge_score | core_score | specialties |")
    lines.append("|---|---:|---|---:|---:|---|")
    for model in payload["model_scan"]["models"]:
        specs = ", ".join(model.get("specialties") or [])
        lines.append(
            f"| {model['name']} | {model['size_b']:.2f} | {model['tier']} | "
            f"{model['edge_score']:.2f} | {model['core_score']:.2f} | {specs} |"
        )
    lines.append("")

    lines.append("## Edge -> Core Self-Improvement Process")
    for stage in payload["fusion_plan"]["self_improvement_stages"]:
        lines.append(f"- {stage['stage']}: {stage['goal']}")
        lines.append(f"  - entry: `{stage['entry_metric']}={stage['entry_value']}`")
        lines.append(f"  - target: `{stage['target']}`")
        lines.append(f"  - exit_gate: `{stage['exit_gate']}`")
        lines.append(f"  - drivers: `{', '.join(stage['drivers'])}`")
        lines.append(f"  - candidate_models: `{', '.join(stage['candidate_models'])}`")
    lines.append("")

    lines.append("## Capability Snapshot")
    snap = payload.get("capability_snapshot", {}).get("metrics", {})
    for k in [
        "edge_schema_valid_rate",
        "edge_delta_schema_valid_rate",
        "bridge_consistency_overall_score",
        "core_systemic_score",
        "core_cv_min_score",
        "formal_logic_closed",
    ]:
        lines.append(f"- {k}: `{snap.get(k)}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Open-source model fusion pathway scanner for H2Q-Evo")
    parser.add_argument("--output-prefix", default="open_source_model_fusion_pathway")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    model_scan = discover_models()
    snapshot = load_capability_snapshot()
    fusion_plan = build_pathway(model_scan["models"], snapshot)

    payload: Dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "model_scan": model_scan,
        "capability_snapshot": snapshot,
        "fusion_plan": fusion_plan,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"
    router_latest = REPORTS / "open_source_model_fusion_router_latest.json"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    out_md.write_text(render_markdown(payload), encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    router_policy = {
        "generated_at_utc": payload["generated_at_utc"],
        "routing_policy": payload["fusion_plan"]["routing_policy"],
        "tier_counts": payload["fusion_plan"]["tier_counts"],
    }
    router_latest.write_text(json.dumps(router_policy, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Router JSON: {router_latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
