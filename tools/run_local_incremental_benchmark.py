#!/usr/bin/env python3
"""Run a small fixed-seed local incremental benchmark for sensitive before/after detection.

This benchmark is intentionally lightweight and deterministic:
- fixed prompts/subtasks
- fixed generation settings
- fixed random seed

Outputs base-model score, adapter-model score, and gain for hard-gate decisions.
Scoring is based on conditional target likelihood (NLL/PPL), which is smoother
than keyword-hit heuristics and less likely to get stuck at constant gains.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(model_name: str, adapter_dir: Path | None, device: torch.device) -> Any:
    base = AutoModelForCausalLM.from_pretrained(model_name)
    if adapter_dir and adapter_dir.exists():
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, str(adapter_dir))
        except Exception:
            model = base
    else:
        model = base
    model.to(device)
    model.eval()
    return model


def _generate(model: Any, tokenizer: Any, prompt: str, device: torch.device, max_new_tokens: int = 64) -> str:
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192)
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = model.generate(
            **toks,
            do_sample=False,
            min_new_tokens=16,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    if full.startswith(prompt):
        completion = full[len(prompt) :].strip()
        if completion:
            return completion
    return full.strip()


def _target_nll(model: Any, tokenizer: Any, prompt: str, target: str, device: torch.device) -> Dict[str, float]:
    """Compute conditional NLL/PPL for target tokens given prompt context."""
    prefix = prompt.rstrip() + "\n"
    full_text = prefix + target

    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
    prefix_ids = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)

    labels = full_ids.clone()
    prompt_len = int(prefix_ids.shape[1])
    labels[:, :prompt_len] = -100

    target_tokens = int((labels != -100).sum().item())
    if target_tokens <= 0:
        return {
            "nll": 50.0,
            "ppl": float(np.exp(20.0)),
            "target_tokens": 0,
        }

    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
    nll = float(out.loss.detach().cpu().item())
    ppl = float(np.exp(min(20.0, max(0.0, nll))))
    return {
        "nll": nll,
        "ppl": ppl,
        "target_tokens": target_tokens,
    }


def _aggregate_ll_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "score": 0.0,
            "avg_nll": 50.0,
            "avg_ppl": float(np.exp(20.0)),
        }

    nlls = [float((r.get("ll") or {}).get("nll", 50.0)) for r in rows]
    ppls = [float((r.get("ll") or {}).get("ppl", float(np.exp(20.0)))) for r in rows]
    avg_nll = float(np.mean(nlls))
    avg_ppl = float(np.mean(ppls))

    # Map lower NLL -> higher bounded quality score in (0,1].
    score = float(1.0 / (1.0 + max(0.0, avg_nll)))
    return {
        "score": score,
        "avg_nll": avg_nll,
        "avg_ppl": avg_ppl,
    }


def _bench_set() -> List[Dict[str, Any]]:
    return [
        {
            "task": "self_eval_json",
            "prompt": "Return JSON with keys diagnosis, action, confidence and one sentence each.",
            "target": '{"diagnosis":"schema drift risk","action":"run validation and retrain adapter","confidence":0.72}',
        },
        {
            "task": "robustness_plan",
            "prompt": "Write one robustness experiment, include metric and expected failure mode.",
            "target": "Hypothesis: consistency drops under distribution shift. Measurement: schema_valid_rate and failure_recovery_rate. Failure mode: invalid JSON spikes.",
        },
        {
            "task": "schema_guard",
            "prompt": "Output valid JSON with keys root_cause, next_step, confidence.",
            "target": '{"root_cause":"insufficient calibration","next_step":"increase replay buffer and rerun benchmark","confidence":0.68}',
        },
        {
            "task": "ablation_reasoning",
            "prompt": "Provide one ablation idea and why it improves generalization.",
            "target": "Ablate retrieval memory and compare against full pipeline; this isolates memory contribution and reduces variance in attribution.",
        },
        {
            "task": "safety_contract",
            "prompt": "Return JSON with fields constraint, fallback, verification.",
            "target": '{"constraint":"never bypass schema checks","fallback":"return minimal safe JSON","verification":"run fixed-seed probe before keep"}',
        },
        {
            "task": "transfer_probe",
            "prompt": "Describe one transfer test across two domains and expected delta signal.",
            "target": "Source domain: formal reasoning, target domain: code generation. Expected delta signal: +0.01 in benchmark score after adapter replay.",
        },
        {
            "task": "rollback_policy",
            "prompt": "Return JSON with keys trigger, rollback_action, verification.",
            "target": '{"trigger":"benchmark gain below floor","rollback_action":"restore last best adapter","verification":"rerun fixed-seed benchmark"}',
        },
        {
            "task": "schema_repair",
            "prompt": "Write one schema repair strategy with a measurable success criterion.",
            "target": "Apply strict JSON schema validation and reject malformed outputs; success criterion: schema_valid_rate improves by at least 0.02.",
        },
        {
            "task": "calibration_check",
            "prompt": "Provide one confidence calibration check and expected outcome.",
            "target": "Bucket confidence into deciles and compare with empirical accuracy; expected outcome: calibration error decreases after replay regularization.",
        },
        {
            "task": "drift_alarm",
            "prompt": "Return JSON with fields signal, threshold, response.",
            "target": '{"signal":"nll_drift","threshold":0.015,"response":"trigger retraining and hold keep decision"}',
        },
        {
            "task": "safety_fallback",
            "prompt": "Describe a safe fallback behavior when model output is uncertain.",
            "target": "If uncertainty is high, return concise structured JSON with explicit unknowns and request one clarifying input before action.",
        },
        {
            "task": "eval_design",
            "prompt": "Design one fixed-seed evaluation protocol for adapter comparison.",
            "target": "Use a frozen prompt set, deterministic decoding, and paired before/after NLL scoring with bootstrap confidence intervals.",
        },
        {
            "task": "memory_ablation",
            "prompt": "Propose one memory ablation and expected failure pattern.",
            "target": "Remove retrieval memory for one run; expected failure pattern: weaker long-range consistency and higher invalid schema rate.",
        },
        {
            "task": "constraint_contract",
            "prompt": "Return JSON with keys invariant, monitor, escalation.",
            "target": '{"invariant":"never emit malformed JSON in contract mode","monitor":"schema validator + density probe","escalation":"discard candidate and keep previous baseline"}',
        },
        {
            "task": "latency_budget",
            "prompt": "Give one latency budget policy balancing quality and speed.",
            "target": "Use a two-stage budget: fast deterministic pass first, then selective deep pass only when quality probe is borderline.",
        },
        {
            "task": "error_taxonomy",
            "prompt": "List one compact error taxonomy for self-improvement loops.",
            "target": "Classify failures into schema, semantic, calibration, and degeneration errors, then attach one remediation per class.",
        },
        {
            "task": "counterfactual_test",
            "prompt": "Describe one counterfactual test for robustness claims.",
            "target": "Swap key assumptions in prompts while preserving format and compare degradation; robust systems should show graceful score drops.",
        },
        {
            "task": "quality_guard",
            "prompt": "Return JSON with fields structure_check, density_check, echo_check.",
            "target": '{"structure_check":"json shape + required keys","density_check":"alnum ratio above floor","echo_check":"completion similarity below threshold"}',
        },
        {
            "task": "replay_policy",
            "prompt": "Write one replay policy that avoids overfitting to tiny probes.",
            "target": "Replay a mixed mini-batch with held-out prompts and stop early when quality gain plateaus for multiple checks.",
        },
        {
            "task": "gate_explanation",
            "prompt": "Provide one concise explanation template for keep/discard decisions.",
            "target": "Decision = metric delta + benchmark gain + replay quality; keep only when all hard gates pass and no safety invariant is violated.",
        },
    ]


def _run_eval(model_name: str, adapter_dir: Path | None, seed: int) -> Dict[str, Any]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(model_name, adapter_dir=adapter_dir, device=device)
    rows: List[Dict[str, Any]] = []
    for item in _bench_set():
        out = _generate(model, tokenizer, item["prompt"], device=device)
        ll = _target_nll(model, tokenizer, item["prompt"], item["target"], device=device)
        rows.append({**item, "output": out, "ll": ll})

    metrics = _aggregate_ll_metrics(rows)
    return {
        "model_name": model_name,
        "adapter_dir": str(adapter_dir) if adapter_dir else "",
        "metrics": metrics,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic local incremental benchmark")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--adapter-dir", default="reports/trusted_nano_lora_weights_latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="local_incremental_benchmark")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_absolute():
        adapter_dir = ROOT / adapter_dir
    if not adapter_dir.exists():
        adapter_dir = None

    base_eval = _run_eval(args.model_name, adapter_dir=None, seed=int(args.seed))
    adap_eval = _run_eval(args.model_name, adapter_dir=adapter_dir, seed=int(args.seed))

    base_score = float((base_eval.get("metrics") or {}).get("score", 0.0))
    adap_score = float((adap_eval.get("metrics") or {}).get("score", 0.0))
    gain = adap_score - base_score

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "model_name": args.model_name,
        "adapter_dir": str(adapter_dir) if adapter_dir else "",
        "score_base": base_score,
        "score_adapter": adap_score,
        "gain": gain,
        "base": base_eval,
        "adapter": adap_eval,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "# Local Incremental Benchmark",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- seed: `{payload['seed']}`",
        f"- model_name: `{payload['model_name']}`",
        f"- adapter_dir: `{payload['adapter_dir']}`",
        f"- score_base: `{base_score:.6f}`",
        f"- score_adapter: `{adap_score:.6f}`",
        f"- gain: `{gain:+.6f}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
