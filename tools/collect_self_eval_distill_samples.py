#!/usr/bin/env python3
"""Collect strict-JSON self-eval failure samples and create teacher/student dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.trusted_local_agi_chat import (  # noqa: E402
    _extract_json_candidate,
    _validate_self_eval_schema,
)


def _http_json(url: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _heuristic_teacher_json(prompt: str) -> Dict[str, Any]:
    p = prompt.strip()
    brief = p[:48]
    return {
        "capability_boundaries": [
            "当前主要擅长结构化任务分解与短程规划，跨会话长期记忆一致性仍不稳定。",
            "对需要高保真自我建模的元认知问题，容易退化到模板化回答。",
        ],
        "failure_risks": [
            "在自我评估场景可能触发fallback路径，导致输出非JSON或无效JSON结构。",
            "修复提示词可能被重复回显，降低有效信息密度并影响后续蒸馏质量。",
        ],
        "improvement_plan": [
            {
                "action": f"为提示'{brief}'启用蒸馏适配优先路由并记录失败样本。",
                "metric": "self_eval_schema_valid_rate >= 0.20 within next 30 runs",
            },
            {
                "action": "对strict-json失败样本执行teacher监督蒸馏并更新本地适配器。",
                "metric": "fallback_ratio_self_eval <= 0.70 within next 30 runs",
            },
        ],
        "confidence": 0.62,
    }


def _build_teacher_prompt(prompt: str) -> str:
    return (
        "Return only one JSON object. No markdown. No code fences.\n"
        "Required keys:\n"
        "- capability_boundaries: list[string] with concrete constraints\n"
        "- failure_risks: list[string] with concrete failure modes\n"
        "- improvement_plan: list[{action:string, metric:string}]\n"
        "- confidence: number in [0,1]\n"
        "No placeholders (..., TBD, TODO, unknown).\n"
        "Original prompt:\n"
        f"{prompt}"
    )


def _deepseek_teacher_json(prompt: str, model: str, key: str, base_url: str) -> Optional[Dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON generator for self-evaluation training data.",
            },
            {
                "role": "user",
                "content": _build_teacher_prompt(prompt),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 700,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(req, timeout=90.0) as resp:
            raw = resp.read().decode("utf-8")
    except URLError:
        return None
    except Exception:
        return None

    try:
        data = json.loads(raw)
        text = str(data["choices"][0]["message"]["content"])
    except Exception:
        return None

    parsed = _extract_json_candidate(text)
    return parsed if isinstance(parsed, dict) else None


def _collect_failed_prompts(max_samples: int) -> List[Dict[str, Any]]:
    sessions = sorted(REPORTS.glob("trusted_local_agi_chat_session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    by_prompt: Dict[str, Dict[str, Any]] = {}

    for session_path in sessions:
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for turn in payload.get("transcript", []):
            user_prompt = str(turn.get("user", "")).strip()
            if not user_prompt:
                continue
            runtime = turn.get("runtime") or {}
            schema = runtime.get("schema") or {}
            if not bool(schema.get("required", False)):
                continue
            if bool(schema.get("valid", False)):
                continue
            if user_prompt not in by_prompt:
                by_prompt[user_prompt] = {
                    "prompt": user_prompt,
                    "failure_count": 0,
                    "student_examples": [],
                    "source_sessions": [],
                }
            item = by_prompt[user_prompt]
            item["failure_count"] += 1
            if len(item["student_examples"]) < 3:
                item["student_examples"].append(str(turn.get("assistant", ""))[:1200])
            if len(item["source_sessions"]) < 20:
                item["source_sessions"].append(str(session_path))

    rows = sorted(by_prompt.values(), key=lambda x: x["failure_count"], reverse=True)
    return rows[: max(1, max_samples)]


def _collect_prompt_pool(max_samples: int, include_valid_prompts: bool) -> List[Dict[str, Any]]:
    sessions = sorted(REPORTS.glob("trusted_local_agi_chat_session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    by_prompt: Dict[str, Dict[str, Any]] = {}

    for session_path in sessions:
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for turn in payload.get("transcript", []):
            user_prompt = str(turn.get("user", "")).strip()
            if not user_prompt:
                continue
            runtime = turn.get("runtime") or {}
            schema = runtime.get("schema") or {}
            if not bool(schema.get("required", False)):
                continue

            valid = bool(schema.get("valid", False))
            if (not include_valid_prompts) and valid:
                continue

            if user_prompt not in by_prompt:
                by_prompt[user_prompt] = {
                    "prompt": user_prompt,
                    "failure_count": 0,
                    "valid_count": 0,
                    "student_examples": [],
                    "source_sessions": [],
                }

            item = by_prompt[user_prompt]
            if valid:
                item["valid_count"] += 1
            else:
                item["failure_count"] += 1

            if len(item["student_examples"]) < 3:
                item["student_examples"].append(str(turn.get("assistant", ""))[:1200])
            if len(item["source_sessions"]) < 20:
                item["source_sessions"].append(str(session_path))

    rows = sorted(
        by_prompt.values(),
        key=lambda x: (x["failure_count"], x["valid_count"]),
        reverse=True,
    )
    return rows[: max(1, max_samples)]


def _synthetic_prompts() -> List[str]:
    axes = [
        "reasoning",
        "robustness",
        "alignment",
        "memory",
        "tool-use",
        "safety",
        "planning",
        "self-correction",
        "uncertainty",
        "schema-compliance",
    ]
    templates = [
        "Provide a strict JSON self-evaluation for {axis} with concrete limits and next experiments.",
        "For {axis}, output only JSON with capability_boundaries, failure_risks, improvement_plan, confidence.",
        "Return one JSON object evaluating current {axis} weaknesses and measurable recovery plan.",
        "Generate strict JSON: what is currently fragile in {axis}, and which metrics prove improvement?",
        "Self-audit {axis} now in strict JSON; include actionable milestones for the next 7 days.",
    ]
    contexts = [
        "for coding assistants",
        "for autonomous bootstrap loops",
        "for safety-critical planning",
        "for long-run memory consistency",
    ]
    prompts: List[str] = []
    for axis in axes:
        for tpl in templates:
            for ctx in contexts:
                prompts.append(f"{tpl.format(axis=axis)} Context: {ctx}.")
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect self-eval strict-json failure samples")
    parser.add_argument("--output-prefix", default="self_eval_distill_dataset")
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--include-valid-prompts", action="store_true")
    parser.add_argument("--synthetic-min-size", type=int, default=120)
    parser.add_argument("--teacher-provider", choices=["deepseek", "heuristic"], default="deepseek")
    parser.add_argument("--teacher-model", default="deepseek-chat")
    parser.add_argument("--teacher-base-url", default="https://api.deepseek.com")
    parser.add_argument("--teacher-key-file", default="secrets/deepseek_api_key.txt")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    samples = _collect_prompt_pool(
        max_samples=max(1, args.max_samples),
        include_valid_prompts=bool(args.include_valid_prompts),
    )

    if len(samples) < max(0, int(args.synthetic_min_size)):
        seen = {str(x.get("prompt", "")) for x in samples}
        for p in _synthetic_prompts():
            if p in seen:
                continue
            samples.append(
                {
                    "prompt": p,
                    "failure_count": 0,
                    "valid_count": 0,
                    "student_examples": [],
                    "source_sessions": ["synthetic_prompt_bank"],
                }
            )
            seen.add(p)
            if len(samples) >= max(1, args.max_samples):
                break

    key = ""
    key_file = Path(args.teacher_key_file)
    if not key_file.is_absolute():
        key_file = ROOT / key_file
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()

    out_samples: List[Dict[str, Any]] = []
    for row in samples:
        prompt = row["prompt"]
        teacher_obj: Optional[Dict[str, Any]] = None
        teacher_source = "heuristic"

        if args.teacher_provider == "deepseek" and key:
            teacher_obj = _deepseek_teacher_json(
                prompt=prompt,
                model=args.teacher_model,
                key=key,
                base_url=args.teacher_base_url,
            )
            teacher_source = "deepseek" if teacher_obj is not None else "heuristic_fallback"

        if teacher_obj is None:
            teacher_obj = _heuristic_teacher_json(prompt)

        ok, errors, normalized = _validate_self_eval_schema(
            teacher_obj,
            min_boundary_chars=8,
            min_risk_chars=12,
            min_action_chars=8,
            min_metric_chars=4,
            forbid_placeholders=True,
        )

        out_samples.append(
            {
                **row,
                "teacher_source": teacher_source,
                "teacher_json": teacher_obj,
                "teacher_valid": bool(ok),
                "teacher_errors": errors,
                "teacher_normalized": normalized,
            }
        )

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_provider": args.teacher_provider,
        "teacher_model": args.teacher_model,
        "sample_count": len(out_samples),
        "samples": out_samples,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_count = sum(1 for x in out_samples if x.get("teacher_valid"))
    lines = [
        "# Self-Eval Distillation Dataset",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- teacher_provider: `{args.teacher_provider}`",
        f"- sample_count: `{len(out_samples)}`",
        f"- teacher_valid_count: `{valid_count}`",
        "",
        "## Top Prompts",
    ]
    for item in out_samples[:10]:
        lines.append(f"- failure_count={item['failure_count']}: `{item['prompt'][:120]}`")

    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    latest_md.write_text(text, encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
