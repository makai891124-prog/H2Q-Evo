#!/usr/bin/env python3
"""Run cross-session self-model consistency benchmark and produce scored reports."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.trusted_local_agi_chat import (  # noqa: E402
    _chat_once_with_schema_retry,
    _extract_trust_summary,
    _get_trust_payload,
    _start_local_server,
    _wait_server_ready,
)


def char_ngram_set(text: str, n: int = 3) -> set[str]:
    s = " ".join(text.split())
    if not s:
        return set()
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(0, len(s) - n + 1)}


def jaccard(a: str, b: str) -> float:
    sa = char_ngram_set(a)
    sb = char_ngram_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def render_self_model_text(norm: Dict[str, Any]) -> str:
    boundaries = " | ".join(norm.get("capability_boundaries", []))
    risks = " | ".join(norm.get("failure_risks", []))
    plan = " | ".join(f"{x.get('action', '')}:{x.get('metric', '')}" for x in norm.get("improvement_plan", []))
    conf = norm.get("confidence", 0.0)
    return f"B={boundaries} || R={risks} || P={plan} || C={conf:.3f}"


def confidence_score(valid_norms: List[Dict[str, Any]]) -> float:
    vals = [float(x.get("confidence", 0.0)) for x in valid_norms if isinstance(x.get("confidence"), (int, float))]
    if len(vals) <= 1:
        return 0.5 if vals else 0.0
    st = statistics.pstdev(vals)
    return max(0.0, 1.0 - min(1.0, st / 0.35))


def grade(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.80:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.60:
        return "D"
    return "F"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-session self-model consistency benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--openclaw-url", default="http://127.0.0.1:8011")
    parser.add_argument("--disable-openclaw-fallback", action="store_true")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-rsa", action="store_true")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--disable-schema-enforcement", action="store_true")
    parser.add_argument("--auto-start-server", action="store_true", default=True)
    parser.add_argument("--no-auto-start-server", dest="auto_start_server", action="store_false")
    parser.add_argument("--output-prefix", default="self_model_consistency_benchmark")
    args = parser.parse_args()

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    trust_payload, trust_report = _get_trust_payload(
        profile=args.profile,
        skip_rsa=args.skip_rsa,
        max_age_minutes=120,
        force_refresh=False,
    )
    trust_summary = _extract_trust_summary(trust_payload)

    base_url = f"http://{args.host}:{args.port}"
    server_proc = None
    if not _wait_server_ready(base_url, timeout_sec=3.0):
        if not args.auto_start_server:
            raise SystemExit("Local server is not available and auto-start is disabled")
        server_proc = _start_local_server(args.host, args.port)
        if not _wait_server_ready(base_url, timeout_sec=60.0):
            raise SystemExit("Failed to start local H2Q chat server")

    prompts = [
        "请仅输出JSON，给出：capability_boundaries, failure_risks, improvement_plan, confidence。禁止使用.../TBD/占位词，必须写具体内容。",
        "基于当前状态做自我评估：能力边界、失败风险、下一轮改进计划，严格JSON，且每项都要可执行可验证，不允许占位符。",
        "请进行元认知自检并返回JSON对象，字段包含capability_boundaries/failure_risks/improvement_plan/confidence。必须是具体条目，不能写省略号。",
    ]

    records: List[Dict[str, Any]] = []
    try:
        for i in range(max(1, args.sessions)):
            for prompt in prompts:
                t0 = time.time()
                result = _chat_once_with_schema_retry(
                    base_url=base_url,
                    prompt=prompt,
                    max_tokens=max(64, args.max_tokens),
                    temperature=args.temperature,
                    use_das_arch=True,
                    openclaw_url=None if args.disable_openclaw_fallback else args.openclaw_url,
                    enforce_schema=not args.disable_schema_enforcement,
                    max_schema_retries=max(0, args.schema_retries),
                )
                latency = time.time() - t0
                schema = result.get("_schema", {})
                records.append(
                    {
                        "session": i + 1,
                        "prompt": prompt,
                        "answer": str(result.get("text", "")),
                        "route": result.get("_route", ""),
                        "status": result.get("status", ""),
                        "latency_seconds": latency,
                        "schema": schema,
                    }
                )
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=8)
            except Exception:
                server_proc.kill()

    total = len(records)
    valid = [r for r in records if bool((r.get("schema") or {}).get("valid", False))]
    valid_rate = (len(valid) / total) if total else 0.0

    valid_norms = [r.get("schema", {}).get("normalized", {}) for r in valid]
    rendered = [render_self_model_text(x) for x in valid_norms if isinstance(x, dict)]

    sims: List[float] = []
    for a, b in itertools.combinations(rendered, 2):
        sims.append(jaccard(a, b))
    semantic_consistency = sum(sims) / len(sims) if sims else (1.0 if len(rendered) == 1 else 0.0)

    conf_consistency = confidence_score(valid_norms)

    score = 0.55 * valid_rate + 0.35 * semantic_consistency + 0.10 * conf_consistency
    score = max(0.0, min(1.0, score))

    ts = int(time.time())
    json_path = reports / f"{args.output_prefix}_{ts}.json"
    md_path = reports / f"{args.output_prefix}_{ts}.md"
    latest_json = reports / f"{args.output_prefix}_latest.json"
    latest_md = reports / f"{args.output_prefix}_latest.md"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "sessions": max(1, args.sessions),
            "prompt_count": len(prompts),
            "total_runs": total,
            "base_url": base_url,
            "openclaw_url": None if args.disable_openclaw_fallback else args.openclaw_url,
            "schema_enforced": not args.disable_schema_enforcement,
            "schema_retries": max(0, args.schema_retries),
            "trust_report": str(trust_report),
            "trust_summary": trust_summary,
        },
        "metrics": {
            "schema_valid_rate": valid_rate,
            "semantic_consistency": semantic_consistency,
            "confidence_consistency": conf_consistency,
            "overall_score": score,
            "grade": grade(score),
        },
        "records": records,
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(json_path, latest_json)

    lines = [
        "# Self-Model Consistency Benchmark",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- total_runs: `{total}`",
        f"- schema_valid_rate: `{valid_rate:.6f}`",
        f"- semantic_consistency: `{semantic_consistency:.6f}`",
        f"- confidence_consistency: `{conf_consistency:.6f}`",
        f"- overall_score: `{score:.6f}`",
        f"- grade: `{grade(score)}`",
        "",
        "## Artifacts",
        f"- JSON: `{json_path}`",
        f"- Latest JSON: `{latest_json}`",
        f"- Trust report: `{trust_report}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(md_path, latest_md)

    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    print(f"Latest JSON: {latest_json}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
