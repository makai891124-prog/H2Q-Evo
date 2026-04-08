#!/usr/bin/env python3
"""Generate a final demo scorecard from the latest 3-step experience artifacts.

Outputs:
- reports/final_demo_scorecard_<ts>.md
- reports/final_demo_scorecard_latest.md
- reports/final_demo_scorecard_<ts>.json
- reports/final_demo_scorecard_latest.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def latest_by_pattern(base: Path, pattern: str) -> Optional[Path]:
    files = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def latest_valid_json_by_pattern(base: Path, pattern: str) -> Optional[Path]:
    files = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            json.loads(raw)
            return path
        except Exception:
            continue
    return None


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def score_step1(step1: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
    if not step1:
        return 0, "missing", {"reason": "step1 artifact not found"}

    checks = {
        "object_is_response": step1.get("object") == "response",
        "status_completed": step1.get("status") == "completed",
        "has_output_text": bool(str(step1.get("output_text", "")).strip()),
        "has_usage": isinstance(step1.get("usage"), dict),
    }
    passed = sum(1 for v in checks.values() if v)
    score = int(round((passed / len(checks)) * 100))
    verdict = "pass" if passed == len(checks) else "partial"
    return score, verdict, checks


def score_step2(step2: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
    if not step2:
        return 0, "missing", {"reason": "step2 artifact not found"}

    artifacts = step2.get("artifacts") or []
    checks = {
        "mode_full": step2.get("mode") == "full",
        "ok_true": bool(step2.get("ok", False)),
        "confidence_present": isinstance(step2.get("confidence"), (int, float)),
        "artifacts_present": isinstance(artifacts, list) and len(artifacts) >= 2,
    }
    passed = sum(1 for v in checks.values() if v)
    score = int(round((passed / len(checks)) * 100))
    verdict = "pass" if passed == len(checks) else "partial"
    return score, verdict, checks


def score_step3(step3: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
    if not step3:
        return 0, "missing", {"reason": "step3 artifact not found"}

    trust = step3.get("trust") or {}
    transcript = step3.get("transcript") or []
    first = transcript[0] if transcript else {}
    runtime = first.get("runtime") or {}
    assistant = str(first.get("assistant", "")).strip()

    checks = {
        "trusted_ready": bool(trust.get("trusted_ready", False)),
        "trust_score_gte_07": float(trust.get("trust_score", 0.0)) >= 0.70,
        "has_transcript": len(transcript) >= 1,
        "assistant_non_empty": bool(assistant),
    }

    quality_flags = {
        "contains_code_fence": "```" in assistant,
        "has_route": bool(runtime.get("route")),
        "latency_recorded": isinstance(runtime.get("latency_seconds"), (int, float)),
    }

    passed = sum(1 for v in checks.values() if v)
    quality_passed = sum(1 for v in quality_flags.values() if v)

    score = int(round(((passed / len(checks)) * 0.8 + (quality_passed / len(quality_flags)) * 0.2) * 100))
    verdict = "pass" if passed == len(checks) else "partial"

    merged: Dict[str, Any] = {}
    merged.update(checks)
    merged.update(quality_flags)
    return score, verdict, merged


def overall_grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    step1_path = latest_valid_json_by_pattern(reports, "one_click_step1_responses_*.json")
    step2_path = latest_valid_json_by_pattern(reports, "one_click_step2_full_*.json")
    step3_path = latest_valid_json_by_pattern(reports, "trusted_local_agi_chat_session_*.json")

    step1 = load_json(step1_path)
    step2 = load_json(step2_path)
    step3 = load_json(step3_path)

    s1, v1, c1 = score_step1(step1)
    s2, v2, c2 = score_step2(step2)
    s3, v3, c3 = score_step3(step3)

    overall = int(round((s1 + s2 + s3) / 3))
    grade = overall_grade(overall)
    ts = int(time.time())
    now = datetime.now(timezone.utc).isoformat()

    payload = {
        "generated_at_utc": now,
        "inputs": {
            "step1": str(step1_path) if step1_path else "",
            "step2": str(step2_path) if step2_path else "",
            "step3": str(step3_path) if step3_path else "",
        },
        "scores": {
            "step1_protocol": {"score": s1, "verdict": v1, "checks": c1},
            "step2_full_chain": {"score": s2, "verdict": v2, "checks": c2},
            "step3_trusted_chat": {"score": s3, "verdict": v3, "checks": c3},
            "overall": overall,
            "grade": grade,
        },
    }

    out_json = reports / f"final_demo_scorecard_{ts}.json"
    out_json_latest = reports / "final_demo_scorecard_latest.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_json_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = [
        "# Final Demo Scorecard",
        "",
        f"- generated_at_utc: `{now}`",
        f"- overall_score: `{overall}`",
        f"- grade: `{grade}`",
        "",
        "## Inputs",
        "",
        f"- step1_protocol_artifact: `{step1_path}`",
        f"- step2_full_chain_artifact: `{step2_path}`",
        f"- step3_trusted_chat_artifact: `{step3_path}`",
        "",
        "## Step Scores",
        "",
        f"- step1_protocol: score=`{s1}`, verdict=`{v1}`",
        f"- step2_full_chain: score=`{s2}`, verdict=`{v2}`",
        f"- step3_trusted_chat: score=`{s3}`, verdict=`{v3}`",
        "",
        "## Key Checks",
        "",
        f"- step1 checks: `{c1}`",
        f"- step2 checks: `{c2}`",
        f"- step3 checks: `{c3}`",
        "",
        "## Notes",
        "",
        "- This scorecard reflects runnable integration quality, not benchmark superiority claims.",
        "- Re-run one_click_agi_experience.sh before regenerating for fresh evidence.",
    ]

    out_md = reports / f"final_demo_scorecard_{ts}.md"
    out_md_latest = reports / "final_demo_scorecard_latest.md"
    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    out_md_latest.write_text(text, encoding="utf-8")

    print(f"Scorecard MD: {out_md}")
    print(f"Scorecard MD Latest: {out_md_latest}")
    print(f"Scorecard JSON: {out_json}")
    print(f"Scorecard JSON Latest: {out_json_latest}")
    print(f"Overall: {overall} ({grade})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
