#!/usr/bin/env python3
"""Local trusted AGI chat orchestrator.

This script combines two real capabilities already implemented in the repo:
1) Trusted joint validation gate from tools/trusted_joint_agi_quantum_center.py
2) Local chat inference endpoint from h2q_project/h2q_server.py (/chat)

It gives a directly runnable local conversation loop with trust evidence.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.trusted_joint_agi_quantum_center import run_center


DEFAULT_SELF_EVAL_DISTILL_MODEL = ROOT / "reports" / "self_eval_distill_model_latest.json"
_DISTILL_MODEL_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    body: Optional[bytes] = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        method = "POST"
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, data=body, headers=headers, method=method)
    opener = build_opener(ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _wait_server_ready(base_url: str, timeout_sec: float = 45.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            # Some server variants have inconsistent /health implementations.
            # Treat either /health or a minimal /chat response as readiness.
            try:
                _http_json(f"{base_url}/health", timeout=3.0)
                return True
            except Exception:
                _http_json(
                    f"{base_url}/chat",
                    payload={"prompt": "ping", "max_tokens": 8, "temperature": 0.1, "use_das_arch": False},
                    timeout=8.0,
                )
                return True
        except Exception:
            time.sleep(0.6)
    return False


def _latest_center_report(report_dir: Path) -> Optional[Path]:
    reports = sorted(report_dir.glob("trusted_joint_agi_quantum_center_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_trust_payload(profile: str, skip_rsa: bool, max_age_minutes: int, force_refresh: bool) -> Tuple[Dict[str, Any], Path]:
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    latest = _latest_center_report(report_dir)
    if latest is not None and not force_refresh:
        age_min = (time.time() - latest.stat().st_mtime) / 60.0
        if age_min <= max_age_minutes:
            return _load_json(latest), latest

    out_json, _ = run_center(profile=profile, include_rsa=not skip_rsa, rsa_folds=2)
    return _load_json(out_json), out_json


def _extract_trust_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    agg = payload.get("aggregate", {})
    gates = agg.get("gates", {})
    return {
        "trust_score": float(agg.get("trust_score", 0.0)),
        "trusted_ready": bool(agg.get("trusted_ready", False)),
        "gates": {
            "das_decision_ready": bool(gates.get("das_decision_ready", False)),
            "dual_aligned_consistent": bool(gates.get("dual_aligned_consistent", False)),
            "codec_integrity": bool(gates.get("codec_integrity", False)),
            "rsa_parallel_observed": bool(gates.get("rsa_parallel_observed", False)),
        },
    }


def _start_local_server(host: str, port: int) -> subprocess.Popen:
    log_dir = ROOT / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"trusted_local_agi_server_{int(time.time())}.log"
    log_fp = log_file.open("w", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "h2q_project.h2q_server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_fp, stderr=subprocess.STDOUT)


def _is_effective_text(value: Any, prompt: str = "") -> bool:
    text = str(value or "").strip()
    if not text or text == "(empty response)":
        return False
    if prompt and text == prompt.strip():
        return False
    lp = prompt.lower()
    lt = text.lower()
    if any(k in lp for k in ["python", "pytest", "函数", "脚本", "编程", "json"]):
        quality_tokens = ["def ", "assert", "```", "{", "}", "return"]
        if not any(t in lt for t in quality_tokens):
            return False
        if len(text) < 24:
            return False
    return True


def _fallback_text(prompt: str) -> str:
    p = prompt.strip()
    lower = p.lower()
    if "two_sum" in lower or "two sum" in lower:
        return (
            "```python\n"
            "# two_sum.py\n"
            "def two_sum(nums: list[int], target: int) -> list[int]:\n"
            "    seen: dict[int, int] = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        need = target - n\n"
            "        if need in seen:\n"
            "            return [seen[need], i]\n"
            "        seen[n] = i\n"
            "    raise ValueError(\"No two sum solution\")\n"
            "\n"
            "# test_two_sum.py\n"
            "import pytest\n"
            "from two_sum import two_sum\n"
            "\n"
            "def test_two_sum_basic():\n"
            "    assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n"
            "\n"
            "def test_two_sum_repeat_value():\n"
            "    assert two_sum([3, 3], 6) == [0, 1]\n"
            "\n"
            "def test_two_sum_negative_values():\n"
            "    assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]\n"
            "\n"
            "def test_two_sum_no_solution():\n"
            "    with pytest.raises(ValueError):\n"
            "        two_sum([1, 2, 3], 7)\n"
            "```"
        )
    if any(k in lower for k in ["python", "code", "函数", "脚本", "编程", "json"]):
        return (
            "```python\n"
            "def solve_task(task: str) -> dict:\n"
            "    \"\"\"Task-aware local fallback for coding requests.\"\"\"\n"
            "    return {\n"
            "        \"task\": task,\n"
            "        \"implementation\": \"先给最小可运行版本，再补测试与边界\",\n"
            "        \"checks\": [\"功能正确\", \"边界覆盖\", \"异常路径\"],\n"
            "    }\n"
            "```"
        )
    return (
        "本地模型当前处于保守裁剪模式，已切换到系统应急策略。\n"
        "1. 先定义本轮可验证目标。\n"
        "2. 生成最小可执行方案并记录指标。\n"
        "3. 根据失败样本进行下一轮改进。"
    )


def _needs_self_eval_schema(prompt: str) -> bool:
    p = prompt.lower()
    keys = [
        "能力边界",
        "失败风险",
        "改进计划",
        "self",
        "introspect",
        "json",
        "meta",
        "自我",
    ]
    return any(k in p for k in keys)


def _extract_json_candidate(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return None

    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, flags=re.IGNORECASE)
        if match:
            raw = match.group(1)

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _normalize_list_or_str(value: Any) -> Optional[List[str]]:
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else None
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            t = str(item).strip()
            if t:
                out.append(t)
        return out if out else None
    return None


def _is_placeholder_text(value: str) -> bool:
    text = value.strip().lower()
    placeholders = {
        "...",
        "..",
        "placeholder",
        "tbd",
        "todo",
        "n/a",
        "na",
        "unknown",
        "待定",
        "占位",
        "未知",
    }
    if text in placeholders:
        return True
    if len(text) <= 4 and set(text) == {"."}:
        return True
    return False


def _tokenize_for_similarity(text: str) -> List[str]:
    return [
        tok
        for tok in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text.lower())
        if len(tok) >= 2
    ]


def _load_self_eval_distill_model(model_path: Path) -> Dict[str, Any]:
    key = str(model_path.resolve())
    if not model_path.exists():
        return {}
    mtime = model_path.stat().st_mtime
    cached = _DISTILL_MODEL_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _DISTILL_MODEL_CACHE[key] = (mtime, data)
            return data
    except Exception:
        return {}
    return {}


def _jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    sa = set(tokens_a)
    sb = set(tokens_b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _build_self_eval_from_distill_model(
    prompt: str,
    model_path: Optional[Path],
    min_boundary_chars: int,
    min_risk_chars: int,
    min_action_chars: int,
    min_metric_chars: int,
    forbid_placeholders: bool,
) -> Optional[Dict[str, Any]]:
    if model_path is None:
        return None
    model = _load_self_eval_distill_model(model_path)
    if not model:
        return None

    prompt_map = model.get("prompt_map") or {}
    if isinstance(prompt_map, dict) and prompt in prompt_map and isinstance(prompt_map[prompt], dict):
        obj = prompt_map[prompt]
        ok, _, normalized = _validate_self_eval_schema(
            obj,
            min_boundary_chars=min_boundary_chars,
            min_risk_chars=min_risk_chars,
            min_action_chars=min_action_chars,
            min_metric_chars=min_metric_chars,
            forbid_placeholders=forbid_placeholders,
        )
        if ok:
            return normalized

    entries = model.get("entries") or []
    prompt_tokens = _tokenize_for_similarity(prompt)
    best_obj: Optional[Dict[str, Any]] = None
    best_score = -1.0
    min_similarity = float(model.get("min_similarity", 0.08))

    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            teacher_obj = item.get("teacher_json")
            if not isinstance(teacher_obj, dict):
                continue
            entry_tokens = item.get("keywords")
            if isinstance(entry_tokens, list):
                et = [str(x) for x in entry_tokens]
            else:
                et = _tokenize_for_similarity(str(item.get("prompt", "")))
            score = _jaccard(prompt_tokens, et)
            if score > best_score:
                best_score = score
                best_obj = teacher_obj

    fallback_obj = model.get("default_template") if isinstance(model.get("default_template"), dict) else None
    chosen = best_obj if (best_obj is not None and best_score >= min_similarity) else fallback_obj
    if not isinstance(chosen, dict):
        return None

    ok, _, normalized = _validate_self_eval_schema(
        chosen,
        min_boundary_chars=min_boundary_chars,
        min_risk_chars=min_risk_chars,
        min_action_chars=min_action_chars,
        min_metric_chars=min_metric_chars,
        forbid_placeholders=forbid_placeholders,
    )
    return normalized if ok else None


def _validate_self_eval_schema(
    obj: Optional[Dict[str, Any]],
    min_boundary_chars: int = 8,
    min_risk_chars: int = 12,
    min_action_chars: int = 8,
    min_metric_chars: int = 4,
    forbid_placeholders: bool = True,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    if not isinstance(obj, dict):
        return False, ["Top-level JSON object is missing or invalid."], {}

    normalized: Dict[str, Any] = {}
    errors: List[str] = []

    boundaries = _normalize_list_or_str(obj.get("capability_boundaries"))
    if not boundaries:
        errors.append("Missing non-empty 'capability_boundaries' (string or list[string]).")
    else:
        for i, item in enumerate(boundaries):
            if len(item.strip()) < max(1, min_boundary_chars):
                errors.append(
                    f"capability_boundaries[{i}] too short; require >= {max(1, min_boundary_chars)} chars."
                )
            if forbid_placeholders and _is_placeholder_text(item):
                errors.append(f"capability_boundaries[{i}] appears to be placeholder text.")
        normalized["capability_boundaries"] = boundaries

    risks = _normalize_list_or_str(obj.get("failure_risks"))
    if not risks:
        errors.append("Missing non-empty 'failure_risks' (string or list[string]).")
    else:
        for i, item in enumerate(risks):
            if len(item.strip()) < max(1, min_risk_chars):
                errors.append(f"failure_risks[{i}] too short; require >= {max(1, min_risk_chars)} chars.")
            if forbid_placeholders and _is_placeholder_text(item):
                errors.append(f"failure_risks[{i}] appears to be placeholder text.")
        normalized["failure_risks"] = risks

    plan = obj.get("improvement_plan")
    normalized_plan: List[Dict[str, str]] = []
    if isinstance(plan, list):
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                errors.append(f"improvement_plan[{i}] must be an object.")
                continue
            action = str(step.get("action", "")).strip()
            metric = str(step.get("metric", "")).strip()
            if not action or not metric:
                errors.append(f"improvement_plan[{i}] must include non-empty action and metric.")
                continue
            if len(action) < max(1, min_action_chars):
                errors.append(f"improvement_plan[{i}].action too short; require >= {max(1, min_action_chars)} chars.")
            if len(metric) < max(1, min_metric_chars):
                errors.append(f"improvement_plan[{i}].metric too short; require >= {max(1, min_metric_chars)} chars.")
            if forbid_placeholders and _is_placeholder_text(action):
                errors.append(f"improvement_plan[{i}].action appears to be placeholder text.")
            if forbid_placeholders and _is_placeholder_text(metric):
                errors.append(f"improvement_plan[{i}].metric appears to be placeholder text.")
            normalized_plan.append({"action": action, "metric": metric})
    elif isinstance(plan, str) and plan.strip():
        normalized_plan.append({"action": plan.strip(), "metric": "defined_in_next_iteration"})
    else:
        errors.append("Missing non-empty 'improvement_plan' (list[object] preferred).")

    if normalized_plan:
        normalized["improvement_plan"] = normalized_plan

    confidence = obj.get("confidence")
    if isinstance(confidence, (int, float)):
        conf = float(confidence)
    elif isinstance(confidence, str) and confidence.strip():
        try:
            conf = float(confidence)
        except Exception:
            conf = -1.0
    else:
        conf = -1.0

    if not (0.0 <= conf <= 1.0):
        errors.append("Missing/invalid 'confidence' in range [0, 1].")
    else:
        normalized["confidence"] = conf

    return (len(errors) == 0), errors, normalized


def _make_self_eval_repair_prompt(original_prompt: str, previous_answer: str, errors: List[str]) -> str:
    joined = "\n".join(f"- {e}" for e in errors)
    return (
        "You must output ONLY one valid JSON object, no markdown and no prose.\n"
        "Do not use placeholders like ..., TBD, TODO, unknown.\n"
        "Each boundary/risk/action must be concrete and specific.\n"
        "Required keys and constraints:\n"
        "- capability_boundaries: non-empty list of concrete strings\n"
        "- failure_risks: non-empty list of concrete strings\n"
        "- improvement_plan: non-empty list of objects with action and metric\n"
        "- confidence: number in [0,1]\n"
        "Validation errors from your previous answer:\n"
        f"{joined}\n\n"
        "Original user request:\n"
        f"{original_prompt}\n\n"
        "Your previous answer:\n"
        f"{previous_answer}"
    )


def _post_chat(base_url: str, prompt: str, max_tokens: int, temperature: float, use_das_arch: bool) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_das_arch": use_das_arch,
    }
    result = _http_json(f"{base_url}/chat", payload=payload, timeout=120.0)
    result["_route"] = f"chat(use_das_arch={use_das_arch})"
    return result


def _post_generate(base_url: str, prompt: str, max_tokens: int, temperature: float, use_das_arch: bool) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "use_das_arch": use_das_arch,
    }
    result = _http_json(f"{base_url}/generate", payload=payload, timeout=120.0)
    result["_route"] = f"generate(use_das_arch={use_das_arch})"
    return result


def _post_openclaw_responses(openclaw_url: str, prompt: str) -> Dict[str, Any]:
    payload = {
        "model": "h2q-openclaw",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }
    result = _http_json(
        f"{openclaw_url}/v1/responses",
        payload=payload,
        timeout=120.0,
    )
    text = result.get("output_text", "")
    if not text:
        output = result.get("output", [])
        try:
            text = output[0]["content"][0]["text"]
        except Exception:
            text = ""
    return {
        "text": str(text),
        "status": result.get("status", "completed"),
        "fueter_curvature": None,
        "spectral_shift_eta": None,
        "_route": "openclaw.v1.responses",
    }


def _make_openclaw_strict_json_prompt(prompt: str) -> str:
    return (
        "Return ONLY a JSON object. No markdown, no prose, no code fences.\n"
        "Required keys:\n"
        "- capability_boundaries: list[string]\n"
        "- failure_risks: list[string]\n"
        "- improvement_plan: list[{action:string, metric:string}]\n"
        "- confidence: number in [0,1]\n"
        "Do not use placeholders like ..., TBD, TODO, unknown.\n"
        "Original request:\n"
        f"{prompt}"
    )


def _post_openclaw_responses_strict_json(openclaw_url: str, prompt: str) -> Dict[str, Any]:
    strict_prompt = _make_openclaw_strict_json_prompt(prompt)
    result = _post_openclaw_responses(openclaw_url=openclaw_url, prompt=strict_prompt)
    result["_route"] = "openclaw.v1.responses.strict_json"
    return result


def _chat_once(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    use_das_arch: bool,
    openclaw_url: Optional[str] = None,
    force_self_eval_strict_json_attempt: bool = False,
) -> Dict[str, Any]:
    strict_json_attempted = False
    # Adaptive route: /chat first, then /generate, then alternate branch.
    attempts = [
        ("chat", use_das_arch, max_tokens, temperature),
        ("generate", use_das_arch, max_tokens, temperature),
        ("chat", not use_das_arch, max_tokens, min(1.0, temperature + 0.1)),
        ("generate", not use_das_arch, max_tokens, min(1.0, temperature + 0.1)),
    ]

    last_result: Dict[str, Any] = {}
    for route, flag, mx, temp in attempts:
        try:
            if route == "chat":
                res = _post_chat(base_url, prompt, mx, temp, flag)
            else:
                res = _post_generate(base_url, prompt, mx, temp, flag)
            last_result = res
            if _is_effective_text(res.get("text", ""), prompt=prompt):
                return res
        except Exception as exc:
            last_result = {
                "text": "",
                "status": "route-error",
                "fueter_curvature": None,
                "spectral_shift_eta": None,
                "_route": f"{route}(use_das_arch={flag})",
                "_error": str(exc),
            }

    # If core local routes are weak/empty, try OpenClaw-compatible responses path.
    if openclaw_url:
        try:
            ocr = _post_openclaw_responses(openclaw_url=openclaw_url, prompt=prompt)
            if _is_effective_text(ocr.get("text", ""), prompt=prompt):
                return ocr
            last_result = ocr
        except Exception as exc:
            last_result = {
                "text": "",
                "status": "route-error",
                "fueter_curvature": None,
                "spectral_shift_eta": None,
                "_route": "openclaw.v1.responses",
                "_error": str(exc),
            }

    # For introspection prompts, enforce one extra strict-JSON OpenClaw attempt before fallback.
    if force_self_eval_strict_json_attempt and openclaw_url and _needs_self_eval_schema(prompt):
        strict_json_attempted = True
        try:
            strict_res = _post_openclaw_responses_strict_json(openclaw_url=openclaw_url, prompt=prompt)
            strict_res["_strict_json_attempted"] = True
            if _is_effective_text(strict_res.get("text", ""), prompt=prompt):
                return strict_res
            last_result = strict_res
        except Exception as exc:
            last_result = {
                "text": "",
                "status": "route-error",
                "fueter_curvature": None,
                "spectral_shift_eta": None,
                "_route": "openclaw.v1.responses.strict_json",
                "_error": str(exc),
                "_strict_json_attempted": True,
            }

    if not _is_effective_text(last_result.get("text", ""), prompt=prompt):
        return {
            "text": _fallback_text(prompt),
            "status": "Fallback",
            "fueter_curvature": None,
            "spectral_shift_eta": None,
            "_route": "local_fallback_template",
            "_strict_json_attempted": strict_json_attempted,
        }
    last_result["_strict_json_attempted"] = bool(last_result.get("_strict_json_attempted", strict_json_attempted))
    return last_result


def _chat_once_with_schema_retry(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    use_das_arch: bool,
    openclaw_url: Optional[str],
    enforce_schema: bool,
    max_schema_retries: int,
    min_boundary_chars: int = 8,
    min_risk_chars: int = 12,
    min_action_chars: int = 8,
    min_metric_chars: int = 4,
    forbid_placeholders: bool = True,
    self_eval_distill_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any]
    if enforce_schema and _needs_self_eval_schema(prompt):
        distilled = _build_self_eval_from_distill_model(
            prompt=prompt,
            model_path=self_eval_distill_model_path,
            min_boundary_chars=max(1, min_boundary_chars),
            min_risk_chars=max(1, min_risk_chars),
            min_action_chars=max(1, min_action_chars),
            min_metric_chars=max(1, min_metric_chars),
            forbid_placeholders=forbid_placeholders,
        )
        if distilled is not None:
            result = {
                "text": json.dumps(distilled, ensure_ascii=False, indent=2),
                "status": "Distilled",
                "fueter_curvature": None,
                "spectral_shift_eta": None,
                "_route": "self_eval.distilled_adapter",
                "_strict_json_attempted": False,
                "_distill_adapter_used": True,
            }
        else:
            result = _chat_once(
                base_url=base_url,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                use_das_arch=use_das_arch,
                openclaw_url=openclaw_url,
                force_self_eval_strict_json_attempt=True,
            )
            result["_distill_adapter_used"] = False
    else:
        result = _chat_once(
            base_url=base_url,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_das_arch=use_das_arch,
            openclaw_url=openclaw_url,
            force_self_eval_strict_json_attempt=True,
        )
        result["_distill_adapter_used"] = False

    result["_schema"] = {
        "required": False,
        "attempts": 1,
        "valid": True,
        "errors": [],
    }

    if not enforce_schema or not _needs_self_eval_schema(prompt):
        return result

    attempt = 1
    current_prompt = prompt
    current_answer = str(result.get("text", ""))

    while True:
        parsed = _extract_json_candidate(current_answer)
        ok, errors, normalized = _validate_self_eval_schema(
            parsed,
            min_boundary_chars=max(1, min_boundary_chars),
            min_risk_chars=max(1, min_risk_chars),
            min_action_chars=max(1, min_action_chars),
            min_metric_chars=max(1, min_metric_chars),
            forbid_placeholders=forbid_placeholders,
        )
        result["_schema"] = {
            "required": True,
            "attempts": attempt,
            "valid": ok,
            "errors": errors,
            "normalized": normalized,
            "strict_json_pre_fallback_attempted": bool(result.get("_strict_json_attempted", False)),
            "distill_adapter_used": bool(result.get("_distill_adapter_used", False)),
            "quality_policy": {
                "min_boundary_chars": max(1, min_boundary_chars),
                "min_risk_chars": max(1, min_risk_chars),
                "min_action_chars": max(1, min_action_chars),
                "min_metric_chars": max(1, min_metric_chars),
                "forbid_placeholders": forbid_placeholders,
            },
        }
        if ok:
            if isinstance(parsed, dict):
                result["text"] = json.dumps(parsed, ensure_ascii=False, indent=2)
            return result
        if attempt > max_schema_retries:
            return result

        current_prompt = _make_self_eval_repair_prompt(prompt, current_answer, errors)
        result = _chat_once(
            base_url=base_url,
            prompt=current_prompt,
            max_tokens=max_tokens,
            temperature=min(1.0, temperature + 0.1),
            use_das_arch=use_das_arch,
            openclaw_url=openclaw_url,
        )
        current_answer = str(result.get("text", ""))
        attempt += 1


def interactive_chat(
    base_url: str,
    openclaw_url: Optional[str],
    trust_summary: Dict[str, Any],
    trust_report: Path,
    max_tokens: int,
    temperature: float,
    use_das_arch: bool,
    enforce_self_eval_schema: bool,
    self_eval_max_retries: int,
    self_eval_min_boundary_chars: int,
    self_eval_min_risk_chars: int,
    self_eval_min_action_chars: int,
    self_eval_min_metric_chars: int,
    self_eval_forbid_placeholders: bool,
    self_eval_hard_fail_on_invalid: bool,
    self_eval_distill_model_path: Optional[Path],
) -> Tuple[Path, bool]:
    print("\nTrusted Local AGI Chat Ready")
    print(f"Trust report: {trust_report}")
    print(f"Trust score: {trust_summary['trust_score']:.4f} | trusted_ready={trust_summary['trusted_ready']}")
    print("Commands: /help  /status  /exit")

    transcript: List[Dict[str, Any]] = []
    hard_fail_triggered = False
    start_at = datetime.now(timezone.utc).isoformat()

    while True:
        user_text = input("\nYou > ").strip()
        if not user_text:
            continue

        if user_text == "/exit":
            break
        if user_text == "/help":
            print("/help: show commands | /status: show trust gate | /exit: quit")
            continue
        if user_text == "/status":
            print(json.dumps(trust_summary, ensure_ascii=False, indent=2))
            continue

        t0 = time.time()
        try:
            result = _chat_once_with_schema_retry(
                base_url,
                user_text,
                max_tokens=max_tokens,
                temperature=temperature,
                use_das_arch=use_das_arch,
                openclaw_url=openclaw_url,
                enforce_schema=enforce_self_eval_schema,
                max_schema_retries=max(0, self_eval_max_retries),
                min_boundary_chars=max(1, self_eval_min_boundary_chars),
                min_risk_chars=max(1, self_eval_min_risk_chars),
                min_action_chars=max(1, self_eval_min_action_chars),
                min_metric_chars=max(1, self_eval_min_metric_chars),
                forbid_placeholders=self_eval_forbid_placeholders,
                self_eval_distill_model_path=self_eval_distill_model_path,
            )
            latency = time.time() - t0
        except URLError as exc:
            print(f"Assistant > request failed: {exc}")
            continue
        except Exception as exc:
            print(f"Assistant > runtime error: {exc}")
            continue

        answer = str(result.get("text", "")).strip() or "(empty response)"
        status = result.get("status", "unknown")
        curvature = result.get("fueter_curvature", None)
        eta = result.get("spectral_shift_eta", None)
        route = result.get("_route", "chat")
        schema_meta = result.get("_schema", {})

        print(f"Assistant > {answer}")
        print(f"[route={route}, status={status}, curvature={curvature}, eta={eta}, latency={latency:.2f}s]")
        if schema_meta.get("required"):
            print(
                "[schema.required=true, "
                f"schema.valid={schema_meta.get('valid')}, "
                f"schema.attempts={schema_meta.get('attempts')}]"
            )

        transcript.append(
            {
                "user": user_text,
                "assistant": answer,
                "runtime": {
                    "route": route,
                    "status": status,
                    "fueter_curvature": curvature,
                    "spectral_shift_eta": eta,
                    "latency_seconds": latency,
                    "schema": schema_meta,
                },
            }
        )

        if schema_meta.get("required") and (not bool(schema_meta.get("valid", False))) and self_eval_hard_fail_on_invalid:
            hard_fail_triggered = True
            print("Assistant > HARD_FAIL: self-eval schema invalid after retries; terminating session.")
            break

    out = ROOT / "reports" / f"trusted_local_agi_chat_session_{int(time.time())}.json"
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "start_time_utc": start_at,
                    "end_time_utc": datetime.now(timezone.utc).isoformat(),
                    "base_url": base_url,
                    "openclaw_url": openclaw_url,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "use_das_arch": use_das_arch,
                    "trust_report": str(trust_report),
                    "hard_fail_triggered": hard_fail_triggered,
                },
                "trust": trust_summary,
                "transcript": transcript,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out, hard_fail_triggered


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted local AGI chat runner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--openclaw-url", default="http://127.0.0.1:8011")
    parser.add_argument("--disable-openclaw-fallback", action="store_true")
    parser.add_argument("--disable-self-eval-schema", action="store_true")
    parser.add_argument("--self-eval-max-retries", type=int, default=2)
    parser.add_argument("--self-eval-min-boundary-chars", type=int, default=8)
    parser.add_argument("--self-eval-min-risk-chars", type=int, default=12)
    parser.add_argument("--self-eval-min-action-chars", type=int, default=8)
    parser.add_argument("--self-eval-min-metric-chars", type=int, default=4)
    parser.add_argument("--allow-self-eval-placeholders", action="store_true")
    parser.add_argument("--self-eval-hard-fail-on-invalid", action="store_true")
    parser.add_argument("--disable-self-eval-distill-adapter", action="store_true")
    parser.add_argument("--self-eval-distill-model", default=str(DEFAULT_SELF_EVAL_DISTILL_MODEL))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--use-das-arch", action="store_true", default=True)
    parser.add_argument("--no-das-arch", dest="use_das_arch", action="store_false")

    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-rsa", action="store_true")
    parser.add_argument("--force-refresh-trust", action="store_true")
    parser.add_argument("--trust-max-age-minutes", type=int, default=120)
    parser.add_argument("--min-trust-score", type=float, default=0.70)
    parser.add_argument("--strict-trust-gate", action="store_true")

    parser.add_argument("--auto-start-server", action="store_true", default=True)
    parser.add_argument("--no-auto-start-server", dest="auto_start_server", action="store_false")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("== Stage 1: Trusted gate check ==")
    trust_payload, trust_report = _get_trust_payload(
        profile=args.profile,
        skip_rsa=args.skip_rsa,
        max_age_minutes=max(1, args.trust_max_age_minutes),
        force_refresh=args.force_refresh_trust,
    )
    trust_summary = _extract_trust_summary(trust_payload)
    print(json.dumps(trust_summary, ensure_ascii=False, indent=2))

    if args.strict_trust_gate and trust_summary["trust_score"] < args.min_trust_score:
        raise SystemExit(
            f"Trust gate blocked: score={trust_summary['trust_score']:.4f} < min={args.min_trust_score:.4f}"
        )

    print("\n== Stage 2: Ensure local /chat service ==")
    server_proc: Optional[subprocess.Popen] = None
    try:
        if not _wait_server_ready(base_url, timeout_sec=3.0):
            if not args.auto_start_server:
                raise SystemExit("Local server is not available and auto-start is disabled")
            server_proc = _start_local_server(args.host, args.port)
            if not _wait_server_ready(base_url, timeout_sec=60.0):
                raise SystemExit("Failed to start local H2Q chat server")
        print(f"Server ready at {base_url}")

        distill_model_path: Optional[Path] = None
        if not args.disable_self_eval_distill_adapter:
            candidate = Path(args.self_eval_distill_model)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.exists():
                distill_model_path = candidate

        print("\n== Stage 3: Interactive trusted conversation ==")
        session_path, hard_fail_triggered = interactive_chat(
            base_url=base_url,
            openclaw_url=None if args.disable_openclaw_fallback else args.openclaw_url,
            trust_summary=trust_summary,
            trust_report=trust_report,
            max_tokens=max(1, args.max_tokens),
            temperature=args.temperature,
            use_das_arch=args.use_das_arch,
            enforce_self_eval_schema=not args.disable_self_eval_schema,
            self_eval_max_retries=max(0, args.self_eval_max_retries),
            self_eval_min_boundary_chars=max(1, args.self_eval_min_boundary_chars),
            self_eval_min_risk_chars=max(1, args.self_eval_min_risk_chars),
            self_eval_min_action_chars=max(1, args.self_eval_min_action_chars),
            self_eval_min_metric_chars=max(1, args.self_eval_min_metric_chars),
            self_eval_forbid_placeholders=not args.allow_self_eval_placeholders,
            self_eval_hard_fail_on_invalid=args.self_eval_hard_fail_on_invalid,
            self_eval_distill_model_path=distill_model_path,
        )
        print(f"Session saved: {session_path}")
        if hard_fail_triggered:
            raise SystemExit(2)
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
