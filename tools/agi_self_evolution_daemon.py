#!/usr/bin/env python3
"""Continuous local AGI self-evolution daemon.

This daemon runs self-evolution rounds periodically:
1) Optional trust gate refresh/reuse.
2) Ensure local AGI service is online.
3) Execute multi-step self-evolution prompts with adaptive inference route.
4) Persist JSON + markdown reports and optional alert files.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.memory_manager import append_with_limit, json_write_round_payload
from tools.trusted_joint_agi_quantum_center import run_center
from tools.trusted_local_agi_chat import _http_json, _start_local_server, _wait_server_ready


EVOLUTION_PROMPTS = [
    "你是本地AGI，请按科研线输出：1) 可证伪假设 2) 实验设计 3) 数据与统计判据 4) 停止条件。",
    "你是本地AGI，请按工程线输出严格JSON：{\"milestones\":[],\"actions\":[],\"metrics\":[],\"risks\":[],\"rollback\":[],\"next_checkpoint\":\"\"}",
    "你是本地AGI，请按产品线输出：用户价值、交付节奏、验收标准、失败告警策略与下一轮路线图。",
]

EXTENDED_EVOLUTION_PROMPTS = [
    "请输出一个可执行的模型对齐实验：包含对照组、指标、最小样本量与显著性阈值。",
    "请输出服务稳定性改进计划：包含SLO、错误预算、熔断策略与回滚触发器。",
    "请给出编程能力进化任务：实现一个函数+pytest用例+边界条件清单。",
    "请给出安全线目标：提示注入防护、敏感信息处理与审计日志方案。",
    "请给出数据线目标：采样策略、数据质量评分、异常样本处理闭环。",
    "请给出评测线目标：构建可复现实验并输出统一评分表结构(JSON)。",
    "请给出工具链线目标：自动化脚本、CI门禁、失败告警与重试策略。",
    "请给出成本优化目标：token预算、推理时延、吞吐与单位成本控制。",
    "请给出长期记忆目标：知识沉淀结构、索引策略、失真检测与修复。",
]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _read_deepseek_api_key(key_file: str) -> str:
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    p = Path(key_file)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return ""


def _deepseek_stream_chat(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 120.0,
    max_response_chars: int = 32768,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是AGI进化辅助规划器。输出必须可执行、可验证、可追踪。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    collected: List[str] = []
    collected_chars = 0
    truncated = False
    usage: Dict[str, Any] = {}
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta = (
                obj.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )
            if delta:
                if max_response_chars <= 0:
                    truncated = True
                    continue
                remain = max_response_chars - collected_chars
                if remain <= 0:
                    truncated = True
                    continue
                if len(delta) > remain:
                    collected.append(delta[:remain])
                    collected_chars += remain
                    truncated = True
                    continue
                collected.append(delta)
                collected_chars += len(delta)

            if "usage" in obj and isinstance(obj["usage"], dict):
                usage = obj["usage"]

    text = "".join(collected).strip()
    prompt_tokens = int(usage.get("prompt_tokens", _estimate_tokens(prompt)))
    completion_tokens = int(usage.get("completion_tokens", _estimate_tokens(text)))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

    return {
        "ok": bool(text),
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model": model,
        "truncated": truncated,
    }


def _external_assist(
    prompt: str,
    assist_cfg: Dict[str, Any],
    traffic_state: Dict[str, Any],
) -> Dict[str, Any]:
    if assist_cfg.get("provider", "none") != "deepseek":
        return {"enabled": False, "ok": False, "reason": "provider-disabled"}

    if traffic_state["calls_this_round"] >= assist_cfg["max_calls_per_round"]:
        return {"enabled": True, "ok": False, "reason": "round-call-budget-exceeded"}
    if traffic_state["tokens_total"] >= assist_cfg["max_tokens_total"]:
        return {"enabled": True, "ok": False, "reason": "total-token-budget-exceeded"}
    if traffic_state["tokens_this_round"] >= assist_cfg["max_tokens_per_round"]:
        return {"enabled": True, "ok": False, "reason": "round-token-budget-exceeded"}

    api_key = _read_deepseek_api_key(assist_cfg["key_file"])
    if not api_key:
        return {"enabled": True, "ok": False, "reason": "missing-api-key"}

    retries = max(0, int(assist_cfg.get("retries", 2)))
    backoff = max(0.0, float(assist_cfg.get("retry_backoff_seconds", 1.5)))
    backoff_max = max(backoff, float(assist_cfg.get("retry_backoff_max_seconds", 8.0)))
    last_err_reason = "assist-unknown-error"
    for attempt in range(retries + 1):
        if attempt > 0 and traffic_state["calls_this_round"] >= assist_cfg["max_calls_per_round"]:
            return {
                "enabled": True,
                "ok": False,
                "reason": "round-call-budget-exceeded-after-retry",
                "retry_count": attempt,
                "model": assist_cfg["model"],
                "total_tokens": 0,
            }
        traffic_state["calls_total"] += 1
        traffic_state["calls_this_round"] += 1
        try:
            result = _deepseek_stream_chat(
                api_key=api_key,
                base_url=assist_cfg["base_url"],
                model=assist_cfg["model"],
                prompt=prompt,
                temperature=assist_cfg["temperature"],
                max_tokens=assist_cfg["max_tokens"],
                max_response_chars=assist_cfg.get("max_response_chars", 32768),
            )
            if not result.get("ok", False):
                last_err_reason = "assist-empty-response"
                if attempt < retries:
                    time.sleep(min(backoff_max, backoff * (2 ** attempt)))
                    continue
                return {
                    "enabled": True,
                    "ok": False,
                    "reason": last_err_reason,
                    "retry_count": attempt,
                    "model": assist_cfg["model"],
                    "total_tokens": int(result.get("total_tokens", 0) or 0),
                }
            break
        except requests.Timeout:
            last_err_reason = "assist-timeout"
        except requests.HTTPError as exc:
            code = int(getattr(getattr(exc, "response", None), "status_code", 0) or 0)
            if code == 429:
                last_err_reason = "assist-http-429-rate-limit"
            elif 500 <= code <= 599:
                last_err_reason = f"assist-http-{code}-server"
            elif code >= 400:
                last_err_reason = f"assist-http-{code}-client"
            else:
                last_err_reason = "assist-http-error"
        except requests.RequestException:
            last_err_reason = "assist-network-error"
        except Exception:
            last_err_reason = "assist-internal-error"

        if attempt < retries:
            time.sleep(min(backoff_max, backoff * (2 ** attempt)))
            continue
        return {
            "enabled": True,
            "ok": False,
            "reason": last_err_reason,
            "retry_count": attempt,
            "model": assist_cfg["model"],
            "total_tokens": 0,
        }

    traffic_state["tokens_total"] += int(result.get("total_tokens", 0))
    traffic_state["tokens_this_round"] += int(result.get("total_tokens", 0))

    return {"enabled": True, "retry_count": attempt, **result}


def _safe_json_list(raw: str, limit: int) -> List[str]:
    text = (raw or "").strip()
    if not text:
        return []

    # Strip markdown fences often returned by chat models.
    text = text.replace("```json", "").replace("```", "").strip()
    if "[" in text and "]" in text:
        s = text.find("[")
        e = text.rfind("]")
        if s >= 0 and e > s:
            text = text[s : e + 1]

    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip()]
            out = [x for x in out if len(x) >= 10]
            return out[:limit]
        if isinstance(data, dict) and isinstance(data.get("goals"), list):
            out = [str(x).strip() for x in data["goals"] if str(x).strip()]
            out = [x for x in out if len(x) >= 10]
            return out[:limit]
    except Exception:
        pass

    lines = [ln.strip("- \t\",") for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if ln in {"[", "]", "{" , "}"}:
            continue
        if re.fullmatch(r"\d+\.?", ln):
            continue
        if len(ln) < 10:
            continue
        cleaned.append(ln)
    return cleaned[:limit]


def _synthesize_external_goals(assist_cfg: Dict[str, Any], count: int) -> List[str]:
    if assist_cfg.get("provider", "none") != "deepseek" or count <= 0:
        return []

    api_key = _read_deepseek_api_key(assist_cfg["key_file"])
    if not api_key:
        return []

    prompt = (
        "你是AGI进化架构师。请设计高价值的无人值守进化目标，覆盖科研/工程/产品/安全/评测。"
        "仅输出JSON数组，每项是可执行的一句话任务，不要解释，不要markdown代码块，不要编号。"
    )
    try:
        res = _deepseek_stream_chat(
            api_key=api_key,
            base_url=assist_cfg["base_url"],
            model=assist_cfg["model"],
            prompt=prompt,
            temperature=min(0.8, max(0.1, assist_cfg["temperature"])),
            max_tokens=min(assist_cfg["max_tokens"], 512),
        )
        if not res.get("ok", False):
            return []
        return _safe_json_list(str(res.get("text", "")), limit=count)
    except Exception:
        return []


def _build_round_prompts(round_id: int, goal_mode: str, external_goals: List[str], external_goal_limit: int) -> List[str]:
    prompts = list(EVOLUTION_PROMPTS)
    if goal_mode == "extended":
        prompts.extend(EXTENDED_EVOLUTION_PROMPTS)

    if external_goals and external_goal_limit > 0:
        limit = min(len(external_goals), external_goal_limit)
        for i in range(limit):
            prompts.append(f"[外部扩展目标#{i+1}] {external_goals[i]}")

    # 每轮固定加入一条循环控制任务，促使系统进行自诊断与闭环。
    prompts.append(
        f"[第{round_id}轮控制任务] 输出下一轮自适应控制建议(JSON): {{\"temperature\":0-1,\"max_tokens\":64-1024,\"focus\":\"...\",\"risk\":\"...\"}}"
    )
    return prompts


def _token_set(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text or ""))


def _looks_placeholder_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) <= 6:
        return True
    if re.fullmatch(r"\[?#?\d+\]?\s*(AI|agi)?", t, flags=re.IGNORECASE):
        return True
    return False


def _analyze_round_failures(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    failed = [e for e in entries if not e.get("nonempty", False)]
    route_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    assist_reason_counts: Dict[str, int] = {}
    placeholder_count = 0

    for e in failed:
        runtime = e.get("runtime", {})
        route = str(runtime.get("route", "unknown"))
        status = str(runtime.get("status", "unknown"))
        route_counts[route] = route_counts.get(route, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

        assist = runtime.get("assist", {})
        reason = str(assist.get("reason", "") or "none")
        assist_reason_counts[reason] = assist_reason_counts.get(reason, 0) + 1

        if _looks_placeholder_text(str(e.get("assistant", ""))):
            placeholder_count += 1

    return {
        "failed_entry_count": len(failed),
        "route_counts": route_counts,
        "status_counts": status_counts,
        "assist_reason_counts": assist_reason_counts,
        "placeholder_like_count": placeholder_count,
    }


def _round_success_from_acceptance(
    entries: List[Dict[str, Any]],
    core_prompt_count: int,
    min_overall_success_ratio: float,
    min_core_success_ratio: float,
) -> Dict[str, Any]:
    total = max(1, len(entries))
    nonempty_total = sum(1 for e in entries if e.get("nonempty", False))
    overall_ratio = float(nonempty_total) / float(total)

    core_n = max(1, min(core_prompt_count, len(entries)))
    core_nonempty = sum(1 for e in entries[:core_n] if e.get("nonempty", False))
    core_ratio = float(core_nonempty) / float(core_n)

    success = overall_ratio >= min_overall_success_ratio and core_ratio >= min_core_success_ratio
    return {
        "success": bool(success),
        "overall_ratio": overall_ratio,
        "core_ratio": core_ratio,
        "nonempty_total": nonempty_total,
        "core_nonempty": core_nonempty,
        "core_prompt_count": core_n,
        "min_overall_success_ratio": min_overall_success_ratio,
        "min_core_success_ratio": min_core_success_ratio,
    }


def _docker_consistency_check(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    docker_image: str,
    min_overlap: float,
) -> Dict[str, Any]:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return {"executed": False, "ok": False, "reason": "docker-not-found"}

    local = adaptive_infer(
        base_url=base_url,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    local_text = str(local.get("text", "")).strip()

    run_candidates = [
        ("/app/h2q_project", "h2q/core/brain.py"),
        ("/app/h2q_project", "h2q_project/h2q/core/brain.py"),
        ("/app/h2q_project/h2q_project", "h2q/core/brain.py"),
    ]
    docker_text = ""
    last_stderr = ""
    used_script = ""
    used_workdir = ""
    for workdir, script_path in run_candidates:
        cmd = [
            docker_bin,
            "run",
            "--rm",
            "-e",
            "PYTHONPATH=/app/h2q_project:/app/h2q_project/h2q_project",
            "-v",
            f"{ROOT}:/app/h2q_project",
            "-w",
            workdir,
            docker_image,
            "python3",
            script_path,
            "--prompt",
            prompt,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return {"executed": True, "ok": False, "reason": "docker-timeout", "image": docker_image}

        if proc.returncode == 0:
            docker_text = (proc.stdout or "").strip()
            used_script = script_path
            used_workdir = workdir
            break
        last_stderr = (proc.stderr or "")[-400:]

    if not docker_text:
        return {
            "executed": True,
            "ok": False,
            "reason": "docker-run-failed",
            "image": docker_image,
            "stderr": last_stderr,
            "tried_runs": [{"workdir": wd, "script": sp} for wd, sp in run_candidates],
        }
    local_ok = _is_effective_text(local_text, prompt=prompt)
    docker_ok = _is_effective_text(docker_text, prompt=prompt)

    a = _token_set(local_text)
    b = _token_set(docker_text)
    overlap = float(len(a & b)) / float(max(1, min(len(a), len(b))))

    ok = local_ok and docker_ok and overlap >= min_overlap
    reason = "ok" if ok else "low-overlap-or-invalid-output"
    return {
        "executed": True,
        "ok": ok,
        "reason": reason,
        "image": docker_image,
        "docker_workdir": used_workdir,
        "docker_script": used_script,
        "local_route": str(local.get("_route", "unknown")),
        "local_ok": local_ok,
        "docker_ok": docker_ok,
        "overlap": overlap,
        "min_overlap": min_overlap,
        "local_preview": local_text[:200],
        "docker_preview": docker_text[:200],
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_effective_text(value: Any, prompt: str = "") -> bool:
    text = str(value or "").strip()
    if not text or text == "(empty response)":
        return False
    if _looks_placeholder_text(text):
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
    lower = prompt.lower()
    if any(k in lower for k in ["python", "pytest", "函数", "脚本", "编程"]):
        return (
            "```python\n"
            "def agi_task_plan(task: str) -> dict:\n"
            "    steps = [\"定义验收标准\", \"实现最小可运行版本\", \"补齐pytest回归\", \"记录失败并迭代\"]\n"
            "    return {\"task\": task, \"steps\": steps, \"metrics\": {\"pass_rate\": 0.0, \"latency_ms\": 0.0}}\n"
            "\n"
            "def test_agi_task_plan_basic():\n"
            "    out = agi_task_plan(\"evolve\")\n"
            "    assert \"steps\" in out and len(out[\"steps\"]) >= 3\n"
            "```"
        )
    if "json" in lower:
        return (
            '{"milestones":["科研线基线验证","工程线回归稳定","产品线日报上线"],'
            '"actions":["运行验证脚本","执行回归测试","生成日报与告警"],'
            '"metrics":["pass_rate","empty_reply_rate","latency_ms"],'
            '"risks":["空响应","回归失败"],'
            '"rollback":["切换fallback路由","降级到稳定分支"],'
            '"next_checkpoint":"T+1轮"}'
        )
    return (
        "进入本地应急进化输出模式：\n"
        "- 科研线：先验证假设并记录统计显著性。\n"
        "- 工程线：先保可用性，再做性能优化。\n"
        "- 产品线：以用户可见价值和稳定交付为主。"
    )


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


def _latest_center_report(report_dir: Path) -> Optional[Path]:
    candidates = sorted(report_dir.glob("trusted_joint_agi_quantum_center_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_trust_payload(profile: str, skip_rsa: bool, max_age_minutes: int, force_refresh: bool, skip_trust_check: bool) -> Tuple[Dict[str, Any], Optional[Path]]:
    if skip_trust_check:
        return {
            "aggregate": {
                "trust_score": 0.0,
                "trusted_ready": False,
                "gates": {
                    "das_decision_ready": False,
                    "dual_aligned_consistent": False,
                    "codec_integrity": False,
                    "rsa_parallel_observed": False,
                },
                "note": "Trust check skipped by configuration.",
            }
        }, None

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    latest = _latest_center_report(report_dir)
    if latest is not None and not force_refresh:
        age_min = (time.time() - latest.stat().st_mtime) / 60.0
        if age_min <= max(1, max_age_minutes):
            return _load_json(latest), latest

    out_json, _ = run_center(profile=profile, include_rsa=not skip_rsa, rsa_folds=2)
    return _load_json(out_json), out_json


def _post_chat(base_url: str, prompt: str, max_tokens: int, temperature: float, use_das_arch: bool) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_das_arch": use_das_arch,
    }
    res = _http_json(f"{base_url}/chat", payload=payload, timeout=120.0)
    res["_route"] = f"chat(use_das_arch={use_das_arch})"
    return res


def _post_generate(base_url: str, prompt: str, max_tokens: int, temperature: float, use_das_arch: bool) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "use_das_arch": use_das_arch,
    }
    res = _http_json(f"{base_url}/generate", payload=payload, timeout=120.0)
    res["_route"] = f"generate(use_das_arch={use_das_arch})"
    return res


def adaptive_infer(base_url: str, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    attempts = [
        ("chat", False, max_tokens, temperature),
        ("chat", True, max_tokens, min(1.0, temperature + 0.1)),
        ("generate", False, max_tokens, temperature),
        ("generate", True, max_tokens, min(1.0, temperature + 0.1)),
    ]

    last: Dict[str, Any] = {}
    for route, use_das, mx, temp in attempts:
        try:
            if route == "chat":
                result = _post_chat(base_url, prompt, mx, temp, use_das)
            else:
                result = _post_generate(base_url, prompt, mx, temp, use_das)
            last = result
            if _is_effective_text(result.get("text", ""), prompt=prompt):
                return result
        except Exception as exc:
            last = {
                "text": "",
                "status": "route-error",
                "_route": f"{route}(use_das_arch={use_das})",
                "_error": str(exc),
                "fueter_curvature": None,
                "spectral_shift_eta": None,
            }
    if not _is_effective_text(last.get("text", ""), prompt=prompt):
        return {
            "text": _fallback_text(prompt),
            "status": "Fallback",
            "_route": "local_fallback_template",
            "fueter_curvature": None,
            "spectral_shift_eta": None,
            "_error": "",
        }
    return last


def run_round(
    base_url: str,
    round_id: int,
    max_tokens: int,
    temperature: float,
    assist_cfg: Dict[str, Any],
    traffic_state: Dict[str, Any],
    goal_mode: str,
    external_goals: List[str],
    external_goal_limit: int,
    min_overall_success_ratio: float,
    min_core_success_ratio: float,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    nonempty = 0
    traffic_state["calls_this_round"] = 0
    traffic_state["tokens_this_round"] = 0

    round_prompts = _build_round_prompts(
        round_id=round_id,
        goal_mode=goal_mode,
        external_goals=external_goals,
        external_goal_limit=external_goal_limit,
    )

    for prompt in round_prompts:
        assist = _external_assist(prompt, assist_cfg=assist_cfg, traffic_state=traffic_state)
        effective_prompt = prompt
        if assist.get("ok", False):
            effective_prompt = (
                prompt
                + "\n\n外部LLM辅助建议(DeepSeek):\n"
                + str(assist.get("text", ""))
                + "\n\n请结合建议输出最终可执行结果。"
            )

        t0 = time.time()
        result = adaptive_infer(base_url=base_url, prompt=effective_prompt, max_tokens=max_tokens, temperature=temperature)
        latency = time.time() - t0
        answer = str(result.get("text", "")).strip()

        # If local route still fails, optionally fall back to DeepSeek answer.
        if not _is_effective_text(answer, prompt=prompt) and assist.get("ok", False) and assist_cfg.get("fallback_on_local_failure", True):
            answer = str(assist.get("text", "")).strip()
            result = {
                "_route": "deepseek_assist_fallback",
                "status": "ExternalAssist",
                "fueter_curvature": None,
                "spectral_shift_eta": None,
                "_error": "",
            }

        ok = _is_effective_text(answer, prompt=prompt)
        nonempty += 1 if ok else 0

        entries.append(
            {
                "prompt": prompt,
                "assistant": answer,
                "nonempty": ok,
                "runtime": {
                    "route": result.get("_route", "unknown"),
                    "status": result.get("status", "unknown"),
                    "fueter_curvature": result.get("fueter_curvature", None),
                    "spectral_shift_eta": result.get("spectral_shift_eta", None),
                    "latency_seconds": latency,
                    "error": result.get("_error", ""),
                    "assist": {
                        "enabled": bool(assist.get("enabled", False)),
                        "ok": bool(assist.get("ok", False)),
                        "reason": assist.get("reason", ""),
                        "model": assist.get("model", ""),
                        "tokens": int(assist.get("total_tokens", 0)) if assist.get("enabled", False) else 0,
                    },
                },
            }
        )

    acceptance = _round_success_from_acceptance(
        entries=entries,
        core_prompt_count=len(EVOLUTION_PROMPTS),
        min_overall_success_ratio=min_overall_success_ratio,
        min_core_success_ratio=min_core_success_ratio,
    )
    failure_analysis = _analyze_round_failures(entries)

    return {
        "round_id": round_id,
        "timestamp_utc": now_utc(),
        "entries": entries,
        "prompt_count": len(round_prompts),
        "nonempty_count": nonempty,
        "success": bool(acceptance["success"]),
        "acceptance": acceptance,
        "failure_analysis": failure_analysis,
    }


def _adapt_inference_params(
    round_payload: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> Tuple[float, int, Dict[str, Any]]:
    prompt_count = max(1, int(round_payload.get("prompt_count", len(round_payload.get("entries", [])))))
    nonempty = int(round_payload.get("nonempty_count", 0))
    ratio = float(nonempty) / float(prompt_count)

    next_temp = temperature
    next_tokens = max_tokens
    action = "hold"

    if ratio < 0.75:
        next_temp = min(1.0, temperature + 0.08)
        next_tokens = min(1024, max_tokens + 64)
        action = "recover-quality"
    elif ratio > 0.95:
        next_temp = max(0.2, temperature - 0.03)
        next_tokens = max(96, max_tokens - 16)
        action = "stabilize-cost"

    detail = {
        "action": action,
        "ratio": ratio,
        "prev": {"temperature": temperature, "max_tokens": max_tokens},
        "next": {"temperature": next_temp, "max_tokens": next_tokens},
    }
    return next_temp, next_tokens, detail


def write_round_report(round_payload: Dict[str, Any], trust_summary: Dict[str, Any], trust_report: Optional[Path]) -> Path:
    out = ROOT / "reports" / f"agi_self_evolution_round_{int(time.time())}.json"
    json_write_round_payload(
        out_path=out,
        round_payload=round_payload,
        trust_summary=trust_summary,
        trust_report=trust_report,
    )
    return out


def write_daily_report(
    all_rounds: List[Dict[str, Any]],
    trust_summary: Dict[str, Any],
    trust_report: Optional[Path],
    alert_files: List[Path],
    aggregate: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    ts = int(time.time())
    out_json = ROOT / "reports" / f"agi_self_evolution_daily_{ts}.json"
    out_md = ROOT / "reports" / f"AGI自我进化日报_{ts}.md"

    if aggregate is None:
        success_rounds = sum(1 for r in all_rounds if r.get("success", False))
        failed_rounds = len(all_rounds) - success_rounds
        total_rounds = len(all_rounds)
        assist_total_calls = 0
        assist_success_calls = 0
        assist_total_tokens = 0
        assist_reasons: Dict[str, int] = {}
        for r in all_rounds:
            for e in r.get("entries", []):
                assist = e.get("runtime", {}).get("assist", {})
                if not assist.get("enabled", False):
                    continue
                assist_total_calls += 1
                if assist.get("ok", False):
                    assist_success_calls += 1
                assist_total_tokens += int(assist.get("tokens", 0) or 0)
                reason = str(assist.get("reason", "") or "ok" if assist.get("ok", False) else assist.get("reason", "unknown"))
                assist_reasons[reason] = assist_reasons.get(reason, 0) + 1
    else:
        total_rounds = int(aggregate.get("total_rounds", len(all_rounds)))
        success_rounds = int(aggregate.get("success_rounds", 0))
        failed_rounds = int(aggregate.get("failed_rounds", 0))
        assist_total_calls = int(aggregate.get("assist_total_calls", 0))
        assist_success_calls = int(aggregate.get("assist_success_calls", 0))
        assist_total_tokens = int(aggregate.get("assist_total_tokens", 0))
        assist_reasons = dict(aggregate.get("assist_reasons", {}))

    assist_success_rate = (
        float(assist_success_calls) / float(assist_total_calls)
        if assist_total_calls > 0
        else 0.0
    )

    payload = {
        "meta": {
            "created_at_utc": now_utc(),
            "round_count": total_rounds,
            "success_rounds": success_rounds,
            "failed_rounds": failed_rounds,
            "retained_rounds_in_memory": len(all_rounds),
            "trust_report": str(trust_report) if trust_report else "",
        },
        "trust": trust_summary,
        "assist_summary": {
            "enabled_calls": assist_total_calls,
            "success_calls": assist_success_calls,
            "success_rate": assist_success_rate,
            "total_tokens": assist_total_tokens,
            "reasons": assist_reasons,
        },
        "rounds": all_rounds,
        "alerts": [str(p) for p in alert_files],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# AGI Local Self-Evolution Daily Report",
        "",
        f"- Created at (UTC): `{payload['meta']['created_at_utc']}`",
        f"- Trust score: `{trust_summary.get('trust_score', 0.0):.4f}`",
        f"- Trusted ready: `{trust_summary.get('trusted_ready', False)}`",
        f"- Rounds: `{total_rounds}` | Success: `{success_rounds}` | Failed: `{failed_rounds}`",
        (
            "- Assist: "
            f"enabled=`{assist_total_calls}` "
            f"success=`{assist_success_calls}` "
            f"success_rate=`{assist_success_rate:.2%}` "
            f"tokens=`{assist_total_tokens}`"
        ),
        f"- Trust report: `{payload['meta']['trust_report']}`",
        "",
        "## Round Summary",
        "",
        "| Round | Success | Non-empty Replies |",
        "|---|---|---:|",
    ]

    for r in all_rounds:
        lines.append(
            f"| {r['round_id']} | {r['success']} | {r['nonempty_count']}/{len(r['entries'])} |"
        )

    lines.extend(["", "## Three-Track Snapshot", ""])
    if all_rounds:
        latest = all_rounds[-1]
        for idx, track in enumerate(["科研线", "工程线", "产品线"]):
            entry = latest["entries"][idx] if idx < len(latest["entries"]) else {}
            lines.append(f"- {track}: route=`{entry.get('runtime', {}).get('route', 'n/a')}` success=`{entry.get('nonempty', False)}`")

    if alert_files:
        lines.extend(["", "## Alerts", ""])
        for p in alert_files:
            lines.append(f"- `{p}`")

    if assist_reasons:
        lines.extend(["", "## Assist Reasons", ""])
        for reason, count in sorted(assist_reasons.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{reason}`: {count}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def write_alert(round_payload: Dict[str, Any]) -> Optional[Path]:
    if round_payload.get("success", False):
        return None

    failed = [e for e in round_payload.get("entries", []) if not e.get("nonempty", False)]
    out = ROOT / "reports" / f"agi_self_evolution_alert_{int(time.time())}.json"
    out.write_text(
        json.dumps(
            {
                "created_at_utc": now_utc(),
                "reason": "empty-or-invalid-response",
                "round_id": round_payload.get("round_id"),
                "failed_entries": failed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous local AGI self-evolution daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval-minutes", type=float, default=10.0)
    parser.add_argument("--rounds", type=int, default=0, help="0 means run forever")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-rsa", action="store_true")
    parser.add_argument("--skip-trust-check", action="store_true")
    parser.add_argument("--force-refresh-trust", action="store_true")
    parser.add_argument("--trust-max-age-minutes", type=int, default=120)

    parser.add_argument("--auto-start-server", action="store_true", default=True)
    parser.add_argument("--no-auto-start-server", dest="auto_start_server", action="store_false")
    parser.add_argument("--fail-on-empty", action="store_true")
    parser.add_argument("--goal-mode", choices=["basic", "extended"], default="extended")
    parser.add_argument("--external-goal-limit", type=int, default=4)
    parser.add_argument("--external-goal-refresh-rounds", type=int, default=6)
    parser.add_argument("--basic-lock-rounds", type=int, default=48, help="When goal-mode=basic, keep basic mode for at least this many rounds before allowing auto-extend")
    parser.add_argument("--min-overall-success-ratio", type=float, default=0.85)
    parser.add_argument("--min-core-success-ratio", type=float, default=1.0)

    # External LLM assist (DeepSeek) options.
    parser.add_argument("--assist-provider", choices=["none", "deepseek"], default="deepseek")
    parser.add_argument("--assist-model", default="deepseek-chat")
    parser.add_argument("--assist-base-url", default="https://api.deepseek.com")
    parser.add_argument("--assist-key-file", default="secrets/deepseek_api_key.txt")
    parser.add_argument("--assist-temperature", type=float, default=0.3)
    parser.add_argument("--assist-max-tokens", type=int, default=512)
    parser.add_argument("--assist-max-calls-per-round", type=int, default=3)
    parser.add_argument("--assist-max-est-tokens-total", type=int, default=120000)
    parser.add_argument("--assist-max-est-tokens-per-round", type=int, default=12000)
    parser.add_argument("--assist-max-calls-cap", type=int, default=16)
    parser.add_argument("--assist-retries", type=int, default=2)
    parser.add_argument("--assist-retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--assist-retry-backoff-max-seconds", type=float, default=8.0)
    parser.add_argument("--assist-max-response-chars", type=int, default=32768)
    parser.add_argument("--no-assist-fallback", dest="assist_fallback", action="store_false")

    parser.add_argument("--enable-docker-consistency-check", action="store_true")
    parser.add_argument("--docker-check-interval-rounds", type=int, default=6)
    parser.add_argument("--docker-image", default=os.getenv("DOCKER_IMAGE", "h2q-sandbox"))
    parser.add_argument("--docker-min-overlap", type=float, default=0.05)
    parser.add_argument("--in-memory-round-window", type=int, default=100)
    parser.add_argument("--in-memory-alert-window", type=int, default=256)
    parser.set_defaults(assist_fallback=True)

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    assist_cfg = {
        "provider": args.assist_provider,
        "model": args.assist_model,
        "base_url": args.assist_base_url,
        "key_file": args.assist_key_file,
        "temperature": max(0.0, min(1.0, args.assist_temperature)),
        "max_tokens": max(16, args.assist_max_tokens),
        "max_calls_per_round": max(0, args.assist_max_calls_per_round),
        "max_tokens_total": max(0, args.assist_max_est_tokens_total),
        "max_tokens_per_round": max(0, args.assist_max_est_tokens_per_round),
        "fallback_on_local_failure": bool(args.assist_fallback),
        "retries": max(0, args.assist_retries),
        "retry_backoff_seconds": max(0.0, args.assist_retry_backoff_seconds),
        "retry_backoff_max_seconds": max(0.0, args.assist_retry_backoff_max_seconds),
        "max_response_chars": max(256, args.assist_max_response_chars),
    }
    traffic_state: Dict[str, Any] = {
        "calls_total": 0,
        "calls_this_round": 0,
        "tokens_total": 0,
        "tokens_this_round": 0,
    }

    external_goals = _synthesize_external_goals(assist_cfg=assist_cfg, count=max(0, args.external_goal_limit))
    current_temperature = max(0.0, min(1.0, args.temperature))
    current_max_tokens = max(1, args.max_tokens)
    current_goal_mode = str(args.goal_mode)
    consecutive_fail_rounds = 0
    consecutive_success_rounds = 0

    trust_payload, trust_path = get_trust_payload(
        profile=args.profile,
        skip_rsa=args.skip_rsa,
        max_age_minutes=max(1, args.trust_max_age_minutes),
        force_refresh=args.force_refresh_trust,
        skip_trust_check=args.skip_trust_check,
    )
    trust_summary = _extract_trust_summary(trust_payload)

    server_proc: Optional[subprocess.Popen] = None
    all_rounds: List[Dict[str, Any]] = []
    alert_files: List[Path] = []
    round_window = max(1, int(args.in_memory_round_window))
    alert_window = max(1, int(args.in_memory_alert_window))
    aggregate_stats: Dict[str, Any] = {
        "total_rounds": 0,
        "success_rounds": 0,
        "failed_rounds": 0,
        "assist_total_calls": 0,
        "assist_success_calls": 0,
        "assist_total_tokens": 0,
        "assist_reasons": {},
    }

    try:
        if not _wait_server_ready(base_url, timeout_sec=3.0):
            if not args.auto_start_server:
                raise SystemExit("Local AGI server is not available and auto-start is disabled")
            server_proc = _start_local_server(args.host, args.port)
            if not _wait_server_ready(base_url, timeout_sec=60.0):
                raise SystemExit("Failed to start local AGI server")

        round_id = 0
        while True:
            round_id += 1

            refresh_every = max(0, args.external_goal_refresh_rounds)
            if refresh_every > 0 and (round_id == 1 or (round_id - 1) % refresh_every == 0):
                external_goals = _synthesize_external_goals(
                    assist_cfg=assist_cfg,
                    count=max(0, args.external_goal_limit),
                )

            round_payload = run_round(
                base_url=base_url,
                round_id=round_id,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                assist_cfg=assist_cfg,
                traffic_state=traffic_state,
                goal_mode=current_goal_mode,
                external_goals=external_goals,
                external_goal_limit=max(0, args.external_goal_limit),
                min_overall_success_ratio=max(0.5, min(1.0, args.min_overall_success_ratio)),
                min_core_success_ratio=max(0.5, min(1.0, args.min_core_success_ratio)),
            )

            next_temp, next_tokens, control = _adapt_inference_params(
                round_payload=round_payload,
                temperature=current_temperature,
                max_tokens=current_max_tokens,
            )
            round_payload["control"] = control

            failure_analysis = round_payload.get("failure_analysis", {})
            round_success = bool(round_payload.get("success", False))
            if round_success:
                consecutive_success_rounds += 1
                consecutive_fail_rounds = 0
            else:
                consecutive_fail_rounds += 1
                consecutive_success_rounds = 0

            self_adjustments: Dict[str, Any] = {
                "prev_goal_mode": current_goal_mode,
                "prev_assist_max_calls": assist_cfg["max_calls_per_round"],
                "consecutive_fail_rounds": consecutive_fail_rounds,
                "consecutive_success_rounds": consecutive_success_rounds,
            }

            budget_blocked = int(failure_analysis.get("assist_reason_counts", {}).get("round-call-budget-exceeded", 0))
            if budget_blocked > 0:
                assist_cfg["max_calls_per_round"] = min(
                    max(1, args.assist_max_calls_cap),
                    assist_cfg["max_calls_per_round"] + 1,
                )

            if consecutive_fail_rounds >= 2 and current_goal_mode == "extended":
                current_goal_mode = "basic"
            elif (
                consecutive_success_rounds >= 2
                and current_goal_mode == "basic"
                and round_id >= max(0, args.basic_lock_rounds)
            ):
                current_goal_mode = "extended"

            if args.enable_docker_consistency_check and max(1, args.docker_check_interval_rounds) > 0:
                if round_id % max(1, args.docker_check_interval_rounds) == 0:
                    round_payload["docker_consistency"] = _docker_consistency_check(
                        base_url=base_url,
                        prompt=EVOLUTION_PROMPTS[0],
                        max_tokens=min(160, current_max_tokens),
                        temperature=min(0.8, current_temperature),
                        docker_image=args.docker_image,
                        min_overlap=max(0.0, min(1.0, args.docker_min_overlap)),
                    )

            self_adjustments["next_goal_mode"] = current_goal_mode
            self_adjustments["next_assist_max_calls"] = assist_cfg["max_calls_per_round"]
            round_payload["self_evolution_adjustments"] = self_adjustments
            current_temperature = next_temp
            current_max_tokens = next_tokens
            append_with_limit(all_rounds, round_payload, round_window)
            aggregate_stats["total_rounds"] += 1
            if round_payload.get("success", False):
                aggregate_stats["success_rounds"] += 1
            else:
                aggregate_stats["failed_rounds"] += 1
            for entry in round_payload.get("entries", []):
                assist = entry.get("runtime", {}).get("assist", {})
                if not assist.get("enabled", False):
                    continue
                aggregate_stats["assist_total_calls"] += 1
                if assist.get("ok", False):
                    aggregate_stats["assist_success_calls"] += 1
                aggregate_stats["assist_total_tokens"] += int(assist.get("tokens", 0) or 0)
                reason = str(assist.get("reason", "") or "ok" if assist.get("ok", False) else assist.get("reason", "unknown"))
                reasons = aggregate_stats["assist_reasons"]
                reasons[reason] = int(reasons.get(reason, 0)) + 1

            round_path = write_round_report(round_payload, trust_summary, trust_path)
            print(f"Round {round_id} report: {round_path}")

            alert = write_alert(round_payload)
            if alert is not None:
                append_with_limit(alert_files, alert, alert_window)
                print(f"Round {round_id} alert: {alert}")

            if args.rounds > 0 and round_id >= args.rounds:
                break

            time.sleep(max(1.0, args.interval_minutes * 60.0))

        daily_json, daily_md = write_daily_report(
            all_rounds,
            trust_summary,
            trust_path,
            alert_files,
            aggregate=aggregate_stats,
        )
        print(f"Daily report JSON: {daily_json}")
        print(f"Daily report MD: {daily_md}")

        if args.fail_on_empty and any(not r.get("success", False) for r in all_rounds):
            raise SystemExit("Self-evolution daemon detected empty/invalid responses")
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
