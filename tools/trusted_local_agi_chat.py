#!/usr/bin/env python3
"""Local trusted AGI chat orchestrator.

This script combines two real capabilities already implemented in the repo:
1) Trusted joint validation gate from tools/trusted_joint_agi_quantum_center.py
2) Local chat inference endpoint from h2q_project/h2q_server.py (/chat)

It gives a directly runnable local conversation loop with trust evidence.
"""

import argparse
import json
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
    if any(k in lower for k in ["python", "code", "函数", "脚本", "编程", "json"]):
        return (
            "```python\n"
            "def agi_iteration_plan(goal: str) -> dict:\n"
            "    steps = [\n"
            "        \"定义可验证目标\",\n"
            "        \"生成最小实现\",\n"
            "        \"执行并记录指标\",\n"
            "        \"基于失败样本迭代\",\n"
            "    ]\n"
            "    metrics = {\"pass_rate\": 0.0, \"latency_ms\": 0.0, \"empty_reply_rate\": 0.0}\n"
            "    return {\"goal\": goal, \"steps\": steps, \"metrics\": metrics, \"stop_condition\": \"pass_rate>=0.9\"}\n"
            "```"
        )
    return (
        "本地模型当前处于保守裁剪模式，已切换到系统应急策略。\n"
        "1. 先定义本轮可验证目标。\n"
        "2. 生成最小可执行方案并记录指标。\n"
        "3. 根据失败样本进行下一轮改进。"
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


def _chat_once(base_url: str, prompt: str, max_tokens: int, temperature: float, use_das_arch: bool) -> Dict[str, Any]:
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

    if not _is_effective_text(last_result.get("text", ""), prompt=prompt):
        return {
            "text": _fallback_text(prompt),
            "status": "Fallback",
            "fueter_curvature": None,
            "spectral_shift_eta": None,
            "_route": "local_fallback_template",
        }
    return last_result


def interactive_chat(
    base_url: str,
    trust_summary: Dict[str, Any],
    trust_report: Path,
    max_tokens: int,
    temperature: float,
    use_das_arch: bool,
) -> Path:
    print("\nTrusted Local AGI Chat Ready")
    print(f"Trust report: {trust_report}")
    print(f"Trust score: {trust_summary['trust_score']:.4f} | trusted_ready={trust_summary['trusted_ready']}")
    print("Commands: /help  /status  /exit")

    transcript: List[Dict[str, Any]] = []
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
            result = _chat_once(base_url, user_text, max_tokens=max_tokens, temperature=temperature, use_das_arch=use_das_arch)
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

        print(f"Assistant > {answer}")
        print(f"[route={route}, status={status}, curvature={curvature}, eta={eta}, latency={latency:.2f}s]")

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
                },
            }
        )

    out = ROOT / "reports" / f"trusted_local_agi_chat_session_{int(time.time())}.json"
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "start_time_utc": start_at,
                    "end_time_utc": datetime.now(timezone.utc).isoformat(),
                    "base_url": base_url,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "use_das_arch": use_das_arch,
                    "trust_report": str(trust_report),
                },
                "trust": trust_summary,
                "transcript": transcript,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted local AGI chat runner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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

        print("\n== Stage 3: Interactive trusted conversation ==")
        session_path = interactive_chat(
            base_url=base_url,
            trust_summary=trust_summary,
            trust_report=trust_report,
            max_tokens=max(1, args.max_tokens),
            temperature=args.temperature,
            use_das_arch=args.use_das_arch,
        )
        print(f"Session saved: {session_path}")
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
