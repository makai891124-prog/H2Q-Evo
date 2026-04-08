#!/usr/bin/env python3
"""OpenClaw-style agent adapter for H2Q-Evo.

Purpose:
- Provide an easy-to-use autonomous-agent entry similar to OpenClaw UX.
- Expose H2Q core capabilities (local agent, dynamic blueprint evolution, release gate)
  through a simple CLI and HTTP interface.

Interfaces:
- CLI one-shot task execution.
- HTTP API:
  - POST /openclaw/agent/run
  - GET  /openclaw/manifest
    - POST /v1/chat/completions  (OpenAI compatibility)
    - POST /v1/responses         (OpenResponses compatibility)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import local H2Q executor for core agent reasoning.
from h2q_project.local_executor import LocalExecutor

REPORTS_DIR = REPO_ROOT / "reports"
PYTHON_BIN = sys.executable


@dataclass
class AdapterResult:
    mode: str
    ok: bool
    summary: str
    output: str
    confidence: float
    elapsed_sec: float
    artifacts: List[str]
    details: Dict[str, Any]


def _run_subprocess(cmd: List[str]) -> Dict[str, Any]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    elapsed = time.time() - start
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_sec": elapsed,
    }


def _extract_latest_path(stdout: str, key: str = "Latest JSON:") -> str:
    for line in stdout.splitlines():
        if key in line:
            return line.split(key, 1)[-1].strip()
    return ""


def run_agent_task(task: str, strategy: str = "auto") -> AdapterResult:
    start = time.time()
    executor = LocalExecutor(enable_precision_gating=True)
    adapter_home = REPO_ROOT / ".openclaw_adapter_home"
    adapter_home.mkdir(parents=True, exist_ok=True)

    try:
        executor.init_knowledge_db(adapter_home)
    except Exception:
        # Self-heal legacy or incompatible schema in isolated adapter storage.
        knowledge_dir = adapter_home / "knowledge"
        if knowledge_dir.exists():
            shutil.rmtree(knowledge_dir, ignore_errors=True)
        executor.init_knowledge_db(adapter_home)

    result = executor.execute(task, strategy=strategy)
    elapsed = time.time() - start

    output = str(result.get("output", ""))
    confidence = float(result.get("confidence", 0.0))
    ok = confidence >= 0.2 or len(output.strip()) > 0

    return AdapterResult(
        mode="agent",
        ok=ok,
        summary="Local autonomous agent task completed" if ok else "Agent execution produced low confidence output",
        output=output,
        confidence=confidence,
        elapsed_sec=elapsed,
        artifacts=[],
        details={
            "strategy_used": result.get("strategy_used", strategy),
            "task_type": result.get("task_type", "unknown"),
            "timestamp": result.get("timestamp"),
        },
    )


def run_dynamic_evolution(cycles: int = 2) -> AdapterResult:
    cmd = [
        PYTHON_BIN,
        "tools/dynamic_blueprint_bootstrap.py",
        "--cycles",
        str(cycles),
        "--max-actions-per-cycle",
        "2",
        "--enable-release-gate-cycle",
        "--strong-release-gate-cycle",
        "--release-gate-retries",
        "2",
        "--release-gate-profile",
        "quick",
        "--release-gate-relax-step",
        "0.05",
        "--min-breadth",
        "0.60",
        "--min-horizon",
        "0.80",
        "--min-robustness",
        "0.60",
        "--output-prefix",
        "openclaw_bridge_blueprint",
    ]
    run = _run_subprocess(cmd)
    ok = run["returncode"] == 0
    latest_json = _extract_latest_path(run["stdout"], key="Latest JSON:")
    latest_md = _extract_latest_path(run["stdout"], key="Latest MD:")

    summary = "Dynamic blueprint evolution completed" if ok else "Dynamic blueprint evolution failed"
    return AdapterResult(
        mode="evolve",
        ok=ok,
        summary=summary,
        output=run["stdout"][-1600:],
        confidence=1.0 if ok else 0.0,
        elapsed_sec=float(run["elapsed_sec"]),
        artifacts=[p for p in [latest_json, latest_md] if p],
        details={
            "returncode": run["returncode"],
            "stderr_tail": run["stderr"][-1000:],
        },
    )


def run_public_validation() -> AdapterResult:
    cmd = [
        PYTHON_BIN,
        "tools/release_gate.py",
        "--profile",
        "quick",
        "--lookback-rounds",
        "48",
        "--assist-provider",
        "none",
        "--min-breadth",
        "0.60",
        "--min-horizon",
        "0.80",
        "--min-robustness",
        "0.60",
        "--output-prefix",
        "openclaw_bridge_release_gate",
    ]
    run = _run_subprocess(cmd)
    ok = run["returncode"] == 0 and ("Gate OK: True" in run["stdout"])
    report_json = ""
    report_md = ""
    for line in run["stdout"].splitlines():
        if line.startswith("JSON:"):
            report_json = line.split("JSON:", 1)[-1].strip()
        if line.startswith("MD:"):
            report_md = line.split("MD:", 1)[-1].strip()

    return AdapterResult(
        mode="validate",
        ok=ok,
        summary="Release gate validation passed" if ok else "Release gate validation failed",
        output=run["stdout"][-1600:],
        confidence=1.0 if ok else 0.0,
        elapsed_sec=float(run["elapsed_sec"]),
        artifacts=[p for p in [report_json, report_md] if p],
        details={
            "returncode": run["returncode"],
            "stderr_tail": run["stderr"][-1000:],
        },
    )


def run_full(task: str, strategy: str, cycles: int) -> AdapterResult:
    start = time.time()
    agent = run_agent_task(task, strategy)
    evolve = run_dynamic_evolution(cycles=cycles)
    validate = run_public_validation()

    ok = agent.ok and evolve.ok and validate.ok
    artifacts = list(agent.artifacts) + list(evolve.artifacts) + list(validate.artifacts)

    output = "\n\n".join(
        [
            f"[agent] {agent.summary}\n{agent.output}",
            f"[evolve] {evolve.summary}\n{evolve.output}",
            f"[validate] {validate.summary}\n{validate.output}",
        ]
    )

    return AdapterResult(
        mode="full",
        ok=ok,
        summary="OpenClaw-style integrated autonomous flow passed" if ok else "Integrated flow has failed stages",
        output=output,
        confidence=(agent.confidence + evolve.confidence + validate.confidence) / 3.0,
        elapsed_sec=time.time() - start,
        artifacts=artifacts,
        details={
            "agent": asdict(agent),
            "evolve": asdict(evolve),
            "validate": asdict(validate),
        },
    )


def get_manifest() -> Dict[str, Any]:
    return {
        "name": "h2q-openclaw-adapter",
        "version": "0.2.0",
        "description": "OpenClaw-compatible autonomous agent adapter for H2Q-Evo core algorithms",
        "http_endpoints": [
            "/openclaw/manifest",
            "/openclaw/agent/run",
            "/v1/chat/completions",
            "/v1/responses",
        ],
        "openclaw_compat": {
            "supports_x_openclaw_agent_id": True,
            "supports_openai_chat_completions": True,
            "supports_openresponses": True,
        },
        "capabilities": [
            {
                "id": "agent.execute",
                "description": "Run local autonomous agent task with precision gating",
                "input_schema": {"task": "string", "strategy": "string"},
            },
            {
                "id": "h2q.dynamic_blueprint.evolve",
                "description": "Run dynamic blueprint bootstrap with strong release gate",
                "input_schema": {"cycles": "integer"},
            },
            {
                "id": "h2q.release_gate.validate",
                "description": "Run public-ready release gate validation",
                "input_schema": {},
            },
            {
                "id": "h2q.integrated.full",
                "description": "Run agent task + evolution + gate validation as one flow",
                "input_schema": {"task": "string", "strategy": "string", "cycles": "integer"},
            },
        ],
        "artifacts_dir": str(REPORTS_DIR),
    }


class AgentRunRequest(BaseModel):
    task: str = Field(..., description="Natural language task")
    mode: str = Field("full", description="agent|evolve|validate|full")
    strategy: str = Field("auto", description="Execution strategy for agent mode")
    cycles: int = Field(2, description="Evolution cycles for evolve/full mode")


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAICompatRequest(BaseModel):
    model: Optional[str] = "h2q-openclaw"
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    user: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512


class OpenResponsesRequest(BaseModel):
    model: Optional[str] = "h2q-openclaw"
    input: Any = None
    stream: Optional[bool] = False
    user: Optional[str] = None


def _extract_task_from_responses_input(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value.strip()

    if isinstance(input_value, dict):
        return _extract_task_from_responses_input([input_value])

    if isinstance(input_value, list):
        user_texts: List[str] = []
        for item in input_value:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = item.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text and role in {"user", "input", "developer"}:
                    user_texts.append(text)
                continue
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if str(part.get("type", "")).strip().lower() in {"input_text", "text"}:
                        text = str(part.get("text", "")).strip()
                        if text and role in {"user", "input", "developer"}:
                            user_texts.append(text)
        if user_texts:
            return user_texts[-1]

    return ""


def dispatch(mode: str, task: str, strategy: str, cycles: int) -> AdapterResult:
    mode = mode.lower().strip()
    if mode == "agent":
        return run_agent_task(task=task, strategy=strategy)
    if mode == "evolve":
        return run_dynamic_evolution(cycles=cycles)
    if mode == "validate":
        return run_public_validation()
    if mode == "full":
        return run_full(task=task, strategy=strategy, cycles=cycles)
    raise ValueError(f"Unsupported mode: {mode}")


app = FastAPI(title="H2Q OpenClaw Adapter", version="0.2.0")


@app.get("/openclaw/manifest")
def manifest() -> Dict[str, Any]:
    return get_manifest()


@app.post("/openclaw/agent/run")
def openclaw_agent_run(request: AgentRunRequest) -> Dict[str, Any]:
    try:
        res = dispatch(request.mode, request.task, request.strategy, request.cycles)
        return asdict(res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/chat/completions")
def openai_compat_chat(
    request: OpenAICompatRequest,
    x_openclaw_agent_id: Optional[str] = Header(default=None, alias="x-openclaw-agent-id"),
) -> Dict[str, Any]:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    user_messages = [m.content for m in request.messages if m.role == "user"]
    task = user_messages[-1] if user_messages else request.messages[-1].content
    if not task.strip():
        raise HTTPException(status_code=400, detail="Missing user message content")

    try:
        res = run_agent_task(task=task, strategy="auto")
        content = res.output or res.summary
        now = int(time.time())
        return {
            "id": f"chatcmpl-h2q-{now}",
            "object": "chat.completion",
            "created": now,
            "model": request.model or "h2q-openclaw",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, len(task.split())),
                "completion_tokens": max(1, len(content.split())),
                "total_tokens": max(2, len(task.split()) + len(content.split())),
            },
            "_openclaw": {
                "agent_id": x_openclaw_agent_id or "main",
                "stream_requested": bool(request.stream),
                "user": request.user,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/responses")
def openresponses_compat(
    request: OpenResponsesRequest,
    x_openclaw_agent_id: Optional[str] = Header(default=None, alias="x-openclaw-agent-id"),
) -> Dict[str, Any]:
    task = _extract_task_from_responses_input(request.input)
    if not task:
        raise HTTPException(status_code=400, detail="input must contain user text")

    try:
        res = run_agent_task(task=task, strategy="auto")
        content = res.output or res.summary
        now = int(time.time())
        response_id = f"resp_h2q_{now}"
        return {
            "id": response_id,
            "object": "response",
            "created": now,
            "model": request.model or "h2q-openclaw",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": f"msg_h2q_{now}",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                        }
                    ],
                }
            ],
            "output_text": content,
            "usage": {
                "input_tokens": max(1, len(task.split())),
                "output_tokens": max(1, len(content.split())),
                "total_tokens": max(2, len(task.split()) + len(content.split())),
            },
            "_openclaw": {
                "agent_id": x_openclaw_agent_id or "main",
                "stream_requested": bool(request.stream),
                "user": request.user,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw-style adapter for H2Q-Evo")
    parser.add_argument("--serve", action="store_true", help="Run HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--mode", default="full", help="agent|evolve|validate|full")
    parser.add_argument("--task", default="请输出一个可执行的本地AGI自举进化计划")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run("tools.openclaw_h2q_adapter:app", host=args.host, port=args.port, reload=False)
        return

    result = dispatch(mode=args.mode, task=args.task, strategy=args.strategy, cycles=args.cycles)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"mode={result.mode} ok={result.ok} confidence={result.confidence:.3f} elapsed={result.elapsed_sec:.2f}s")
        print(f"summary: {result.summary}")
        print("artifacts:")
        for p in result.artifacts:
            print(f"- {p}")
        print("output:")
        print(result.output[:4000])


if __name__ == "__main__":
    main()
