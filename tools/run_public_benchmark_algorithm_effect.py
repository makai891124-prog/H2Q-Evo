#!/usr/bin/env python3
"""Run a public benchmark harness with infra probing and multi-solver comparison."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _extract_json_candidates(raw_text: str) -> List[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return []

    candidates: List[Dict[str, Any]] = []

    # 1) Parse fenced JSON blocks first.
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE):
        chunk = m.group(1).strip()
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            pass

    # 2) Parse outermost raw JSON object from text.
    obj = _extract_json_object(text)
    if obj:
        candidates.append(obj)

    # 3) Parse escaped JSON snippets embedded in strings.
    for m in re.finditer(r'\{\\"x\\"\s*:\s*[-]?\d+\s*,\s*\\"y\\"\s*:\s*[-]?\d+\}', text):
        chunk = m.group(0).encode("utf-8").decode("unicode_escape")
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            pass

    return candidates


def _has_integer_xy(payload: Dict[str, Any], model_text: str) -> bool:
    # Direct payload fields
    if isinstance(payload.get("x"), int) and isinstance(payload.get("y"), int):
        return True

    # Candidates parsed from model text
    for obj in _extract_json_candidates(model_text):
        if isinstance(obj.get("x"), int) and isinstance(obj.get("y"), int):
            return True

    # Try one level nested dict values if endpoint wraps JSON object in a field.
    for v in payload.values():
        if isinstance(v, dict) and isinstance(v.get("x"), int) and isinstance(v.get("y"), int):
            return True

    return False


def _extract_model_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("response"), str):
        return payload["response"]
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("message"), str):
        return payload["message"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            if isinstance(c0.get("text"), str):
                return c0["text"]
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    return ""


def _probe_infra(health_endpoint: str, model_endpoint: str, timeout_sec: float) -> Dict[str, Any]:
    opener = request.build_opener(request.ProxyHandler({}))

    health_ok = False
    health_error = ""
    try:
        req = request.Request(health_endpoint, method="GET")
        with opener.open(req, timeout=max(1.0, timeout_sec)) as resp:
            health_ok = int(getattr(resp, "status", 200)) < 400
    except Exception as exc:  # pragma: no cover
        health_error = f"{type(exc).__name__}: {exc}"

    json_ok = False
    json_error = ""
    response_preview = ""
    try:
        probe_body = {
            "prompt": "Return only JSON object with integer x and y fields. Example: {\"x\":1,\"y\":0}",
            "max_new_tokens": 64,
            "temperature": 0.2,
            "use_das_arch": False,
        }
        req = request.Request(
            model_endpoint,
            data=json.dumps(probe_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(req, timeout=max(1.0, timeout_sec)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = _extract_model_text(payload)
        response_preview = (text or json.dumps(payload, ensure_ascii=False))[:240]
        json_ok = _has_integer_xy(payload, text)
        if not json_ok:
            json_error = "JSON probe response does not contain integer x/y fields"
    except Exception as exc:  # pragma: no cover
        json_error = f"{type(exc).__name__}: {exc}"

    infra_invalid = not (health_ok and json_ok)
    return {
        "health_ok": health_ok,
        "health_error": health_error,
        "json_probe_ok": json_ok,
        "json_probe_error": json_error,
        "json_probe_preview": response_preview,
        "infra_invalid": infra_invalid,
        "summary": (
            "infra_invalid: service health/json-protocol probe failed"
            if infra_invalid
            else "infra_ok"
        ),
    }


def _run(cmd: List[str], timeout_sec: int) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout_sec)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-40:]),
    }


def _solver_entry(name: str, run: Dict[str, Any], artifact: Path | None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "solver": name,
        "status": "ok" if int(run.get("returncode", 1)) == 0 else "failed",
        "run": run,
        "artifact": str(artifact) if artifact else "",
        "metrics": {},
    }
    if artifact and artifact.exists() and entry["status"] == "ok":
        payload = _load_json(artifact)
        entry["metrics"] = payload.get("metrics") or {}
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Public benchmark algorithm-effect harness")
    parser.add_argument("--task-file", default="benchmarks/interactive_reasoning/tasks_v1.json")
    parser.add_argument("--health-endpoint", default="http://127.0.0.1:8000/health")
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--model-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-steps-multiplier", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--output-prefix", default="public_benchmark_algorithm_effect")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    task_file = str((ROOT / args.task_file).resolve())

    probe = _probe_infra(args.health_endpoint, args.model_endpoint, args.model_timeout_seconds)

    solver_runs: Dict[str, Dict[str, Any]] = {}

    bfs_prefix = f"{args.output_prefix}_bfs"
    bfs_cmd = [
        str(PY),
        "tools/run_interactive_reasoning_benchmark.py",
        "--solver",
        "bfs",
        "--task-file",
        task_file,
        "--min-success-rate",
        "0.99",
        "--output-prefix",
        bfs_prefix,
    ]
    bfs_run = _run(bfs_cmd, timeout_sec=args.timeout_sec)
    bfs_artifact = REPORTS / f"{bfs_prefix}_latest.json"
    solver_runs["bfs"] = _solver_entry("bfs", bfs_run, bfs_artifact if bfs_artifact.exists() else None)

    if probe["infra_invalid"]:
        solver_runs["model"] = {
            "solver": "model",
            "status": "infra_invalid",
            "reason": probe["summary"],
            "artifact": "",
            "metrics": {},
        }
    else:
        model_prefix = f"{args.output_prefix}_model"
        model_cmd = [
            str(PY),
            "tools/run_interactive_reasoning_benchmark.py",
            "--solver",
            "model",
            "--task-file",
            task_file,
            "--model-endpoint",
            args.model_endpoint,
            "--model-timeout-seconds",
            str(max(1.0, args.model_timeout_seconds)),
            "--max-steps-multiplier",
            str(max(1, args.max_steps_multiplier)),
            "--min-success-rate",
            "0.0",
            "--output-prefix",
            model_prefix,
        ]
        model_run = _run(model_cmd, timeout_sec=args.timeout_sec)
        model_artifact = REPORTS / f"{model_prefix}_latest.json"
        solver_runs["model"] = _solver_entry(
            "model",
            model_run,
            model_artifact if model_artifact.exists() else None,
        )

    guard_prefix = f"{args.output_prefix}_model_json_guard"
    guard_cmd = [
        str(PY),
        "tools/run_interactive_reasoning_benchmark.py",
        "--solver",
        "model-json-guard",
        "--task-file",
        task_file,
        "--model-endpoint",
        args.model_endpoint,
        "--model-timeout-seconds",
        str(max(1.0, args.model_timeout_seconds)),
        "--max-steps-multiplier",
        str(max(1, args.max_steps_multiplier)),
        "--min-success-rate",
        "0.0",
        "--output-prefix",
        guard_prefix,
    ]
    guard_run = _run(guard_cmd, timeout_sec=args.timeout_sec)
    guard_artifact = REPORTS / f"{guard_prefix}_latest.json"
    solver_runs["model-json-guard"] = _solver_entry(
        "model-json-guard",
        guard_run,
        guard_artifact if guard_artifact.exists() else None,
    )
    solver_runs["model-json-guard"]["infra_invalid_context"] = bool(probe["infra_invalid"])

    bfs_metrics = solver_runs.get("bfs", {}).get("metrics") or {}
    model_metrics = solver_runs.get("model", {}).get("metrics") or {}
    guard_metrics = solver_runs.get("model-json-guard", {}).get("metrics") or {}

    model_has_metrics = bool(model_metrics)
    guard_has_metrics = bool(guard_metrics)
    comparison = {
        "model_minus_bfs_success_rate": (
            float(model_metrics.get("success_rate", 0.0)) - float(bfs_metrics.get("success_rate", 0.0))
            if model_has_metrics
            else None
        ),
        "guard_minus_bfs_success_rate": (
            float(guard_metrics.get("success_rate", 0.0)) - float(bfs_metrics.get("success_rate", 0.0))
            if guard_has_metrics
            else None
        ),
        "guard_minus_bfs_parse_errors": (
            int(guard_metrics.get("parse_errors", 0) or 0) - int(bfs_metrics.get("parse_errors", 0) or 0)
            if guard_has_metrics
            else None
        ),
        "guard_minus_model_success_rate": (
            float(guard_metrics.get("success_rate", 0.0)) - float(model_metrics.get("success_rate", 0.0))
            if model_has_metrics and guard_has_metrics
            else None
        ),
        "guard_minus_model_parse_errors": (
            int(guard_metrics.get("parse_errors", 0) or 0) - int(model_metrics.get("parse_errors", 0) or 0)
            if model_has_metrics and guard_has_metrics
            else None
        ),
    }

    payload = {
        "generated_at_utc": _now_utc(),
        "infra_probe": probe,
        "task_file": task_file,
        "solver_runs": solver_runs,
        "comparison": comparison,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    lines: List[str] = [
        "# Public Benchmark Algorithm Effect",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- task_file: `{task_file}`",
        f"- infra_invalid: `{probe['infra_invalid']}`",
        f"- health_ok: `{probe['health_ok']}`",
        f"- json_probe_ok: `{probe['json_probe_ok']}`",
    ]

    if probe.get("health_error"):
        lines.append(f"- health_error: `{probe['health_error']}`")
    if probe.get("json_probe_error"):
        lines.append(f"- json_probe_error: `{probe['json_probe_error']}`")

    lines.extend(["", "## Solver Summary"])
    for name in ["bfs", "model", "model-json-guard"]:
        entry = solver_runs.get(name) or {}
        lines.append(f"- {name}: status=`{entry.get('status', 'missing')}` artifact=`{entry.get('artifact', '')}`")
        metrics = entry.get("metrics") or {}
        if metrics:
            lines.append(
                f"  success_rate={float(metrics.get('success_rate', 0.0)):.6f}, parse_errors={int(metrics.get('parse_errors', 0) or 0)}, avg_steps={float(metrics.get('avg_steps', 0.0)):.2f}"
            )

    lines.extend(
        [
            "",
            "## Comparison",
            f"- model_minus_bfs_success_rate: `{comparison['model_minus_bfs_success_rate']}`",
            f"- guard_minus_bfs_success_rate: `{comparison['guard_minus_bfs_success_rate']}`",
            f"- guard_minus_bfs_parse_errors: `{comparison['guard_minus_bfs_parse_errors']}`",
            f"- guard_minus_model_success_rate: `{comparison['guard_minus_model_success_rate']}`",
            f"- guard_minus_model_parse_errors: `{comparison['guard_minus_model_parse_errors']}`",
        ]
    )

    if probe["infra_invalid"]:
        lines.extend(
            [
                "",
                "## Interpretation",
                "- Model-family results are marked `infra_invalid`; this run should not be interpreted as algorithm regression.",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Infra invalid: {probe['infra_invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
