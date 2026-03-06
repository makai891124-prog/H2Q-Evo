#!/usr/bin/env python3
"""Run interactive reasoning benchmark tasks.

The benchmark evaluates multi-step grid navigation with ordered checkpoints,
which acts as a lightweight proxy for interactive reasoning and memory.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _neighbors(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [(nx, ny) for nx, ny in cand if 0 <= nx < width and 0 <= ny < height]


def _bfs(
    width: int,
    height: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    if start in obstacles or goal in obstacles:
        return None
    q: deque[Tuple[int, int]] = deque([start])
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            path: List[Tuple[int, int]] = []
            node: Optional[Tuple[int, int]] = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        for nxt in _neighbors(cur[0], cur[1], width, height):
            if nxt in obstacles or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    return None


def _solve_task(task: Dict[str, Any]) -> Dict[str, Any]:
    width = int(task["width"])
    height = int(task["height"])
    start = (int(task["start"][0]), int(task["start"][1]))
    goal = (int(task["goal"][0]), int(task["goal"][1]))
    checkpoints = [
        (int(p[0]), int(p[1]))
        for p in task.get("checkpoints", [])
    ]
    obstacles = {(int(p[0]), int(p[1])) for p in task.get("obstacles", [])}

    ordered_targets = checkpoints + [goal]
    cursor = start
    full_path: List[Tuple[int, int]] = [start]
    reached = []

    for target in ordered_targets:
        seg = _bfs(width=width, height=height, start=cursor, goal=target, obstacles=obstacles)
        if not seg:
            return {
                "task_id": task["id"],
                "success": False,
                "reason": "unreachable",
                "steps": 0,
                "reached": reached,
                "goal": list(goal),
            }
        full_path.extend(seg[1:])
        cursor = target
        reached.append(list(target))

    return {
        "task_id": task["id"],
        "success": True,
        "reason": "ok",
        "steps": max(0, len(full_path) - 1),
        "reached": reached,
        "goal": list(goal),
    }


def _clamp_pos(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
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


def _model_next_move(
    *,
    model_endpoint: str,
    model_timeout_seconds: float,
    model_use_chat_api: bool,
    task: Dict[str, Any],
    width: int,
    height: int,
    obstacles: set[Tuple[int, int]],
    current: Tuple[int, int],
    next_target: Tuple[int, int],
    step_index: int,
    max_total_steps: int,
) -> Tuple[int, int, str]:
    obstacle_list = sorted([[x, y] for x, y in obstacles])
    prompt = (
        "You are controlling an agent on a grid. Return only JSON with integer fields x and y for the next position. "
        "One-step Manhattan move only. Avoid obstacles and boundaries.\n"
        f"grid_width={width}, grid_height={height}\n"
        f"current={[current[0], current[1]]}, target={[next_target[0], next_target[1]]}\n"
        f"obstacles={obstacle_list}\n"
        f"step={step_index}, max_steps={max_total_steps}, task_id={task.get('id','')}"
    )

    if model_use_chat_api:
        body = {
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "max_new_tokens": 64,
            "temperature": 0.2,
        }
    else:
        body = {
            "prompt": prompt,
            "max_new_tokens": 64,
            "temperature": 0.2,
            "use_das_arch": False,
        }

    req = request.Request(
        model_endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = request.build_opener(request.ProxyHandler({}))

    try:
        with opener.open(req, timeout=max(1.0, model_timeout_seconds)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return current[0], current[1], f"api-error:{type(exc).__name__}"

    text = ""
    if isinstance(data, dict):
        if isinstance(data.get("response"), str):
            text = data["response"]
        elif isinstance(data.get("text"), str):
            text = data["text"]
        elif isinstance(data.get("message"), str):
            text = data["message"]
        elif isinstance(data.get("choices"), list) and data["choices"]:
            choice0 = data["choices"][0]
            if isinstance(choice0, dict):
                if isinstance(choice0.get("text"), str):
                    text = choice0["text"]
                elif isinstance(choice0.get("message"), dict):
                    msg = choice0.get("message", {})
                    if isinstance(msg.get("content"), str):
                        text = msg["content"]

    move = _extract_json_object(text)
    if not move:
        return current[0], current[1], "parse-failed"

    try:
        nx = int(move.get("x"))
        ny = int(move.get("y"))
    except (TypeError, ValueError):
        return current[0], current[1], "non-integer"

    nx, ny = _clamp_pos(nx, ny, width, height)
    if abs(nx - current[0]) + abs(ny - current[1]) != 1:
        return current[0], current[1], "non-adjacent"
    if (nx, ny) in obstacles:
        return current[0], current[1], "hit-obstacle"
    return nx, ny, "ok"


def _solve_task_with_model(
    task: Dict[str, Any],
    *,
    model_endpoint: str,
    model_timeout_seconds: float,
    model_use_chat_api: bool,
    max_steps_multiplier: int,
) -> Dict[str, Any]:
    width = int(task["width"])
    height = int(task["height"])
    start = (int(task["start"][0]), int(task["start"][1]))
    goal = (int(task["goal"][0]), int(task["goal"][1]))
    checkpoints = [(int(p[0]), int(p[1])) for p in task.get("checkpoints", [])]
    obstacles = {(int(p[0]), int(p[1])) for p in task.get("obstacles", [])}
    ordered_targets = checkpoints + [goal]

    if start in obstacles or goal in obstacles:
        return {
            "task_id": task["id"],
            "success": False,
            "reason": "invalid-start-or-goal",
            "steps": 0,
            "reached": [],
            "goal": list(goal),
            "policy": "model",
        }

    max_total_steps = max(8, width * height * max(1, max_steps_multiplier))
    current = start
    reached: List[List[int]] = []
    target_idx = 0
    parse_errors = 0

    for step_index in range(max_total_steps):
        if target_idx >= len(ordered_targets):
            return {
                "task_id": task["id"],
                "success": True,
                "reason": "ok",
                "steps": step_index,
                "reached": reached,
                "goal": list(goal),
                "policy": "model",
                "parse_errors": parse_errors,
            }

        next_target = ordered_targets[target_idx]
        nx, ny, move_reason = _model_next_move(
            model_endpoint=model_endpoint,
            model_timeout_seconds=model_timeout_seconds,
            model_use_chat_api=model_use_chat_api,
            task=task,
            width=width,
            height=height,
            obstacles=obstacles,
            current=current,
            next_target=next_target,
            step_index=step_index,
            max_total_steps=max_total_steps,
        )
        if move_reason != "ok":
            parse_errors += 1

        current = (nx, ny)
        if current == next_target:
            reached.append([current[0], current[1]])
            target_idx += 1

    return {
        "task_id": task["id"],
        "success": False,
        "reason": "step-budget-exceeded",
        "steps": max_total_steps,
        "reached": reached,
        "goal": list(goal),
        "policy": "model",
        "parse_errors": parse_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive reasoning benchmark runner")
    parser.add_argument(
        "--task-file",
        default="benchmarks/interactive_reasoning/tasks_v1.json",
        help="Path to task JSON file",
    )
    parser.add_argument("--solver", choices=["bfs", "model"], default="bfs")
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--model-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--model-use-chat-api", action="store_true")
    parser.add_argument("--max-steps-multiplier", type=int, default=4)
    parser.add_argument("--min-success-rate", type=float, default=0.80)
    parser.add_argument("--output-prefix", default="interactive_reasoning_benchmark")
    args = parser.parse_args()

    task_path = ROOT / args.task_file
    if not task_path.exists():
        raise SystemExit(f"Task file not found: {task_path}")

    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("Task file must be a non-empty JSON array")

    REPORTS.mkdir(parents=True, exist_ok=True)

    if args.solver == "model":
        results = [
            _solve_task_with_model(
                t,
                model_endpoint=args.model_endpoint,
                model_timeout_seconds=max(1.0, args.model_timeout_seconds),
                model_use_chat_api=bool(args.model_use_chat_api),
                max_steps_multiplier=max(1, args.max_steps_multiplier),
            )
            for t in tasks
        ]
    else:
        results = [_solve_task(t) for t in tasks]

    success_count = sum(1 for r in results if r.get("success", False))
    task_count = len(results)
    success_rate = float(success_count) / float(task_count)
    steps = [int(r.get("steps", 0) or 0) for r in results if r.get("success", False)]

    payload = {
        "meta": {
            "created_at_utc": _now_utc(),
            "task_file": str(task_path),
            "task_count": task_count,
            "solver": args.solver,
            "min_success_rate": max(0.0, args.min_success_rate),
        },
        "metrics": {
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_steps": (sum(steps) / float(len(steps))) if steps else 0.0,
            "max_steps": max(steps) if steps else 0,
            "min_steps": min(steps) if steps else 0,
            "parse_errors": sum(int(r.get("parse_errors", 0) or 0) for r in results),
            "passed": success_rate >= max(0.0, args.min_success_rate),
        },
        "results": results,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_latest = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    out_latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    lines = [
        "# Interactive Reasoning Benchmark",
        "",
        f"- created_at_utc: `{payload['meta']['created_at_utc']}`",
        f"- task_count: `{task_count}`",
        f"- success_count: `{success_count}`",
        f"- success_rate: `{success_rate:.2%}`",
        f"- solver: `{args.solver}`",
        f"- avg_steps: `{payload['metrics']['avg_steps']:.2f}`",
        f"- parse_errors: `{payload['metrics']['parse_errors']}`",
        f"- pass_threshold: `{max(0.0, args.min_success_rate):.2%}`",
        f"- passed: `{payload['metrics']['passed']}`",
        "",
    ]

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_latest_md.write_text("\n".join(lines), encoding="utf-8")

    print("Interactive reasoning benchmark completed")
    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")

    if not payload["metrics"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
