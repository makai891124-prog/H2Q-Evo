#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启发式模块进化执行器
- 从本地文档提取上下文
- 使用 Gemini CLI 生成 todo list
- 按 todo 执行代码变更
- 运行校验机制（语法检查 + 作弊检测）

遵循：真实编程、无作弊、无欺骗
"""

import json
import os
import time
import logging
import argparse
import subprocess
import sys
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque

from h2q_project.h2q.agi.gemini_cli_integration import GeminiCLIIntegration

try:
    from h2q_project.h2q.agi.gemini_verifier import GeminiCodeSupervisor
    SUPERVISOR_AVAILABLE = True
except Exception:
    SUPERVISOR_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = PROJECT_ROOT / "evo_state.json"
DOC_FILE = PROJECT_ROOT / "Upload Project File For Analysis"
SUMMARY_FILE = PROJECT_ROOT / "HEURISTIC_EVOLUTION_SUMMARY.md"
SANDBOX_ROOT = PROJECT_ROOT / ".evo_sandbox"
ALLOWED_TARGET_ROOTS = set(os.getenv("EVOLUTION_ALLOWED_ROOTS", "h2q_project").split(","))
DOCKER_REQUIRED = os.getenv("REQUIRE_DOCKER_VALIDATION", "1") == "1"

logger = logging.getLogger(__name__)


class CodeValidator:
    @staticmethod
    def validate_syntax(code: str, filename: str) -> bool:
        if not filename.endswith(".py"):
            return True
        try:
            compile(code, filename, "exec")
            return True
        except SyntaxError:
            return False


class HeuristicEvolutionDriver:
    def __init__(self):
        self.gemini = GeminiCLIIntegration(api_key=os.getenv("GEMINI_API_KEY"))
        self.supervisor = GeminiCodeSupervisor(os.getenv("GEMINI_API_KEY")) if SUPERVISOR_AVAILABLE else None
        self.state = self._load_state()
        self._action_timestamps = deque()
        self._last_success_ts = self.state.get("last_success_ts", 0.0)
        self._consecutive_failures = int(self.state.get("consecutive_failures", 0))

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return {"generation": 0, "last_task_id": 0, "todo_list": [], "history": []}

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _extract_doc_context(self, max_chars: int = 6000) -> str:
        parts: List[str] = []
        if DOC_FILE.exists():
            text = DOC_FILE.read_text(encoding="utf-8", errors="ignore")
            key = "todo_list"
            idx = text.find(key)
            if idx != -1:
                start = max(0, idx - 2000)
                end = min(len(text), idx + 2000)
                parts.append(text[start:end])
            parts.append(text[: max_chars])
        if SUMMARY_FILE.exists():
            parts.append(SUMMARY_FILE.read_text(encoding="utf-8", errors="ignore")[:max_chars])
        return "\n\n".join(parts)[:max_chars]

    def generate_todo_list(self, limit: int = 5) -> List[Dict[str, Any]]:
        context = self._extract_doc_context()
        logger.info("生成待办列表...")
        failure_summary = self._build_failure_summary()
        allowed_hint = ", ".join(sorted(ALLOWED_TARGET_ROOTS))
        prompt = f"""
你是H2Q-Evo的进化规划器。请基于以下上下文生成待办任务列表（JSON数组）。
要求：
- 仅输出JSON数组，不要额外文字。
- 每个任务包含：task, priority(high|medium|low), target_files(数组), rationale。
- 任务应可执行、可验证、非作弊。
    - target_files 必须落在以下目录之一：{allowed_hint}
    - 只选择本仓库核心模块，不要修改第三方依赖或外部路径。
    - 优先解决最近失败原因，避免重复无效任务。

上下文：
{context}

    近期失败摘要：
    {failure_summary}
"""
        result = self.gemini.query(prompt, use_cache=False, response_mime_type="application/json")
        if result.get("status") != "success":
            logger.error("生成待办列表失败: %s", result.get("error", "unknown"))
            return []
        raw = result.get("response", "")
        tasks = self._parse_json_payload(raw, expect_array=True)
        if tasks is None:
            logger.error("待办JSON解析失败")
            self._dump_raw_response("todo_parse_failure", raw)
            return []

        tasks = self._filter_safe_todos(tasks)
        logger.info("生成待办数量: %d", len(tasks))
        return tasks[:limit]

    def inject_todos(self, todos: List[Dict[str, Any]]):
        last_id = int(self.state.get("last_task_id", 0))
        for t in todos:
            last_id += 1
            self.state.setdefault("todo_list", []).append({
                "id": last_id,
                "task": t.get("task", ""),
                "priority": t.get("priority", "medium"),
                "status": "pending",
                "target_files": t.get("target_files", []),
                "rationale": t.get("rationale", ""),
                "source": "auto_module_evolution",
                "retry_count": 0
            })
        self.state["last_task_id"] = last_id
        self._save_state()

    def _filter_safe_todos(self, todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        safe: List[Dict[str, Any]] = []
        for t in todos:
            targets = t.get("target_files", [])
            if not targets:
                continue
            if any(self._is_safe_relative_path(p) for p in targets):
                safe.append(t)
        if not safe:
            logger.warning("未生成可用任务（目标路径不安全）")
        return safe

    def _select_next_task(self) -> Optional[Dict[str, Any]]:
        todos = [t for t in self.state.get("todo_list", []) if t.get("status") == "pending"]
        if not todos:
            logger.info("暂无待办任务")
            return None
        filtered: List[Dict[str, Any]] = []
        for t in todos:
            targets = t.get("target_files", [])
            if not targets:
                self._mark_task_failed(t, "missing_targets")
                continue
            safe = any(self._is_safe_relative_path(p) for p in targets)
            if not safe:
                self._mark_task_failed(t, "unsafe_targets")
                continue
            filtered.append(t)
        if not filtered:
            logger.info("待办任务均不安全，等待重生成")
            return None
        priority_order = {"high": 0, "medium": 1, "low": 2}
        filtered.sort(key=lambda t: priority_order.get(t.get("priority", "medium"), 1))
        return filtered[0]

    def _generate_code_change(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        logger.info("生成代码变更: %s", task.get("task", ""))
        allowed_hint = ", ".join(sorted(ALLOWED_TARGET_ROOTS))
        prompt = f"""
你是H2Q-Evo自动改进器。请为以下任务生成代码变更：
任务: {task.get('task')}
要求：
- 输出JSON对象：{{"file_path": "...", "content": "完整文件内容"}}。
- 必须复用现有项目抽象，避免硬编码答案与作弊。
- 修改尽量小且可验证。
    - file_path 必须是相对路径，并且位于以下目录之一：{allowed_hint}
    - 不允许修改第三方依赖目录或外部路径。
"""
        result = self.gemini.query(prompt, use_cache=False, response_mime_type="application/json")
        if result.get("status") != "success":
            logger.error("代码生成失败: %s", result.get("error", "unknown"))
            return None
        raw = result.get("response", "")
        plan = self._parse_json_payload(raw, expect_array=False)
        if plan is None:
            logger.error("代码生成JSON解析失败")
            self._dump_raw_response("code_parse_failure", raw)
            return None
        if isinstance(plan, list):
            for item in plan:
                if not isinstance(item, dict):
                    logger.error("代码生成JSON格式错误")
                    return None
                if not self._is_safe_relative_path(item.get("file_path", "")):
                    logger.error("生成的路径不安全，拒绝应用")
                    return None
            return plan
        if not isinstance(plan, dict):
            logger.error("代码生成JSON格式错误")
            return None
        if not self._is_safe_relative_path(plan.get("file_path", "")):
            logger.error("生成的路径不安全，拒绝应用")
            return None
        return plan

    def _validate_change(self, code_plan: Any, task: Dict[str, Any]) -> bool:
        if isinstance(code_plan, list):
            for item in code_plan:
                if not self._validate_change(item, task):
                    return False
            return True
        if not isinstance(code_plan, dict):
            logger.error("代码变更无效: 非法格式")
            return False
        file_path = code_plan.get("file_path")
        content = code_plan.get("content", "")
        if not file_path or not content:
            logger.error("代码变更无效: 缺少文件或内容")
            return False
        if not self._is_safe_relative_path(file_path):
            logger.error("代码变更无效: 路径不安全 %s", file_path)
            return False
        if not CodeValidator.validate_syntax(content, file_path):
            logger.error("语法校验失败: %s", file_path)
            return False
        if self.supervisor:
            ok, _ = self.supervisor.supervise_code_generation(content, task_description=task.get("task", ""))
            if not ok:
                logger.error("监督校验未通过")
                return False
        return True

    def _apply_change(self, code_plan: Any) -> bool:
        if isinstance(code_plan, list):
            for item in code_plan:
                if not self._apply_change(item):
                    return False
            return True
        if not isinstance(code_plan, dict):
            logger.error("应用变更失败: 非法格式")
            return False
        file_path = code_plan.get("file_path")
        content = code_plan.get("content", "")
        if not file_path or not content:
            logger.error("应用变更失败: 空内容")
            return False
        if not self._is_safe_relative_path(file_path):
            logger.error("应用变更失败: 路径不安全 %s", file_path)
            return False
        if not self._sandbox_validate(file_path, content):
            logger.error("沙箱验证失败: %s", file_path)
            return False
        target = PROJECT_ROOT / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = target.with_suffix(target.suffix + ".bak")
            backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        target.write_text(content, encoding="utf-8")
        logger.info("已合并进主项目: %s", target)
        return True

    def _is_safe_relative_path(self, file_path: str) -> bool:
        path = Path(file_path)
        if path.is_absolute():
            return False
        try:
            resolved = (PROJECT_ROOT / path).resolve()
            if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
                return False
            parts = path.parts
            if not parts:
                return False
            return parts[0] in ALLOWED_TARGET_ROOTS
        except Exception:
            return False

    def _sandbox_validate(self, file_path: str, content: str) -> bool:
        sandbox_path = SANDBOX_ROOT / file_path
        sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox_path.write_text(content, encoding="utf-8")
        if not CodeValidator.validate_syntax(content, file_path):
            return False
        if file_path.endswith(".py"):
            try:
                subprocess.run(
                    [sys.executable, "-m", "py_compile", str(sandbox_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except Exception as e:
                logger.error("沙箱编译失败: %s", e)
                return False
            if self._docker_validate(str(sandbox_path), file_path) is False:
                logger.error("Docker 验证失败: %s", file_path)
                return False
        logger.info("沙箱验证通过: %s", sandbox_path)
        return True

    def _docker_validate(self, sandbox_abs_path: str, file_path: str) -> bool:
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                capture_output=True,
                text=True
            )
        except Exception:
            if DOCKER_REQUIRED:
                logger.error("Docker 不可用，无法通过验证")
                return False
            logger.warning("Docker 不可用，跳过 Docker 验证")
            return True

        try:
            subprocess.run(
                ["docker", "image", "inspect", "h2q-sandbox"],
                check=True,
                capture_output=True,
                text=True
            )
        except Exception as e:
            logger.error("Docker 镜像 h2q-sandbox 不可用: %s", e)
            return False

        rel_path = str(Path(file_path))
        try:
            subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{SANDBOX_ROOT}:/app",
                    "-w", "/app",
                    "h2q-sandbox",
                    "python3", "-m", "py_compile", rel_path
                ],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Docker 验证通过: %s", file_path)
            return True
        except Exception as e:
            logger.error("Docker 验证异常: %s", e)
            return False

    def evolve_once(self) -> bool:
        task = self._select_next_task()
        if not task:
            return False
        self._rate_limit()
        logger.info("开始执行任务ID %s", task.get("id"))
        code_plan = self._generate_code_change(task)
        if not code_plan:
            logger.error("代码生成失败，任务未完成")
            self._mark_task_failed(task, "code_generation_failed")
            return True
        if not self._validate_change(code_plan, task):
            logger.error("代码验证失败，任务未完成")
            self._mark_task_failed(task, "validation_failed")
            return True
        if not self._apply_change(code_plan):
            logger.error("应用变更失败，任务未完成")
            self._mark_task_failed(task, "apply_failed")
            return True

        task["status"] = "completed"
        self.state["generation"] = int(self.state.get("generation", 0)) + 1
        self._consecutive_failures = 0
        self._last_success_ts = time.time()
        self.state["consecutive_failures"] = self._consecutive_failures
        self.state["last_success_ts"] = self._last_success_ts
        file_changed = None
        if isinstance(code_plan, list):
            file_changed = [item.get("file_path") for item in code_plan if isinstance(item, dict)]
        elif isinstance(code_plan, dict):
            file_changed = code_plan.get("file_path")
        self.state.setdefault("history", []).append({
            "generation": self.state["generation"],
            "task_id": task.get("id"),
            "file_changed": file_changed,
            "timestamp": time.time()
        })
        self._save_state()
        logger.info("任务完成: %s", task.get("id"))
        return True

    def _rate_limit(self, max_per_minute: int = 5):
        now = time.time()
        while self._action_timestamps and now - self._action_timestamps[0] > 60:
            self._action_timestamps.popleft()
        if len(self._action_timestamps) >= max_per_minute:
            wait = 60 - (now - self._action_timestamps[0])
            if wait > 0:
                logger.info("限速: 等待 %.1f 秒 (每分钟最多 %d 次)", wait, max_per_minute)
                time.sleep(wait)
            now = time.time()
            while self._action_timestamps and now - self._action_timestamps[0] > 60:
                self._action_timestamps.popleft()
        self._action_timestamps.append(time.time())

    def _mark_task_failed(self, task: Dict[str, Any], reason: str):
        retry = int(task.get("retry_count", 0)) + 1
        task["retry_count"] = retry
        task["last_error"] = reason
        self._consecutive_failures += 1
        self.state["consecutive_failures"] = self._consecutive_failures
        self.state["last_failure_ts"] = time.time()
        if retry >= 3:
            task["status"] = "failed"
            logger.error("任务失败并停止重试: %s", task.get("id"))
        else:
            task["status"] = "pending"
            logger.info("任务将重试: %s (retry=%d)", task.get("id"), retry)
        self._save_state()

    def run(self, max_cycles: int = 3, sleep_seconds: int = 2, loop_forever: bool = False):
        gate = self.state.get("benchmark_gate") or self.state.get("last_benchmark_gate")
        if isinstance(gate, dict) and gate.get("passed") is False:
            logger.warning("基准门禁未通过，进化循环停止")
            return
        self._cleanup_pending_tasks()
        if not self.state.get("todo_list"):
            todos = self.generate_todo_list()
            if todos:
                self.inject_todos(todos)

        while True:
            for _ in range(max_cycles):
                if not self.evolve_once():
                    if loop_forever:
                        if self._should_trigger_feedback():
                            logger.warning("持续失败，触发反馈任务生成")
                        logger.info("待办为空，重新生成...")
                        todos = self.generate_todo_list()
                        if todos:
                            self.inject_todos(todos)
                            continue
                    return
                time.sleep(sleep_seconds)
            if not loop_forever:
                break

    def _cleanup_pending_tasks(self):
        updated = False
        for t in self.state.get("todo_list", []):
            if t.get("status") != "pending":
                continue
            targets = t.get("target_files", [])
            if not targets:
                t["status"] = "failed"
                t["last_error"] = "missing_targets"
                updated = True
                continue
            if not any(self._is_safe_relative_path(p) for p in targets):
                t["status"] = "failed"
                t["last_error"] = "unsafe_targets"
                updated = True
        if updated:
            self._save_state()

    def _parse_json_payload(self, raw: str, expect_array: bool) -> Optional[Any]:
        candidates: List[str] = []
        if "```json" in raw:
            start = raw.find("```json") + len("```json")
            end = raw.find("```", start)
            if end != -1:
                candidates.append(raw[start:end].strip())
        if "```" in raw and not candidates:
            start = raw.find("```") + len("```")
            end = raw.find("```", start)
            if end != -1:
                candidates.append(raw[start:end].strip())
        if not candidates:
            array_start = raw.find("[")
            array_end = raw.rfind("]") + 1
            obj_start = raw.find("{")
            obj_end = raw.rfind("}") + 1
            if array_start != -1 and array_end != -1 and array_end > array_start:
                candidates.append(raw[array_start:array_end])
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                candidates.append(raw[obj_start:obj_end])

        for text in candidates:
            try:
                return json.loads(text)
            except Exception:
                try:
                    value = ast.literal_eval(text)
                    return value
                except Exception:
                    continue
        return None

    def _build_failure_summary(self) -> str:
        recent = []
        for t in reversed(self.state.get("todo_list", [])):
            if t.get("status") == "failed":
                recent.append(f"id={t.get('id')} reason={t.get('last_error')}")
            if len(recent) >= 5:
                break
        if not recent:
            return "无"
        return "\n".join(recent)

    def _should_trigger_feedback(self) -> bool:
        if self._consecutive_failures >= 5:
            return True
        if self._last_success_ts and (time.time() - self._last_success_ts) > 300:
            return True
        return False

    def _dump_raw_response(self, tag: str, raw: str):
        try:
            dump_path = SANDBOX_ROOT / f"last_response_{tag}.txt"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(raw, encoding="utf-8")
            logger.info("已保存原始响应: %s", dump_path)
        except Exception as e:
            logger.error("保存原始响应失败: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Heuristic Module Evolution Driver")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--sleep", type=int, default=2)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    driver = HeuristicEvolutionDriver()
    driver.run(max_cycles=args.max_cycles, sleep_seconds=args.sleep, loop_forever=args.loop)
