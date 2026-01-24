import os
import sys
import json
import time
import shutil
import re
import asyncio
import logging
import ast
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
load_dotenv() 

from google import genai
from google.genai import types
import docker
import aiofiles

try:
    from m24_protocol import apply_m24_wrapper
    from project_graph import generate_interface_map
    from task_schema import EvolutionTask
except ImportError:
    pass

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("evolution.log"), logging.StreamHandler()]
)
logger = logging.getLogger("H2Q-Evo")

class Config:
    API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "./h2q_project")).resolve()
    DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "h2q-sandbox")
    MEMORY_FILE = "project_memory.json"
    STATE_FILE = "evo_state.json"
    DOCKER_MEM_LIMIT = "8g"
    MAX_RETRIES = 3
    INFERENCE_MODE = os.getenv("INFERENCE_MODE", "api").lower()

class CodeValidator:
    @staticmethod
    def validate_syntax(code: str, filename: str) -> bool:
        if not filename.endswith('.py'): return True
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax Error in {filename}: {e}")
            return False

class H2QNexus:
    def __init__(self):
        logger.info(f"Initializing H2Q-Evo v11.1 [Bootstrap-Fix] | Mode: {Config.INFERENCE_MODE.upper()}")
        self.client = genai.Client(api_key=Config.API_KEY) if Config.API_KEY else None

        # Optional Docker client - don't fail if not available
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False

        self.state = self._load_json(Config.STATE_FILE, {
            "generation": 0, "last_task_id": 0, "todo_list": [], "history": []
        })
        # 确保可以导入 h2q_project 下的统一数学架构
        try:
            sys.path.insert(0, str(Config.PROJECT_ROOT))
        except Exception as e:
            logger.warning(f"Failed to add PROJECT_ROOT to sys.path: {e}")

        self._check_source_integrity()
        # Skip Docker environment check if Docker is not available
        if self.docker_available:
            self._ensure_env()
        else:
            logger.info("Skipping Docker environment check (Docker not available)")
        self._update_task_gates()

        # 初始化数学架构进化集成（UnifiedH2QMathematicalArchitecture 集成）
        try:
            from h2q_project.src.h2q.core.evolution_integration import create_mathematical_core_for_evolution_system, get_h2q_evolution_integration
            # 说明：通过 Evolution Bridge 间接使用 UnifiedH2QMathematicalArchitecture
            # 关键词：UnifiedH2QMathematicalArchitecture
            self.math_bridge = create_mathematical_core_for_evolution_system(
                dim=256, action_dim=64, project_root=str(Path.cwd())
            )
            self.math_integration = get_h2q_evolution_integration(str(Path.cwd()))
            logger.info("✅ Mathematical architecture integration initialized")
        except Exception as e:
            self.math_bridge = None
            self.math_integration = None
            logger.warning(f"Mathematical integration unavailable: {e}")

    async def local_inference(self, prompt: str) -> str:
        if not self.docker_available:
            logger.info("🐳 Docker not available, falling back to API inference...")
            return await self.api_inference(prompt)
        logger.info("🧠 Using LOCAL H2Q BRAIN for inference...")
        # 直接调用 brain.py，它会加载最新权重并训练一步
        cmd = (
            f"docker run --rm "
            f"-v {Config.PROJECT_ROOT}:/app/h2q_project "
            f"-w /app/h2q_project {Config.DOCKER_IMAGE} "
            f"python3 h2q/core/brain.py --prompt \"{prompt}\""
        )
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return stdout.decode()
        else:
            logger.error(f"❌ Local inference failed:\n{stderr.decode()}")
            raise Exception(f"Local inference failed: {stderr.decode()}")

    async def api_inference(self, prompt: str) -> str:
        """Fallback API inference when Docker is not available."""
        if not self.client:
            raise Exception("No API client available (missing GEMINI_API_KEY)")

        logger.info("🔮 Using API inference (Gemini)...")
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"❌ API inference failed: {e}")
            raise Exception(f"API inference failed: {e}")

    async def run(self):
        life_process = None
        if Config.INFERENCE_MODE == 'local':
            logger.info("🚀 Starting independent Life Cycle process...")
            # 【核心修复】调用 heartbeat.py 脚本，而不是复杂的单行命令
            cmd = (
                f"docker run --rm --name h2q_life_cycle "
                f"-v {Config.PROJECT_ROOT}:/app/h2q_project "
                f"-w /app/h2q_project {Config.DOCKER_IMAGE} "
                f"python3 -u tools/heartbeat.py" # -u 确保日志不缓存
            )
            life_process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            asyncio.create_task(self._stream_logs(life_process.stdout, "[LifeCycle]"))
            asyncio.create_task(self._stream_logs(life_process.stderr, "[LifeCycle ERR]"))
            
        try:
            while True:
                # 简化版主循环
                await asyncio.sleep(60) # 在本地模式下，主程序可以轮询得慢一点
                logger.info("Supervisor check...")
                self._update_task_gates()
                # 实际的进化逻辑将由本地模型在后台自我触发（通过 Curiosity 模块）
                # 这里我们保持主程序存活即可
                # 数学架构进化一轮（记录指标，增强可观测性）
                try:
                    if self.math_bridge is not None:
                        import torch
                        state = torch.randn(1, 256)
                        learning_signal = torch.tensor([0.1])
                        results = self.math_bridge(state, learning_signal)
                        # 将关键指标写入状态文件
                        self.state.setdefault("math_metrics", [])
                        self.state["math_metrics"].append({
                            "timestamp": time.time(),
                            "generation": results.get("generation"),
                            "output_norm": results.get("evolution_metrics", {}).get("output_norm", 0.0),
                            "state_change": results.get("evolution_metrics", {}).get("state_change", 0.0),
                        })
                        self._save_json(Config.STATE_FILE, self.state)
                        # 定期保存集成状态
                        if len(self.state["math_metrics"]) % 5 == 0 and self.math_integration is not None:
                            self.math_integration.save_integration_state()
                except Exception as e:
                    logger.warning(f"Mathematical evolution step failed: {e}")
        finally:
            if life_process:
                logger.info("🛑 Shutting down Life Cycle process...")
                try:
                    # 使用 docker stop 命令优雅地停止容器
                    stop_process = await asyncio.create_subprocess_shell(f"docker stop h2q_life_cycle")
                    await stop_process.wait()
                except Exception as e:
                    logger.error(f"Failed to stop container: {e}")

    async def _stream_logs(self, stream, prefix):
        while True:
            line = await stream.readline()
            if not line: break
            logger.info(f"{prefix} {line.decode().strip()}")

    # --- 完整的辅助函数 ---
    def _check_source_integrity(self):
        if not Config.PROJECT_ROOT.exists(): exit(1)
        if not (Config.PROJECT_ROOT / ".git").exists(): subprocess.run(["git", "init"], cwd=Config.PROJECT_ROOT, check=False)

    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        return default

    def _save_json(self, path, data):
        with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)

    def _ensure_env(self):
        if not self.docker_available:
            logger.info("Skipping Docker environment check (Docker not available)")
            return
        try:
            self.docker_client.images.get(Config.DOCKER_IMAGE)
            logger.info(f"Docker image {Config.DOCKER_IMAGE} found")
        except:
            logger.info(f"Building Docker image {Config.DOCKER_IMAGE}...")
            self.docker_client.api.build(path=".", tag=Config.DOCKER_IMAGE, rm=True)
            logger.info(f"Docker image {Config.DOCKER_IMAGE} built successfully")

    def _update_task_gates(self):
        gate_state = self._load_json("honest_evolution_state.json", {})
        gate = gate_state.get("last_benchmark_gate", {})
        passed = gate.get("passed", False)

        self.state["benchmark_gate"] = {
            "passed": passed,
            "public_only": gate.get("public_only", True),
            "min_questions_per_benchmark": gate.get("min_questions_per_benchmark", 0),
            "multi_select_scoring": gate.get("multi_select_scoring", True),
            "timestamp": gate.get("timestamp")
        }

        updated = False
        todos = self.state.get("todo_list", [])
        for task in todos:
            status = task.get("status")
            if not passed and status in ("pending", "in_progress", "ready"):
                task["status"] = "blocked_by_gate"
                updated = True
            if passed and status == "blocked_by_gate":
                task["status"] = "pending"
                updated = True

        if updated:
            self.state["todo_list"] = todos
            self._save_json(Config.STATE_FILE, self.state)

    def _extract_json(self, text):
        try:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            raw = match.group(1) if match else text
            start, end = raw.find('{'), raw.rfind('}') + 1
            if start != -1 and end != -1:
                res = json.loads(raw[start:end])
                return res[0] if isinstance(res, list) else res
        except: pass
        return None

if __name__ == "__main__":
    nexus = H2QNexus()
    asyncio.run(nexus.run())