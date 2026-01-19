import os
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
        self.docker_client = docker.from_env()
        self.state = self._load_json(Config.STATE_FILE, {
            "generation": 0, "last_task_id": 0, "todo_list": [], "history": []
        })
        self._check_source_integrity()
        self._ensure_env()

    async def local_inference(self, prompt: str) -> str:
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
            return ""

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
                # 实际的进化逻辑将由本地模型在后台自我触发（通过 Curiosity 模块）
                # 这里我们保持主程序存活即可
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
        try: self.docker_client.images.get(Config.DOCKER_IMAGE)
        except: self.docker_client.api.build(path=".", tag=Config.DOCKER_IMAGE, rm=True)

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