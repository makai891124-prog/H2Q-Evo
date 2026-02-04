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
import inspect
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
import docker
import aiofiles

try:
    from h2q_project.fractal_memory import FractalMemory
    from h2q_project.tool_synthesizer import ToolSynthesizer
    from h2q_project.precision_gated_executor import PrecisionGatedExecutor
except Exception:
    FractalMemory = None
    ToolSynthesizer = None
    PrecisionGatedExecutor = None

# DAS和M24核心导入
from h2q_project.das_core import DASCore
from m24_protocol import apply_m24_wrapper
from das_agi_autonomous_system import get_das_agi_system

try:
    from project_graph import generate_interface_map
    from task_schema import EvolutionTask
    from agi_evolution_loss_metrics import (
        AGI_EvolutionLossSystem,
        CapabilityMetrics,
        MathematicalCoreMetrics,
        EvolutionLossComponents
    )
    from deepseek_local_integration import get_deepseek_evolution_integration
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

        # 初始化成本跟踪
        self.cost_savings = 0.0  # DeepSeek本地推理节省的成本
        self.api_costs = 0.0     # API调用产生的成本

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

        # 初始化DAS AGI自主系统
        try:
            self.das_agi_system = get_das_agi_system(dimension=256)
            logger.info("✅ DAS AGI Autonomous System initialized")
        except Exception as e:
            self.das_agi_system = None
            logger.warning(f"DAS AGI System unavailable: {e}")

        # 初始化DAS数学架构进化集成
        try:
            from h2q_project.das_core import create_das_based_architecture
            # DAS架构直接用于进化系统
            self.math_bridge = create_das_based_architecture(dim=256)
            logger.info("✅ DAS mathematical architecture integration initialized")
        except Exception as e:
            self.math_bridge = None
            logger.warning(f"DAS integration unavailable: {e}")

        # 初始化AGI进化损失指标系统
        try:
            self.loss_system = AGI_EvolutionLossSystem()
            logger.info("✅ AGI Evolution Loss Metrics System initialized")
        except Exception as e:
            self.loss_system = None
            logger.warning(f"AGI Evolution Loss System unavailable: {e}")

        # 初始化DeepSeek本地推理集成
        try:
            self.deepseek_integration = get_deepseek_evolution_integration()
            logger.info("✅ DeepSeek Local Integration initialized")
        except Exception as e:
            self.deepseek_integration = None
            logger.warning(f"DeepSeek Integration unavailable: {e}")

        # --- DAS Cycle Integration ---
        self.custom_tools: Dict[str, Any] = {}
        self._llm_wrapper = self._build_sync_llm_wrapper()

        if FractalMemory is not None:
            self.fractal_memory = FractalMemory()
        else:
            self.fractal_memory = None

        if ToolSynthesizer is not None:
            self.tool_synthesizer = ToolSynthesizer(
                llm_client=self._llm_wrapper,
                toolbox_register=self._register_tool,
            )
        else:
            self.tool_synthesizer = None

        if PrecisionGatedExecutor is not None:
            self.precision_executor = PrecisionGatedExecutor(llm_client=self._llm_wrapper)
        else:
            self.precision_executor = None

    def _build_sync_llm_wrapper(self):
        class _LLMWrapper:
            def __init__(self, outer):
                self.outer = outer

            def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
                return {"text": self.outer._sync_generate(prompt, **kwargs)}

        return _LLMWrapper(self)

    def _sync_generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.client:
            return ""

        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 512)
        response = self.client.models.generate_content(
            model=Config.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text

    def _register_tool(self, tool_name: str, tool_fn: Any) -> None:
        self.custom_tools[tool_name] = tool_fn

    def _semantic_vector(self, task: str) -> np.ndarray:
        tokens = task.lower().split()
        length_score = len(tokens)
        math_score = sum(1 for t in tokens if any(k in t for k in ["math", "计算", "solve", "equation", "+", "-", "*", "/"]))
        logic_score = sum(1 for t in tokens if any(k in t for k in ["prove", "logic", "推理", "证明", "therefore", "if", "then"]))
        return np.array([length_score, math_score, logic_score], dtype=np.float64)

    def _execute_custom_tool(self, tool_name: str, query: str) -> Optional[Any]:
        try:
            from h2q_project import custom_tools  # type: ignore
        except Exception:
            try:
                import custom_tools  # type: ignore
            except Exception:
                return None

        if not hasattr(custom_tools, tool_name):
            return None

        tool_fn = getattr(custom_tools, tool_name)
        try:
            sig = inspect.signature(tool_fn)
            if len(sig.parameters) == 0:
                return tool_fn()
            return tool_fn(query)
        except Exception as e:
            logger.warning(f"Tool execution failed: {e}")
            return None

    async def run_das_cycle(self, query: str) -> Dict[str, Any]:
        """
        DAS Cycle:
        Wake -> Recall -> Fast Path -> Precision Check -> Expansion -> Closure
        """
        logger.info("[DAS] Wake Up: Loading FractalMemory & ToolSynthesizer")
        if self.fractal_memory is None or self.tool_synthesizer is None or self.precision_executor is None:
            return {"error": "DAS components unavailable"}

        semantic_vec = self._semantic_vector(query)
        recall = self.fractal_memory.retrieve(query, semantic_vec, top_k=1)

        if recall:
            logger.info("[DAS] Memory Hit -> Using Stored Tool/Logic")
            memory_item = recall[0]
            check = self.precision_executor.execute_with_precision_gating(query)
            if not check.get("probe", {}).get("is_high_entropy", True):
                return {
                    "output": memory_item.solution,
                    "source": "memory",
                    "check": check,
                }

        logger.info("[DAS] Memory Miss -> Coding New Tool")
        synthesis = self.tool_synthesizer.synthesize(query)
        if not synthesis.success:
            return {"error": synthesis.error or "tool_synthesis_failed"}

        logger.info("[DAS] Tool Verified -> Executing")
        tool_output = self._execute_custom_tool(synthesis.tool_name, query)

        check = self.precision_executor.execute_with_precision_gating(str(tool_output))
        if not check.get("probe", {}).get("is_high_entropy", True):
            logger.info("[DAS] Memory Updated")
            self.fractal_memory.store(
                task=query,
                solution=str(tool_output),
                confidence=0.9,
                semantic_vec=semantic_vec,
            )
            return {
                "output": tool_output,
                "source": "synthesized_tool",
                "tool_name": synthesis.tool_name,
                "check": check,
            }

        return {
            "output": tool_output,
            "source": "synthesized_tool_unverified",
            "tool_name": synthesis.tool_name,
            "check": check,
        }

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
        """API推理：优先使用DeepSeek本地推理，节省费用"""
        # 优先尝试DeepSeek本地推理
        if self.deepseek_integration is not None:
            try:
                logger.info("🧠 尝试DeepSeek本地推理...")
                result = await self.deepseek_integration.evolutionary_inference(
                    prompt, task_type='general'
                )

                if result['success']:
                    logger.info("✅ DeepSeek本地推理成功")
                    # 记录成本节省
                    self.cost_savings += 0.001  # 假设每次API调用成本0.001美元
                    return result['response']
                else:
                    logger.warning(f"⚠️ DeepSeek本地推理失败: {result.get('error_message', '未知错误')}")

            except Exception as e:
                logger.warning(f"⚠️ DeepSeek本地推理异常: {e}")

        # 如果DeepSeek不可用或失败，回退到Gemini API
        logger.info("🔮 回退到Gemini API推理...")
        if not self.client:
            raise Exception("DeepSeek本地推理和Gemini API都不可用")

        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            # 记录API使用成本
            self.api_costs += 0.001  # 假设每次API调用成本0.001美元
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini API推理失败: {e}")
            raise Exception(f"所有推理方法都失败: {e}")

    def get_cost_stats(self) -> Dict[str, float]:
        """获取成本统计信息"""
        total_costs = self.api_costs
        net_savings = self.cost_savings - self.api_costs
        return {
            "cost_savings": self.cost_savings,
            "api_costs": self.api_costs,
            "total_costs": total_costs,
            "net_savings": net_savings
        }

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
                        
                        # 计算AGI进化损失指标（如果有的话，暂时跳过）
                        
                        # 将DAS指标写入状态文件
                        self.state.setdefault("das_metrics", [])
                        self.state["das_metrics"].append({
                            "timestamp": time.time(),
                            "generation": results.get("generation", 0),
                            "invariant_distances": results.get("invariant_distances", 0.0),
                            "manifold_size": results.get("manifold_size", 1),
                            "group_hierarchy_depth": results.get("group_hierarchy_depth", 1),
                        })
                        self._save_json(Config.STATE_FILE, self.state)
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

    async def start_das_agi_evolution(self) -> None:
        """
        启动DAS驱动AGI自主进化系统

        这是M24验证的核心功能：真正的AGI自我进化和生长
        """
        if not self.das_agi_system:
            logger.error("DAS AGI系统不可用")
            return

        logger.info("🚀 启动DAS驱动AGI自主进化系统")
        logger.info("M24验证：这不是模拟，而是基于DAS的真实AGI进化")

        try:
            # 保存当前状态
            self._save_json(Config.STATE_FILE, self.state)

            # 启动AGI进化
            await self.das_agi_system.start_autonomous_evolution()

        except Exception as e:
            logger.error(f"DAS AGI进化失败: {e}")
            raise

    def get_das_agi_status(self) -> Dict[str, Any]:
        """
        获取DAS AGI系统状态

        Returns:
            AGI系统状态字典
        """
        if not self.das_agi_system:
            return {"error": "DAS AGI系统不可用"}


def process(prompt: str) -> Dict[str, Any]:
    """
    Synchronous wrapper to run a single DAS cycle.

    Returns:
        Result dict from run_das_cycle.
    """
    nexus = H2QNexus()
    try:
        return asyncio.run(nexus.run_das_cycle(prompt))
    except RuntimeError:
        # If already inside an event loop, fallback to creating a new task
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(nexus.run_das_cycle(prompt))

        return self.das_agi_system.get_system_status()

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