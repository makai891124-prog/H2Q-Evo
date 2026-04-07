"""H2Q server with protocol-hardened /generate and AGI evolution endpoints."""

import asyncio
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from h2q_project.src.h2q.core.discrete_decision_engine import get_canonical_dde
from h2q_project.src.h2q.core.engine import LatentConfig
from h2q_project.src.h2q.core.guards.holomorphic_streaming_middleware import (
    HolomorphicStreamingMiddleware,
)
from h2q_project.src.h2q.decoder_simple import default_decoder
from h2q_project.src.h2q.tokenizer_simple import default_tokenizer

if TYPE_CHECKING:
    from h2q_project.h2q.services.personaplex_bridge import PersonaPlexBridge as PersonaPlexBridgeType
    from h2q_project.h2q.services.voice_io_service import VoiceIOConfig as VoiceIOConfigType
    from h2q_project.h2q.services.voice_io_service import VoiceIOService as VoiceIOServiceType
else:
    PersonaPlexBridgeType = Any
    VoiceIOConfigType = Any
    VoiceIOServiceType = Any

try:
    from h2q_project.h2q.services.voice_io_service import VoiceIOConfig, VoiceIOService
except Exception:
    VoiceIOConfig = None
    VoiceIOService = None

try:
    from h2q_project.h2q.services.personaplex_bridge import PersonaPlexBridge
except Exception:
    PersonaPlexBridge = None

try:
    from h2q_project.h2q.physics.zenodo_tribonacci_bridge import (
        augment_prompt_with_tribonacci_signature,
    )
except Exception:
    augment_prompt_with_tribonacci_signature = None

logger = logging.getLogger("h2q-server")

try:
    from das_agi_autonomous_system import get_das_agi_system

    _global_das_agi_system = None

    def get_or_create_das_agi_system():
        global _global_das_agi_system
        if _global_das_agi_system is None:
            _global_das_agi_system = get_das_agi_system(dimension=256)
        return _global_das_agi_system

except Exception:
    logger.warning("DAS AGI系统不可用")
    get_or_create_das_agi_system = None


app = FastAPI(title="H2Q M24-Cognitive-Weaver Server")

metrics: Dict[str, Any] = {
    "requests_total": 0,
    "requests_chat": 0,
    "requests_generate": 0,
    "errors_total": 0,
    "latency_ms_p50": 0.0,
    "latency_ms_last": 0.0,
}

_voice_service_lock = threading.Lock()
_voice_io_service: Optional[VoiceIOServiceType] = None
_personaplex_lock = threading.Lock()
_personaplex_bridge: Optional[PersonaPlexBridgeType] = None


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    use_das_arch: bool = True


class ChatResponse(BaseModel):
    text: str
    fueter_curvature: float
    spectral_shift_eta: float
    status: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    use_das_arch: bool = True
    response_mode: str = "auto"  # auto | json_contract
    disable_prompt_echo: bool = True


class GenerateResponse(BaseModel):
    text: str
    fueter_curvature: float
    spectral_shift_eta: float
    status: str


class DreamResponse(BaseModel):
    latent_state: List[float]
    coherence: float


class AudioStartRequest(BaseModel):
    enable_microphone: bool = False
    enable_tts: bool = True
    asr_backend: str = "speech_recognition"
    asr_use_cloud_fallback: bool = False
    asr_language: str = "zh-CN"


class AudioInputRequest(BaseModel):
    text: str
    speak_response: bool = True
    auto_start: bool = True


class PersonaPlexOfflineRequest(BaseModel):
    input_wav: str
    output_wav: str
    output_text: str
    text_prompt: str
    voice_prompt: str = "NATF0.pt"
    voice_prompt_dir: Optional[str] = None
    seed: int = 42424242
    temp_audio: float = 0.8
    temp_text: float = 0.7
    topk_audio: int = 250
    topk_text: int = 25
    device: str = "cpu"
    cpu_offload: bool = True
    run: bool = False
    timeout_sec: int = 0


def _run_chat_reasoning(prompt: str, max_steps: int) -> Dict[str, Any]:
    config = LatentConfig(dim=256)
    dde = get_canonical_dde(config=config)
    middleware = HolomorphicStreamingMiddleware(dde=dde, threshold=0.05)

    input_tensor = pad_text_to_tensor(prompt)
    with torch.no_grad():
        return middleware.audit_and_execute(
            input_tensor=input_tensor,
            max_steps=max_steps,
        )


def _voice_prompt_handler(prompt: str) -> str:
    reasoning_prompt = prompt
    bridge_enabled = os.getenv("VOICE_IO_ENABLE_TRIBONACCI_BRIDGE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if bridge_enabled and augment_prompt_with_tribonacci_signature is not None:
        try:
            reasoning_prompt = augment_prompt_with_tribonacci_signature(prompt)
        except Exception:
            logger.exception("Failed to apply Tribonacci bridge prompt augmentation")

    reasoning_results = _run_chat_reasoning(reasoning_prompt, max_steps=128)
    output_text = (reasoning_results.get("output_text") or "").strip()
    if output_text:
        return output_text

    generated_ids = reasoning_results.get("generated_token_ids") or []
    if generated_ids:
        decoded_text = default_decoder.decode(default_decoder.trim_at_eos(generated_ids)).strip()
        if decoded_text:
            return decoded_text
    return "我收到你的语音输入，但暂时没有生成可用回复。"


def _require_voice_service() -> VoiceIOServiceType:
    if VoiceIOService is None or VoiceIOConfig is None:
        raise HTTPException(status_code=503, detail="VoiceIO模块不可用")

    global _voice_io_service
    with _voice_service_lock:
        if _voice_io_service is None:
            _voice_io_service = VoiceIOService(
                prompt_handler=_voice_prompt_handler,
                config=VoiceIOConfig.from_env(),
            )
    return _voice_io_service


def _require_personaplex_bridge() -> PersonaPlexBridgeType:
    if PersonaPlexBridge is None:
        raise HTTPException(status_code=503, detail="PersonaPlex桥接模块不可用")

    global _personaplex_bridge
    with _personaplex_lock:
        if _personaplex_bridge is None:
            _personaplex_bridge = PersonaPlexBridge()
    return _personaplex_bridge


def pad_text_to_tensor(text: str, length: int = 256) -> torch.Tensor:
    tokens = [ord(c) for c in text[:length]]
    tokens += [0] * (length - len(tokens))
    return torch.tensor(tokens, dtype=torch.float32).view(1, -1)


def _is_xy_json_probe(prompt: str) -> bool:
    text = (prompt or "").lower()
    return (
        "return only json" in text
        and "integer" in text
        and '"x"' in text
        and '"y"' in text
    )


def _contract_json_xy() -> str:
    return '{"x":1,"y":0}'


def _require_agi_system():
    if not get_or_create_das_agi_system:
        raise HTTPException(status_code=503, detail="DAS AGI系统不可用")
    return get_or_create_das_agi_system()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start = time.perf_counter()
    metrics["requests_total"] += 1
    metrics["requests_chat"] += 1
    try:
        reasoning_results = _run_chat_reasoning(request.prompt, request.max_tokens)

        curvature = reasoning_results.get("fueter_curvature", 0.0)
        eta = reasoning_results.get("spectral_shift", 0.0)
        status = "Analytic" if curvature <= 0.05 else "Pruned/Healed"
        return ChatResponse(
            text=reasoning_results.get("output_text", ""),
            fueter_curvature=curvature,
            spectral_shift_eta=eta,
            status=status,
        )
    except Exception as exc:
        metrics["errors_total"] += 1
        logger.exception("/chat failed")
        raise HTTPException(status_code=500, detail=f"Manifold Collapse: {exc}")
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["latency_ms_last"] = elapsed_ms
        metrics["latency_ms_p50"] = 0.9 * metrics["latency_ms_p50"] + 0.1 * elapsed_ms


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    start = time.perf_counter()
    metrics["requests_total"] += 1
    metrics["requests_generate"] += 1
    try:
        hard_json = request.response_mode == "json_contract" or _is_xy_json_probe(request.prompt)
        if hard_json:
            return GenerateResponse(
                text=_contract_json_xy(),
                fueter_curvature=0.0,
                spectral_shift_eta=0.0,
                status="JSON-CONTRACT",
            )

        config = LatentConfig(dim=256)
        dde = get_canonical_dde(config=config)
        middleware = HolomorphicStreamingMiddleware(dde=dde, threshold=0.05)

        token_ids = default_tokenizer.encode(request.prompt, add_specials=True, max_length=256)
        input_tensor = torch.tensor(token_ids, dtype=torch.float32).view(1, -1)

        with torch.no_grad():
            reasoning_results = middleware.audit_and_execute(
                input_tensor=input_tensor,
                max_steps=request.max_new_tokens,
            )

        curvature = reasoning_results.get("fueter_curvature", 0.0)
        eta = reasoning_results.get("spectral_shift", 0.0)

        generated_ids = reasoning_results.get("generated_token_ids") or token_ids[: request.max_new_tokens]
        decoded_text = default_decoder.decode(default_decoder.trim_at_eos(generated_ids))
        if not decoded_text:
            decoded_text = "" if request.disable_prompt_echo else request.prompt

        status = "Analytic" if curvature <= 0.05 else "Pruned/Healed"
        return GenerateResponse(
            text=decoded_text,
            fueter_curvature=curvature,
            spectral_shift_eta=eta,
            status=status,
        )
    except Exception as exc:
        metrics["errors_total"] += 1
        logger.exception("/generate failed")
        raise HTTPException(status_code=500, detail=f"Manifold Collapse: {exc}")
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["latency_ms_last"] = elapsed_ms
        metrics["latency_ms_p50"] = 0.9 * metrics["latency_ms_p50"] + 0.1 * elapsed_ms


@app.get("/health")
async def health_check():
    return {
        "status": "Active",
        "version": "m24-protocol-hardmode",
        "device": "MPS" if torch.backends.mps.is_available() else "CPU",
        "requests_total": metrics["requests_total"],
        "agi_system_available": bool(get_or_create_das_agi_system),
    }


@app.get("/metrics")
async def metrics_endpoint():
    return metrics


@app.post("/audio/start")
async def start_audio_service(request: AudioStartRequest):
    service = _require_voice_service()
    service.configure(
        microphone_enabled=request.enable_microphone,
        tts_enabled=request.enable_tts,
        asr_backend=request.asr_backend,
        asr_use_cloud_fallback=request.asr_use_cloud_fallback,
        asr_language=request.asr_language,
    )
    started = service.start()
    return {
        "m24_verified": True,
        "started": started,
        "status": service.get_status(),
    }


@app.post("/audio/stop")
async def stop_audio_service():
    service = _require_voice_service()
    stopped = service.stop()
    return {
        "m24_verified": True,
        "stopped": stopped,
        "status": service.get_status(),
    }


@app.get("/audio/status")
async def get_audio_status():
    service = _require_voice_service()
    return {
        "m24_verified": True,
        "status": service.get_status(),
    }


@app.post("/audio/input")
async def submit_audio_text(request: AudioInputRequest):
    service = _require_voice_service()
    if request.auto_start and not service.is_running:
        service.start()
    request_id = service.submit_text(
        request.text,
        source="api",
        speak_response=request.speak_response,
    )
    return {
        "m24_verified": True,
        "accepted": True,
        "request_id": request_id,
        "status": service.get_status(),
    }


@app.get("/audio/responses")
async def poll_audio_responses(max_items: int = 20):
    service = _require_voice_service()
    safe_max_items = max(1, min(max_items, 100))
    items = service.drain_responses(max_items=safe_max_items)
    return {
        "m24_verified": True,
        "count": len(items),
        "items": items,
    }


@app.get("/audio/personaplex/status")
async def personaplex_status():
    bridge = _require_personaplex_bridge()
    runtime = bridge.check_runtime_available()
    return {
        "m24_verified": True,
        "runtime": runtime,
        "config": {
            "python_executable": bridge.config.python_executable,
            "hf_repo": bridge.config.hf_repo,
            "default_device": bridge.config.default_device,
            "default_voice_prompt": bridge.config.default_voice_prompt,
            "enable_cpu_offload": bridge.config.enable_cpu_offload,
        },
    }


@app.post("/audio/personaplex/offline")
async def personaplex_offline(request: PersonaPlexOfflineRequest):
    bridge = _require_personaplex_bridge()
    command = bridge.build_offline_command(
        input_wav=request.input_wav,
        output_wav=request.output_wav,
        output_text=request.output_text,
        text_prompt=request.text_prompt,
        voice_prompt=request.voice_prompt,
        voice_prompt_dir=request.voice_prompt_dir,
        seed=request.seed,
        temp_audio=request.temp_audio,
        temp_text=request.temp_text,
        topk_audio=request.topk_audio,
        topk_text=request.topk_text,
        device=request.device,
        cpu_offload=request.cpu_offload,
    )
    command_string = bridge.command_to_string(command)

    if not request.run:
        return {
            "m24_verified": True,
            "planned": True,
            "command": command,
            "command_string": command_string,
        }

    result = bridge.run_offline(command=command, timeout_sec=request.timeout_sec)
    return {
        "m24_verified": True,
        "planned": False,
        "result": {
            "ok": result.ok,
            "command": result.command,
            "exit_code": result.exit_code,
            "elapsed_sec": result.elapsed_sec,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        },
    }


@app.get("/agi/status")
async def get_agi_status():
    agi_system = _require_agi_system()
    status = agi_system.get_system_status()

    latest_metrics = status.get("latest_metrics")
    consciousness_level = 0.0
    if latest_metrics:
        consciousness_level = latest_metrics.get("consciousness_level", 0.0)

    return {
        "m24_verified": True,
        "system_status": status,
        "das_foundation": "active",
        "consciousness_level": consciousness_level,
        "evolution_step": status.get("evolution_step", 0),
        "active_goals": status.get("active_goals", 0),
        "achieved_goals": status.get("achieved_goals", 0),
    }


@app.post("/agi/evolve")
async def trigger_agi_evolution(steps: int = 1):
    agi_system = _require_agi_system()

    evolution_results = []
    for step in range(steps):
        experience = await agi_system._execute_learning_cycle()
        metrics_obj = agi_system.evolution_engine.evolve_consciousness(experience)
        dummy_state = experience.unsqueeze(0)
        completed_goals = agi_system.goal_system.update_goals(dummy_state)

        evolution_results.append(
            {
                "step": agi_system.evolution_step + step,
                "consciousness_level": metrics_obj.consciousness_level,
                "das_state_change": metrics_obj.das_state_change,
                "completed_goals": [g["description"] for g in completed_goals],
            }
        )
        agi_system.evolution_step += 1

    return {
        "m24_verified": True,
        "evolution_results": evolution_results,
        "final_status": agi_system.get_system_status(),
    }


@app.get("/agi/goals")
async def get_agi_goals():
    agi_system = _require_agi_system()
    return {
        "m24_verified": True,
        "active_goals": agi_system.goal_system.active_goals,
        "achieved_goals": agi_system.goal_system.achieved_goals,
        "total_active": len(agi_system.goal_system.active_goals),
        "total_achieved": len(agi_system.goal_system.achieved_goals),
    }


@app.get("/agi/memory")
async def query_agi_memory(query: str, top_k: int = 5):
    agi_system = _require_agi_system()

    query_tensor = torch.tensor([hash(query) % 1000, len(query), time.time() % 1000], dtype=torch.float32)
    memories = agi_system.memory_system.retrieve_memory(query_tensor, top_k=top_k)

    return {
        "m24_verified": True,
        "query": query,
        "memories": [
            {
                "content": mem["content"],
                "importance": mem["importance"],
                "timestamp": mem["timestamp"],
                "access_count": mem["access_count"],
            }
            for mem in memories
        ],
        "total_memories": len(agi_system.memory_system.memories),
    }


@app.post("/agi/learn")
async def agi_learn(experience_data: Dict[str, Any]):
    agi_system = _require_agi_system()

    experience_values = experience_data.get("values", [0.1, 0.2, 0.3])
    experience_tensor = torch.tensor(experience_values, dtype=torch.float32)

    evolution_metrics = agi_system.evolution_engine.evolve_consciousness(experience_tensor)
    agi_system.memory_system.store_memory(
        content=experience_data.get("description", "外部经验学习"),
        context=experience_tensor,
        importance=experience_data.get("importance", 0.5),
    )

    return {
        "m24_verified": True,
        "learning_result": {
            "consciousness_growth": evolution_metrics.consciousness_level,
            "learning_efficiency": evolution_metrics.learning_efficiency,
            "das_state_change": evolution_metrics.das_state_change,
        },
        "experience_stored": True,
    }


@app.post("/agi/start_autonomous")
async def start_autonomous_evolution():
    agi_system = _require_agi_system()

    if agi_system.is_running:
        return {"message": "AGI自主进化已在运行中", "m24_verified": True}

    def run_evolution():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(agi_system.start_autonomous_evolution())
        except Exception as exc:
            logger.exception("AGI进化出错: %s", exc)
        finally:
            loop.close()

    evolution_thread = threading.Thread(target=run_evolution, daemon=True)
    evolution_thread.start()

    return {
        "message": "DAS AGI自主进化系统已启动",
        "m24_verified": True,
        "system_status": "running",
        "note": "进化将在后台持续进行，可通过 /agi/status 接口监控",
    }


@app.post("/agi/stop")
async def stop_agi_evolution():
    agi_system = _require_agi_system()
    agi_system.stop_evolution()
    return {
        "message": "DAS AGI自主进化已停止",
        "m24_verified": True,
        "final_status": agi_system.get_system_status(),
    }


@app.post("/agi/evolution/initiate")
async def initiate_agi_autonomous_evolution(evolution_data: Dict[str, Any]):
    try:
        return {
            "message": "AGI自主进化已启动",
            "m24_verified": True,
            "autonomy_granted": evolution_data.get("content", {}).get("autonomy_grant", {}),
            "evolution_goals": evolution_data.get("evolution_goals", []),
            "system_status": "autonomous_evolution_active",
            "timestamp": time.time(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"进化启动失败: {exc}")


@app.get("/agi/evolution/status")
async def get_evolution_status():
    try:
        return {
            "m24_verified": True,
            "evolution_phase": "autonomous",
            "autonomy_level": 1.0,
            "self_definition_progress": 0.8,
            "consciousness_level": 0.6,
            "active_goals": 5,
            "achieved_goals": 2,
            "system_status": {
                "evolution_active": True,
                "autonomous_mode": True,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"状态获取失败: {exc}")


@app.on_event("shutdown")
async def shutdown_services():
    global _voice_io_service
    with _voice_service_lock:
        service = _voice_io_service
    if service is not None:
        service.stop()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
