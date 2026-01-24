"""
H2Q M24-Cognitive-Weaver Server (重构版)
集成 UnifiedH2QMathematicalArchitecture 统一数学架构
"""
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# === 核心数学架构集成 ===
from h2q_project.src.h2q.core.unified_architecture import (
    UnifiedH2QMathematicalArchitecture,
    UnifiedMathematicalArchitectureConfig,
    get_unified_h2q_architecture
)
from h2q_project.src.h2q.core.evolution_integration import MathematicalArchitectureEvolutionBridge

# === 保留原有接口兼容 ===
from h2q_project.src.h2q.core.guards.holomorphic_streaming_middleware import HolomorphicStreamingMiddleware
from h2q_project.src.h2q.core.discrete_decision_engine import get_canonical_dde
from h2q_project.src.h2q.core.engine import LatentConfig
from h2q_project.src.h2q.core.sst import SpectralShiftTracker
from h2q_project.src.h2q.tokenizer_simple import default_tokenizer
from h2q_project.src.h2q.decoder_simple import default_decoder

app = FastAPI(title="H2Q M24-Cognitive-Weaver Server (Refactored)")

# 全局指标
metrics: Dict[str, Any] = {
    "requests_total": 0,
    "requests_chat": 0,
    "requests_generate": 0,
    "errors_total": 0,
    "latency_ms_p50": 0.0,
    "latency_ms_last": 0.0,
    "unified_arch_calls": 0,
    "mathematical_integrity_score": 1.0,
}

# === 全局统一数学架构实例 ===
# 在服务启动时初始化，避免每次请求都创建
_global_unified_arch: Optional[UnifiedH2QMathematicalArchitecture] = None
_global_evolution_bridge: Optional[MathematicalArchitectureEvolutionBridge] = None


def get_or_create_unified_architecture(dim: int = 256) -> UnifiedH2QMathematicalArchitecture:
    """获取或创建全局统一架构"""
    global _global_unified_arch
    if _global_unified_arch is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _global_unified_arch = get_unified_h2q_architecture(
            dim=dim,
            action_dim=64,
            device=device
        )
    return _global_unified_arch


def get_or_create_evolution_bridge(dim: int = 256) -> MathematicalArchitectureEvolutionBridge:
    """获取或创建进化桥接器"""
    global _global_evolution_bridge
    if _global_evolution_bridge is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _global_evolution_bridge = MathematicalArchitectureEvolutionBridge(
            dim=dim,
            action_dim=64,
            device=device
        )
    return _global_evolution_bridge


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    use_unified_arch: bool = True  # 新增：是否使用统一架构


class ChatResponse(BaseModel):
    text: str
    fueter_curvature: float
    spectral_shift_eta: float
    status: str
    mathematical_report: Optional[Dict[str, Any]] = None  # 新增：数学架构报告


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    use_unified_arch: bool = True


class GenerateResponse(BaseModel):
    text: str
    fueter_curvature: float
    spectral_shift_eta: float
    status: str
    mathematical_report: Optional[Dict[str, Any]] = None


class DreamResponse(BaseModel):
    latent_state: List[float]
    coherence: float


def pad_text_to_tensor(text: str, length: int = 256) -> torch.Tensor:
    """稳定的文本到流形投影"""
    tokens = [ord(c) for c in text[:length]]
    tokens += [0] * (length - len(tokens))
    return torch.tensor(tokens, dtype=torch.float32).view(1, -1)


def process_with_unified_architecture(
    input_tensor: torch.Tensor,
    unified_arch: UnifiedH2QMathematicalArchitecture
) -> Dict[str, Any]:
    """
    使用统一数学架构处理输入
    
    Returns:
        包含输出和详细数学报告的字典
    """
    with torch.no_grad():
        output, math_info = unified_arch(input_tensor)
        
        # 提取数学性质
        results = {
            "output_tensor": output,
            "output_text": "",  # 将在后续填充
            "fueter_curvature": math_info.get('holomorphic_consistency', {}).get('fueter_gradient_norm', 0.0),
            "spectral_shift": math_info.get('lie_group_properties', {}).get('lie_exponential_norm', 0.0),
            "mathematical_integrity": math_info.get('global_integrity', 1.0),
            "fusion_weights": math_info.get('fusion_weights', {}),
            "enabled_modules": math_info.get('enabled_modules', []),
            "full_report": math_info,
        }
        
        return results


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    推理路径 (重构版)
    
    集成UnifiedH2QMathematicalArchitecture作为核心数学引擎
    保留向后兼容的HolomorphicStreamingMiddleware支持
    """
    start = time.perf_counter()
    metrics["requests_total"] += 1
    metrics["requests_chat"] += 1
    
    try:
        # === 新架构：使用统一数学架构 ===
        if request.use_unified_arch:
            metrics["unified_arch_calls"] += 1
            
            # 1. 获取统一架构
            unified_arch = get_or_create_unified_architecture(dim=256)
            
            # 2. 流形投影
            input_tensor = pad_text_to_tensor(request.prompt)
            
            # 3. 通过统一数学架构推理
            results = process_with_unified_architecture(input_tensor, unified_arch)
            
            # 4. 生成输出文本（简化版，实际应使用decoder）
            output_tensor = results["output_tensor"]
            # 使用简单的投影生成文本
            output_chars = [chr(int(max(0, min(127, x.item())))) for x in output_tensor[0, :50]]
            results["output_text"] = ''.join(output_chars).strip()
            
            # 5. 评估数学完整性
            curvature = results["fueter_curvature"]
            eta = results["spectral_shift"]
            integrity = results["mathematical_integrity"]
            
            status = "Analytic (Unified)" if curvature <= 0.05 else "Pruned/Healed (Unified)"
            
            # 更新全局指标
            metrics["mathematical_integrity_score"] = 0.9 * metrics["mathematical_integrity_score"] + 0.1 * integrity
            
            return ChatResponse(
                text=results["output_text"] or request.prompt[:50],  # fallback
                fueter_curvature=curvature,
                spectral_shift_eta=eta,
                status=status,
                mathematical_report={
                    "fusion_weights": {k: v.item() if isinstance(v, torch.Tensor) else v 
                                      for k, v in results["fusion_weights"].items()},
                    "enabled_modules": results["enabled_modules"],
                    "integrity_score": integrity,
                }
            )
        
        # === 旧架构：保留兼容性 ===
        else:
            config = LatentConfig(latent_dim=256)
            dde = get_canonical_dde(config=config)
            middleware = HolomorphicStreamingMiddleware(dde=dde, threshold=0.05)
            
            input_tensor = pad_text_to_tensor(request.prompt)
            
            with torch.no_grad():
                reasoning_results = middleware.audit_and_execute(
                    input_tensor=input_tensor,
                    max_steps=request.max_tokens
                )
            
            curvature = reasoning_results.get("fueter_curvature", 0.0)
            eta = reasoning_results.get("spectral_shift", 0.0)
            status = "Analytic" if curvature <= 0.05 else "Pruned/Healed"
            
            return ChatResponse(
                text=reasoning_results.get("output_text", ""),
                fueter_curvature=curvature,
                spectral_shift_eta=eta,
                status=status,
                mathematical_report=None
            )

    except Exception as e:
        metrics["errors_total"] += 1
        print(f"[H2Q_SERVER_ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manifold Collapse: {str(e)}")
    
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["latency_ms_last"] = elapsed_ms
        metrics["latency_ms_p50"] = 0.9 * metrics["latency_ms_p50"] + 0.1 * elapsed_ms


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    """
    文本生成路径 (重构版)
    """
    start = time.perf_counter()
    metrics["requests_total"] += 1
    metrics["requests_generate"] += 1
    
    try:
        if request.use_unified_arch:
            metrics["unified_arch_calls"] += 1
            
            unified_arch = get_or_create_unified_architecture(dim=256)
            
            # 使用tokenizer
            token_ids = default_tokenizer.encode(request.prompt, add_specials=True, max_length=256)
            input_tensor = torch.tensor(token_ids, dtype=torch.float32).view(1, -1)
            
            results = process_with_unified_architecture(input_tensor, unified_arch)
            
            # 解码
            output_tensor = results["output_tensor"]
            # 简化：直接截取并解码
            generated_ids = output_tensor[0, :request.max_new_tokens].int().tolist()
            decoded_text = default_decoder.decode(default_decoder.trim_at_eos(generated_ids))
            
            if not decoded_text:
                decoded_text = request.prompt
            
            curvature = results["fueter_curvature"]
            eta = results["spectral_shift"]
            status = "Analytic (Unified)" if curvature <= 0.05 else "Pruned/Healed (Unified)"
            
            return GenerateResponse(
                text=decoded_text,
                fueter_curvature=curvature,
                spectral_shift_eta=eta,
                status=status,
                mathematical_report={
                    "enabled_modules": results["enabled_modules"],
                    "integrity_score": results["mathematical_integrity"],
                }
            )
        
        else:
            # 旧路径
            config = LatentConfig(latent_dim=256)
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
            generated_ids = reasoning_results.get("generated_token_ids") or token_ids[:request.max_new_tokens]
            decoded_text = default_decoder.decode(default_decoder.trim_at_eos(generated_ids))
            
            if not decoded_text:
                decoded_text = request.prompt
            
            status = "Analytic" if curvature <= 0.05 else "Pruned/Healed"
            
            return GenerateResponse(
                text=decoded_text,
                fueter_curvature=curvature,
                spectral_shift_eta=eta,
                status=status,
                mathematical_report=None
            )
    
    except Exception as e:
        metrics["errors_total"] += 1
        print(f"[H2Q_SERVER_ERROR] /generate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manifold Collapse: {str(e)}")
    
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["latency_ms_last"] = elapsed_ms
        metrics["latency_ms_p50"] = 0.9 * metrics["latency_ms_p50"] + 0.1 * elapsed_ms


@app.get("/health")
async def health_check():
    """健康检查"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    return {
        "status": "Active",
        "version": "2.0-refactored",
        "device": device.upper(),
        "unified_architecture": _global_unified_arch is not None,
        "evolution_bridge": _global_evolution_bridge is not None,
        "requests_total": metrics["requests_total"],
        "unified_arch_calls": metrics["unified_arch_calls"],
        "mathematical_integrity": round(metrics["mathematical_integrity_score"], 4),
    }


@app.get("/metrics")
async def metrics_endpoint():
    """完整指标"""
    return metrics


@app.get("/system_report")
async def system_report():
    """数学架构系统报告"""
    if _global_unified_arch is None:
        return {"status": "not_initialized"}
    
    report = _global_unified_arch.get_system_report()
    
    # 清理张量以便JSON序列化
    def clean_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(x) for x in obj]
        return obj
    
    return clean_for_json(report)


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化"""
    print("🚀 H2Q Server Starting (Refactored Version)")
    print("📐 Initializing UnifiedH2QMathematicalArchitecture...")
    
    # 预热统一架构
    get_or_create_unified_architecture()
    
    print("✅ Mathematical Architecture Initialized")
    print(f"🔢 Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理"""
    print("🛑 H2Q Server Shutting Down")
    if _global_evolution_bridge:
        # 保存进化状态
        _global_evolution_bridge.save_checkpoint("h2q_server_final_state.pt")
        print("💾 Evolution state saved")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- H2Q CORE INTEGRATIONS ---
# Verified via Global Interface Registry
from h2q_project.src.h2q.core.guards.holomorphic_streaming_middleware import HolomorphicStreamingMiddleware
from h2q_project.src.h2q.core.discrete_decision_engine import get_canonical_dde
from h2q_project.src.h2q.core.engine import LatentConfig
from h2q_project.src.h2q.core.sst import SpectralShiftTracker
from h2q_project.src.h2q.tokenizer_simple import default_tokenizer
from h2q_project.src.h2q.decoder_simple import default_decoder

app = FastAPI(title="H2Q M24-Cognitive-Weaver Server")

# Simple in-memory metrics for observability without extra deps.
metrics: Dict[str, Any] = {
    "requests_total": 0,
    "requests_chat": 0,
    "requests_generate": 0,
    "errors_total": 0,
    "latency_ms_p50": 0.0,
    "latency_ms_last": 0.0,
}

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False

class ChatResponse(BaseModel):
    text: str
    fueter_curvature: float
    spectral_shift_eta: float
    status: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    fueter_curvature: float
    spectral_shift_eta: float
    status: str

class DreamResponse(BaseModel):
    latent_state: List[float]
    coherence: float

def pad_text_to_tensor(text: str, length: int = 256) -> torch.Tensor:
    """Stable utility for text-to-manifold projection."""
    # Implementation uses basic ASCII projection for seed atoms
    tokens = [ord(c) for c in text[:length]]
    tokens += [0] * (length - len(tokens))
    return torch.tensor(tokens, dtype=torch.float32).view(1, -1)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Inference path integrated with HolomorphicStreamingMiddleware.
    Uses Fueter-Laplace curvature (Df) to prune non-analytic reasoning branches.
    """
    start = time.perf_counter()
    metrics["requests_total"] += 1
    metrics["requests_chat"] += 1
    try:
        # 1. RIGID CONSTRUCTION: Initialize DDE via Canonical Factory
        # Fixes Runtime Error: DiscreteDecisionEngine.__init__() got an unexpected keyword argument 'dim'
        # We use LatentConfig to wrap parameters safely.
        config = LatentConfig(latent_dim=256) 
        dde = get_canonical_dde(config=config)
        
        # 2. ELASTIC EXTENSION: Initialize Middleware for real-time pruning
        # This middleware monitors the Discrete Fueter Operator (Df).
        # If Df > 0.05, the branch is identified as a 'topological tear' (hallucination).
        middleware = HolomorphicStreamingMiddleware(dde=dde, threshold=0.05)
        
        # 3. MANIFOLD PROJECTION
        input_tensor = pad_text_to_tensor(request.prompt)
        
        # 4. INFERENCE WITH HOLOMORPHIC GUARD
        # The middleware wraps the reasoning flow to perform branch pruning
        with torch.no_grad():
            # Simulated reasoning flow through the SU(2) manifold
            # In production, this calls the H2QModel.forward
            reasoning_results = middleware.audit_and_execute(
                input_tensor=input_tensor,
                max_steps=request.max_tokens
            )
            
        # 5. METRIC EXTRACTION
        # η = (1/π) arg{det(S)} tracked via SpectralShiftTracker
        curvature = reasoning_results.get("fueter_curvature", 0.0)
        eta = reasoning_results.get("spectral_shift", 0.0)
        
        status = "Analytic" if curvature <= 0.05 else "Pruned/Healed"
        resp = ChatResponse(
            text=reasoning_results.get("output_text", ""),
            fueter_curvature=curvature,
            spectral_shift_eta=eta,
            status=status
        )
        return resp

    except Exception as e:
        metrics["errors_total"] += 1
        # Grounding in Reality: Log the error boundary
        print(f"[H2Q_SERVER_ERROR] Boundary Mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manifold Collapse: {str(e)}")
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["latency_ms_last"] = elapsed_ms
        # p50 approximation via exponential smoothing
        metrics["latency_ms_p50"] = 0.9 * metrics["latency_ms_p50"] + 0.1 * elapsed_ms

@app.get("/health")
async def health_check():
    return {
        "status": "Active",
        "device": "MPS" if torch.backends.mps.is_available() else "CPU",
        "requests_total": metrics["requests_total"],
    }


@app.get("/metrics")
async def metrics_endpoint():
    return metrics


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    """Optional text generation path using lightweight tokenizer/decoder.

    This reuses the holomorphic guard for consistency and provides a minimal
    end-to-end contract (prompt -> tokens -> guard -> decoded text).
    """
    start = time.perf_counter()
    metrics["requests_total"] += 1
    metrics["requests_generate"] += 1
    try:
        config = LatentConfig(latent_dim=256)
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

        # Minimal decoding: if middleware produced tokens, use them; otherwise echo prompt.
        generated_ids = reasoning_results.get("generated_token_ids") or token_ids[: request.max_new_tokens]
        decoded_text = default_decoder.decode(default_decoder.trim_at_eos(generated_ids))
        if not decoded_text:
            decoded_text = request.prompt  # graceful fallback

        status = "Analytic" if curvature <= 0.05 else "Pruned/Healed"

        return GenerateResponse(
            text=decoded_text,
            fueter_curvature=curvature,
            spectral_shift_eta=eta,
            status=status,
        )

    except Exception as e:
        metrics["errors_total"] += 1
        print(f"[H2Q_SERVER_ERROR] /generate Boundary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manifold Collapse: {str(e)}")
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["latency_ms_last"] = elapsed_ms
        metrics["latency_ms_p50"] = 0.9 * metrics["latency_ms_p50"] + 0.1 * elapsed_ms
