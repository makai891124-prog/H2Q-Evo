import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- H2Q CORE INTEGRATIONS ---
# Verified via Global Interface Registry
from h2q.core.guards.holomorphic_streaming_middleware import HolomorphicStreamingMiddleware
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.engine import LatentConfig
from h2q.core.sst import SpectralShiftTracker
from h2q.tokenizer_simple import default_tokenizer
from h2q.decoder_simple import default_decoder

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
