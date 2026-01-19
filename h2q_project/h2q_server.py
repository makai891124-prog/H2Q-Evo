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

app = FastAPI(title="H2Q M24-Cognitive-Weaver Server")

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

        return ChatResponse(
            text=reasoning_results.get("output_text", ""),
            fueter_curvature=curvature,
            spectral_shift_eta=eta,
            status=status
        )

    except Exception as e:
        # Grounding in Reality: Log the error boundary
        print(f"[H2Q_SERVER_ERROR] Boundary Mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manifold Collapse: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "Active", "device": "MPS" if torch.backends.mps.is_available() else "CPU"}
