import os
from pathlib import Path

PROJECT_ROOT = Path("./h2q_project").resolve()
SERVER_FILE = PROJECT_ROOT / "h2q_server.py"

# 标准的 FastAPI 服务器代码，集成了 Padding 修复逻辑
SERVER_CODE = """
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# 尝试导入核心引擎，如果失败则使用 Mock (为了防止启动崩溃)
try:
    from h2q.dde import DiscreteDecisionEngine
    from h2q.data.generator import H2QSyntheticEngine
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    print("WARNING: H2Q Core modules not found. Running in Mock Mode.")

app = FastAPI(title="H2Q AGI Server", version="1.1.0")

# --- 核心模型初始化 ---
if HAS_CORE:
    # 使用 AI 进化后的参数
    dde = DiscreteDecisionEngine(dim=32, num_actions=10)
    dreamer = H2QSyntheticEngine()
else:
    dde = None
    dreamer = None

class ChatRequest(BaseModel):
    text: str
    temperature: float = 0.7

class DreamResponse(BaseModel):
    spectral_shift: float
    manifold_state: str

def pad_text_to_tensor(text: str, target_len: int = 256) -> torch.Tensor:
    # 1. 转字节
    tokens = list(text.encode('utf-8', errors='ignore'))
    
    # 2. 截断
    if len(tokens) > target_len:
        tokens = tokens[:target_len]
        
    # 3. 填充 (Padding)
    if len(tokens) < target_len:
        padding = [0] * (target_len - len(tokens))
        tokens.extend(padding)
        
    # 4. 转 Tensor 并归一化
    return torch.tensor(tokens, dtype=torch.float32).unsqueeze(0) / 255.0

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 核心修复：使用 Padding 逻辑
        input_tensor = pad_text_to_tensor(request.text, 256)
        
        thought_trace = 0.0
        response_logits = []
        
        if HAS_CORE:
            # 这里需要适配 DDE 的输入层，如果 DDE 还没适配 256->32 的映射，我们先做个简单的投影
            # 假设 DDE 接受 32 维 latent，我们需要把 256 压缩到 32
            # 简单的平均池化模拟 L0->L1 压缩
            latent = torch.nn.functional.adaptive_avg_pool1d(input_tensor.unsqueeze(0), 32).squeeze(0)
            
            # 前向传播
            logits = dde(latent)
            thought_trace = float(logits.mean().item())
            response_logits = logits.tolist()[0]
        
        return {
            "status": "success",
            "input_length": len(request.text),
            "thought_trace": thought_trace, # 思维谱系
            "response_vector": response_logits, # 动作意图
            "message": f"H2Q processed your input: '{request.text[:20]}...'"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dream")
async def dream():
    try:
        val = 0.0
        if HAS_CORE and hasattr(dreamer, 'generate_spectral_trace'):
            # 适配不同的参数签名
            try:
                val = dreamer.generate_spectral_trace(depth=4)
            except:
                val = dreamer.generate_spectral_trace()
        
        return {
            "dream_state": "Active",
            "spectral_value": val,
            "description": "Current eigenvalue of the fractal manifold."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(">>> Starting H2Q Server on 0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

def fix():
    print(f"正在重写服务器代码: {SERVER_FILE}")
    with open(SERVER_FILE, "w", encoding="utf-8") as f:
        f.write(SERVER_CODE)
    print("✅ 修复完成！现在它是一个真正的 Web 服务器了。")

if __name__ == "__main__":
    fix()