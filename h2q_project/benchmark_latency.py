# benchmark_latency.py

import torch
import time
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config
from h2q.system import AutonomousSystem
from h2q.knot_kernel import H2Q_Knot_Kernel

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 1
SEQ_LEN = 1024 # 测试长序列，这是 Transformer 的痛点
VOCAB_SIZE = 50257
LOOPS = 50 # 跑50次取平均

def benchmark_model(name, model, input_tensor):
    print(f"\n🔥 测试 {name} ...")
    model.eval()
    
    # 预热 (Warmup)
    with torch.no_grad():
        _ = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if hasattr(torch.backends, 'mps'): torch.mps.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(LOOPS):
            start = time.time()
            _ = model(input_tensor)
            
            if hasattr(torch.backends, 'mps'): torch.mps.synchronize()
            end = time.time()
            times.append(end - start)
            
    avg_time = np.mean(times)
    print(f"   ⏱️ 平均延迟: {avg_time*1000:.2f} ms")
    print(f"   ⚡️ 吞吐量: {BATCH_SIZE * SEQ_LEN / avg_time:.2f} tokens/s")
    return avg_time

def run_comparison():
    print(f"🚀 [H2Q vs GPT-2] 内核延迟基准测试")
    print(f"   配置: Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, Device={DEVICE}")
    
    # 构造输入
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
    
    # --- 1. GPT-2 Large (774M) ---
    # 我们只加载配置，不加载权重，只测计算量
    config = GPT2Config.from_pretrained("gpt2-large")
    gpt2 = GPT2LMHeadModel(config).to(DEVICE)
    time_gpt2 = benchmark_model("GPT-2 Large (774M)", gpt2, dummy_input)
    del gpt2
    
    # --- 2. H2Q Knot Kernel (256 Dim) ---
    # 模拟同样的词表输出
    h2q_sys = AutonomousSystem(context_dim=256, action_dim=256)
    h2q_sys.dde.kernel = H2Q_Knot_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=12)
    h2q_sys.dde.to(DEVICE)
    
    # 包装一下 forward 以匹配接口
    class H2QWrapper(torch.nn.Module):
        def __init__(self, sys):
            super().__init__()
            self.sys = sys
        def forward(self, x):
            return self.sys.dde.kernel(x)
            
    h2q_model = H2QWrapper(h2q_sys)
    time_h2q = benchmark_model("H2Q Knot Kernel (256 Dim)", h2q_model, dummy_input)
    
    # --- 总结 ---
    print(f"\n🏆 最终结果:")
    print(f"   H2Q 比 GPT-2 Large 快: {time_gpt2 / time_h2q:.2f} 倍")
    
    # 估算显存优势 (理论值)
    # GPT2-Large: ~3GB 参数
    # H2Q: ~50MB 参数 (256维)
    print(f"   显存占用优势 (估算): ~60x")

if __name__ == "__main__":
    run_comparison()