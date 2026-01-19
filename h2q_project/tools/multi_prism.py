# tools/multi_prism.py

import torch
import torch.nn.functional as F
from transformers import AutoModel
import os

# --- [中国区加速] ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def generate_crystals():
    model_name = "gpt2"
    dims = [256, 64, 16] # 我们要测试的三个维度
    
    print(f"🔮 正在加载基座模型: {model_name} ...")
    hf_model = AutoModel.from_pretrained(model_name)
    embeddings = hf_model.get_input_embeddings().weight.detach()
    
    print("📉 开始多维 SVD 提取...")
    # 一次性计算最大的 SVD (256)，然后切片即可
    U, S, V = torch.svd_lowrank(embeddings, q=256)
    
    for d in dims:
        print(f"   ⚡️ 处理维度: {d} ...")
        # 切片提取前 d 个特征
        compressed = U[:, :d] @ torch.diag(S[:d])
        
        # 射影归一化
        projected = F.normalize(compressed, p=2, dim=-1)
        
        crystal = {
            "source_model": model_name,
            "dim": d,
            "geometric_embeddings": projected,
            "projection_matrix": V[:, :d]
        }
        
        save_path = f"h2q_memory_{d}.pt"
        torch.save(crystal, save_path)
        print(f"      ✅ 已保存: {save_path}")

if __name__ == "__main__":
    generate_crystals()