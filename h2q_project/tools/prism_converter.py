# tools/prism_converter.py
import os
# --- [中国区加速配置] ---
# 使用 HF-Mirror 镜像站，解决下载连接超时问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ----------------------
import torch
import torch.nn.functional as F
from transformers import AutoModel


def extract_and_crystallize(model_name, target_dim=256, save_path="h2q_memory.pt"):
    print(f"🔮 正在加载开源模型: {model_name} ...")
    try:
        # 我们只需要 Embedding 层，因为它包含了模型对世界的静态认知
        hf_model = AutoModel.from_pretrained(model_name)
        embeddings = hf_model.get_input_embeddings().weight.detach() # Shape: [Vocab, Hidden]
        print(f"   原始维度: {embeddings.shape}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print("📉 正在进行 SVD 降维 (提取核心语义)...")
    # 使用 SVD 将高维语义压缩到我们的 256 维空间
    # U, S, V = torch.svd(embeddings) # 注意：对于大矩阵，这可能很慢
    # 我们使用低秩近似
    U, S, V = torch.svd_lowrank(embeddings, q=target_dim)
    
    # 取前 target_dim 个特征
    compressed_emb = U[:, :target_dim] @ torch.diag(S[:target_dim])
    print(f"   压缩后维度: {compressed_emb.shape}")

    print("🌐 正在进行射影几何映射 (Projective Mapping)...")
    # 核心步骤：强制投影到单位超球面
    projected_emb = F.normalize(compressed_emb, p=2, dim=-1)

    # 由于 H2Q 是基于 Byte (0-255) 的，我们需要将 Token 空间的知识
    # 映射到 Byte 空间。这是一个复杂的话题，
    # 简化版策略：我们只保存这个投影矩阵，作为 DDE 的“外部知识库”
    
    crystal = {
        "source_model": model_name,
        "geometric_embeddings": projected_emb, # [Vocab, 256]
        "projection_matrix": V[:, :target_dim] # 保存投影矩阵，用于转换输入
    }

    torch.save(crystal, save_path)
    print(f"✅ 记忆晶体已保存至: {save_path}")
    print("   这块晶体包含了原模型最核心的语义几何结构。")

if __name__ == "__main__":
    # 你可以换成 'gpt2', 'bert-base-uncased', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    extract_and_crystallize("gpt2")