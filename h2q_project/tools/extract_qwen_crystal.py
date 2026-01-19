# tools/extract_qwen_crystal.py

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os

# --- [中国区加速] ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def extract_crystal():
    # 使用 Qwen2.5-0.5B，它非常小(1GB左右)，但中英代码能力极强
    model_name = "Qwen/Qwen2.5-0.5B"
    save_path = "h2q_qwen_crystal.pt"
    target_dim = 256

    print(f"🔮 正在加载教师模型: {model_name} ...")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("请确保网络通畅，或手动下载模型。")
        return

    # 获取原始嵌入矩阵 [Vocab_Size, Hidden_Dim] (通常是 151936 x 896)
    embeddings = model.get_input_embeddings().weight.detach()
    vocab_size, hidden_dim = embeddings.shape
    print(f"   原始维度: {vocab_size} x {hidden_dim}")

    print("📉 正在进行 SVD 结晶 (提取核心拓扑)...")
    # 使用低秩 SVD 提取最重要的 256 个语义维度
    # float32 精度计算以保证质量
    U, S, V = torch.svd_lowrank(embeddings.float(), q=target_dim)
    
    # 压缩后的嵌入 [Vocab, 256]
    compressed_emb = U @ torch.diag(S)
    
    # 射影归一化 (投影到 H2Q 超球面)
    projected_emb = F.normalize(compressed_emb, p=2, dim=-1)

    crystal = {
        "source": model_name,
        "embeddings": projected_emb.half(), # 转回 fp16 节省空间
        "projection_matrix": V.half(),      # 保存投影矩阵
        "vocab_size": vocab_size
    }

    torch.save(crystal, save_path)
    print(f"✅ 真理晶体已保存: {save_path}")
    print("   这块晶体包含了 Qwen 对中文、英文和代码的全部底层认知。")

if __name__ == "__main__":
    extract_crystal()