# train_vision_core.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from h2q.knot_kernel import H2Q_Knot_Kernel
from h2q.hierarchical_decoder import ConceptDecoder
from tools.vision_loader import get_vision_dataloader

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64 
SEQ_LEN = 3072 # CIFAR-10 (32*32*3)
LR = 5e-4
STEPS = 3000
VOCAB_SIZE = 257 # 像素值 0-255

# 保存路径
VISION_KERNEL_PATH = "h2q_vision_kernel.pth"
VISION_DECODER_PATH = "h2q_vision_decoder.pth"

def train_vision():
    print(f"🚀 [H2Q-Vision] 启动 CIFAR-10 视觉流形训练... 设备: {DEVICE}")
    
    # 1. 初始化视觉系统
    # L0 拼写核：负责理解像素间的拓扑关系
    kernel = H2Q_Knot_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=6).to(DEVICE)
    
    # Decoder：负责还原像素
    # Stride=1: 先做 1:1 无损还原验证
    decoder = ConceptDecoder(dim=256, vocab_size=VOCAB_SIZE, stride=1).to(DEVICE)
    
    # 2. 数据
    # 首次运行会自动下载 CIFAR-10 (~160MB)
    loader = get_vision_dataloader(split="train", batch_size=BATCH_SIZE)
    
    # 3. 优化器
    optimizer = optim.AdamW(list(kernel.parameters()) + list(decoder.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    kernel.train()
    decoder.train()
    
    progress_bar = tqdm(range(STEPS), desc="Vision Topology Learning")
    data_iter = iter(loader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter).to(DEVICE)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter).to(DEVICE)
            
        # 编码
        features, stab = kernel(batch, return_features=True)
        
        # 解码
        logits = decoder(features)
        
        # 损失
        recon_loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), batch.reshape(-1))
        total_loss = recon_loss + 0.01 * stab
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Loss": f"{recon_loss.item():.4f}", "Stab": f"{stab.item():.4f}"})

    print("✅ 视觉核心训练完成。")
    torch.save(kernel.state_dict(), VISION_KERNEL_PATH)
    torch.save(decoder.state_dict(), VISION_DECODER_PATH)
    print(f"💾 权重已保存。")

if __name__ == "__main__":
    train_vision()