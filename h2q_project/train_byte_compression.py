# train_byte_compression.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from h2q.system import AutonomousSystem
from tools.byte_loader import get_byte_dataloader

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32  # 字节级可以加大 Batch
SEQ_LEN = 256    # 字节级需要更长的序列
LR = 5e-4        # 字节级通常需要稍大的学习率
STEPS = 1000     # 跑 1000 步

def run_byte_training(dim):
    print(f"\n🚀 [Byte-Experiment] 启动维度 Dim={dim} 的训练...")
    
    # 1. 初始化系统 (Vocab=257: 0-255 + EOS)
    # 注意：这里不再加载 GPT-2 晶体，因为那是 Token 级的
    # 我们从零开始训练几何结构
    system = AutonomousSystem(context_dim=dim, action_dim=dim)
    
    # 强制重置内核为 Byte 模式
    from h2q.gut_kernel import H2Q_Geometric_Kernel
    system.dde.kernel = H2Q_Geometric_Kernel(dim=dim, vocab_size=257, depth=12)
    system.dde.to(DEVICE)
    
    # 2. 数据与优化器
    train_loader = get_byte_dataloader(split="train", batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    optimizer = optim.AdamW(system.dde.parameters(), lr=LR)
    
    losses = []
    system.dde.train()
    
    progress_bar = tqdm(range(STEPS), desc=f"Dim={dim}")
    data_iter = iter(train_loader)
    
    for _ in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        inputs = batch.to(DEVICE)
        context = inputs[:, :-1]
        targets = inputs[:, 1:]
        
        logits, _ = system.dde.kernel(context)
        # Vocab = 257
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, 257), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return losses

def main():
    # 我们对比 256 (全维), 64 (4x压缩), 16 (16x压缩)
    dims = [256, 64, 16]
    results = {}
    
    for d in dims:
        results[d] = run_byte_training(d)
        
    print("\n📊 正在绘制字节级对比图...")
    plt.figure(figsize=(12, 8))
    
    colors = {256: 'blue', 64: 'orange', 16: 'red'}
    
    for d in dims:
        plt.plot(results[d], label=f'Dim = {d}', color=colors[d], alpha=0.8, linewidth=1.5)
        
    plt.title("H2Q Byte-Level Dimensional Collapse: The True Isomorphism")
    plt.xlabel("Steps")
    plt.ylabel("Byte-Level Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()