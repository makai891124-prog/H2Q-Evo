# train_fractal.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 导入系统和数据加载器
from h2q.system import AutonomousSystem
from tools.byte_loader import get_byte_dataloader

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32   # 字节级 Batch 可以大一点
SEQ_LEN = 256     # 字节流需要更长的上下文
LR = 3e-4         # 学习率
STEPS = 2000      # 训练步数
VOCAB_SIZE = 257  # 0-255 + EOS

def train_fractal():
    print(f"🚀 [H2Q-Fractal] 启动分形架构验证... 设备: {DEVICE}")
    print("   核心假设: 智能源于 2->256 的对称性破缺展开")
    
    # 1. 初始化系统
    # DDE 会自动调用我们刚刚更新的 H2Q_Geometric_Kernel (包含分形嵌入)
    system = AutonomousSystem(context_dim=256, action_dim=256)
    
    # 强制确保内核参数正确 (Byte-Level)
    from h2q.gut_kernel import H2Q_Geometric_Kernel
    system.dde.kernel = H2Q_Geometric_Kernel(dim=256, vocab_size=VOCAB_SIZE, depth=12)
    system.dde.to(DEVICE)
    
    # 2. 数据准备 (WikiText Byte Stream)
    train_loader = get_byte_dataloader(split="train", batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    
    # 3. 优化器
    optimizer = optim.AdamW(system.dde.parameters(), lr=LR)
    
    # 4. 训练循环
    history = []
    system.dde.train()
    
    progress_bar = tqdm(range(STEPS), desc="Fractal Evolution")
    data_iter = iter(train_loader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        inputs = batch.to(DEVICE)
        context = inputs[:, :-1]
        targets = inputs[:, 1:]
        
        # 前向传播 (触发分形展开)
        logits, _ = system.dde.kernel(context)
        
        # 计算 Loss
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # --- 生成演示 (每500步) ---
        if step % 500 == 0 and step > 0:
            system.dde.eval()
            # 种子: "The " 的字节编码
            seed = torch.tensor([list(b"The ")], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                for _ in range(50): # 生成50个字节
                    logits, _ = system.dde.kernel(seed)
                    next_byte_logits = logits[:, -1, :]
                    probs = torch.nn.functional.softmax(next_byte_logits, dim=-1)
                    next_byte = torch.multinomial(probs, num_samples=1)
                    seed = torch.cat([seed, next_byte], dim=1)
            
            # 解码字节流
            generated_bytes = seed[0].cpu().tolist()
            try:
                text = bytes(generated_bytes).decode('utf-8', errors='ignore')
                tqdm.write(f"\n🔍 [Step {step}] 分形生成: {repr(text)}")
            except:
                pass
            system.dde.train()

    print("✅ 分形验证完成。")
    
    # 5. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='purple', alpha=0.7, label='Fractal Loss')
    plt.title("H2Q Fractal Embedding: Convergence Analysis")
    plt.xlabel("Steps")
    plt.ylabel("Byte-Level Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    train_fractal()