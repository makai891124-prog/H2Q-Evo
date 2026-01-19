# train_spacetime.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from h2q.spacetime_kernel import H2Q_Spacetime_Kernel
from tools.byte_loader import get_byte_dataloader

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
SEQ_LEN = 64
LR = 5e-4
STEPS = 3000
VOCAB_SIZE = 257

# 数学数据生成器
def get_math_batch(bs):
    data = []
    for _ in range(bs):
        a, b = random.randint(0, 999), random.randint(0, 999)
        text = f"{a}+{b}={a+b}"
        b_data = list(text.encode('utf-8'))
        b_data += [0] * (SEQ_LEN - len(b_data))
        data.append(b_data[:SEQ_LEN])
    return torch.tensor(data, dtype=torch.long).to(DEVICE)

def train():
    print(f"🚀 [H2Q-4D] 启动时空波形演化... 设备: {DEVICE}")
    
    model = H2Q_Spacetime_Kernel(max_dim=256, vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    lang_loader = get_byte_dataloader(split="train", batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    lang_iter = iter(lang_loader)
    
    history = {"math_stab": [], "lang_stab": []}
    
    progress_bar = tqdm(range(STEPS), desc="Spacetime Evolution")
    
    for step in progress_bar:
        # 随机混合训练
        is_math = random.random() > 0.5
        
        if is_math:
            inputs = get_math_batch(BATCH_SIZE)
            task_type = "Math"
        else:
            try:
                inputs = next(lang_iter).to(DEVICE)
            except StopIteration:
                lang_iter = iter(lang_loader)
                inputs = next(lang_iter).to(DEVICE)
            task_type = "Lang"
            
        context = inputs[:, :-1]
        targets = inputs[:, 1:]
        
        logits, stability = model(context)
        
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        # 总损失：任务 + 波形稳定性
        total_loss = loss + 0.1 * stability
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录不同任务的稳定性特征
        if is_math:
            history["math_stab"].append(stability.item())
            # 补齐另一个列表以保持长度一致(用于绘图)
            if len(history["lang_stab"]) > 0:
                history["lang_stab"].append(history["lang_stab"][-1])
            else:
                history["lang_stab"].append(0)
        else:
            history["lang_stab"].append(stability.item())
            if len(history["math_stab"]) > 0:
                history["math_stab"].append(history["math_stab"][-1])
            else:
                history["math_stab"].append(0)
                
        progress_bar.set_postfix({"Loss": f"{loss.item():.3f}", "Stab": f"{stability.item():.4f}"})

    print("✅ 演化完成。")
    
    # 绘图：观察数学和语言的波形稳定性区别
    plt.figure(figsize=(10, 6))
    plt.plot(history["math_stab"], label='Math Waveform Stability', alpha=0.7)
    plt.plot(history["lang_stab"], label='Lang Waveform Stability', alpha=0.7)
    plt.title("H2Q Spacetime: Waveform Characteristics by Domain")
    plt.xlabel("Steps")
    plt.ylabel("Stability (Std Dev of Time Component)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()