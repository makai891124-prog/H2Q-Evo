# train_reversible_vision.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import psutil
import os

# 导入可逆内核和视觉数据加载器
from h2q.reversible_kernel import H2Q_Reversible_Kernel
from tools.vision_loader import get_vision_dataloader

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128 # [升级] 尝试更大的 Batch Size，测试内存优化效果
SEQ_LEN = 3072  # CIFAR-10 (32*32*3)
LR = 5e-4
STEPS = 3000
VOCAB_SIZE = 257

# 保存路径
VISION_KERNEL_PATH = "h2q_vision_reversible_kernel.pth"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def train_reversible():
    print(f"🚀 [H2Q-Reversible] 启动内存优化视觉训练... 设备: {DEVICE}")
    
    # 1. 初始化可逆模型
    model = H2Q_Reversible_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=6).to(DEVICE)
    
    # 2. 数据
    loader = get_vision_dataloader(split="train", batch_size=BATCH_SIZE)
    
    # 3. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    progress_bar = tqdm(range(STEPS), desc="Reversible Vision Training")
    data_iter = iter(loader)
    
    initial_mem = get_memory_usage()
    print(f"   初始内存: {initial_mem:.2f} MB")
    
    for step in progress_bar:
        try:
            batch = next(data_iter).to(DEVICE)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter).to(DEVICE)
            
        # 前向传播 (内部使用 checkpoint)
        logits, stab = model(batch)
        
        # 损失
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), batch.reshape(-1))
        total_loss = loss + 0.1 * stab
        
        # 反向传播 (PyTorch 会自动重计算)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 监控内存
        if step % 100 == 0:
            mem_usage = get_memory_usage()
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Mem": f"{mem_usage:.2f} MB"
            })

    final_mem = get_memory_usage()
    print("✅ 训练完成。")
    print(f"   最终内存: {final_mem:.2f} MB")
    print(f"   内存增量: {final_mem - initial_mem:.2f} MB (相比非可逆版本应显著降低)")
    
    # 保存权重
    torch.save(model.state_dict(), VISION_KERNEL_PATH)
    print(f"💾 可逆视觉核已保存: {VISION_KERNEL_PATH}")

if __name__ == "__main__":
    # 安装 psutil
    try:
        import psutil
    except ImportError:
        print("请先安装 psutil: pip install psutil")
    else:
        train_reversible()