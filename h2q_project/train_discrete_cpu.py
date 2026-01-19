# train_discrete_cpu.py

import torch
import torch.nn as nn
from tqdm import tqdm
import psutil
import os

# 导入内核和数据加载器
from h2q.knot_kernel import H2Q_Knot_Kernel
from tools.vision_loader import get_vision_dataloader

# --- 配置 ---
DEVICE = torch.device("cpu") # [核心] 强制使用 CPU
BATCH_SIZE = 32
SEQ_LEN = 1024 # 保持一个合理的序列长度
LR = 1e-3      # 离散梯度需要调整学习率
STEPS = 1000
VOCAB_SIZE = 257
GRAD_THRESHOLD = 1e-5 # 离散导数阈值

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def train_discrete_cpu():
    print(f"🚀 [H2Q-Discrete] 启动 CPU 离散导数训练... 设备: {DEVICE}")
    
    # 1. 初始化模型
    model = H2Q_Knot_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=6).to(DEVICE)
    
    # 2. 数据
    loader = get_vision_dataloader(split="train", batch_size=BATCH_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    progress_bar = tqdm(range(STEPS), desc="Discrete CPU Training")
    data_iter = iter(loader)
    
    initial_mem = get_memory_usage()
    print(f"   初始内存: {initial_mem:.2f} MB")
    
    for step in progress_bar:
        try:
            batch = next(data_iter).to(DEVICE)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter).to(DEVICE)
            
        # --- [核心] 离散梯度手动计算 (在 no_grad 下) ---
        with torch.no_grad():
            
            # 1. 计算基准 Loss
            logits_base, stab_base = model(batch)
            loss_base = loss_fn(logits_base.reshape(-1, VOCAB_SIZE), batch.reshape(-1))
            
            # 2. 遍历参数，计算离散梯度
            for param in model.parameters():
                
                # 创建一个与参数同形的梯度张量
                grad_discrete = torch.zeros_like(param.data)
                
                # 迭代参数的每一个元素 (这很慢，但能精确控制)
                # 优化：我们可以对整个张量进行扰动，然后用 sign()
                
                # --- 优化版：张量级扰动 ---
                
                # a. 计算扰动
                # 我们使用一个小的固定扰动，而不是随机扰动
                perturbation = torch.full_like(param.data, 1e-4)
                
                # b. 计算 L(w + p)
                param.data.add_(perturbation)
                logits_plus, _ = model(batch)
                loss_plus = loss_fn(logits_plus.reshape(-1, VOCAB_SIZE), batch.reshape(-1))
                
                # c. 计算梯度方向
                # grad_direction > 0 -> 增加参数会增加 Loss
                # grad_direction < 0 -> 增加参数会减小 Loss
                grad_direction = loss_plus - loss_base
                
                # d. [核心] 离散化梯度
                # 如果梯度变化大于阈值，则标记为 +1 或 -1
                # 否则标记为 0 (截断)
                grad_discrete[grad_direction > GRAD_THRESHOLD] = 1.0
                grad_discrete[grad_direction < -GRAD_THRESHOLD] = -1.0
                
                # e. 手动更新参数
                # 我们希望减小 Loss，所以要沿着梯度的反方向更新
                param.data.sub_(LR * grad_discrete)
                
                # 还原参数以便下一次迭代 (虽然我们已经更新了，但为了逻辑清晰)
                # 实际上，我们应该先计算所有梯度，再统一更新
                # 这里为了简化，我们采用“在线更新”
                
            # 监控
            if step % 10 == 0:
                mem_usage = get_memory_usage()
                progress_bar.set_postfix({
                    "Loss": f"{loss_base.item():.4f}", 
                    "Mem": f"{mem_usage:.2f} MB"
                })

    final_mem = get_memory_usage()
    print("✅ 训练完成。")
    print(f"   最终内存: {final_mem:.2f} MB")
    print(f"   内存增量: {final_mem - initial_mem:.2f} MB")

if __name__ == "__main__":
    train_discrete_cpu()