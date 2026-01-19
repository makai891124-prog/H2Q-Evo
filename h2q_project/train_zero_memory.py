# train_zero_memory.py

import torch
import torch.nn as nn
from tqdm import tqdm
import psutil
import os
import copy

# 导入内核和数据加载器
from h2q.knot_kernel import H2Q_Knot_Kernel
from tools.vision_loader import get_vision_dataloader

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64 # 我们可以保持一个合理的 Batch Size
SEQ_LEN = 3072
LR = 1e-3       # 有限差分法需要稍大的学习率
STEPS = 500     # 演示 500 步
VOCAB_SIZE = 257
EPSILON = 1e-4  # 扰动大小

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def train_zero_memory():
    print(f"🚀 [H2Q-ZeroMem] 启动零内存梯度训练... 设备: {DEVICE}")
    
    # 1. 初始化模型
    # 我们使用最简单的 Knot Kernel 进行验证
    model = H2Q_Knot_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=6).to(DEVICE)
    
    # 2. 数据
    loader = get_vision_dataloader(split="train", batch_size=BATCH_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    progress_bar = tqdm(range(STEPS), desc="Zero-Memory Training")
    data_iter = iter(loader)
    
    initial_mem = get_memory_usage()
    print(f"   初始内存: {initial_mem:.2f} MB")
    
    for step in progress_bar:
        try:
            batch = next(data_iter).to(DEVICE)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter).to(DEVICE)
            
        # --- [核心] 零内存梯度计算 ---
        # 整个过程在 no_grad 下进行
        with torch.no_grad():
            
            # 遍历模型中的每一个参数
            for param in model.parameters():
                
                # 1. 创建一个用于存储梯度的空间
                if not hasattr(param, 'grad_manual'):
                    param.grad_manual = torch.zeros_like(param.data)
                
                # 2. 迭代计算每个元素的梯度
                # (这是一个非常慢的过程，仅用于验证内存)
                # 优化：我们可以对整个参数张量进行扰动
                
                # 保存原始参数值
                original_val = param.data.clone()
                
                # 3. 计算 L(w + e)
                param.data.add_(EPSILON)
                logits_plus, _ = model(batch)
                loss_plus = loss_fn(logits_plus.reshape(-1, VOCAB_SIZE), batch.reshape(-1))
                
                # 4. 计算 L(w - e)
                # 还原并减去 epsilon
                param.data.copy_(original_val)
                param.data.sub_(EPSILON)
                logits_minus, _ = model(batch)
                loss_minus = loss_fn(logits_minus.reshape(-1, VOCAB_SIZE), batch.reshape(-1))
                
                # 5. 计算梯度
                grad = (loss_plus - loss_minus) / (2 * EPSILON)
                
                # 存储梯度
                # 注意：这里我们计算的是整个参数张量的平均梯度，这是一个简化
                # 真正的有限差分需要对每个元素操作，会更慢
                param.grad_manual.fill_(grad)
                
                # 还原参数
                param.data.copy_(original_val)

            # --- 手动更新权重 ---
            for param in model.parameters():
                param.data.sub_(LR * param.grad_manual)

        # 监控
        if step % 10 == 0:
            mem_usage = get_memory_usage()
            # 我们用 loss_plus 作为当前 loss 的近似值
            progress_bar.set_postfix({
                "Loss": f"{loss_plus.item():.4f}", 
                "Mem": f"{mem_usage:.2f} MB"
            })

    final_mem = get_memory_usage()
    print("✅ 训练完成。")
    print(f"   最终内存: {final_mem:.2f} MB")
    print(f"   内存增量: {final_mem - initial_mem:.2f} MB")

if __name__ == "__main__":
    train_zero_memory()