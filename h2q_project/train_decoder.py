# train_decoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from tools.byte_loader import get_byte_dataloader
from h2q.hierarchical_system import H2Q_Hierarchical_System
from h2q.hierarchical_decoder import ConceptDecoder

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
SEQ_LEN = 256 # 字符长度
LR = 1e-3
STEPS = 2000
SPELLING_WEIGHTS = "h2q_model_knot.pth"
HIERARCHY_WEIGHTS = "h2q_model_hierarchy.pth" # 上一步保存的层级权重

def train_decoder():
    print(f"🚀 [H2Q-Decoder] 启动概念解码训练... 设备: {DEVICE}")
    
    # 1. 加载已训练好的层级系统 (Encoder)
    encoder = H2Q_Hierarchical_System(vocab_size=257, dim=256, spelling_weights_path=SPELLING_WEIGHTS)
    
    # 加载 L1 概念层权重
    if os.path.exists(HIERARCHY_WEIGHTS):
        print(f"🧊 加载概念层权重: {HIERARCHY_WEIGHTS}")
        encoder.load_state_dict(torch.load(HIERARCHY_WEIGHTS), strict=False)
    else:
        print("⚠️ 未找到概念层权重，解码器将基于随机概念进行训练（效果会差）")
        
    encoder.to(DEVICE)
    encoder.eval() # 编码器全冻结！
    
    # 2. 初始化解码器 (Decoder)
    decoder = ConceptDecoder(dim=256, vocab_size=257, stride=8)
    decoder.to(DEVICE)
    
    # 3. 数据
    train_loader = get_byte_dataloader(split="train", batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    
    # 4. 优化器 (只优化解码器)
    optimizer = optim.AdamW(decoder.parameters(), lr=LR)
    
    # 5. 训练循环
    progress_bar = tqdm(range(STEPS), desc="Decoder Training")
    data_iter = iter(train_loader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        inputs = batch.to(DEVICE) # [B, 256]
        
        # --- Encoder 前向 (获取概念) ---
        with torch.no_grad():
            # 我们只需要 concept_stream (真实的概念流)，不需要 pred_concepts
            _, concept_stream = encoder(inputs) 
            # concept_stream: [B, 32, 256]
            
        # --- Decoder 前向 (重构字符) ---
        # 我们试图从概念流还原回原始的 inputs
        recon_logits = decoder(concept_stream) # [B, 256, 257]
        
        # --- 重构损失 ---
        loss = nn.CrossEntropyLoss()(recon_logits.reshape(-1, 257), inputs.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"ReconLoss": f"{loss.item():.4f}"})
        
        # --- 生成演示 (验证解码能力) ---
        if step % 500 == 0 and step > 0:
            # 取第一个样本的前 32 个字符 (4个概念) 进行展示
            orig_bytes = inputs[0, :32].cpu().tolist()
            
            # 解码预测
            pred_probs = torch.softmax(recon_logits[0, :32], dim=-1)
            pred_bytes = torch.argmax(pred_probs, dim=-1).cpu().tolist()
            
            try:
                orig_str = bytes(orig_bytes).decode('utf-8', errors='ignore').replace('\n', ' ')
                pred_str = bytes(pred_bytes).decode('utf-8', errors='ignore').replace('\n', ' ')
                tqdm.write(f"\n🔍 [Step {step}]")
                tqdm.write(f"   原文: {orig_str}")
                tqdm.write(f"   重构: {pred_str}")
            except: pass

    print("✅ 解码器训练完成。")
    torch.save(decoder.state_dict(), "h2q_model_decoder.pth")

if __name__ == "__main__":
    train_decoder()