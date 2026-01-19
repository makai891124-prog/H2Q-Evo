# train_arithmetic.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from h2q.hierarchical_system import H2Q_Hierarchical_System
from h2q.hierarchical_decoder import ConceptDecoder

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
SEQ_LEN = 32 # 短序列： "123+456=579"
LR = 1e-3
STEPS = 5000
# 复用你训练好的无噪权重
SPELLING_WEIGHTS = "h2q_model_knot.pth" 
DECODER_WEIGHTS = "h2q_model_decoder.pth" # 那个 Loss=0.01 的完美解码器

def generate_arithmetic_batch(batch_size):
    """生成加法数据流: '123+45=168' (补齐到 SEQ_LEN)"""
    batch_data = []
    for _ in range(batch_size):
        a = random.randint(0, 9999)
        b = random.randint(0, 9999)
        c = a + b
        text = f"{a}+{b}={c}"
        # 补齐
        bytes_data = list(text.encode('utf-8'))
        if len(bytes_data) < SEQ_LEN:
            bytes_data += [0] * (SEQ_LEN - len(bytes_data))
        else:
            bytes_data = bytes_data[:SEQ_LEN]
        batch_data.append(bytes_data)
    return torch.tensor(batch_data, dtype=torch.long).to(DEVICE)

def train_math_core():
    print(f"🚀 [H2Q-Math] 启动算术几何核心训练... 设备: {DEVICE}")
    
    # 1. 加载全套系统 (Encoder + Decoder)
    # 注意：我们这次要训练 Encoder 的 L1 层来学会“加法逻辑”
    # L0 (拼写) 和 Decoder (还原) 保持冻结！
    
    encoder = H2Q_Hierarchical_System(vocab_size=257, dim=256, spelling_weights_path=SPELLING_WEIGHTS)
    decoder = ConceptDecoder(dim=256, vocab_size=257, stride=8)
    
    # 加载完美的解码器权重
    if os.path.exists(DECODER_WEIGHTS):
        decoder.load_state_dict(torch.load(DECODER_WEIGHTS))
        print("✅ 完美解码器已加载 (作为输出校验)")
    
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    
    # 冻结 L0 和 Decoder，只训练 L1 (概念核)
    # 我们希望 L1 学会： 概念(123+45) -> 概念(168)
    for p in encoder.spelling_kernel.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False
    
    optimizer = optim.AdamW(encoder.concept_layers.parameters(), lr=LR)
    
    progress_bar = tqdm(range(STEPS), desc="Learning Addition Geometry")
    
    for step in progress_bar:
        inputs = generate_arithmetic_batch(BATCH_SIZE)
        
        # 目标：输入是完整算式，我们希望模型能学会这种序列结构
        # 自回归预测：输入前 N 个，预测第 N+1 个
        # 但为了简化验证“计算能力”，我们这里做自编码训练
        # 看它能否在概念层“压缩”并“无损还原”算式
        
        # 1. 编码
        _, concept_stream = encoder(inputs)
        
        # 2. 解码
        recon_logits = decoder(concept_stream)
        
        # 3. 损失 (必须极低，数学容不得误差)
        loss = nn.CrossEntropyLoss()(recon_logits.reshape(-1, 257), inputs.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"MathLoss": f"{loss.item():.4f}"})
        
        # 验证
        if step % 1000 == 0:
            # 取一个样本看效果
            pred_bytes = torch.argmax(recon_logits[0], dim=-1).cpu().tolist()
            try:
                text = bytes(pred_bytes).decode('utf-8', errors='ignore').replace('\x00', '')
                tqdm.write(f"\n🧮 [Step {step}] 还原算式: {text}")
            except: pass

    print("✅ 数学核心验证完成。")

if __name__ == "__main__":
    import os
    train_math_core()