# train_multilingual.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from h2q.system import AutonomousSystem
from h2q.hierarchical_decoder import ConceptDecoder
from tools.byte_loader import get_byte_dataloader
import os

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
SEQ_LEN = 256
LR = 3e-4
STEPS = 3000
VOCAB_SIZE = 257
CORPUS_PATH = "mix_corpus.txt"

# 保存路径 (覆盖旧的，或者存新的)
KNOT_PATH = "h2q_model_knot.pth"
DECODER_PATH = "h2q_model_decoder.pth"

def train():
    print(f"🚀 [H2Q-Babel] 启动多语言混合训练... 设备: {DEVICE}")
    
    # 1. 生成语料
    if not os.path.exists(CORPUS_PATH):
        from tools.mix_corpus_generator import generate_mix_corpus
        generate_mix_corpus()
    
    # 2. 初始化系统 (L0 + Decoder)
    # 我们这次同时训练 L0 和 Decoder，为了快速适配
    system = AutonomousSystem(context_dim=256, action_dim=256)
    
    from h2q.knot_kernel import H2Q_Knot_Kernel
    system.dde.kernel = H2Q_Knot_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=6)
    
    decoder = ConceptDecoder(dim=256, vocab_size=VOCAB_SIZE, stride=1) # 注意：先训练1:1还原
    # 为了支持多语言，我们先训练 L0 的自编码能力 (不压缩)，确保 L0 能看懂中文
    # 等 L0 稳定了，再做压缩训练。这里为了演示，我们直接训练 L0 + Decoder(Stride=1)
    
    system.dde.to(DEVICE)
    decoder.to(DEVICE)
    
    # 3. 数据
    train_loader = get_byte_dataloader(file_path=CORPUS_PATH, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    
    # 4. 优化器
    optimizer = optim.AdamW(list(system.dde.parameters()) + list(decoder.parameters()), lr=LR)
    
    progress_bar = tqdm(range(STEPS), desc="Multilingual Training")
    data_iter = iter(train_loader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        inputs = batch.to(DEVICE)
        
        # L0 编码 (获取特征)
        features, stab_loss = system.dde.kernel(inputs, return_features=True)
        
        # Decoder 解码
        logits = decoder(features)
        
        # 损失
        recon_loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), inputs.reshape(-1))
        total_loss = recon_loss + 0.1 * stab_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Recon": f"{recon_loss.item():.4f}"})
        
        if step % 500 == 0:
             # 验证中文还原
            orig_bytes = inputs[0, :20].cpu().tolist()
            pred_bytes = torch.argmax(logits[0, :20], dim=-1).cpu().tolist()
            try:
                print(f"\n原文: {bytes(orig_bytes).decode('utf-8', 'ignore')}")
                print(f"还原: {bytes(pred_bytes).decode('utf-8', 'ignore')}")
            except: pass

    print("✅ 多语言训练完成。")
    torch.save(system.dde.kernel.state_dict(), KNOT_PATH)
    # 注意：这里的 Decoder 是 Stride=1 的，仅用于验证 L0 的表达能力
    # 实际使用中，L0 训练好后，需要重新训练 Stride=8 的 Decoder
    
if __name__ == "__main__":
    train()