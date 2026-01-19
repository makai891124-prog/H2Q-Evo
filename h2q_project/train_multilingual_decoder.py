# train_multilingual_decoder.py

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
SEQ_LEN = 256
LR = 1e-3
STEPS = 3000 # 多语言需要多练一会儿
VOCAB_SIZE = 257
CORPUS_PATH = "mix_corpus.txt" # [关键] 必须使用混合语料

# 权重路径
SPELLING_WEIGHTS = "h2q_model_knot.pth"
HIERARCHY_WEIGHTS = "h2q_model_hierarchy.pth"
DECODER_SAVE_PATH = "h2q_model_decoder.pth" # 覆盖旧的解码器

def train_decoder():
    print(f"🚀 [H2Q-Decoder] 启动多语言解码训练... 设备: {DEVICE}")
    
    # 1. 加载编码器 (L0 + L1) - 冻结状态
    encoder = H2Q_Hierarchical_System(vocab_size=257, dim=256, spelling_weights_path=SPELLING_WEIGHTS)
    
    if os.path.exists(HIERARCHY_WEIGHTS):
        print(f"🧊 加载概念层权重: {HIERARCHY_WEIGHTS}")
        encoder.load_state_dict(torch.load(HIERARCHY_WEIGHTS), strict=False)
    else:
        print("❌ 严重错误：未找到概念层权重！请先运行 train_hierarchy.py")
        return
        
    encoder.to(DEVICE)
    encoder.eval() # 编码器全冻结
    
    # 2. 初始化解码器 (Stride=8, 3级展开)
    decoder = ConceptDecoder(dim=256, vocab_size=257, stride=8)
    decoder.to(DEVICE)
    
    # 3. 数据加载 (混合语料)
    train_loader = get_byte_dataloader(file_path=CORPUS_PATH, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    
    # 4. 优化器
    optimizer = optim.AdamW(decoder.parameters(), lr=LR)
    
    # 5. 训练循环
    progress_bar = tqdm(range(STEPS), desc="Multilingual Decoding")
    data_iter = iter(train_loader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        inputs = batch.to(DEVICE) # [B, 256]
        
        # --- Encoder 前向 (获取概念流) ---
        with torch.no_grad():
            # concept_stream: [B, 32, 256]
            _, concept_stream = encoder(inputs) 
            
        # --- Decoder 前向 (还原字符) ---
        recon_logits = decoder(concept_stream) # [B, 256, 257]
        
        # --- 损失 ---
        loss = nn.CrossEntropyLoss()(recon_logits.reshape(-1, VOCAB_SIZE), inputs.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Recon": f"{loss.item():.4f}"})
        
        # --- 实时验证 (中英混排) ---
        if step % 500 == 0 and step > 0:
            # 取第一个样本的前 48 个字节 (6个概念)
            orig_bytes = inputs[0, :48].cpu().tolist()
            pred_bytes = torch.argmax(recon_logits[0, :48], dim=-1).cpu().tolist()
            
            try:
                # 使用 replace 避免换行符破坏打印格式
                orig_str = bytes(orig_bytes).decode('utf-8', errors='ignore').replace('\n', '⏎')
                pred_str = bytes(pred_bytes).decode('utf-8', errors='ignore').replace('\n', '⏎')
                tqdm.write(f"\n🔍 [Step {step}]")
                tqdm.write(f"   原文: {orig_str}")
                tqdm.write(f"   还原: {pred_str}")
            except: pass

    print("✅ 多语言解码器训练完成。")
    torch.save(decoder.state_dict(), DECODER_SAVE_PATH)
    print(f"💾 模型已保存至: {DECODER_SAVE_PATH}")

if __name__ == "__main__":
    train_decoder()