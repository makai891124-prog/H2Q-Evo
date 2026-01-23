# train_knot.py (混合语料版)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from h2q.system import AutonomousSystem
from tools.byte_loader import get_byte_dataloader

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
SEQ_LEN = 256
LR = 3e-4
STEPS = 3000 # 增加步数
VOCAB_SIZE = 257
CORPUS_PATH = "mix_corpus.txt" # [关键修改]
MODEL_SAVE_PATH = "h2q_model_knot.pth"

def train_knot():
    print(f"🚀 [H2Q-Knot] 启动混合语料 L0 训练... 设备: {DEVICE}")
    
    system = AutonomousSystem(context_dim=256, action_dim=256)
    from h2q.knot_kernel import H2Q_Knot_Kernel
    system.dde.kernel = H2Q_Knot_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=6)
    system.dde.to(DEVICE)
    
    # 加载混合语料
    train_loader = get_byte_dataloader(file_path=CORPUS_PATH, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    optimizer = optim.AdamW(system.dde.parameters(), lr=LR)
    
    system.dde.train()
    progress_bar = tqdm(range(STEPS), desc="L0 Training")
    data_iter = iter(train_loader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        inputs = batch.to(DEVICE)
        logits, stability_loss = system.dde.kernel(inputs)
        task_loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), inputs.reshape(-1))
        total_loss = task_loss + 0.1 * stability_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Task": f"{task_loss.item():.4f}", "Stab": f"{stability_loss.item():.4f}"})

    print("✅ L0 训练完成。")
    torch.save(system.dde.kernel.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 L0 权重已保存: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_knot()