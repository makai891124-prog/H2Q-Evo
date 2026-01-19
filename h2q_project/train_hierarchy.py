# train_hierarchy.py (混合语料版)

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from tools.byte_loader import get_byte_dataloader
from h2q.hierarchical_system import H2Q_Hierarchical_System

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
SEQ_LEN = 256
LR = 1e-3
STEPS = 2000
SPELLING_WEIGHTS = "h2q_model_knot.pth" 
SAVE_PATH = "h2q_model_hierarchy.pth"
CORPUS_PATH = "mix_corpus.txt" # [关键修改]

def train_hierarchy():
    print(f"🚀 [H2Q-Hierarchy] 启动混合语料 L1 训练... 设备: {DEVICE}")
    
    if not os.path.exists(SPELLING_WEIGHTS):
        print("❌ 错误: 请先运行 train_knot.py")
        return

    model = H2Q_Hierarchical_System(vocab_size=257, dim=256, spelling_weights_path=SPELLING_WEIGHTS)
    model.to(DEVICE)
    
    train_loader = get_byte_dataloader(file_path=CORPUS_PATH, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR)
    
    model.train() 
    progress_bar = tqdm(range(STEPS), desc="L1 Training")
    data_iter = iter(train_loader)
    contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        inputs = batch.to(DEVICE)
        pred_concepts, true_concepts = model(inputs)
        
        preds = pred_concepts[:, :-1, :].reshape(-1, 256)
        targets = true_concepts[:, 1:, :].reshape(-1, 256)
        
        indices = torch.randperm(targets.shape[0]).to(DEVICE)
        neg_targets = targets[indices]
        
        target_labels = torch.ones(preds.shape[0]).to(DEVICE)
        loss_align = contrastive_loss_fn(preds, targets, target_labels)
        
        neg_labels = -1 * torch.ones(preds.shape[0]).to(DEVICE)
        loss_contrast = contrastive_loss_fn(preds, neg_targets, neg_labels)
        
        loss = loss_align + loss_contrast
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    print("✅ L1 训练完成。")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 L1 权重已保存: {SAVE_PATH}")

if __name__ == "__main__":
    train_hierarchy()