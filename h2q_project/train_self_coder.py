import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 配置 ---
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5e-5
DATASET_FILE = "/app/h2q_evo/h2q_evolution_dataset.jsonl"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "h2q_coder_v1.pt")

class H2QCoderLM(nn.Module):
    def __init__(self, vocab_size=257, embed_dim=256, n_heads=4, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # 【核心修复】调用 self.out，而不是返回它
        return self.out(x)

class CodeDataset(Dataset):
    def __init__(self, file_path, max_len=1024):
        self.samples = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集未找到: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = f"[INST] {data['instruction']} [CODE] {data['output']}"
                    self.samples.append(text)
                except: pass
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = list(text.encode('utf-8', errors='ignore')) + [0]
        tokens = tokens[:1024]
        tokens += [0] * (1024 - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 开始自我编程训练 (在 {device} 上)...")
    dataset = CodeDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = H2QCoderLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            inputs, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)
            optimizer.zero_grad()
            logits = model(inputs) # 现在 logits 是一个 Tensor
            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {i} | Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"✅ 训练完成！本地大脑已保存: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()