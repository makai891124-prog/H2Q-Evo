import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque

try:
    from h2q.data.universal_stream import UniversalStreamLoader
    from h2q.knot_kernel import H2Q_Knot_Kernel
except ImportError as e:
    # Fallback Mocks with DAS integration
    import sys
    sys.path.insert(0, '/app/h2q_project')
    from das_core import create_das_based_architecture

    class H2Q_Knot_Kernel(nn.Module):
        def __init__(self, vocab_size=257, max_dim=256):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, max_dim)
            self.das_arch = create_das_based_architecture(dim=max_dim)
            self.head = nn.Linear(max_dim, vocab_size)

        def forward(self, x):
            # 使用DAS架构增强嵌入
            emb = self.emb(x)
            das_result = self.das_arch(emb.view(-1, emb.size(-1)))
            enhanced_emb = das_result.get('output', emb.view(-1, emb.size(-1))).view(emb.shape)
            return self.head(enhanced_emb), torch.tensor(0.0)

    class UniversalStreamLoader:
        def __iter__(self):
            yield {"l0_indices": torch.randint(0, 256, (1, 512))}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = Path("/app/h2q_project/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "h2q_brain_latest.pt"

class H2QBrain:
    def __init__(self, batch_size=4, lr=1e-4):
        print(f"🧠 [H2Q-Brain] Initializing on {DEVICE}...")
        self.batch_size = batch_size
        self.model = H2Q_Knot_Kernel(vocab_size=257, max_dim=256).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.load_memory()
        self.stream_loader = UniversalStreamLoader()
        self.data_iter = iter(self.stream_loader)
        self.step = 0
        
    def load_memory(self):
        if CHECKPOINT_PATH.exists():
            state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            current = self.model.state_dict()
            filtered = {}
            dropped = []
            for k, v in state_dict.items():
                if k in current and current[k].shape == v.shape:
                    filtered[k] = v
                else:
                    dropped.append(k)
            self.model.load_state_dict(filtered, strict=False)
            if dropped:
                print(f"   ⚠️ Skipped incompatible keys: {len(dropped)}")
    
    def save_memory(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), CHECKPOINT_PATH)
        print(f"   💾 Memory crystal saved.")
        
    def live_and_learn(self):
        try:
            self.model.train()
            
            # 【核心修复】收集样本，组成一个 Batch
            batch_buffer = []
            for _ in range(self.batch_size):
                sample = next(self.data_iter)
                # sample['l0_indices'] 的形状是 [1, 512]
                batch_buffer.append(sample['l0_indices'])

            # 将样本列表堆叠成一个 Batch 张量，形状 [Batch, 512]
            inputs = torch.cat(batch_buffer, dim=0).to(DEVICE)

            self.optimizer.zero_grad()
            targets = inputs[:, 1:]
            logits, stability_loss = self.model(inputs[:, :-1]) 
            
            loss = self.criterion(logits.reshape(-1, 257), targets.reshape(-1))
            total_loss = loss + stability_loss * 0.1
            total_loss.backward()
            self.optimizer.step()
            
            self.step += 1
            if self.step % 10 == 0:
                print(f"   💓 Living... Step {self.step} | Loss: {total_loss.item():.4f}")
            if self.step % 100 == 0:
                self.save_memory()
                
        except StopIteration:
            self.data_iter = iter(self.stream_loader)
        except Exception as e:
            print(f"   ⚠️ Brain function error: {e}")

    # ... (推理方法保持不变)
    def generate_code_plan(self, prompt: str, max_new_tokens: int = 1024) -> str:
        self.model.eval()
        tokens = list(prompt.encode('utf-8'))
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids = torch.tensor(tokens[-256:], dtype=torch.long).unsqueeze(0).to(DEVICE)
                logits, _ = self.model(input_ids)
                next_token = torch.argmax(logits[0, -1, :]).item()
                if next_token == 0: break
                tokens.append(next_token)
        return bytes(tokens).decode('utf-8', errors='ignore')

# ... (接口保持不变)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    
    brain = H2QBrain()
    brain.live_and_learn()
    response = brain.generate_code_plan(args.prompt)
    print(response)