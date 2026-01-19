# tools/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import os

# --- [中国区加速] ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class H2QTextDataset(Dataset):
    def __init__(self, split="train", seq_len=128, max_samples=None):
        print(f"📚 正在加载 WikiText-2 ({split}) 数据集...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        
        # 加载数据集
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # 预处理：合并所有文本并切分
        text = "\n".join([t for t in dataset["text"] if len(t) > 0])
        tokens = self.tokenizer.encode(text)
        
        # 切分成固定长度的块
        self.examples = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            self.examples.append(tokens[i : i + seq_len])
            if max_samples and len(self.examples) >= max_samples:
                break
                
        print(f"✅ 数据集加载完成: {len(self.examples)} 个样本 (Seq_Len={seq_len})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 返回的是 LongTensor [Seq_Len]
        return torch.tensor(self.examples[idx], dtype=torch.long)

def get_dataloader(split="train", batch_size=8, seq_len=128):
    dataset = H2QTextDataset(split, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # 测试代码
    loader = get_dataloader(batch_size=2, seq_len=32)
    batch = next(iter(loader))
    print(f"Batch Shape: {batch.shape}")
    print(f"Sample: {batch[0]}")