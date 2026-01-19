# tools/byte_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os

class H2QByteDataset(Dataset):
    def __init__(self, split="train", seq_len=256, file_path=None):
        """
        支持从 HuggingFace 或 本地文件 加载
        """
        self.seq_len = seq_len
        
        if file_path and os.path.exists(file_path):
            print(f"📚 [Byte-Level] 正在加载本地文件: {file_path} ...")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"📚 [Byte-Level] 正在加载 WikiText-2 ({split}) ...")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n".join([t for t in dataset["text"] if len(t) > 0])
        
        self.bytes_data = list(text.encode('utf-8'))
        self.num_samples = len(self.bytes_data) // seq_len
            
        print(f"✅ 字节流加载完成: 总字节 {len(self.bytes_data)}, 样本数 {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.bytes_data[start:end]
        return torch.tensor(chunk, dtype=torch.long)

def get_byte_dataloader(split="train", batch_size=32, seq_len=256, file_path=None):
    dataset = H2QByteDataset(split, seq_len, file_path=file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)