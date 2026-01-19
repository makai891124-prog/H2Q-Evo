import torch
from datasets import load_dataset, interleave_datasets
from typing import Iterator, Dict

class UniversalStreamLoader:
    def __init__(self, sequence_length: int = 512):
        self.sequence_length = sequence_length
        
        math_ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        code_ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

        self.combined_stream = interleave_datasets([math_ds, code_ds])

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = ""
        
        for example in self.combined_stream:
            text = example.get('text', example.get('content', ''))
            if text and isinstance(text, str):
                buffer += text
            
            # 只要 buffer 长度足够，就不断切片
            while len(buffer) >= self.sequence_length:
                # 1. 精准切片
                chunk = buffer[:self.sequence_length]
                buffer = buffer[self.sequence_length:]
                
                # 2. 编码
                tokens = list(chunk.encode('utf-8', errors='ignore'))
                
                # 【核心修复】强制截断或填充到完全一致的长度
                # 编码后长度可能 > sequence_length
                if len(tokens) > self.sequence_length:
                    tokens = tokens[:self.sequence_length]
                # 编码后长度可能 < sequence_length
                elif len(tokens) < self.sequence_length:
                    tokens += [0] * (self.sequence_length - len(tokens))
                
                # 3. 输出带 Batch 维度的张量
                yield {"l0_indices": torch.tensor(tokens, dtype=torch.long).unsqueeze(0)}