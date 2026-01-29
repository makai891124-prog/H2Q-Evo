#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader, Dataset
from h2q_project.src.h2q.tokenizer_simple import default_tokenizer
from core_machine_integrated_completion import CoreMachineCodeTransformer

FILES = ["README.md", "00_START_HERE.md", "DAS_AGI_README.md"]
MAX_LEN = 512
BATCH_SIZE = 2
EPOCHS = 2
MAX_STEPS = 10

class SeqDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        seq = self.chunks[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


def main():
    texts = []
    for f in FILES:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                texts.append(fp.read())

    raw = "\n\n".join(texts)
    if not raw:
        raise SystemExit("no data")

    ids = default_tokenizer.encode(raw, add_specials=True, max_length=100000)
    chunks = [ids[i:i+MAX_LEN] for i in range(0, len(ids)-MAX_LEN, MAX_LEN)]
    dataset = SeqDataset(chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CoreMachineCodeTransformer(
        vocab_size=default_tokenizer.vocab_size,
        hidden_dim=256,
        num_layers=2,
        num_heads=4
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=default_tokenizer.pad_id)

    steps = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for input_ids, target_ids in loader:
            logits = model(input_ids)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps >= MAX_STEPS:
                break
        print(f"epoch {epoch+1} loss {total_loss / max(1, steps):.4f}")
        if steps >= MAX_STEPS:
            break

    print("batch_experiment_done", {"samples": len(dataset), "steps": steps})


if __name__ == "__main__":
    main()
