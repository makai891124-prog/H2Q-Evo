import torch
import os
import sys

class H2QCoderLM(torch.nn.Module):
    def __init__(self, vocab_size=257, embed_dim=256, n_heads=4, n_layers=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # 【核心修复】调用 self.out，而不是返回它
        return self.out(x)

class H2QCoder:
    def __init__(self, checkpoint_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = H2QCoderLM().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        prompt = f"[INST] {prompt} [CODE] "
        tokens = list(prompt.encode('utf-8', errors='ignore'))
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                logits = self.model(input_ids) # 现在 logits 是一个 Tensor
                next_token = torch.argmax(logits[0, -1, :]).item()
                if next_token == 0: break
                tokens.append(next_token)
        
        full_text = bytes(tokens).decode('utf-8', errors='ignore')
        return full_text.split("[CODE] ", 1)[-1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    coder = H2QCoder("checkpoints/h2q_coder_v1.pt")
    print(coder.generate(args.prompt))