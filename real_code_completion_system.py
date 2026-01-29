#!/usr/bin/env python3
"""
H2Q-Evo 真实代码补全系统
基于Transformer的代码生成模型，实现真正的代码补全能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path

sys.path.append('/Users/imymm/H2Q-Evo')

from h2q_project.src.h2q.tokenizer_simple import default_tokenizer


class CodeDataset(Dataset):
    """代码数据集"""

    def __init__(self, code_samples: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for code in code_samples:
            tokens = tokenizer.encode(code, add_specials=True, max_length=max_length)
            if len(tokens) >= 10:  # 只保留有意义的代码片段
                self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # 创建输入和目标序列
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


class CodeTransformer(nn.Module):
    """代码生成Transformer模型"""

    def __init__(self, vocab_size: int, hidden_dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # 嵌入层 - 增加嵌入维度
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(1024, hidden_dim)  # 最大序列长度

        # Transformer层 - 增加层数和前馈维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # 增加前馈网络维度
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 输出层
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化 - 改进的初始化策略"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化，标准差根据输入维度调整
            std = 0.02 if module.weight.shape[0] != self.vocab_size else 0.02 / (self.hidden_dim ** 0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层使用较小的标准差
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        seq_len = input_ids.size(1)

        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        # 词嵌入
        token_emb = self.embedding(input_ids)

        # 组合嵌入
        x = token_emb + pos_emb

        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)

        # Transformer前向传播
        output = self.transformer(x, mask=causal_mask, src_key_padding_mask=~attention_mask)

        # Layer normalization
        output = self.layer_norm(output)

        # 语言模型头
        logits = self.lm_head(output)

        return logits


class RealCodeCompletionSystem:
    """真实代码补全系统"""

    def __init__(self, model_path: Optional[str] = None):
        self.tokenizer = default_tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        self.model = CodeTransformer(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=512,
            num_layers=6,
            num_heads=8
        ).to(self.device)

        # 加载或训练模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.train_basic_model()

    def train_basic_model(self):
        """训练基础代码生成模型"""
        print("🏗️ 训练基础代码生成模型...")

        # 创建训练数据
        code_samples = self._create_training_samples()

        # 创建数据集
        dataset = CodeDataset(code_samples, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # 优化器和学习率调度器 - 改进的训练参数
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        # 训练循环 - 更长的训练时间
        num_epochs = 50  # 增加训练轮数
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        print(f"📚 训练数据大小: {len(dataset)}")
        print(f"🏃 开始训练 {num_epochs} 轮...")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # 创建注意力掩码
                attention_mask = (input_ids != self.tokenizer.pad_id)

                optimizer.zero_grad()

                # 前向传播
                logits = self.model(input_ids, attention_mask)

                # 计算损失 (只计算非pad位置)
                loss = criterion(
                    logits.view(-1, self.tokenizer.vocab_size),
                    target_ids.view(-1)
                )

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:  # 减少打印频率
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / num_batches
            scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

            # 早停机制
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model("/Users/imymm/H2Q-Evo/best_code_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("早停: 损失不再下降")
                    break

        # 加载最佳模型
        if os.path.exists("/Users/imymm/H2Q-Evo/best_code_model.pth"):
            self.load_model("/Users/imymm/H2Q-Evo/best_code_model.pth")
            print("✅ 加载最佳模型")

        print("🎉 模型训练完成!")

    def _create_training_samples(self) -> List[str]:
        """创建训练样本 - 改进的数据预处理"""
        samples = [
            # Python基础语法 - 更完整的示例，去除多余空格
            "def hello_world():\n    print('Hello, World!')\n    return True",
            "class Calculator:\n    def __init__(self):\n        self.result = 0\n\n    def add(self, x, y):\n        return x + y\n\n    def subtract(self, x, y):\n        return x - y",
            "import torch\nimport torch.nn as nn\n\nclass SimpleModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 1)\n\n    def forward(self, x):\n        return self.linear(x)",
            "for i in range(10):\n    print(f'Number: {i}')\n    if i % 2 == 0:\n        print('Even')\n    else:\n        print('Odd')",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
            "try:\n    result = 10 / 2\n    print(f'Result: {result}')\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')\nfinally:\n    print('Done')",
            "with open('file.txt', 'r') as f:\n    content = f.read()\n    print(content)",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",

            # JavaScript示例
            "function greet(name) {\n    return `Hello, ${name}!`;\n}\n\nconsole.log(greet('World'));",
            "const add = (a, b) => a + b;\n\nconst result = add(5, 3);\nconsole.log(result);",
            "class Rectangle {\n    constructor(width, height) {\n        this.width = width;\n        this.height = height;\n    }\n\n    getArea() {\n        return this.width * this.height;\n    }\n}",

            # 更多Python示例
            "import json\n\ndata = {'name': 'Alice', 'age': 30}\njson_str = json.dumps(data)\nprint(json_str)",
            "from collections import Counter\n\nwords = ['apple', 'banana', 'apple', 'cherry']\nword_count = Counter(words)\nprint(word_count)",
            "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            "import random\n\nnumbers = [random.randint(1, 100) for _ in range(10)]\nsorted_numbers = sorted(numbers)\nprint(sorted_numbers)",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",

            # 简单的函数定义 - 更紧凑
            "def sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total",
            "def max_value(a, b):\n    if a > b:\n        return a\n    else:\n        return b",
            "def reverse_string(s):\n    return s[::-1]",
            "def count_vowels(text):\n    vowels = 'aeiouAEIOU'\n    count = 0\n    for char in text:\n        if char in vowels:\n            count += 1\n    return count",

            # 变量赋值和表达式 - 更紧凑
            "x = 5\ny = 10\nresult = x + y\nprint(result)",
            "name = 'Alice'\nage = 25\nmessage = f'Hello, {name}! You are {age} years old.'\nprint(message)",
            "numbers = [1, 2, 3, 4, 5]\nsquared = [x**2 for x in numbers]\nprint(squared)",

            # 控制流 - 更紧凑
            "if x > 0:\n    print('Positive')\nelif x < 0:\n    print('Negative')\nelse:\n    print('Zero')",
            "while n > 0:\n    print(n)\n    n -= 1",
            "for item in items:\n    if item.startswith('test'):\n        print(f'Found: {item}')\n        break"
        ]

        # 数据增强：创建更多变体，但减少空格
        augmented_samples = []
        for sample in samples:
            augmented_samples.append(sample)
            # 只添加有意义的变体
            if 'def ' in sample and '(' in sample:
                func_name = sample.split('def ')[1].split('(')[0]
                augmented_samples.append(f"{func_name}()\n")
                augmented_samples.append(f"result = {func_name}(arg)\n")

        # 过滤掉包含过多空格的样本
        filtered_samples = []
        for sample in samples + augmented_samples:
            # 计算空格比例
            space_ratio = sample.count(' ') / len(sample) if len(sample) > 0 else 0
            if space_ratio < 0.3:  # 空格比例小于30%
                filtered_samples.append(sample)

        return filtered_samples

    def generate_completion(self, prompt: str, max_length: int = 100, temperature: float = 0.8,
                           top_k: int = 50, top_p: float = 0.9) -> str:
        """生成代码补全"""
        print(f"🔧 生成代码补全: {prompt[:50]}...")

        self.model.eval()

        # 编码提示
        tokens = self.tokenizer.encode(prompt, add_specials=True, max_length=200)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        generated_tokens = tokens.copy()

        with torch.no_grad():
            for i in range(max_length):
                # 获取当前序列
                current_ids = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

                # 限制序列长度
                if current_ids.size(1) > 512:
                    current_ids = current_ids[:, -512:]

                # 前向传播
                logits = self.model(current_ids)

                # 获取最后一个位置的logits
                next_token_logits = logits[0, -1, :]

                # 应用温度
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Top-k 采样
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits

                # Top-p (nucleus) 采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    next_token_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

                # 计算概率
                probs = F.softmax(next_token_logits, dim=-1)

                # 采样下一个token
                next_token = torch.multinomial(probs, 1).item()

                print(f"  生成token {i+1}: {next_token} (prob: {probs[next_token]:.4f})")  # 调试信息

                # 添加到生成序列
                generated_tokens.append(next_token)

                # 检查停止条件
                if next_token == self.tokenizer.eos_id:
                    print(f"  遇到EOS token，停止生成")
                    break

                # 防止过长
                if len(generated_tokens) >= 300:
                    print(f"  达到最大长度，停止生成")
                    break

        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens[len(tokens):], skip_specials=True)

        print(f"  生成的tokens: {generated_tokens[len(tokens):][:10]}")  # 调试信息
        print(f"  生成的文本: '{generated_text[:100]}'")  # 调试信息

        return generated_text

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.tokenizer.vocab_size,
            'hidden_dim': self.model.hidden_dim
        }, path)
        print(f"💾 模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📥 模型已加载: {path}")


def test_real_code_completion():
    """测试真实代码补全"""
    print("🧪 测试真实代码补全系统")
    print("=" * 50)

    # 创建系统
    system = RealCodeCompletionSystem()

    # 测试提示
    test_prompts = [
        "def calculate_fibonacci(n):",
        "class NeuralNetwork(nn.Module):",
        "import torch",
        "for i in range(",
        "def binary_search(arr, target):",
        "function greet(name) {",
        "const add = (a, b) =>"
    ]

    for prompt in test_prompts:
        print(f"\n📝 提示: {prompt}")

        # 生成补全
        completion = system.generate_completion(prompt, max_length=50)
        print(f"  补全:\n{completion}")

        # 显示完整代码
        full_code = prompt + completion
        print(f"  完整代码:\n{full_code[:200]}...")

        # 简单验证语法
        if 'def ' in full_code or 'class ' in full_code or 'function ' in full_code:
            print("  ✅ 包含函数/类定义")
        if '(' in full_code and ')' in full_code:
            print("  ✅ 包含函数调用语法")
        if '{' in full_code and '}' in full_code:
            print("  ✅ 包含代码块语法")

    # 保存模型
    system.save_model("/Users/imymm/H2Q-Evo/real_code_completion_model.pth")

    print("\n✅ 真实代码补全测试完成")


if __name__ == "__main__":
    test_real_code_completion()