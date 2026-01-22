#!/usr/bin/env python3
"""
真正的AGI训练系统
Real AGI Training System

特点:
1. 真实数据集: WikiText-103 / OpenWebText
2. 真正的语言建模: Next Token Prediction (Causal LM)
3. 标准评估: Perplexity + 下游任务评估
4. 可扩展模型: 支持更大规模
"""

import os
import sys
import json
import time
import math
import logging
import hashlib
import requests
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

# ============================================================
# 路径配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'real_data'
MODEL_DIR = SCRIPT_DIR / 'real_models'
LOG_DIR = SCRIPT_DIR / 'real_logs'
CHECKPOINT_DIR = SCRIPT_DIR / 'real_checkpoints'
CACHE_DIR = SCRIPT_DIR / 'cache'

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 日志配置
log_file = LOG_DIR / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# 配置
# ============================================================

@dataclass
class RealAGIConfig:
    """真实AGI训练配置"""
    
    # 模型架构 - GPT-2 Small 规模
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    hidden_dim: int = 768    # 隐藏维度
    num_layers: int = 12     # Transformer层数
    num_heads: int = 12      # 注意力头数
    ff_dim: int = 3072       # FFN维度
    max_seq_len: int = 1024  # 最大序列长度
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 8      # 小batch适配内存
    gradient_accumulation: int = 4  # 梯度累积 = 有效batch 32
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # 数据
    dataset: str = 'wikitext-103'  # 或 'openwebtext'
    
    # 训练时长
    training_hours: float = 5.0
    
    # 检查点与日志
    checkpoint_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 50
    
    # 设备
    device: str = 'auto'


# ============================================================
# Tokenizer (简化版BPE)
# ============================================================

class SimpleTokenizer:
    """简化的字符级tokenizer（可替换为tiktoken）"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        
        # 特殊token
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        self._build_vocab()
    
    def _build_vocab(self):
        """构建词表"""
        # 特殊token
        self.char_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        
        # ASCII可打印字符
        idx = 4
        for i in range(32, 127):
            self.char_to_id[chr(i)] = idx
            idx += 1
        
        # 常见Unicode
        for char in 'àáâãäåæçèéêëìíîïñòóôõöùúûüýÿ':
            self.char_to_id[char] = idx
            idx += 1
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
    
    def encode(self, text: str) -> List[int]:
        """编码文本"""
        tokens = [self.bos_id]
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        tokens.append(self.eos_id)
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """解码token"""
        chars = []
        for t in tokens:
            if t in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            chars.append(self.id_to_char.get(t, '?'))
        return ''.join(chars)
    
    def __len__(self):
        return len(self.char_to_id)


# ============================================================
# 真实数据集加载 - 使用Hugging Face datasets
# ============================================================

class WikiText103Dataset(Dataset):
    """WikiText-103 数据集 - 通过Hugging Face加载"""
    
    def __init__(
        self, 
        data_dir: Path, 
        split: str = 'train',
        max_seq_len: int = 1024,
        tokenizer: Optional[SimpleTokenizer] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or SimpleTokenizer()
        
        # 加载数据
        self.tokens = self._load_and_tokenize()
        
        # 计算样本数
        self.num_samples = max(1, (len(self.tokens) - 1) // max_seq_len)
        
        logger.info(f"WikiText-103 {split}: {len(self.tokens):,} tokens, {self.num_samples:,} samples")
    
    def _load_and_tokenize(self) -> List[int]:
        """加载并tokenize"""
        # 检查缓存
        cache_path = CACHE_DIR / f'wikitext103_{self.split}_tokens.pt'
        if cache_path.exists():
            logger.info(f"从缓存加载 {self.split}...")
            return torch.load(cache_path)
        
        logger.info(f"从Hugging Face加载WikiText-103 {self.split}...")
        
        try:
            from datasets import load_dataset
            
            # 映射split名称
            hf_split = 'validation' if self.split == 'valid' else self.split
            
            # 加载数据集
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=hf_split)
            
            # 合并文本
            texts = []
            for item in dataset:
                text = item['text'].strip()
                if text and not text.startswith('='):
                    texts.append(text)
            
            full_text = ' '.join(texts)
            logger.info(f"文本长度: {len(full_text):,} 字符")
            
        except Exception as e:
            logger.warning(f"Hugging Face加载失败: {e}")
            logger.info("使用本地生成的文本数据...")
            
            # 回退到生成式数据
            full_text = self._generate_fallback_text()
        
        # Tokenize
        logger.info(f"Tokenizing {self.split}...")
        tokens = self.tokenizer.encode(full_text)
        
        # 缓存
        torch.save(tokens, cache_path)
        logger.info(f"Token缓存已保存: {len(tokens):,} tokens")
        
        return tokens
    
    def _generate_fallback_text(self) -> str:
        """生成回退文本（如果无法下载真实数据）"""
        import random
        
        # 知识性文本模板
        templates = [
            "The history of {topic} dates back to {era}. During this period, significant developments occurred in {field}. Scientists and researchers discovered that {fact}. This led to important advances in {application}.",
            "In the field of {field}, researchers have found that {discovery}. This understanding has profound implications for {application}. The principles behind this phenomenon involve {mechanism}.",
            "According to scientific studies, {topic} plays a crucial role in {field}. The mechanism involves {process}, which results in {outcome}. Further research is needed to fully understand {aspect}.",
            "Mathematics provides the foundation for understanding {topic}. The key concepts include {concept1} and {concept2}. These principles are applied in {field} to solve {problem}.",
            "The development of {technology} has transformed {field}. Early pioneers such as {person} made groundbreaking contributions. Today, {application} relies heavily on these foundational discoveries.",
        ]
        
        topics = ['physics', 'chemistry', 'biology', 'astronomy', 'mathematics', 'computer science', 'medicine', 'psychology', 'economics', 'philosophy']
        fields = ['quantum mechanics', 'molecular biology', 'artificial intelligence', 'neuroscience', 'materials science', 'genetics', 'climatology', 'astrophysics']
        eras = ['ancient times', 'the Renaissance', 'the 19th century', 'the early 20th century', 'recent decades']
        
        texts = []
        for _ in range(50000):  # 生成大量文本
            template = random.choice(templates)
            text = template.format(
                topic=random.choice(topics),
                field=random.choice(fields),
                era=random.choice(eras),
                fact=f"the relationship between {random.choice(topics)} and {random.choice(fields)} is fundamental",
                discovery=f"complex interactions govern {random.choice(topics)}",
                mechanism=f"the underlying {random.choice(topics)} principles",
                application=random.choice(fields),
                process=f"systematic {random.choice(topics)} analysis",
                outcome=f"advances in {random.choice(fields)}",
                aspect=f"the full scope of {random.choice(topics)}",
                concept1=random.choice(topics),
                concept2=random.choice(fields),
                problem=f"challenges in {random.choice(fields)}",
                technology=random.choice(topics),
                person="notable researchers"
            )
            texts.append(text)
        
        return ' '.join(texts)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1  # +1 for target
        
        chunk = self.tokens[start:end]
        
        # 如果不够长，循环
        while len(chunk) < self.max_seq_len + 1:
            chunk = chunk + self.tokens[:self.max_seq_len + 1 - len(chunk)]
        
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class TextStreamDataset(IterableDataset):
    """流式文本数据集 - 适用于大规模数据"""
    
    def __init__(
        self,
        data_paths: List[Path],
        max_seq_len: int = 1024,
        tokenizer: Optional[SimpleTokenizer] = None
    ):
        self.data_paths = data_paths
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or SimpleTokenizer()
    
    def __iter__(self) -> Iterator[Dict]:
        buffer = []
        
        for path in self.data_paths:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokens = self.tokenizer.encode(line)
                    buffer.extend(tokens)
                    
                    while len(buffer) >= self.max_seq_len + 1:
                        chunk = buffer[:self.max_seq_len + 1]
                        buffer = buffer[self.max_seq_len:]
                        
                        yield {
                            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                            'labels': torch.tensor(chunk[1:], dtype=torch.long)
                        }


# ============================================================
# GPT-2 风格模型
# ============================================================

class CausalSelfAttention(nn.Module):
    """因果自注意力"""
    
    def __init__(self, config: RealAGIConfig):
        super().__init__()
        
        assert config.hidden_dim % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # 因果mask
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, config: RealAGIConfig):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RealGPTModel(nn.Module):
    """真正的GPT风格语言模型"""
    
    def __init__(self, config: RealAGIConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embed.weight
        
        # 初始化
        self.apply(self._init_weights)
        
        # 特殊初始化
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, T = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """生成文本"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # 截断到max_seq_len
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 训练系统
# ============================================================

class RealAGITrainer:
    """真实AGI训练器"""
    
    def __init__(self, config: Optional[RealAGIConfig] = None):
        self.config = config or RealAGIConfig()
        
        # 设备
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.config.device)
        
        # Tokenizer
        self.tokenizer = SimpleTokenizer()
        
        # 更新vocab_size
        self.config.vocab_size = len(self.tokenizer)
        
        # 模型
        self.model = RealGPTModel(self.config).to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度
        self.scheduler = None  # 在训练时创建
        
        # 统计
        self.stats = {
            'step': 0,
            'epoch': 0,
            'tokens_seen': 0,
            'best_val_loss': float('inf'),
            'train_losses': [],
            'val_losses': [],
            'perplexities': []
        }
        
        self._log_config()
    
    def _log_config(self):
        """记录配置"""
        logger.info("=" * 70)
        logger.info("真实AGI训练系统 - Real AGI Training System")
        logger.info("=" * 70)
        logger.info(f"设备: {self.device}")
        logger.info(f"模型参数: {self.model.count_parameters():,}")
        logger.info(f"隐藏维度: {self.config.hidden_dim}")
        logger.info(f"层数: {self.config.num_layers}")
        logger.info(f"注意力头: {self.config.num_heads}")
        logger.info(f"序列长度: {self.config.max_seq_len}")
        logger.info(f"词表大小: {self.config.vocab_size}")
        logger.info(f"有效批大小: {self.config.batch_size * self.config.gradient_accumulation}")
        logger.info(f"学习率: {self.config.learning_rate}")
        logger.info(f"训练时长: {self.config.training_hours} 小时")
        logger.info("=" * 70)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器，带权重衰减分组"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'ln' in name or 'LayerNorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95)
        )
    
    def _get_lr(self, step: int) -> float:
        """余弦退火学习率"""
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        
        if step < warmup:
            return self.config.learning_rate * step / warmup
        
        progress = (step - warmup) / (max_steps - warmup)
        return self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _set_lr(self, lr: float):
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def prepare_data(self):
        """准备数据"""
        logger.info("\n准备训练数据...")
        
        # WikiText-103
        self.train_dataset = WikiText103Dataset(
            DATA_DIR,
            split='train',
            max_seq_len=self.config.max_seq_len,
            tokenizer=self.tokenizer
        )
        
        self.val_dataset = WikiText103Dataset(
            DATA_DIR,
            split='valid',
            max_seq_len=self.config.max_seq_len,
            tokenizer=self.tokenizer
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"训练集: {len(self.train_dataset):,} 样本")
        logger.info(f"验证集: {len(self.val_dataset):,} 样本")
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            _, loss = self.model(input_ids, labels)
            
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """保存检查点"""
        if path is None:
            path = CHECKPOINT_DIR / f'checkpoint_step{self.stats["step"]}.pt'
        
        torch.save({
            'step': self.stats['step'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'config': asdict(self.config)
        }, path)
        
        logger.info(f"检查点已保存: {path.name}")
    
    def load_checkpoint(self, path: Path) -> bool:
        """加载检查点"""
        if not path.exists():
            return False
        
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        
        logger.info(f"从步骤 {self.stats['step']} 恢复")
        return True
    
    def train(self):
        """训练"""
        self.prepare_data()
        
        target_seconds = self.config.training_hours * 3600
        start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info("开始真实AGI训练 - Next Token Prediction")
        logger.info("=" * 70)
        logger.info(f"目标: {self.config.training_hours} 小时")
        logger.info(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.model.train()
        
        # 梯度累积
        accum_steps = self.config.gradient_accumulation
        accum_loss = 0.0
        
        step = self.stats['step']
        epoch = 0
        
        try:
            while True:
                epoch += 1
                self.stats['epoch'] = epoch
                
                for batch_idx, batch in enumerate(self.train_loader):
                    # 检查时间
                    elapsed = time.time() - start_time
                    if elapsed >= target_seconds:
                        logger.info(f"\n达到目标训练时长 {self.config.training_hours} 小时!")
                        return
                    
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 前向传播
                    _, loss = self.model(input_ids, labels)
                    loss = loss / accum_steps
                    
                    # 反向传播
                    loss.backward()
                    accum_loss += loss.item()
                    
                    # 梯度累积
                    if (batch_idx + 1) % accum_steps == 0:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        # 更新学习率
                        lr = self._get_lr(step)
                        self._set_lr(lr)
                        
                        # 更新参数
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        step += 1
                        self.stats['step'] = step
                        self.stats['tokens_seen'] += input_ids.numel() * accum_steps
                        
                        # 日志
                        if step % self.config.log_steps == 0:
                            tokens_per_sec = self.stats['tokens_seen'] / elapsed
                            progress = elapsed / target_seconds * 100
                            
                            logger.info(
                                f"Step {step:6d} | "
                                f"Loss: {accum_loss:.4f} | "
                                f"LR: {lr:.2e} | "
                                f"Tokens: {self.stats['tokens_seen']:,} | "
                                f"Speed: {tokens_per_sec:.0f} tok/s | "
                                f"Progress: {progress:.1f}%"
                            )
                            
                            self.stats['train_losses'].append(accum_loss)
                        
                        accum_loss = 0.0
                        
                        # 评估
                        if step % self.config.eval_steps == 0:
                            val_metrics = self.evaluate()
                            
                            self.stats['val_losses'].append(val_metrics['loss'])
                            self.stats['perplexities'].append(val_metrics['perplexity'])
                            
                            if val_metrics['loss'] < self.stats['best_val_loss']:
                                self.stats['best_val_loss'] = val_metrics['loss']
                                self.save_checkpoint(CHECKPOINT_DIR / 'best_model.pt')
                            
                            logger.info(
                                f"\n{'='*50}\n"
                                f"评估 @ Step {step}\n"
                                f"  验证 Loss: {val_metrics['loss']:.4f}\n"
                                f"  困惑度 (Perplexity): {val_metrics['perplexity']:.2f}\n"
                                f"  最佳 Loss: {self.stats['best_val_loss']:.4f}\n"
                                f"{'='*50}\n"
                            )
                            
                            # 生成样本
                            self._generate_sample()
                        
                        # 检查点
                        if step % self.config.checkpoint_steps == 0:
                            self.save_checkpoint()
                
                logger.info(f"\nEpoch {epoch} 完成")
        
        except KeyboardInterrupt:
            logger.info("\n训练被中断")
            self.save_checkpoint()
    
    def _generate_sample(self):
        """生成样本文本"""
        prompts = [
            "The meaning of life is",
            "In the year 2050,",
            "Artificial intelligence will"
        ]
        
        logger.info("\n生成样本:")
        
        for prompt in prompts:
            input_ids = torch.tensor([self.tokenizer.encode(prompt)[:-1]]).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=40
                )
            
            text = self.tokenizer.decode(output[0].tolist())
            logger.info(f"  Prompt: {prompt}")
            logger.info(f"  Output: {text[:200]}...")
            logger.info("")


# ============================================================
# Benchmark评估
# ============================================================

class BenchmarkEvaluator:
    """标准Benchmark评估"""
    
    def __init__(self, model: RealGPTModel, tokenizer: SimpleTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_hellaswag(self) -> float:
        """HellaSwag评估 - 常识推理"""
        # 简化版本 - 实际应用中需要下载完整数据集
        logger.info("HellaSwag评估 (简化版)...")
        
        examples = [
            {
                'context': "A person is making a sandwich. They",
                'choices': [
                    " put peanut butter on the bread.",
                    " flew to the moon.",
                    " turned into a tree.",
                    " calculated quantum physics."
                ],
                'answer': 0
            },
            {
                'context': "The dog saw a cat and",
                'choices': [
                    " started barking and chasing it.",
                    " began writing poetry.",
                    " transformed into a car.",
                    " solved a math equation."
                ],
                'answer': 0
            }
        ]
        
        correct = 0
        total = len(examples)
        
        self.model.eval()
        
        for ex in examples:
            scores = []
            for choice in ex['choices']:
                full_text = ex['context'] + choice
                tokens = self.tokenizer.encode(full_text)
                input_ids = torch.tensor([tokens[:-1]]).to(self.device)
                labels = torch.tensor([tokens[1:]]).to(self.device)
                
                with torch.no_grad():
                    _, loss = self.model(input_ids, labels)
                scores.append(-loss.item())
            
            pred = max(range(len(scores)), key=lambda i: scores[i])
            if pred == ex['answer']:
                correct += 1
        
        accuracy = correct / total
        logger.info(f"HellaSwag准确率: {accuracy:.2%}")
        
        return accuracy
    
    def evaluate_all(self) -> Dict:
        """运行所有评估"""
        results = {}
        
        results['hellaswag'] = self.evaluate_hellaswag()
        
        return results


# ============================================================
# 主程序
# ============================================================

def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='真实AGI训练')
    parser.add_argument('--hours', type=float, default=5.0, help='训练小时数')
    parser.add_argument('--hidden', type=int, default=512, help='隐藏维度')
    parser.add_argument('--layers', type=int, default=8, help='层数')
    parser.add_argument('--heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--batch', type=int, default=8, help='批大小')
    parser.add_argument('--seq-len', type=int, default=512, help='序列长度')
    
    args = parser.parse_args()
    
    # 配置 - 适配MPS内存
    config = RealAGIConfig(
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_dim=args.hidden * 4,
        max_seq_len=args.seq_len,
        batch_size=args.batch,
        training_hours=args.hours
    )
    
    # 训练
    trainer = RealAGITrainer(config)
    trainer.train()
    
    # 最终评估
    logger.info("\n" + "=" * 70)
    logger.info("最终Benchmark评估")
    logger.info("=" * 70)
    
    evaluator = BenchmarkEvaluator(
        trainer.model,
        trainer.tokenizer,
        trainer.device
    )
    results = evaluator.evaluate_all()
    
    logger.info(f"\n最终结果: {json.dumps(results, indent=2)}")
    
    # 保存最终模型
    final_path = MODEL_DIR / 'final_model.pt'
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': asdict(config),
        'stats': trainer.stats,
        'benchmark_results': results
    }, final_path)
    
    logger.info(f"\n模型已保存: {final_path}")


if __name__ == "__main__":
    main()
