#!/usr/bin/env python3
"""
H2Q-Evo 本地训练与进化系统
===================================

安全的本地模型训练和进化
- 完全离线，无联网
- 使用本地数据集进行训练
- 自我进化算法
- 性能监控和安全约束
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
H2Q_PROJECT = PROJECT_ROOT / "h2q_project"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(H2Q_PROJECT))

# 导入现有组件
try:
    from local_long_text_generator import LocalLongTextGenerator
    from local_memory_index import OfflineMemoryIndex
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 10
    sequence_length: int = 512
    save_interval: int = 5
    eval_interval: int = 2
    max_grad_norm: float = 1.0
    warmup_steps: int = 100


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    perplexity: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_processed: int = 0
    training_time: float = 0.0


class LocalTextDataset(Dataset):
    """本地文本数据集"""

    def __init__(self, data_dir: Path, sequence_length: int = 512):
        self.sequence_length = sequence_length
        self.data = []

        # 加载本地数据
        self._load_local_data(data_dir)

        # 简单字符级编码
        self.vocab_size = 256  # ASCII字符
        self.pad_token = 0

    def _load_local_data(self, data_dir: Path):
        """加载本地数据"""
        print(f"📚 加载本地训练数据: {data_dir}")

        if not data_dir.exists():
            # 创建示例数据
            data_dir.mkdir(parents=True, exist_ok=True)
            self._create_sample_data(data_dir)

        total_files = 0
        total_chars = 0

        # 递归加载所有文本文件
        for txt_file in data_dir.rglob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:  # 只使用较长的文件
                        self.data.append(content)
                        total_chars += len(content)
                        total_files += 1
            except Exception as e:
                print(f"⚠️ 跳过文件 {txt_file}: {e}")

        print(f"✓ 加载了 {total_files} 个文件，共 {total_chars:,} 个字符")

        if not self.data:
            print("⚠️ 没有找到训练数据，创建示例数据")
            self._create_sample_data(data_dir)
            self.data = ["这是一个示例训练文本，用于测试本地模型训练功能。"] * 10

    def _create_sample_data(self, data_dir: Path):
        """创建示例训练数据"""
        sample_texts = [
            "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。",
            "量子计算利用量子力学的原理，如叠加和纠缠，来进行计算。",
            "机器学习是人工智能的一个子领域，通过数据训练模型来做出预测。",
            "深度学习使用多层神经网络来解决复杂的模式识别问题。",
            "自然语言处理是让计算机理解和生成人类语言的技术。",
            "计算机视觉是让机器能够理解和解释视觉信息的技术。",
            "强化学习通过试错来学习最优策略的机器学习方法。",
            "神经网络是受生物神经系统启发的计算模型。",
            "大数据是指规模巨大、类型多样、处理速度快的海量数据。",
            "算法是解决特定问题的一系列明确指令。"
        ]

        train_dir = data_dir / "training"
        train_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(sample_texts):
            with open(train_dir / f"sample_{i}.txt", 'w', encoding='utf-8') as f:
                # 重复文本以增加数据量
                f.write((text + "\n") * 50)

    def __len__(self):
        return len(self.data) * 10  # 每个文本生成10个序列

    def __getitem__(self, idx):
        # 随机选择一个文本
        text = self.data[idx % len(self.data)]

        # 随机选择起始位置
        start_pos = np.random.randint(0, max(1, len(text) - self.sequence_length - 1))
        chunk = text[start_pos:start_pos + self.sequence_length + 1]

        # 字符级编码
        tokens = [ord(c) % self.vocab_size for c in chunk]

        # 填充或截断
        if len(tokens) < self.sequence_length + 1:
            tokens.extend([self.pad_token] * (self.sequence_length + 1 - len(tokens)))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


class LocalModelTrainer:
    """本地模型训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataset = None
        self.dataloader = None

        # 训练状态
        self.metrics = TrainingMetrics()
        self.best_loss = float('inf')
        self.training_log = []

        print(f"🏋️ 本地训练器初始化完成 | 设备: {self.device}")

    def setup_training(self, data_dir: Path):
        """设置训练环境"""
        print("🔧 设置训练环境...")

        # 创建数据集
        self.dataset = LocalTextDataset(data_dir, self.config.sequence_length)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # 本地训练使用单线程
        )

        # 初始化模型（使用简单的Transformer）
        self._init_model()

        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs * len(self.dataloader)
        )

        print(f"✓ 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✓ 训练数据: {len(self.dataset)} 个序列")
        print(f"✓ 批次大小: {self.config.batch_size}")

    def _init_model(self):
        """初始化模型"""
        vocab_size = 256
        embed_dim = 256
        n_heads = 8
        n_layers = 6

        self.model = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    embed_dim, n_heads, batch_first=True, dropout=0.1
                ),
                num_layers=n_layers
            ),
            nn.Linear(embed_dim, vocab_size)
        ).to(self.device)

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(self.dataloader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # 前向传播
            outputs = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1),
                ignore_index=0  # 忽略填充token
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()

            # 更新指标
            total_loss += loss.item()
            num_batches += 1
            self.metrics.step += 1
            self.metrics.tokens_processed += input_ids.numel()

            # 定期报告
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                perplexity = torch.exp(loss).item()
                print(f"  批次 {batch_idx:3d} | 损失: {loss.item():.4f} | 困惑度: {perplexity:.2f} | LR: {current_lr:.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in self.dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                outputs = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1),
                    ignore_index=0
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def save_checkpoint(self, epoch: int, loss: float):
        """保存检查点"""
        checkpoint_dir = PROJECT_ROOT / "training_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': asdict(self.config),
            'metrics': asdict(self.metrics)
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")

        # 保存最佳模型
        if loss < self.best_loss:
            self.best_loss = loss
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(self.model.state_dict(), best_model_path)
            print(f"🏆 最佳模型已更新: {best_model_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """加载检查点"""
        if not checkpoint_path.exists():
            print(f"⚠️ 检查点不存在: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.metrics.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('loss', float('inf'))

        print(f"📂 检查点已加载: {checkpoint_path}")

    def train(self, data_dir: Path, resume: bool = False):
        """开始训练"""
        print("\n" + "="*60)
        print("🚀 H2Q-Evo 本地模型训练开始")
        print("="*60)
        print("🛡️ 安全保证：完全离线，无联网")
        print("📊 训练配置：")
        print(f"  - 学习率: {self.config.learning_rate}")
        print(f"  - 批次大小: {self.config.batch_size}")
        print(f"  - 序列长度: {self.config.sequence_length}")
        print(f"  - 最大轮数: {self.config.max_epochs}")
        print("="*60 + "\n")

        # 设置训练环境
        self.setup_training(data_dir)

        # 恢复训练（如果需要）
        if resume:
            checkpoint_dir = PROJECT_ROOT / "training_checkpoints"
            latest_checkpoint = max(checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                                  key=lambda x: int(x.stem.split('_')[-1]), default=None)
            if latest_checkpoint:
                self.load_checkpoint(latest_checkpoint)

        start_time = time.time()

        for epoch in range(self.metrics.epoch, self.config.max_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.config.max_epochs}")
            print("-" * 40)

            # 训练
            epoch_start = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start

            # 评估
            if (epoch + 1) % self.config.eval_interval == 0:
                eval_loss, perplexity = self.evaluate()
                print(f"📊 评估损失: {eval_loss:.4f} | 困惑度: {perplexity:.2f}")
            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, train_loss)

            # 更新指标
            self.metrics.epoch = epoch + 1
            self.metrics.loss = train_loss
            self.metrics.perplexity = torch.exp(torch.tensor(train_loss)).item()
            self.metrics.training_time = time.time() - start_time

            # 记录训练日志
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'eval_loss': eval_loss if (epoch + 1) % self.config.eval_interval == 0 else None,
                'perplexity': perplexity if (epoch + 1) % self.config.eval_interval == 0 else None,
                'epoch_time': epoch_time,
                'total_time': self.metrics.training_time
            }
            self.training_log.append(log_entry)

        total_time = time.time() - start_time
        print("\n🎉 训练完成！")
        print(f"⏱️ 总训练时间: {total_time:.2f} 秒")
        print(f"📉 最终损失: {self.metrics.loss:.4f}")
        print(f"🎯 最终困惑度: {self.metrics.perplexity:.2f}")
        print(f"📊 处理token数: {self.metrics.tokens_processed:,}")
        # 保存最终模型
        final_model_path = PROJECT_ROOT / "h2q_project" / "h2q_trained_model.pt"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"💾 最终模型已保存: {final_model_path}")

        # 保存训练日志
        log_path = PROJECT_ROOT / "training_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)
        print(f"📝 训练日志已保存: {log_path}")


class SelfEvolutionEngine:
    """自我进化引擎"""

    def __init__(self):
        self.trainer = LocalModelTrainer(TrainingConfig())
        self.memory_index = None
        self.generation_stats = {
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'average_improvement': 0.0
        }

        print("🧬 自我进化引擎已初始化")

    def initialize_knowledge_base(self):
        """初始化知识库"""
        data_dir = PROJECT_ROOT / "data" / "training_data"
        self.memory_index = OfflineMemoryIndex(data_dir)
        self.memory_index.build(max_files=100)

        print(f"🧠 知识库初始化完成 | 索引文件: {len(self.memory_index.index)}")

    def evolutionary_training_cycle(self):
        """进化训练周期"""
        print("\n🔄 开始进化训练周期...")

        # 1. 评估当前能力
        baseline_metrics = self._evaluate_current_capabilities()

        # 2. 生成训练数据
        training_data = self._generate_training_data()

        # 3. 执行训练
        self.trainer.train(training_data, resume=True)

        # 4. 评估改进
        improved_metrics = self._evaluate_current_capabilities()
        improvement = self._calculate_improvement(baseline_metrics, improved_metrics)

        # 5. 更新进化统计
        self.generation_stats['total_evolutions'] += 1
        if improvement > 0:
            self.generation_stats['successful_evolutions'] += 1
        else:
            self.generation_stats['failed_evolutions'] += 1

        self.generation_stats['average_improvement'] = (
            (self.generation_stats['average_improvement'] * (self.generation_stats['total_evolutions'] - 1)) +
            improvement
        ) / self.generation_stats['total_evolutions']

        print("\n📊 进化结果:")
        print(f"  改进程度: {improvement:.4f}")
        print(f"  成功进化: {'是' if improvement > 0 else '否'}")
        return improvement > 0

    def _evaluate_current_capabilities(self) -> Dict[str, float]:
        """评估当前能力"""
        # 使用简单的指标评估
        text_generator = LocalLongTextGenerator()

        # 生成测试文本
        test_prompts = [
            "解释人工智能的基本原理",
            "什么是量子计算",
            "机器学习的工作原理"
        ]

        total_length = 0
        total_diversity = 0

        for prompt in test_prompts:
            generated = text_generator.generate_long_text(prompt, max_tokens=200)
            total_length += len(generated)
            # 简单多样性度量
            unique_chars = len(set(generated))
            total_diversity += unique_chars / len(generated) if generated else 0

        return {
            'avg_length': total_length / len(test_prompts),
            'avg_diversity': total_diversity / len(test_prompts)
        }

    def _generate_training_data(self) -> Path:
        """生成训练数据"""
        data_dir = PROJECT_ROOT / "data" / "training_data" / "evolution"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 从现有知识库生成训练数据
        if self.memory_index and self.memory_index.index:
            # 选择高质量的文档进行训练
            selected_docs = sorted(
                self.memory_index.index,
                key=lambda x: len(x['content']),
                reverse=True
            )[:10]  # 选择最长的10个文档

            for i, doc in enumerate(selected_docs):
                # 生成变体数据用于训练
                variants = self._create_training_variants(doc['content'])
                for j, variant in enumerate(variants):
                    with open(data_dir / f"evolution_{i}_{j}.txt", 'w', encoding='utf-8') as f:
                        f.write(variant)

        return data_dir.parent

    def _create_training_variants(self, text: str) -> List[str]:
        """创建训练数据变体"""
        variants = [text]  # 原始文本

        # 创建一些简单的变体
        words = text.split()
        if len(words) > 10:
            # 重新排列句子
            mid = len(words) // 2
            variant1 = ' '.join(words[mid:] + words[:mid])
            variants.append(variant1)

            # 截取子串
            variant2 = ' '.join(words[:len(words)//2])
            variants.append(variant2)

        return variants

    def _calculate_improvement(self, baseline: Dict[str, float], current: Dict[str, float]) -> float:
        """计算改进程度"""
        improvement = 0.0
        for key in baseline:
            if key in current:
                improvement += (current[key] - baseline[key]) / max(baseline[key], 1e-6)
        return improvement / len(baseline) if baseline else 0.0

    def run_evolution_cycles(self, num_cycles: int = 3):
        """运行多个进化周期"""
        print("\n" + "="*60)
        print("🧬 H2Q-Evo 自我进化之旅开始")
        print("="*60)
        print(f"🎯 目标：运行 {num_cycles} 个进化周期")
        print("🛡️ 安全：完全本地，无联网")
        print("="*60 + "\n")

        self.initialize_knowledge_base()

        successful_cycles = 0

        for cycle in range(num_cycles):
            print(f"\n🔄 进化周期 {cycle + 1}/{num_cycles}")
            print("-" * 40)

            try:
                success = self.evolutionary_training_cycle()
                if success:
                    successful_cycles += 1
                    print(f"✅ 周期 {cycle + 1} 进化成功")
                else:
                    print(f"⚠️ 周期 {cycle + 1} 进化未见显著改进")
            except Exception as e:
                print(f"❌ 周期 {cycle + 1} 进化失败: {e}")

        print("\n🎊 进化周期完成！")
        print(f"📈 成功周期: {successful_cycles}/{num_cycles}")
        print(f"📊 平均改进: {self.generation_stats['average_improvement']:.2f}")
        print(f"🔬 总进化次数: {self.generation_stats['total_evolutions']}")
        print(f"✅ 成功进化: {self.generation_stats['successful_evolutions']}")
        print(f"❌ 失败进化: {self.generation_stats['failed_evolutions']}")
        # 保存进化统计
        stats_path = PROJECT_ROOT / "evolution_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.generation_stats, f, indent=2, ensure_ascii=False)
        print(f"💾 进化统计已保存: {stats_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="H2Q-Evo 本地训练与进化系统")
    parser.add_argument("--mode", choices=["train", "evolve"], default="train",
                       help="运行模式：train(训练) 或 evolve(进化)")
    parser.add_argument("--data_dir", type=str,
                       help="训练数据目录（默认为自动创建）")
    parser.add_argument("--epochs", type=int, default=5,
                       help="训练轮数")
    parser.add_argument("--cycles", type=int, default=3,
                       help="进化周期数")

    args = parser.parse_args()

    if args.mode == "train":
        # 基础训练模式
        config = TrainingConfig(max_epochs=args.epochs)
        trainer = LocalModelTrainer(config)

        data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "training_data"
        trainer.train(data_dir)

    elif args.mode == "evolve":
        # 自我进化模式
        evolution_engine = SelfEvolutionEngine()
        evolution_engine.run_evolution_cycles(args.cycles)


if __name__ == "__main__":
    main()