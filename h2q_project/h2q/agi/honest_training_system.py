#!/usr/bin/env python3
"""
Honest AGI Training System - 诚实的AGI训练系统
==============================================

修复了之前训练系统的"伪训练"问题：
1. 真正的完整Epoch遍历 - 每个epoch遍历全部数据
2. 正确的样本计数和时间追踪 - 精确记录实际训练量
3. 防作弊验证机制 - 自动检测异常训练模式

作者: H2Q-Evo Team
日期: 2026-01-22
"""

import os
import sys
import json
import time
import math
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 路径配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'honest_data'
MODEL_DIR = SCRIPT_DIR / 'honest_models'
LOG_DIR = SCRIPT_DIR / 'honest_logs'

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'honest_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 训练配置
# ============================================================

@dataclass
class HonestTrainingConfig:
    """诚实训练配置 - 所有参数都有明确的含义"""
    
    # 模型架构
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    # 数据集目标
    min_samples_per_dataset: int = 5000  # 最小样本数
    target_samples_per_epoch: int = 5000  # 每epoch目标样本数
    
    # 训练目标
    target_training_hours: float = 1.0  # 目标训练时长(小时)
    min_epochs: int = 10  # 最小epoch数
    
    # 防作弊阈值
    max_accuracy_jump: float = 0.20  # 单epoch最大准确率跳跃
    min_samples_per_second: float = 10  # 最小样本处理速度
    max_samples_per_second: float = 1000  # 最大样本处理速度
    
    def validate(self):
        """验证配置合理性"""
        assert self.batch_size >= 8, "batch_size太小"
        assert self.min_samples_per_dataset >= 1000, "数据集太小"
        assert self.target_training_hours >= 0.1, "训练时间太短"


# ============================================================
# 防作弊验证器
# ============================================================

class AntiCheatValidator:
    """防作弊验证器 - 检测异常训练模式"""
    
    def __init__(self, config: HonestTrainingConfig):
        self.config = config
        self.history = []
        self.warnings = []
        self.violations = []
    
    def record_epoch(self, metrics: Dict):
        """记录每个epoch的指标"""
        self.history.append({
            **metrics,
            'timestamp': time.time()
        })
    
    def check_accuracy_jump(self, prev_acc: float, curr_acc: float, epoch: int) -> bool:
        """检查准确率跳跃是否异常"""
        jump = curr_acc - prev_acc
        if jump > self.config.max_accuracy_jump:
            self.warnings.append({
                'type': 'accuracy_jump',
                'epoch': epoch,
                'jump': jump,
                'message': f"Epoch {epoch}: 准确率跳跃 {jump:.2%} 超过阈值 {self.config.max_accuracy_jump:.2%}"
            })
            return False
        return True
    
    def check_training_speed(self, samples: int, duration: float, epoch: int) -> bool:
        """检查训练速度是否合理"""
        if duration <= 0:
            self.violations.append({
                'type': 'zero_duration',
                'epoch': epoch,
                'message': f"Epoch {epoch}: 训练时间为0，可能存在作弊"
            })
            return False
        
        speed = samples / duration
        
        if speed < self.config.min_samples_per_second:
            self.warnings.append({
                'type': 'too_slow',
                'epoch': epoch,
                'speed': speed,
                'message': f"Epoch {epoch}: 训练速度 {speed:.1f} samples/s 过慢"
            })
        
        if speed > self.config.max_samples_per_second:
            self.violations.append({
                'type': 'too_fast',
                'epoch': epoch,
                'speed': speed,
                'message': f"Epoch {epoch}: 训练速度 {speed:.1f} samples/s 异常快，可能未真正训练"
            })
            return False
        
        return True
    
    def check_sample_count(self, claimed: int, actual: int, epoch: int) -> bool:
        """检查样本计数是否一致"""
        if actual < claimed * 0.95:
            self.violations.append({
                'type': 'sample_mismatch',
                'epoch': epoch,
                'claimed': claimed,
                'actual': actual,
                'message': f"Epoch {epoch}: 声称处理 {claimed} 样本，实际只有 {actual}"
            })
            return False
        return True
    
    def generate_report(self) -> Dict:
        """生成防作弊报告"""
        total_samples = sum(h.get('samples_processed', 0) for h in self.history)
        total_time = sum(h.get('duration', 0) for h in self.history)
        
        return {
            'is_valid': len(self.violations) == 0,
            'total_epochs': len(self.history),
            'total_samples': total_samples,
            'total_time_seconds': total_time,
            'avg_speed': total_samples / total_time if total_time > 0 else 0,
            'warnings': self.warnings,
            'violations': self.violations,
            'verdict': self._get_verdict()
        }
    
    def _get_verdict(self) -> str:
        if self.violations:
            return "❌ 训练无效 - 检测到作弊行为"
        elif len(self.warnings) > 5:
            return "⚠️ 训练可疑 - 多个警告需要审查"
        else:
            return "✅ 训练有效 - 通过防作弊检查"


# ============================================================
# 真实数据集
# ============================================================

class HonestDataset(Dataset):
    """诚实的数据集 - 完整遍历，无作弊"""
    
    def __init__(self, name: str, num_samples: int, vocab_size: int = 32000):
        self.name = name
        self.vocab_size = vocab_size
        self.samples = []
        
        logger.info(f"生成 {name} 数据集: {num_samples} 样本...")
        self._generate_samples(num_samples)
        logger.info(f"  完成: {len(self.samples)} 样本")
    
    def _generate_samples(self, num_samples: int):
        """生成合成样本 - 这里可以替换为真实数据加载"""
        for i in range(num_samples):
            # 生成问题token序列
            seq_len = random.randint(50, 200)
            tokens = [random.randint(1, self.vocab_size - 1) for _ in range(seq_len)]
            
            # 生成标签 (4分类)
            label = random.randint(0, 3)
            
            self.samples.append({
                'tokens': tokens,
                'label': label,
                'id': f"{self.name}_{i}"
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 转换为tensor，填充到固定长度
        tokens = sample['tokens'][:256]
        if len(tokens) < 256:
            tokens = tokens + [0] * (256 - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'id': sample['id']
        }


class CombinedDataset(Dataset):
    """组合多个数据集"""
    
    def __init__(self, datasets: List[HonestDataset]):
        self.datasets = datasets
        self.samples = []
        
        for ds in datasets:
            self.samples.extend(ds.samples)
        
        random.shuffle(self.samples)
        logger.info(f"组合数据集: {len(self.samples)} 总样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        tokens = sample['tokens'][:256]
        if len(tokens) < 256:
            tokens = tokens + [0] * (256 - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'id': sample['id']
        }


# ============================================================
# 模型架构
# ============================================================

class HonestModel(nn.Module):
    """诚实的模型 - 标准Transformer分类器"""
    
    def __init__(self, config: HonestTrainingConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 4)
        )
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        
        # Embedding
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)
        
        # Transformer编码
        x = self.encoder(x)
        
        # 池化 (使用[CLS]位置或平均池化)
        x = x.mean(dim=1)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 诚实的训练器
# ============================================================

class HonestTrainer:
    """诚实的训练器 - 完整epoch遍历，精确计数"""
    
    def __init__(self, model: HonestModel, config: HonestTrainingConfig):
        self.model = model
        self.config = config
        
        # 设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 防作弊验证器
        self.validator = AntiCheatValidator(config)
        
        # 统计
        self.stats = {
            'total_samples_processed': 0,
            'total_training_time': 0.0,
            'epoch_history': [],
            'best_accuracy': 0.0
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """
        训练一个完整的epoch
        
        关键改进:
        1. 遍历完整的DataLoader (所有样本)
        2. 精确记录处理的样本数
        3. 精确记录训练时间
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        batch_count = 0
        
        # 记录开始时间
        start_time = time.time()
        
        # 遍历完整的DataLoader
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            batch_count += 1
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 计算指标
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct / total_samples if total_samples > 0 else 0
        samples_per_second = total_samples / duration if duration > 0 else 0
        
        # 更新学习率
        self.scheduler.step()
        
        # 更新全局统计
        self.stats['total_samples_processed'] += total_samples
        self.stats['total_training_time'] += duration
        
        # 防作弊验证
        prev_acc = self.stats['epoch_history'][-1]['accuracy'] if self.stats['epoch_history'] else 0
        self.validator.check_accuracy_jump(prev_acc, accuracy, epoch)
        self.validator.check_training_speed(total_samples, duration, epoch)
        
        # 记录epoch指标
        epoch_metrics = {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'batches': batch_count,
            'duration': duration,
            'samples_per_second': samples_per_second,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self.stats['epoch_history'].append(epoch_metrics)
        self.validator.record_epoch(epoch_metrics)
        
        if accuracy > self.stats['best_accuracy']:
            self.stats['best_accuracy'] = accuracy
        
        return epoch_metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            'accuracy': correct / total_samples if total_samples > 0 else 0,
            'samples': total_samples
        }
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            'total_samples_processed': self.stats['total_samples_processed'],
            'total_training_time': self.stats['total_training_time'],
            'total_epochs': len(self.stats['epoch_history']),
            'best_accuracy': self.stats['best_accuracy'],
            'avg_samples_per_second': (
                self.stats['total_samples_processed'] / self.stats['total_training_time']
                if self.stats['total_training_time'] > 0 else 0
            ),
            'anti_cheat_report': self.validator.generate_report()
        }


# ============================================================
# 主训练系统
# ============================================================

class HonestTrainingSystem:
    """诚实的训练系统 - 完整、可验证的训练流程"""
    
    def __init__(self, config: Optional[HonestTrainingConfig] = None):
        self.config = config or HonestTrainingConfig()
        self.config.validate()
        
        # 初始化模型
        self.model = HonestModel(self.config)
        self.trainer = HonestTrainer(self.model, self.config)
        
        # 数据集
        self.datasets = {}
        self.train_loader = None
        self.val_loader = None
        
        # 状态
        self.state = {
            'start_time': None,
            'end_time': None,
            'current_epoch': 0,
            'is_complete': False
        }
        
        logger.info("=" * 60)
        logger.info("诚实AGI训练系统初始化")
        logger.info("=" * 60)
        logger.info(f"设备: {self.trainer.device}")
        logger.info(f"模型参数: {self.model.count_parameters():,}")
        logger.info(f"目标训练时长: {self.config.target_training_hours} 小时")
        logger.info(f"最小样本数: {self.config.min_samples_per_dataset}")
    
    def prepare_data(self):
        """准备数据集"""
        logger.info("\n准备数据集...")
        
        # 创建多个数据集
        dataset_configs = [
            ('mmlu', self.config.min_samples_per_dataset),
            ('gsm8k', self.config.min_samples_per_dataset),
            ('arc', self.config.min_samples_per_dataset // 2),
            ('hellaswag', self.config.min_samples_per_dataset // 2),
        ]
        
        all_datasets = []
        for name, num_samples in dataset_configs:
            ds = HonestDataset(name, num_samples, self.config.vocab_size)
            self.datasets[name] = ds
            all_datasets.append(ds)
        
        # 组合数据集
        combined = CombinedDataset(all_datasets)
        
        # 划分训练/验证集
        total = len(combined)
        train_size = int(0.9 * total)
        val_size = total - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            combined, [train_size, val_size]
        )
        
        # 创建DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False  # 不丢弃最后一个不完整的batch
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"训练集: {train_size} 样本, {len(self.train_loader)} batches")
        logger.info(f"验证集: {val_size} 样本, {len(self.val_loader)} batches")
        
        # 验证数据集大小
        expected_samples = sum(num for _, num in dataset_configs)
        actual_samples = total
        
        if actual_samples < expected_samples * 0.95:
            logger.warning(f"数据集样本数不足: 预期 {expected_samples}, 实际 {actual_samples}")
    
    def run_training(self, num_epochs: Optional[int] = None, 
                     target_hours: Optional[float] = None) -> Dict:
        """
        运行训练
        
        参数:
            num_epochs: 指定epoch数，或None使用时间限制
            target_hours: 目标训练时长(小时)
        """
        if self.train_loader is None:
            self.prepare_data()
        
        target_hours = target_hours or self.config.target_training_hours
        target_seconds = target_hours * 3600
        
        logger.info("\n" + "=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)
        logger.info(f"目标时长: {target_hours} 小时 ({target_seconds:.0f} 秒)")
        logger.info(f"每epoch样本: {len(self.train_loader.dataset)}")
        
        self.state['start_time'] = time.time()
        epoch = 0
        
        while True:
            epoch += 1
            self.state['current_epoch'] = epoch
            
            # 训练一个epoch
            metrics = self.trainer.train_epoch(self.train_loader, epoch)
            
            # 验证
            val_metrics = self.trainer.evaluate(self.val_loader)
            
            # 计算进度
            elapsed = time.time() - self.state['start_time']
            progress = elapsed / target_seconds * 100
            
            # 日志
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Train Acc: {metrics['accuracy']:.2%} | "
                f"Val Acc: {val_metrics['accuracy']:.2%} | "
                f"Samples: {metrics['samples_processed']:,} | "
                f"Speed: {metrics['samples_per_second']:.1f}/s | "
                f"Progress: {progress:.1f}%"
            )
            
            # 检查停止条件
            if num_epochs and epoch >= num_epochs:
                logger.info(f"达到目标epoch数: {num_epochs}")
                break
            
            if elapsed >= target_seconds:
                logger.info(f"达到目标训练时长: {target_hours} 小时")
                break
            
            if epoch >= self.config.min_epochs and val_metrics['accuracy'] > 0.95:
                logger.info(f"提前停止: 验证准确率达到 {val_metrics['accuracy']:.2%}")
                break
        
        self.state['end_time'] = time.time()
        self.state['is_complete'] = True
        
        return self._generate_final_report()
    
    def _generate_final_report(self) -> Dict:
        """生成最终报告"""
        summary = self.trainer.get_training_summary()
        
        total_time = self.state['end_time'] - self.state['start_time']
        
        report = {
            'training_complete': True,
            'total_epochs': self.state['current_epoch'],
            'total_time_seconds': total_time,
            'total_time_formatted': str(timedelta(seconds=int(total_time))),
            'total_samples_processed': summary['total_samples_processed'],
            'avg_samples_per_second': summary['avg_samples_per_second'],
            'best_accuracy': summary['best_accuracy'],
            'anti_cheat_report': summary['anti_cheat_report'],
            'model_parameters': self.model.count_parameters(),
            'config': asdict(self.config)
        }
        
        # 保存报告
        report_path = MODEL_DIR / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 保存模型
        model_path = MODEL_DIR / 'honest_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'training_stats': self.trainer.stats
        }, model_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("训练完成 - 最终报告")
        logger.info("=" * 60)
        logger.info(f"总时长: {report['total_time_formatted']}")
        logger.info(f"总Epochs: {report['total_epochs']}")
        logger.info(f"总样本处理: {report['total_samples_processed']:,}")
        logger.info(f"平均速度: {report['avg_samples_per_second']:.1f} samples/s")
        logger.info(f"最佳准确率: {report['best_accuracy']:.2%}")
        logger.info(f"\n防作弊验证: {summary['anti_cheat_report']['verdict']}")
        
        if summary['anti_cheat_report']['violations']:
            logger.warning("发现违规:")
            for v in summary['anti_cheat_report']['violations']:
                logger.warning(f"  - {v['message']}")
        
        logger.info(f"\n模型保存至: {model_path}")
        logger.info(f"报告保存至: {report_path}")
        
        return report


# ============================================================
# 快速验证函数
# ============================================================

def quick_validation_test():
    """快速验证测试 - 验证训练系统是否正常工作"""
    print("\n" + "=" * 70)
    print("   快速验证测试 - 诚实训练系统")
    print("=" * 70)
    
    # 小规模配置 (测试模式允许更小的数据集)
    config = HonestTrainingConfig(
        min_samples_per_dataset=1000,  # 最小允许值
        target_training_hours=0.1,  # 6分钟
        min_epochs=3,
        batch_size=16
    )
    
    system = HonestTrainingSystem(config)
    report = system.run_training(num_epochs=5)
    
    print("\n" + "=" * 70)
    print("   验证结果")
    print("=" * 70)
    
    # 验证点
    checks = [
        ("完整epoch遍历", report['total_samples_processed'] >= config.min_samples_per_dataset * 2),
        ("时间追踪正常", report['total_time_seconds'] > 1),
        ("样本速度合理", 10 < report['avg_samples_per_second'] < 5000),
        ("防作弊通过", report['anti_cheat_report']['is_valid']),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "-" * 70)
    if all_passed:
        print("  ✅ 所有验证通过 - 训练系统工作正常")
    else:
        print("  ❌ 存在问题 - 请检查日志")
    
    return all_passed


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数 - 运行完整训练"""
    print("\n" + "=" * 70)
    print("   诚实AGI训练系统")
    print("   Honest AGI Training System")
    print("=" * 70)
    
    # 配置 - 目标1小时训练
    config = HonestTrainingConfig(
        min_samples_per_dataset=5000,  # 每个数据集5000样本
        target_training_hours=1.0,     # 1小时训练
        min_epochs=10,
        batch_size=32,
        hidden_dim=256,
        num_layers=4
    )
    
    system = HonestTrainingSystem(config)
    report = system.run_training()
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        quick_validation_test()
    else:
        main()
