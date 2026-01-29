#!/usr/bin/env python3
"""
真实AGI训练系统 - 集成标准数据集
支持标准机器学习基准、性能基准测试和交叉验证
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import psutil
import gc
import atexit
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

# 导入高级谱稳定性控制器
try:
    from advanced_spectral_controller import AdvancedSpectralController, RiemannSpectralLoss
    ADVANCED_SPECTRAL_AVAILABLE = True
except ImportError:
    ADVANCED_SPECTRAL_AVAILABLE = False
    print("警告: 高级谱稳定性控制器不可用，将使用传统谱移跟踪器")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("real_agi_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Real-AGI-Training")

class StandardDatasetLoader:
    """标准数据集加载器"""

    def __init__(self, dataset_name: str, batch_size: int = 32, device: str = "cpu"):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_dataset(self):
        """加载标准数据集"""
        if self.dataset_name.lower() == "mnist":
            return self._load_mnist()
        elif self.dataset_name.lower() == "cifar10":
            return self._load_cifar10()
        elif self.dataset_name.lower() == "cifar100":
            return self._load_cifar100()
        elif self.dataset_name.lower() == "fashion_mnist":
            return self._load_fashion_mnist()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

    def _load_mnist(self):
        """加载MNIST数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )

        # 划分训练和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader

    def _load_cifar10(self):
        """加载CIFAR-10数据集"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

        # 划分训练和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader

    def _load_cifar100(self):
        """加载CIFAR-100数据集"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )

        # 划分训练和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader

    def _load_fashion_mnist(self):
        """加载Fashion-MNIST数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )

        # 划分训练和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader

class BenchmarkModel(nn.Module):
    """基准测试模型 - 支持标准数据集"""

    def __init__(self, dataset_name: str, num_classes: int = 10):
        super(BenchmarkModel, self).__init__()
        self.dataset_name = dataset_name

        if dataset_name.lower() in ['mnist', 'fashion_mnist']:
            # MNIST风格的网络
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:
            # CIFAR风格的网络
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SpectralStabilityTracker:
    """谱稳定性跟踪器"""

    def __init__(self):
        self.stability_history = []
        self.max_history = 100

    def update_stability(self, spectral_shift: float, loss: float) -> float:
        """更新谱稳定性指标"""
        # 计算谱稳定性分数（基于谱移和损失的相关性）
        stability_score = -abs(spectral_shift) * (1.0 / (1.0 + loss))

        self.stability_history.append({
            'spectral_shift': spectral_shift,
            'loss': loss,
            'stability_score': stability_score,
            'timestamp': time.time()
        })

        if len(self.stability_history) > self.max_history:
            self.stability_history = self.stability_history[-self.max_history:]

        return stability_score

    def get_correlation(self) -> float:
        """计算谱稳定性与学习效果的相关性"""
        if len(self.stability_history) < 10:
            return 0.0

        recent_data = self.stability_history[-50:]  # 使用最近50个数据点
        spectral_shifts = [d['spectral_shift'] for d in recent_data]
        losses = [d['loss'] for d in recent_data]

        # 计算皮尔逊相关系数
        if len(set(spectral_shifts)) <= 1 or len(set(losses)) <= 1:
            return 0.0

        spectral_shifts = np.array(spectral_shifts)
        losses = np.array(losses)

        correlation = np.corrcoef(spectral_shifts, losses)[0, 1]
        return correlation

class RealAGITrainer:
    """真实AGI训练器 - 集成标准数据集和基准测试"""

    def __init__(self, dataset_name: str = "cifar10", device: str = "cpu"):
        self.dataset_name = dataset_name
        self.device = device

        # 数据集配置
        self.dataset_config = {
            'mnist': {'num_classes': 10, 'input_shape': (1, 28, 28)},
            'fashion_mnist': {'num_classes': 10, 'input_shape': (1, 28, 28)},
            'cifar10': {'num_classes': 10, 'input_shape': (3, 32, 32)},
            'cifar100': {'num_classes': 100, 'input_shape': (3, 32, 32)}
        }

        # 初始化组件
        self.dataset_loader = StandardDatasetLoader(dataset_name, device=device)
        self.train_loader, self.val_loader, self.test_loader = self.dataset_loader.load_dataset()

        # 创建模型
        config = self.dataset_config[dataset_name.lower()]
        self.model = BenchmarkModel(dataset_name, config['num_classes']).to(device)

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # 谱稳定性控制器 - 现在启用并适配标准数据集
        self.spectral_controller = None
        self.riemann_loss = None
        if ADVANCED_SPECTRAL_AVAILABLE:
            try:
                # 为不同数据集创建合适维度的控制器
                if dataset_name.lower() in ['mnist', 'fashion_mnist']:
                    # MNIST类数据集输出维度为10
                    self.spectral_controller = AdvancedSpectralController(dim=10)
                elif dataset_name.lower() == 'cifar10':
                    # CIFAR-10输出维度为10
                    self.spectral_controller = AdvancedSpectralController(dim=10)
                elif dataset_name.lower() == 'cifar100':
                    # CIFAR-100输出维度为100
                    self.spectral_controller = AdvancedSpectralController(dim=100)
                else:
                    # 默认维度
                    self.spectral_controller = AdvancedSpectralController(dim=64)

                self.riemann_loss = RiemannSpectralLoss()
                logger.info(f"✅ 谱稳定性控制器已启用 - 维度: {self.spectral_controller.dim}")
            except Exception as e:
                logger.warning(f"谱稳定性控制器初始化失败: {e}")
                self.spectral_controller = None
        else:
            logger.info("ℹ️ 高级谱稳定性控制器不可用，使用传统稳定性跟踪")

        # 稳定性跟踪器
        self.stability_tracker = SpectralStabilityTracker()

        # 训练状态
        self.step = 0
        self.best_accuracy = 0.0
        self.best_loss = float('inf')

        # 基准测试结果
        self.benchmark_results = []

        logger.info(f"🎯 初始化真实AGI训练器 - 数据集: {dataset_name}")

    def train_step(self) -> Dict[str, float]:
        """执行训练步骤"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 添加谱稳定性损失
            if self.spectral_controller is not None:
                try:
                    # 计算谱稳定性指标
                    stability_score, stability_metrics = self.spectral_controller.compute_stability(outputs)
                    spectral_loss = self.riemann_loss(stability_metrics)
                    loss = loss + 0.1 * spectral_loss
                except Exception as e:
                    # 如果谱稳定性计算失败，使用简单的正则化
                    logger.warning(f"谱稳定性计算失败，使用替代方案: {e}")
                    spectral_loss = 0.01 * torch.norm(outputs, p=2)
                    loss = loss + spectral_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx >= 10:  # 限制每个epoch的批次数量
                break

        accuracy = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)

        # 计算谱稳定性指标
        spectral_shift = 0.0
        if self.spectral_controller is not None:
            try:
                stability_score, _ = self.spectral_controller.compute_stability(outputs)
                spectral_shift = stability_score
            except Exception as e:
                logger.warning(f"谱稳定性指标计算失败: {e}")
                spectral_shift = 0.0

        # 更新稳定性跟踪
        stability_score = self.stability_tracker.update_stability(spectral_shift, avg_loss)

        self.step += 1

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'spectral_shift': spectral_shift,
            'stability_score': stability_score
        }

    def validate(self) -> Dict[str, float]:
        """验证模型性能"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def benchmark_test(self) -> Dict[str, float]:
        """在测试集上进行基准测试"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_accuracy = 100. * correct / total

        # 记录基准测试结果
        result = {
            'dataset': self.dataset_name,
            'test_accuracy': test_accuracy,
            'step': self.step,
            'timestamp': datetime.now().isoformat()
        }
        self.benchmark_results.append(result)

        return result

    def cross_validate_stability(self) -> Dict[str, float]:
        """交叉验证谱稳定性指标与学习效果的相关性"""
        correlation = self.stability_tracker.get_correlation()

        # 计算稳定性指标的预测能力
        if len(self.stability_tracker.stability_history) >= 20:
            recent_data = self.stability_tracker.stability_history[-20:]
            stability_scores = [d['stability_score'] for d in recent_data]
            losses = [d['loss'] for d in recent_data]

            # 计算稳定性分数对损失的预测相关性
            stability_correlation = np.corrcoef(stability_scores, losses)[0, 1]

            # 计算稳定性改善趋势
            stability_trend = np.polyfit(range(len(stability_scores)), stability_scores, 1)[0]

            return {
                'spectral_loss_correlation': correlation,
                'stability_prediction_correlation': stability_correlation,
                'stability_trend': stability_trend,
                'validation_samples': len(recent_data)
            }

        return {
            'spectral_loss_correlation': correlation,
            'stability_prediction_correlation': 0.0,
            'stability_trend': 0.0,
            'validation_samples': len(self.stability_tracker.stability_history)
        }

    def save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = f"real_agi_checkpoint_{self.dataset_name}_{self.step}.pth"
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'benchmark_results': self.benchmark_results
        }, checkpoint_path)
        logger.info(f"💾 检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.step = checkpoint['step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.benchmark_results = checkpoint.get('benchmark_results', [])
            logger.info(f"✅ 检查点已加载: {checkpoint_path}")
            return True
        return False

def main():
    """主训练函数"""
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ 使用设备: {device}")

    # 支持的数据集
    supported_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    # 为每个数据集创建训练器并运行基准测试
    benchmark_summary = {}

    for dataset_name in supported_datasets:
        try:
            logger.info(f"🚀 开始训练数据集: {dataset_name}")

            # 创建训练器
            trainer = RealAGITrainer(dataset_name=dataset_name, device=device)

            # 尝试加载检查点
            checkpoint_path = f"real_agi_checkpoint_{dataset_name}_latest.pth"
            trainer.load_checkpoint(checkpoint_path)

            # 训练循环
            for epoch in range(5):  # 每个数据集训练5个epoch
                logger.info(f"📈 Epoch {epoch + 1}/5 - 数据集: {dataset_name}")

                # 训练步骤
                train_metrics = trainer.train_step()
                val_metrics = trainer.validate()

                logger.info(f"   训练损失: {train_metrics['loss']:.4f}")
                logger.info(f"   训练准确率: {train_metrics['accuracy']:.2f}%")
                logger.info(f"   验证损失: {val_metrics['val_loss']:.4f}")
                logger.info(f"   验证准确率: {val_metrics['val_accuracy']:.2f}%")
                logger.info(f"   谱移η实部: {train_metrics['spectral_shift']:.4f}")

                # 更新最佳性能
                if val_metrics['val_accuracy'] > trainer.best_accuracy:
                    trainer.best_accuracy = val_metrics['val_accuracy']
                    trainer.best_loss = val_metrics['val_loss']

                # 每10步进行一次交叉验证
                if trainer.step % 10 == 0:
                    stability_metrics = trainer.cross_validate_stability()
                    logger.info(f"   谱稳定性相关性: {stability_metrics['spectral_loss_correlation']:.4f}")
                    logger.info(f"   稳定性预测相关性: {stability_metrics['stability_prediction_correlation']:.4f}")

            # 最终基准测试
            benchmark_result = trainer.benchmark_test()
            logger.info(f"🎯 {dataset_name} 最终测试准确率: {benchmark_result['test_accuracy']:.2f}%")

            # 保存结果
            trainer.save_checkpoint()
            benchmark_summary[dataset_name] = benchmark_result

            # 内存清理
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ 训练数据集 {dataset_name} 失败: {e}")
            continue

    # 生成基准测试报告
    generate_benchmark_report(benchmark_summary)

def generate_benchmark_report(results: Dict[str, Dict]):
    """生成基准测试报告"""
    report_path = "real_agi_benchmark_report.json"

    # 计算平均性能
    accuracies = [result['test_accuracy'] for result in results.values() if 'test_accuracy' in result]
    avg_accuracy = np.mean(accuracies) if accuracies else 0.0

    # 找到最佳表现的数据集
    best_dataset = None
    if results:
        best_dataset = max(results.keys(), key=lambda k: results[k].get('test_accuracy', 0))
    else:
        best_dataset = "none"

    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'datasets_tested': len(results),
            'average_accuracy': avg_accuracy,
            'best_performing_dataset': best_dataset
        },
        'detailed_results': results,
        'comparison_with_baselines': {
            'mnist': {
                'h2q_evo_accuracy': results.get('mnist', {}).get('test_accuracy', 0),
                'baseline_cnn': 99.2,  # 典型CNN基准
                'baseline_resnet': 99.6,  # ResNet基准
                'improvement_over_cnn': results.get('mnist', {}).get('test_accuracy', 0) - 99.2
            },
            'cifar10': {
                'h2q_evo_accuracy': results.get('cifar10', {}).get('test_accuracy', 0),
                'baseline_cnn': 78.5,  # 典型CNN基准
                'baseline_resnet': 92.1,  # ResNet基准
                'improvement_over_cnn': results.get('cifar10', {}).get('test_accuracy', 0) - 78.5
            }
        }
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"📊 基准测试报告已生成: {report_path}")
    logger.info(f"📈 平均准确率: {avg_accuracy:.2f}%")

if __name__ == "__main__":
    main()