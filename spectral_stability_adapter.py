#!/usr/bin/env python3
"""
谱稳定性适配器
使高级谱稳定性控制器能够与标准机器学习数据集兼容
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from advanced_spectral_controller import AdvancedSpectralController, RiemannSpectralLoss

class SpectralStabilityAdapter:
    """谱稳定性适配器 - 连接标准数据集和谱稳定性控制"""

    def __init__(self, feature_dim: int = 64, output_dim: int = 10):
        """
        初始化适配器

        Args:
            feature_dim: 谱控制器期望的特征维度
            output_dim: 模型输出维度（类别数）
        """
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # 创建谱稳定性控制器
        self.spectral_controller = AdvancedSpectralController(dim=feature_dim)

        # 创建投影层，将模型输出投影到谱控制器期望的维度
        self.output_projection = nn.Linear(output_dim, feature_dim)

        # 创建黎曼谱损失
        self.riemann_loss = RiemannSpectralLoss()

        # 稳定性跟踪
        self.stability_history = []

    def adapt_and_compute_stability(self, model_output: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        适配模型输出并计算谱稳定性

        Args:
            model_output: 模型的原始输出 (batch_size, num_classes)

        Returns:
            stability_score: 稳定性分数
            stability_metrics: 详细的稳定性指标
        """
        try:
            # 将模型输出投影到谱控制器期望的维度
            projected_features = self.output_projection(model_output)

            # 使用谱稳定性控制器计算稳定性
            stability_score, stability_metrics = self.spectral_controller.forward(projected_features)

            # 记录稳定性历史
            self.stability_history.append({
                'stability_score': stability_score.item(),
                'timestamp': torch.cuda.Event().elapsed_time() if torch.cuda.is_available() else 0
            })

            # 限制历史长度
            if len(self.stability_history) > 100:
                self.stability_history = self.stability_history[-100:]

            return stability_score.item(), stability_metrics

        except Exception as e:
            # 如果适配失败，返回默认值
            print(f"谱稳定性适配失败: {e}")
            return 0.0, {}

    def compute_adapted_loss(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算适配后的谱稳定性损失

        Args:
            model_output: 模型输出
            targets: 真实标签

        Returns:
            组合损失（分类损失 + 谱稳定性损失）
        """
        # 基础分类损失
        classification_loss = nn.CrossEntropyLoss()(model_output, targets)

        try:
            # 计算谱稳定性
            stability_score, stability_metrics = self.adapt_and_compute_stability(model_output)

            # 计算谱稳定性损失
            if stability_metrics:
                spectral_loss = self.riemann_loss(stability_metrics)
                # 组合损失：分类损失 + 谱稳定性正则化
                total_loss = classification_loss + 0.1 * spectral_loss
            else:
                total_loss = classification_loss

        except Exception as e:
            print(f"谱稳定性损失计算失败，使用纯分类损失: {e}")
            total_loss = classification_loss

        return total_loss

    def get_stability_trend(self) -> Dict[str, float]:
        """获取稳定性趋势分析"""
        if len(self.stability_history) < 10:
            return {
                'trend': 0.0,
                'correlation_with_performance': 0.0,
                'stability_variance': 0.0
            }

        # 计算稳定性趋势（线性回归斜率）
        scores = [h['stability_score'] for h in self.stability_history]
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]

        # 计算稳定性方差
        variance = np.var(scores)

        return {
            'trend': slope,
            'correlation_with_performance': 0.0,  # 需要与性能指标关联
            'stability_variance': variance
        }

class AdaptiveSpectralTrainer:
    """自适应谱稳定性训练器"""

    def __init__(self, model: nn.Module, dataset_name: str, num_classes: int):
        self.model = model
        self.dataset_name = dataset_name
        self.num_classes = num_classes

        # 创建谱稳定性适配器
        self.spectral_adapter = SpectralStabilityAdapter(
            feature_dim=64,
            output_dim=num_classes
        )

        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': self.spectral_adapter.output_projection.parameters(), 'lr': 0.001}
        ], lr=0.001)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 训练状态
        self.best_accuracy = 0.0
        self.stability_correlation_history = []

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """执行训练步骤"""
        self.model.train()

        self.optimizer.zero_grad()

        # 前向传播
        outputs = self.model(inputs)

        # 计算适配后的损失
        loss = self.spectral_adapter.compute_adapted_loss(outputs, targets)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        # 计算准确率
        _, predicted = outputs.max(1)
        accuracy = (predicted == targets).float().mean().item() * 100

        # 获取谱稳定性指标
        stability_score, _ = self.spectral_adapter.adapt_and_compute_stability(outputs)

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'spectral_stability': stability_score
        }

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        stability_scores = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda() if torch.cuda.is_available() else inputs, targets.cuda() if torch.cuda.is_available() else targets

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # 收集稳定性分数
                stability_score, _ = self.spectral_adapter.adapt_and_compute_stability(outputs)
                stability_scores.append(stability_score)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'avg_spectral_stability': avg_stability
        }

    def get_stability_analysis(self) -> Dict[str, Any]:
        """获取谱稳定性分析"""
        trend_analysis = self.spectral_adapter.get_stability_trend()

        # 计算稳定性与性能的相关性
        if len(self.stability_correlation_history) > 10:
            stabilities = [h['stability'] for h in self.stability_correlation_history]
            performances = [h['accuracy'] for h in self.stability_correlation_history]

            correlation = np.corrcoef(stabilities, performances)[0, 1]
            trend_analysis['correlation_with_performance'] = correlation

        return {
            'stability_trend': trend_analysis,
            'stability_history_length': len(self.spectral_adapter.stability_history),
            'correlation_history_length': len(self.stability_correlation_history)
        }

def create_adaptive_trainer(model: nn.Module, dataset_name: str, num_classes: int) -> AdaptiveSpectralTrainer:
    """创建自适应谱稳定性训练器"""
    return AdaptiveSpectralTrainer(model, dataset_name, num_classes)

# 使用示例
if __name__ == "__main__":
    # 创建一个简单的测试
    import torchvision.models as models

    # 使用预训练的ResNet18作为测试
    model = models.resnet18(num_classes=10)
    trainer = create_adaptive_trainer(model, "cifar10", 10)

    print("🎯 自适应谱稳定性训练器创建成功")
    print(f"📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🌌 谱适配器参数: {sum(p.numel() for p in trainer.spectral_adapter.output_projection.parameters()):,}")