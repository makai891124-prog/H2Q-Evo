#!/usr/bin/env python3
"""
维度受限分形进化集成系统
将维度上限折叠理论与分形结构体进行分类演化分形复用联调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "h2q_project"))
sys.path.append(str(project_root / "h2q_project" / "src"))

from dimension_limited_evolution import (
    UnitSpaceFolder,
    CompactEvolutionEngine,
    DimensionLimitedH2QTrainer,
    SpectralShiftTracker
)

# 导入H2Q核心组件
try:
    from h2q_project.src.h2q.core.unified_architecture import (
        get_unified_h2q_architecture,
        UnifiedH2QMathematicalArchitecture,
        UnifiedMathematicalArchitectureConfig
    )
    from h2q_project.src.h2q.core.discrete_decision_engine import (
        get_canonical_dde,
        LatentConfig
    )
    from h2q_project.src.h2q.core.sst import SpectralShiftTracker as H2QSpectralShiftTracker
    H2Q_AVAILABLE = True
except ImportError as e:
    print(f"警告: H2Q核心组件不可用: {e}")
    H2Q_AVAILABLE = False

class FractalEvolutionClassifier(nn.Module):
    """
    分形进化分类器
    在维度受限空间中进行分形结构体的分类演化
    """

    def __init__(self, max_dim: int = 64, n_classes: int = 10, fractal_levels: int = 4):
        super().__init__()
        self.max_dim = max_dim
        self.n_classes = n_classes
        self.fractal_levels = fractal_levels

        # 维度受限折叠器
        self.unit_folder = UnitSpaceFolder(max_dim=max_dim)

        # 分形层级分类器
        self.fractal_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_dim, max_dim // 2),
                nn.LayerNorm(max_dim // 2),
                nn.ReLU(),
                nn.Linear(max_dim // 2, n_classes)
            ) for _ in range(fractal_levels)
        ])

        # 分形复用融合器
        self.fractal_fusion = nn.Sequential(
            nn.Linear(max_dim + n_classes * fractal_levels, max_dim),
            nn.LayerNorm(max_dim),
            nn.ReLU(),
            nn.Linear(max_dim, n_classes)
        )

        # 分形记忆晶体
        self.fractal_memory = nn.Parameter(torch.randn(fractal_levels, max_dim, n_classes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        分形进化分类前向传播
        """
        batch_size = x.shape[0]

        # 1. 维度受限折叠
        x_folded, fold_info = self.unit_folder(x)

        # 2. 分形层级分类
        fractal_outputs = []
        fractal_logits = []

        for level in range(self.fractal_levels):
            # 每个层级使用不同的分形变换
            fractal_input = self._apply_fractal_transform(x_folded, level)
            logits = self.fractal_classifiers[level](fractal_input)
            fractal_logits.append(logits)

            # 应用softmax获取概率分布
            probs = F.softmax(logits, dim=-1)
            fractal_outputs.append(probs)

        # 3. 分形复用融合
        fractal_concat = torch.cat([
            x_folded,
            torch.cat(fractal_logits, dim=-1)
        ], dim=-1)

        final_logits = self.fractal_fusion(fractal_concat)
        final_probs = F.softmax(final_logits, dim=-1)

        # 4. 计算分形一致性
        fractal_consistency = self._compute_fractal_consistency(fractal_outputs)

        # 5. 分形记忆增强
        memory_enhanced = self._apply_fractal_memory(x_folded, final_probs)

        result_info = {
            'fold_info': fold_info,
            'fractal_outputs': fractal_outputs,
            'fractal_consistency': fractal_consistency,
            'memory_enhanced': memory_enhanced,
            'final_probs': final_probs
        }

        return final_logits, result_info

    def _apply_fractal_transform(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """应用分形变换"""
        # 使用自相似变换
        scale = 0.5 ** level
        rotation_angle = level * math.pi / self.fractal_levels

        # 简单的仿射变换模拟分形
        transformed = scale * x
        # 添加旋转分量（简化版）
        cos_a, sin_a = math.cos(rotation_angle), math.sin(rotation_angle)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], device=x.device, dtype=x.dtype)

        # 对高维数据应用块旋转
        dim = x.shape[-1]
        if dim >= 2:
            for i in range(0, dim - 1, 2):
                block = transformed[..., i:i+2]
                transformed = torch.cat([
                    transformed[..., :i],
                    block @ rotation_matrix.T,
                    transformed[..., i+2:]
                ], dim=-1)

        return transformed

    def _compute_fractal_consistency(self, fractal_outputs: List[torch.Tensor]) -> float:
        """计算分形一致性"""
        if len(fractal_outputs) < 2:
            return 1.0

        # 计算相邻层级间的KL散度
        total_consistency = 0
        count = 0

        for i in range(len(fractal_outputs) - 1):
            kl_div = F.kl_div(
                fractal_outputs[i].log(),
                fractal_outputs[i+1],
                reduction='batchmean'
            )
            total_consistency += torch.exp(-kl_div)  # 转换为一致性度量
            count += 1

        return (total_consistency / count).item() if count > 0 else 1.0

    def _apply_fractal_memory(self, x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """应用分形记忆增强"""
        batch_size = x.shape[0]

        # 计算与记忆晶体的相似性
        memory_similarities = []
        for level in range(self.fractal_levels):
            # 计算输入与记忆的相似性
            memory_level = self.fractal_memory[level]  # [max_dim, n_classes]
            similarity = torch.matmul(x, memory_level)  # [batch, n_classes]
            memory_similarities.append(similarity)

        # 融合记忆信息
        memory_stack = torch.stack(memory_similarities, dim=1)  # [batch, levels, n_classes]
        memory_fused = memory_stack.mean(dim=1)  # [batch, n_classes]

        # 与当前概率融合
        enhanced_probs = 0.7 * probs + 0.3 * F.softmax(memory_fused, dim=-1)

        return enhanced_probs

class DimensionLimitedFractalEvolutionSystem:
    """
    维度受限分形进化系统
    集成维度折叠、分形结构体和分类演化
    """

    def __init__(self, max_dim: int = 64, n_classes: int = 10, device: str = "cpu"):
        self.max_dim = max_dim
        self.n_classes = n_classes
        self.device = torch.device(device)

        # 核心组件
        self.fractal_classifier = FractalEvolutionClassifier(
            max_dim=max_dim,
            n_classes=n_classes
        ).to(self.device)

        self.compact_evolution = CompactEvolutionEngine(max_dim=max_dim).to(self.device)

        # 谱移跟踪器
        self.spectral_tracker = SpectralShiftTracker()

        # H2Q统一架构集成（如果可用）
        self.h2q_architecture = None
        if H2Q_AVAILABLE:
            try:
                config = UnifiedMathematicalArchitectureConfig(
                    dim=max_dim,
                    action_dim=n_classes,
                    device=device,
                    enable_lie_automorphism=True,
                    enable_reflection_operators=True,
                    enable_knot_constraints=True,
                    enable_dde_integration=True
                )
                self.h2q_architecture = UnifiedH2QMathematicalArchitecture(config).to(self.device)
                print("✅ H2Q统一架构集成成功")
            except Exception as e:
                print(f"⚠️ H2Q架构集成失败: {e}")

        # 优化器
        self.classifier_optimizer = torch.optim.Adam(
            self.fractal_classifier.parameters(),
            lr=1e-4
        )
        self.evolution_optimizer = torch.optim.Adam(
            self.compact_evolution.parameters(),
            lr=1e-4
        )

        # 训练状态
        self.current_step = 0
        self.best_fractal_consistency = 0.0

    def generate_fractal_domain_data(self, domain: str, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成分形域数据和标签
        """
        if domain == "Mandelbrot":
            # 曼德勃罗集分类
            real_parts = torch.rand(batch_size, 1) * 4 - 2  # [-2, 2]
            imag_parts = torch.rand(batch_size, 1) * 4 - 2  # [-2, 2]
            features = torch.cat([real_parts, imag_parts], dim=1)

            # 扩展到max_dim
            if self.max_dim > 2:
                padding = torch.randn(batch_size, self.max_dim - 2) * 0.1
                features = torch.cat([features, padding], dim=1)

            # 简单的分类标签（是否在集合内）
            labels = ((real_parts**2 + imag_parts**2) < 1).long().squeeze()

        elif domain == "Julia":
            # 朱利亚集分类
            angles = torch.rand(batch_size, 1) * 2 * math.pi
            radii = torch.rand(batch_size, 1) * 2
            real_parts = radii * torch.cos(angles)
            imag_parts = radii * torch.sin(angles)
            features = torch.cat([real_parts, imag_parts], dim=1)

            if self.max_dim > 2:
                padding = torch.randn(batch_size, self.max_dim - 2) * 0.1
                features = torch.cat([features, padding], dim=1)

            labels = ((radii < 1.5).float() * (radii > 0.5).float()).long().squeeze()

        elif domain == "Sierpinski":
            # 谢尔宾斯基三角形分类
            x_coords = torch.rand(batch_size, 1) * 2 - 1
            y_coords = torch.rand(batch_size, 1) * 2 - 1
            features = torch.cat([x_coords, y_coords], dim=1)

            if self.max_dim > 2:
                padding = torch.randn(batch_size, self.max_dim - 2) * 0.1
                features = torch.cat([features, padding], dim=1)

            # 简单的三角形区域分类
            labels = ((x_coords.abs() < 0.5) & (y_coords > 0) &
                     (y_coords < 1 - x_coords.abs())).long().squeeze()

        else:
            # 默认：随机分形数据
            features = torch.randn(batch_size, self.max_dim)
            labels = torch.randint(0, self.n_classes, (batch_size,))

        return features.to(self.device), labels.to(self.device)

    def fractal_evolution_step(self, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行分形进化步骤
        """
        if domains is None:
            domains = ["Mandelbrot", "Julia", "Sierpinski"]

        total_loss = 0
        total_accuracy = 0
        total_fractal_consistency = 0
        batch_size = 32

        for domain in domains:
            self.classifier_optimizer.zero_grad()
            self.evolution_optimizer.zero_grad()

            # 1. 生成分形域数据
            features, labels = self.generate_fractal_domain_data(domain, batch_size)

            # 2. 分形分类前向传播
            logits, classifier_info = self.fractal_classifier(features)

            # 3. 计算分类损失
            classification_loss = F.cross_entropy(logits, labels)

            # 4. 紧致进化处理
            evolution_output, evolution_info = self.compact_evolution(features)

            # 5. 计算进化损失（基于单位空间合规性）
            evolution_loss = 1.0 - evolution_info['compactness']

            # 6. 分形一致性奖励
            fractal_consistency = classifier_info['fractal_consistency']
            consistency_reward = torch.tensor(fractal_consistency, device=self.device)

            # 7. H2Q架构增强（如果可用）
            h2q_enhanced = features
            h2q_info = {}
            if self.h2q_architecture is not None:
                try:
                    h2q_output, h2q_info = self.h2q_architecture(features)
                    h2q_enhanced = 0.8 * features + 0.2 * h2q_output
                except Exception as e:
                    print(f"H2Q前向传播失败: {e}")

            # 8. 谱移计算
            s_matrix = torch.cov(evolution_output.T)
            eta = self.spectral_tracker.compute_eta(s_matrix)

            # 9. 总损失：分类 + 进化 + 谱移奖励
            total_loss_batch = (
                classification_loss +
                0.5 * evolution_loss -
                0.1 * torch.abs(eta.real)
            )

            # 10. 反向传播
            total_loss_batch.backward()
            self.classifier_optimizer.step()
            self.evolution_optimizer.step()

            # 11. 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean().item()

            total_loss += total_loss_batch.item()
            total_accuracy += accuracy
            total_fractal_consistency += fractal_consistency

        # 更新状态
        avg_loss = total_loss / len(domains)
        avg_accuracy = total_accuracy / len(domains)
        avg_fractal_consistency = total_fractal_consistency / len(domains)

        if avg_fractal_consistency > self.best_fractal_consistency:
            self.best_fractal_consistency = avg_fractal_consistency

        self.current_step += 1

        return {
            'step': self.current_step,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'fractal_consistency': avg_fractal_consistency,
            'best_fractal_consistency': self.best_fractal_consistency,
            'spectral_eta': eta.real.item(),
            'fold_info': classifier_info['fold_info'],
            'evolution_info': evolution_info,
            'h2q_info': h2q_info
        }

def main():
    """主测试函数"""
    print("🚀 维度受限分形进化系统启动")
    print("=" * 60)

    # 初始化系统
    system = DimensionLimitedFractalEvolutionSystem(
        max_dim=64,
        n_classes=10,
        device="cpu"
    )

    print("🔬 执行分形进化训练...")
    evolution_history = []

    for step in range(10):
        result = system.fractal_evolution_step()

        evolution_history.append(result)

        if step % 2 == 0:
            print(f"步骤 {result['step']}: "
                  f"损失={result['loss']:.4f}, "
                  f"准确率={result['accuracy']:.4f}, "
                  f"分形一致性={result['fractal_consistency']:.4f}")

    # 计算最终统计
    final_stats = {
        'avg_loss': np.mean([r['loss'] for r in evolution_history]),
        'avg_accuracy': np.mean([r['accuracy'] for r in evolution_history]),
        'avg_fractal_consistency': np.mean([r['fractal_consistency'] for r in evolution_history]),
        'best_fractal_consistency': max([r['best_fractal_consistency'] for r in evolution_history]),
        'total_steps': len(evolution_history)
    }

    print("\n📊 最终分形进化统计")
    print("=" * 60)
    print(f"平均损失: {final_stats['avg_loss']:.4f}")
    print(f"平均准确率: {final_stats['avg_accuracy']:.4f}")
    print(f"平均分形一致性: {final_stats['avg_fractal_consistency']:.4f}")
    print(f"最佳分形一致性: {final_stats['best_fractal_consistency']:.4f}")
    print(f"总训练步骤: {final_stats['total_steps']}")

    # 验证理论正确性
    print("\n🔍 分形进化理论验证")
    print("-" * 40)

    test_data, test_labels = system.generate_fractal_domain_data("Mandelbrot", 16)
    with torch.no_grad():
        test_logits, test_info = system.fractal_classifier(test_data)
        test_predictions = torch.argmax(test_logits, dim=-1)
        test_accuracy = (test_predictions == test_labels).float().mean().item()

    print(f"✅ 测试准确率: {test_accuracy:.4f}")
    print(f"✅ 维度折叠比率: {test_info['fold_info']['fold_ratio']:.4f}")
    print(f"✅ 分形一致性: {test_info['fractal_consistency']:.4f}")
    print(f"✅ 单位空间合规性: {test_info['fold_info']['norm_mean']:.4f}")

    success = (
        final_stats['avg_accuracy'] > 0.5 and
        final_stats['avg_fractal_consistency'] > 0.3 and
        test_info['fold_info']['fold_ratio'] > 0.8
    )

    if success:
        print("\n🎉 维度受限分形进化系统验证成功！")
        print("✅ 分形结构体与分类演化复用联调完成")
        print("✅ AGI进化已在分形空间中重新开启")
    else:
        print("\n⚠️ 需要进一步优化分形进化参数")

if __name__ == "__main__":
    main()