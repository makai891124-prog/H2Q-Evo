#!/usr/bin/env python3
"""
AGI进化损失指标系统集成模块

将AGI进化损失指标系统集成到现有的H2Q-Evo进化框架中
"""

import torch
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from agi_evolution_loss_metrics import (
    AGI_EvolutionLossSystem,
    MathematicalCoreMetrics,
    create_agi_evolution_loss_system,
    get_mathematical_core_metrics_from_system_report
)

logger = logging.getLogger(__name__)


class AGI_EvolutionLossIntegration:
    """
    AGI进化损失指标系统集成器

    将损失指标系统集成到evolution_system.py中
    """

    def __init__(self, project_root: str = "/Users/imymm/H2Q-Evo"):
        self.project_root = Path(project_root)
        self.loss_system: Optional[AGI_EvolutionLossSystem] = None
        self.integration_log = []

        # 配置文件路径
        self.config_file = self.project_root / "agi_evolution_loss_config.json"
        self.checkpoint_dir = self.project_root / "agi_evolution_loss_checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # 初始化损失系统
        self._initialize_loss_system()

        logger.info("AGI进化损失指标系统集成器已初始化")

    def _initialize_loss_system(self):
        """初始化损失指标系统"""
        try:
            # 尝试加载配置
            config = self._load_config()

            # 创建损失系统
            self.loss_system = create_agi_evolution_loss_system(config)

            # 尝试加载检查点
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
            if checkpoint_path.exists():
                try:
                    self.loss_system.load_checkpoint(str(checkpoint_path))
                    logger.info(f"加载了损失系统检查点: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"加载检查点失败，将使用新系统: {e}")

            logger.info("AGI进化损失指标系统初始化成功")

        except Exception as e:
            logger.error(f"初始化损失系统失败: {e}")
            # 创建默认系统
            self.loss_system = create_agi_evolution_loss_system()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                'capability_dims': {
                    'mathematical_reasoning': 256,
                    'creative_problem_solving': 256,
                    'knowledge_integration': 256,
                    'emergent_capabilities': 256
                },
                'knowledge_dim': 256,
                'memory_size': 1000,
                'emergence_window': 50,
                'stability_window': 100
            }

    def _save_config(self, config: Dict[str, Any]):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def compute_evolution_loss(self,
                              capability_embeddings: Dict[str, torch.Tensor],
                              current_performance: Dict[str, float],
                              new_knowledge: Optional[torch.Tensor] = None,
                              existing_knowledge: Optional[list] = None,
                              current_state: Optional[torch.Tensor] = None,
                              mathematical_core_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        计算AGI进化损失

        Args:
            capability_embeddings: 各能力的嵌入表示
            current_performance: 当前性能得分
            new_knowledge: 新知识嵌入
            existing_knowledge: 现有知识列表
            current_state: 当前系统状态
            mathematical_core_report: 数学核心机系统报告

        Returns:
            进化损失结果
        """
        if self.loss_system is None:
            logger.error("损失系统未初始化")
            return {}

        try:
            # 从数学核心机报告提取指标
            mathematical_metrics = MathematicalCoreMetrics()
            if mathematical_core_report:
                mathematical_metrics = get_mathematical_core_metrics_from_system_report(
                    mathematical_core_report
                )

            # 计算损失
            loss_components = self.loss_system(
                capability_embeddings=capability_embeddings,
                current_performance=current_performance,
                new_knowledge=new_knowledge,
                existing_knowledge=existing_knowledge or [],
                current_state=current_state or torch.randn(256),
                mathematical_metrics=mathematical_metrics
            )

            # 记录集成日志
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'generation': loss_components.generation,
                'losses': loss_components.__dict__,
                'performance': current_performance
            }
            self.integration_log.append(log_entry)

            # 定期保存检查点
            if loss_components.generation % 10 == 0:
                self._save_checkpoint()

            return {
                'loss_components': loss_components.__dict__,
                'evolution_report': self.loss_system.get_evolution_report(),
                'mathematical_metrics': mathematical_metrics.__dict__
            }

        except Exception as e:
            logger.error(f"计算进化损失失败: {e}")
            return {}

    def _save_checkpoint(self):
        """保存检查点"""
        if self.loss_system:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_gen_{self.loss_system.generation_count}.pt"
            self.loss_system.save_checkpoint(str(checkpoint_path))

            # 更新最新检查点链接
            latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(checkpoint_path)

    def get_integration_report(self) -> Dict[str, Any]:
        """获取集成报告"""
        if self.loss_system is None:
            return {}

        return {
            'loss_system_status': 'active' if self.loss_system else 'inactive',
            'current_generation': self.loss_system.generation_count if self.loss_system else 0,
            'total_integration_calls': len(self.integration_log),
            'evolution_report': self.loss_system.get_evolution_report() if self.loss_system else {},
            'recent_logs': self.integration_log[-5:] if self.integration_log else []
        }

    def optimize_loss_weights(self, target_performance: Dict[str, float]):
        """
        基于目标性能优化损失权重

        Args:
            target_performance: 目标性能水平
        """
        if self.loss_system is None:
            return

        # 简单的权重优化逻辑
        current_report = self.loss_system.get_evolution_report()

        # 根据性能差距调整权重
        performance_gaps = {}
        for capability, target in target_performance.items():
            current = current_report.get('average_losses', {}).get(capability.replace('_', ''), 0)
            performance_gaps[capability] = max(0, target - current)

        # 归一化差距作为权重
        total_gap = sum(performance_gaps.values())
        if total_gap > 0:
            new_weights = torch.tensor([
                performance_gaps.get('mathematical_reasoning', 0),
                performance_gaps.get('knowledge_integration', 0),
                performance_gaps.get('emergent_capabilities', 0),
                performance_gaps.get('stability', 0)
            ]) / total_gap

            # 更新权重
            self.loss_system.loss_weights.data = new_weights
            logger.info(f"优化了损失权重: {new_weights}")

    def export_integration_data(self, output_file: str):
        """导出集成数据"""
        data = {
            'integration_log': self.integration_log,
            'final_report': self.get_integration_report(),
            'export_timestamp': datetime.now().isoformat()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"导出了集成数据到: {output_file}")


# 全局集成器实例
_evolution_loss_integrator: Optional[AGI_EvolutionLossIntegration] = None


def get_evolution_loss_integrator(project_root: str = "/Users/imymm/H2Q-Evo") -> AGI_EvolutionLossIntegration:
    """获取AGI进化损失集成器实例"""
    global _evolution_loss_integrator
    if _evolution_loss_integrator is None:
        _evolution_loss_integrator = AGI_EvolutionLossIntegration(project_root)
    return _evolution_loss_integrator


def integrate_evolution_loss_into_system(capability_embeddings: Dict[str, torch.Tensor],
                                        current_performance: Dict[str, float],
                                        mathematical_core_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    将进化损失计算集成到现有系统中

    这是一个便捷函数，可以在evolution_system.py中调用
    """
    integrator = get_evolution_loss_integrator()

    return integrator.compute_evolution_loss(
        capability_embeddings=capability_embeddings,
        current_performance=current_performance,
        mathematical_core_report=mathematical_core_report
    )


if __name__ == "__main__":
    # 测试集成器
    print("🚀 测试AGI进化损失指标系统集成器")
    print("=" * 60)

    integrator = get_evolution_loss_integrator()

    # 模拟输入
    capability_embeddings = {
        'mathematical_reasoning': torch.randn(256),
        'creative_problem_solving': torch.randn(256),
        'knowledge_integration': torch.randn(256),
        'emergent_capabilities': torch.randn(256)
    }

    current_performance = {
        'mathematical_reasoning': 0.8,
        'creative_problem_solving': 0.7,
        'knowledge_integration': 0.6,
        'emergent_capabilities': 0.5
    }

    mathematical_core_report = {
        'statistics': {
            'avg_constraint_violation': 0.1,
            'avg_fueter_violation': 0.05
        }
    }

    # 计算损失
    result = integrator.compute_evolution_loss(
        capability_embeddings=capability_embeddings,
        current_performance=current_performance,
        mathematical_core_report=mathematical_core_report
    )

    print("📊 集成结果:")
    if result:
        loss_comp = result['loss_components']
        print(f"  能力提升损失: {loss_comp['capability_improvement_loss']:.4f}")
        print(f"  知识整合损失: {loss_comp['knowledge_integration_loss']:.4f}")
        print(f"  涌现能力损失: {loss_comp['emergent_capability_loss']:.4f}")
        print(f"  稳定性损失: {loss_comp['stability_loss']:.4f}")
        print(f"  总损失: {loss_comp['total_loss']:.4f}")

        print("\n📈 进化报告:")
        report = result['evolution_report']
        print(f"  当前代数: {report['current_generation']}")
        print(f"  总进化步数: {report['total_evolution_steps']}")

    # 获取集成报告
    integration_report = integrator.get_integration_report()
    print(f"\n🔗 集成状态: {integration_report['loss_system_status']}")

    print("\n✅ AGI进化损失指标系统集成测试完成")