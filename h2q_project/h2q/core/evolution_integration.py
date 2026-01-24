"""
数学架构到Evolution System的连接器

将统一的数学架构集成到H2Q-Evo进化系统
"""

import torch
import torch.nn as nn
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .unified_architecture import (
    UnifiedH2QMathematicalArchitecture,
    get_unified_h2q_architecture
)

logger = logging.getLogger(__name__)


class MathematicalArchitectureEvolutionBridge(nn.Module):
    """
    数学架构进化桥接器
    连接统一的数学架构与evolution_system
    """
    
    def __init__(
        self,
        dim: int = 256,
        action_dim: int = 64,
        device: str = "mps",
        checkpoint_dir: Optional[str] = None
    ):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # 获取统一架构
        self.unified_arch = get_unified_h2q_architecture(
            dim=dim,
            action_dim=action_dim,
            device=device
        )
        
        # 进化状态追踪
        self.evolution_history = []
        self.generation_count = 0
        self.mathematical_metrics = {}
        
        logger.info(f"Initialized Mathematical Architecture Evolution Bridge at {datetime.now()}")
    
    def process_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        处理状态通过数学架构
        """
        output, results = self.unified_arch(state)
        
        # 记录指标
        results['generation'] = self.generation_count
        results['timestamp'] = datetime.now().isoformat()
        
        return output, results
    
    def evolution_step(
        self,
        state: torch.Tensor,
        learning_signal: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        进化步骤：应用数学变换并获得改进
        """
        self.generation_count += 1
        
        # 处理状态
        output, results = self.process_state(state)
        
        # 计算进化指标
        evolution_metrics = {
            'generation': self.generation_count,
            'input_norm': torch.norm(state).item(),
            'output_norm': torch.norm(output).item(),
            'state_change': torch.norm(output - state).item(),
        }
        
        # 添加学习信号
        if learning_signal is not None:
            evolution_metrics['learning_signal'] = learning_signal.item() if learning_signal.numel() == 1 else learning_signal.mean().item()
            
            # 应用学习反馈调整融合权重
            self.adjust_fusion_weights(learning_signal)
        
        # 记录历史
        self.evolution_history.append(evolution_metrics)
        
        results['evolution_metrics'] = evolution_metrics
        results['system_report'] = self.unified_arch.get_system_report()
        
        return results
    
    def adjust_fusion_weights(self, signal: torch.Tensor):
        """
        根据学习信号调整融合权重
        """
        # 简单的梯度上升调整
        signal_val = signal.item() if signal.numel() == 1 else signal.mean().item()
        
        with torch.no_grad():
            for key in self.unified_arch.module_fusion_weights:
                param = self.unified_arch.module_fusion_weights[key]
                param.data = param.data + 0.01 * signal_val * torch.randn_like(param)
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """保存检查点"""
        if path is None:
            if self.checkpoint_dir:
                path = f"{self.checkpoint_dir}/math_arch_ckpt_gen{self.generation_count}.pt"
            else:
                path = f"math_arch_ckpt_gen{self.generation_count}.pt"
        
        checkpoint = {
            'generation': self.generation_count,
            'model_state': self.state_dict(),
            'evolution_history': self.evolution_history,
            'mathematical_metrics': self.mathematical_metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        return path
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state'])
        self.generation_count = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_history']
        self.mathematical_metrics = checkpoint['mathematical_metrics']
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def export_metrics_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """导出度量报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'generation_count': self.generation_count,
            'total_steps': len(self.evolution_history),
            'unified_architecture_report': self.unified_arch.get_system_report(),
            'evolution_statistics': self._compute_evolution_statistics(),
            'mathematical_metrics': self.mathematical_metrics,
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Exported metrics to {output_file}")
        
        return report
    
    def _compute_evolution_statistics(self) -> Dict[str, float]:
        """计算进化统计"""
        if not self.evolution_history:
            return {}
        
        stats = {}
        
        # 计算各个指标的统计
        for key in ['input_norm', 'output_norm', 'state_change']:
            values = [e.get(key, 0.0) for e in self.evolution_history]
            if values:
                stats[f'{key}_mean'] = sum(values) / len(values)
                stats[f'{key}_max'] = max(values)
                stats[f'{key}_min'] = min(values)
        
        return stats
    
    def forward(
        self,
        state: torch.Tensor,
        learning_signal: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """前向传播：完整的进化步骤"""
        return self.evolution_step(state, learning_signal)


class H2QEvolutionSystemIntegration:
    """
    H2Q进化系统集成
    将数学架构集成到evolution_system.py
    """
    
    def __init__(self, project_root: str = "/Users/imymm/H2Q-Evo"):
        self.project_root = Path(project_root)
        self.bridge = None
        self.integration_log = []
        
    def initialize_mathematical_architecture(
        self,
        dim: int = 256,
        action_dim: int = 64
    ):
        """初始化数学架构"""
        checkpoint_dir = self.project_root / "training_checkpoints" / "mathematical_architecture"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.bridge = MathematicalArchitectureEvolutionBridge(
            dim=dim,
            action_dim=action_dim,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        self.integration_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'Mathematical Architecture Initialized',
            'config': {
                'dim': dim,
                'action_dim': action_dim,
                'checkpoint_dir': str(checkpoint_dir),
            }
        })
        
        logger.info("Mathematical Architecture initialized successfully")
    
    def process_evolution_cycle(
        self,
        state: torch.Tensor,
        learning_signal: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        处理一个完整的进化循环
        """
        if self.bridge is None:
            raise RuntimeError("Mathematical architecture not initialized")
        
        results = self.bridge(state, learning_signal)
        
        self.integration_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'Evolution Cycle',
            'generation': results['generation'],
            'metrics': results.get('evolution_metrics', {}),
        })
        
        return results
    
    def save_integration_state(self, path: Optional[str] = None) -> str:
        """保存集成状态"""
        if path is None:
            path = str(self.project_root / "mathematical_architecture_state.json")
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'integration_log': self.integration_log,
            'bridge_generation': self.bridge.generation_count if self.bridge else 0,
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved integration state to {path}")
        return path
    
    def export_full_report(self) -> Dict[str, Any]:
        """导出完整报告"""
        if self.bridge is None:
            return {'status': 'not_initialized'}
        
        report = self.bridge.export_metrics_report()
        report['integration_log'] = self.integration_log
        
        return report


# 全局实例（可被evolution_system导入使用）
_global_math_integration = None

def get_h2q_evolution_integration(project_root: str = "/Users/imymm/H2Q-Evo") -> H2QEvolutionSystemIntegration:
    """获取全局数学架构集成"""
    global _global_math_integration
    if _global_math_integration is None:
        _global_math_integration = H2QEvolutionSystemIntegration(project_root)
    return _global_math_integration


def create_mathematical_core_for_evolution_system(
    dim: int = 256,
    action_dim: int = 64,
    project_root: str = "/Users/imymm/H2Q-Evo"
) -> MathematicalArchitectureEvolutionBridge:
    """
    为evolution_system创建数学核心
    """
    integration = get_h2q_evolution_integration(project_root)
    integration.initialize_mathematical_architecture(dim, action_dim)
    return integration.bridge
