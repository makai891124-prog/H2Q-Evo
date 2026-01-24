#!/usr/bin/env python3
"""
H2Q-Evo AGI模型检查点系统
提供训练状态保存、恢复、版本管理和回滚功能
"""

import os
import json
import time
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import torch
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('AGI-Checkpoint')

@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    version: str
    timestamp: float
    generation: int
    accuracy: float
    loss: float
    model_hash: str
    data_hash: str
    config_hash: str
    size_bytes: int
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)

@dataclass
class TrainingState:
    """训练状态"""
    model_weights: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    epoch: int
    step: int
    best_accuracy: float
    best_loss: float
    training_history: List[Dict[str, Any]]
    hyperparameters: Dict[str, Any]

class ModelCheckpointManager:
    """模型检查点管理器"""

    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 50):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata: Dict[str, CheckpointMetadata] = {}

        self._load_metadata()

    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = {
                        k: CheckpointMetadata.from_dict(v) for k, v in data.items()
                    }
                logger.info(f"加载了 {len(self.metadata)} 个检查点元数据")
            except Exception as e:
                logger.error(f"加载检查点元数据失败: {e}")

    def _save_metadata(self):
        """保存元数据"""
        try:
            data = {k: v.to_dict() for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存检查点元数据失败: {e}")

    def _calculate_hash(self, data: Any) -> str:
        """计算数据哈希"""
        if isinstance(data, dict):
            # 对字典进行排序以确保一致性
            data_str = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(data, default=str)
        elif hasattr(data, '__dict__'):
            data_str = json.dumps(data.__dict__, sort_keys=True, default=str)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _get_checkpoint_path(self, version: str) -> Path:
        """获取检查点文件路径"""
        return self.checkpoint_dir / f"checkpoint_{version}.pkl"

    def save_checkpoint(self,
                       training_state: TrainingState,
                       generation: int,
                       accuracy: float,
                       loss: float,
                       description: str = "") -> str:
        """保存检查点"""
        timestamp = time.time()
        version = f"{generation}_{int(timestamp)}"

        # 计算哈希
        model_hash = self._calculate_hash(training_state.model_weights)
        data_hash = self._calculate_hash({
            'epoch': training_state.epoch,
            'step': training_state.step,
            'history': training_state.training_history[-10:]  # 最近10个历史记录
        })
        config_hash = self._calculate_hash(training_state.hyperparameters)

        # 创建元数据
        metadata = CheckpointMetadata(
            version=version,
            timestamp=timestamp,
            generation=generation,
            accuracy=accuracy,
            loss=loss,
            model_hash=model_hash,
            data_hash=data_hash,
            config_hash=config_hash,
            size_bytes=0,  # 稍后计算
            description=description
        )

        checkpoint_path = self._get_checkpoint_path(version)

        try:
            # 保存检查点
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(training_state, f)

            # 计算文件大小
            metadata.size_bytes = checkpoint_path.stat().st_size

            # 保存元数据
            self.metadata[version] = metadata
            self._save_metadata()

            # 清理旧检查点
            self._cleanup_old_checkpoints()

            logger.info(f"检查点已保存: {version} (准确率: {accuracy:.4f}, 损失: {loss:.4f})")
            return version

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            # 清理失败的文件
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise

    def load_checkpoint(self, version: str) -> Optional[TrainingState]:
        """加载检查点"""
        if version not in self.metadata:
            logger.warning(f"检查点版本不存在: {version}")
            return None

        checkpoint_path = self._get_checkpoint_path(version)

        if not checkpoint_path.exists():
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                training_state = pickle.load(f)

            metadata = self.metadata[version]
            logger.info(f"检查点已加载: {version} (生成: {metadata.generation}, "
                       f"准确率: {metadata.accuracy:.4f})")
            return training_state

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None

    def get_best_checkpoint(self, metric: str = 'accuracy') -> Optional[str]:
        """获取最佳检查点版本"""
        if not self.metadata:
            return None

        if metric == 'accuracy':
            best_version = max(self.metadata.keys(),
                             key=lambda v: self.metadata[v].accuracy)
        elif metric == 'loss':
            best_version = min(self.metadata.keys(),
                             key=lambda v: self.metadata[v].loss)
        else:
            logger.warning(f"未知的指标: {metric}")
            return None

        return best_version

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点版本"""
        if not self.metadata:
            return None

        return max(self.metadata.keys(),
                  key=lambda v: self.metadata[v].timestamp)

    def list_checkpoints(self, sort_by: str = 'timestamp',
                        reverse: bool = True) -> List[CheckpointMetadata]:
        """列出所有检查点"""
        checkpoints = list(self.metadata.values())

        if sort_by == 'timestamp':
            checkpoints.sort(key=lambda x: x.timestamp, reverse=reverse)
        elif sort_by == 'accuracy':
            checkpoints.sort(key=lambda x: x.accuracy, reverse=reverse)
        elif sort_by == 'loss':
            checkpoints.sort(key=lambda x: x.loss, reverse=not reverse)  # 损失降序
        elif sort_by == 'generation':
            checkpoints.sort(key=lambda x: x.generation, reverse=reverse)

        return checkpoints

    def delete_checkpoint(self, version: str) -> bool:
        """删除检查点"""
        if version not in self.metadata:
            logger.warning(f"检查点版本不存在: {version}")
            return False

        checkpoint_path = self._get_checkpoint_path(version)

        try:
            # 删除文件
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            # 删除元数据
            del self.metadata[version]
            self._save_metadata()

            logger.info(f"检查点已删除: {version}")
            return True

        except Exception as e:
            logger.error(f"删除检查点失败: {e}")
            return False

    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        if len(self.metadata) <= self.max_checkpoints:
            return

        # 按时间戳排序，保留最新的
        sorted_versions = sorted(self.metadata.keys(),
                               key=lambda v: self.metadata[v].timestamp,
                               reverse=True)

        versions_to_delete = sorted_versions[self.max_checkpoints:]

        for version in versions_to_delete:
            self.delete_checkpoint(version)

        logger.info(f"清理了 {len(versions_to_delete)} 个旧检查点")

    def get_checkpoint_info(self, version: str) -> Optional[Dict[str, Any]]:
        """获取检查点信息"""
        if version not in self.metadata:
            return None

        metadata = self.metadata[version]
        checkpoint_path = self._get_checkpoint_path(version)

        info = metadata.to_dict()
        info['file_exists'] = checkpoint_path.exists()
        info['file_size_mb'] = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0

        return info

    def export_checkpoint(self, version: str, export_path: str) -> bool:
        """导出检查点"""
        training_state = self.load_checkpoint(version)
        if training_state is None:
            return False

        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            with open(export_file, 'wb') as f:
                pickle.dump(training_state, f)

            # 导出元数据
            metadata_file = export_file.with_suffix('.metadata.json')
            metadata = self.metadata[version]
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"检查点已导出: {version} -> {export_file}")
            return True

        except Exception as e:
            logger.error(f"导出检查点失败: {e}")
            return False

    def import_checkpoint(self, import_path: str, description: str = "") -> Optional[str]:
        """导入检查点"""
        import_file = Path(import_path)
        if not import_file.exists():
            logger.error(f"导入文件不存在: {import_path}")
            return None

        try:
            # 加载训练状态
            with open(import_file, 'rb') as f:
                training_state = pickle.load(f)

            # 尝试加载元数据
            metadata_file = import_file.with_suffix('.metadata.json')
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                metadata = CheckpointMetadata.from_dict(metadata_dict)
                metadata.description = description or metadata.description
            else:
                # 创建新的元数据
                timestamp = time.time()
                version = f"imported_{int(timestamp)}"
                metadata = CheckpointMetadata(
                    version=version,
                    timestamp=timestamp,
                    generation=getattr(training_state, 'epoch', 0),
                    accuracy=getattr(training_state, 'best_accuracy', 0.0),
                    loss=getattr(training_state, 'best_loss', float('inf')),
                    model_hash=self._calculate_hash(training_state.model_weights),
                    data_hash=self._calculate_hash({
                        'epoch': getattr(training_state, 'epoch', 0),
                        'step': getattr(training_state, 'step', 0)
                    }),
                    config_hash=self._calculate_hash(getattr(training_state, 'hyperparameters', {})),
                    size_bytes=import_file.stat().st_size,
                    description=description
                )

            # 保存到检查点目录
            checkpoint_path = self._get_checkpoint_path(metadata.version)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(training_state, f)

            # 更新元数据
            self.metadata[metadata.version] = metadata
            self._save_metadata()

            logger.info(f"检查点已导入: {metadata.version}")
            return metadata.version

        except Exception as e:
            logger.error(f"导入检查点失败: {e}")
            return None

class CheckpointRollbackManager:
    """检查点回滚管理器"""

    def __init__(self, checkpoint_manager: ModelCheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.rollback_history: List[Dict[str, Any]] = []
        self.max_rollback_history = 10

    def rollback_to_checkpoint(self, version: str, reason: str = "") -> bool:
        """回滚到指定检查点"""
        training_state = self.checkpoint_manager.load_checkpoint(version)
        if training_state is None:
            return False

        # 记录回滚历史
        rollback_record = {
            'timestamp': time.time(),
            'from_version': self.checkpoint_manager.get_latest_checkpoint(),
            'to_version': version,
            'reason': reason,
            'metadata': self.checkpoint_manager.get_checkpoint_info(version)
        }

        self.rollback_history.append(rollback_record)

        # 限制历史记录大小
        if len(self.rollback_history) > self.max_rollback_history:
            self.rollback_history = self.rollback_history[-self.max_rollback_history:]

        logger.info(f"已回滚到检查点: {version} (原因: {reason})")
        return True

    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """获取回滚历史"""
        return self.rollback_history.copy()

    def suggest_rollback(self, current_metrics: Dict[str, Any]) -> Optional[str]:
        """根据当前指标建议回滚"""
        # 简单的回滚建议逻辑
        current_accuracy = current_metrics.get('accuracy', 0)
        current_loss = current_metrics.get('loss', float('inf'))

        # 查找更好的检查点
        best_accuracy_version = self.checkpoint_manager.get_best_checkpoint('accuracy')
        best_loss_version = self.checkpoint_manager.get_best_checkpoint('loss')

        if best_accuracy_version:
            best_accuracy = self.checkpoint_manager.metadata[best_accuracy_version].accuracy
            if best_accuracy > current_accuracy * 1.1:  # 当前准确率比最佳低10%以上
                return best_accuracy_version

        if best_loss_version:
            best_loss = self.checkpoint_manager.metadata[best_loss_version].loss
            if current_loss > best_loss * 1.1:  # 当前损失比最佳高10%以上
                return best_loss_version

        return None

# 全局检查点管理器实例
checkpoint_manager = ModelCheckpointManager()
rollback_manager = CheckpointRollbackManager(checkpoint_manager)

def create_training_state(model_weights: Dict[str, Any],
                         optimizer_state: Dict[str, Any],
                         epoch: int = 0,
                         step: int = 0,
                         hyperparameters: Dict[str, Any] = None) -> TrainingState:
    """创建训练状态对象"""
    return TrainingState(
        model_weights=model_weights,
        optimizer_state=optimizer_state,
        scheduler_state=None,
        epoch=epoch,
        step=step,
        best_accuracy=0.0,
        best_loss=float('inf'),
        training_history=[],
        hyperparameters=hyperparameters or {}
    )

def save_model_checkpoint(training_state: TrainingState,
                         generation: int,
                         accuracy: float,
                         loss: float,
                         description: str = "") -> str:
    """保存模型检查点"""
    return checkpoint_manager.save_checkpoint(
        training_state, generation, accuracy, loss, description
    )

def load_model_checkpoint(version: str) -> Optional[TrainingState]:
    """加载模型检查点"""
    return checkpoint_manager.load_checkpoint(version)

def get_checkpoint_manager() -> ModelCheckpointManager:
    """获取检查点管理器"""
    return checkpoint_manager

def get_rollback_manager() -> CheckpointRollbackManager:
    """获取回滚管理器"""
    return rollback_manager

if __name__ == "__main__":
    # 测试检查点系统
    print("测试AGI检查点系统...")

    # 创建模拟训练状态
    mock_weights = {'layer1': np.random.randn(100, 50), 'layer2': np.random.randn(50, 10)}
    mock_optimizer = {'lr': 0.001, 'momentum': 0.9}
    mock_hyperparams = {'batch_size': 32, 'learning_rate': 0.001}

    training_state = create_training_state(
        model_weights=mock_weights,
        optimizer_state=mock_optimizer,
        epoch=1,
        step=100,
        hyperparameters=mock_hyperparams
    )

    # 保存检查点
    version = save_model_checkpoint(
        training_state, generation=1, accuracy=0.85, loss=0.45,
        description="初始训练检查点"
    )
    print(f"检查点已保存: {version}")

    # 加载检查点
    loaded_state = load_model_checkpoint(version)
    if loaded_state:
        print(f"检查点已加载: epoch={loaded_state.epoch}, step={loaded_state.step}")

    # 列出检查点
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"总检查点数: {len(checkpoints)}")

    print("检查点系统测试完成")