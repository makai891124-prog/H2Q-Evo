"""
H2Q-Evo 核心算法版本控制系统
提供算法快照、版本追踪和回滚功能，确保生产环境的稳定性
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import hashlib
from pathlib import Path
from enum import Enum

class AlgorithmStatus(Enum):
    """算法状态"""
    EXPERIMENTAL = "experimental"  # 实验性
    BETA = "beta"                 # 测试版
    STABLE = "stable"             # 稳定版
    DEPRECATED = "deprecated"      # 已弃用
    PRODUCTION = "production"      # 生产版

@dataclass
class AlgorithmVersion:
    """算法版本信息"""
    name: str
    version: str
    status: AlgorithmStatus
    description: str
    created_at: str
    author: str
    checkpoint_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    breaking_changes: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmVersion':
        """从字典创建"""
        data['status'] = AlgorithmStatus(data['status'])
        return cls(**data)

class AlgorithmVersionControl:
    """算法版本控制管理器"""
    
    def __init__(self, storage_dir: str = "algorithm_versions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, List[AlgorithmVersion]] = {}
        self.load_registry()
        
    def register_algorithm(
        self,
        name: str,
        version: str,
        status: AlgorithmStatus,
        description: str,
        module: nn.Module,
        config: Dict[str, Any],
        author: str = "H2Q-Evo Team"
    ) -> AlgorithmVersion:
        """注册新算法版本"""
        
        # 创建版本信息
        algo_version = AlgorithmVersion(
            name=name,
            version=version,
            status=status,
            description=description,
            created_at=datetime.now().isoformat(),
            author=author,
            config=config
        )
        
        # 保存模型检查点
        checkpoint_path = self._save_checkpoint(name, version, module, config)
        algo_version.checkpoint_path = str(checkpoint_path)
        
        # 计算签名
        algo_version.signature = self._compute_signature(module)
        
        # 添加到注册表
        if name not in self.versions:
            self.versions[name] = []
        self.versions[name].append(algo_version)
        
        # 保存注册表
        self.save_registry()
        
        print(f"✅ 已注册算法: {name} v{version} ({status.value})")
        return algo_version
        
    def _save_checkpoint(
        self,
        name: str,
        version: str,
        module: nn.Module,
        config: Dict[str, Any]
    ) -> Path:
        """保存模型检查点"""
        checkpoint_dir = self.storage_dir / name / version
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        
        checkpoint = {
            'model_state_dict': module.state_dict(),
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
        
    def _compute_signature(self, module: nn.Module) -> str:
        """计算模块签名（用于验证完整性）"""
        # 使用模型参数计算哈希
        hasher = hashlib.sha256()
        for param in module.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]
        
    def get_version(self, name: str, version: str) -> Optional[AlgorithmVersion]:
        """获取特定版本"""
        if name in self.versions:
            for v in self.versions[name]:
                if v.version == version:
                    return v
        return None
        
    def get_latest_stable(self, name: str) -> Optional[AlgorithmVersion]:
        """获取最新稳定版本"""
        if name not in self.versions:
            return None
            
        stable_versions = [
            v for v in self.versions[name]
            if v.status in [AlgorithmStatus.STABLE, AlgorithmStatus.PRODUCTION]
        ]
        
        if not stable_versions:
            return None
            
        # 按版本号排序（简单的字符串比较）
        return sorted(stable_versions, key=lambda x: x.version, reverse=True)[0]
        
    def load_checkpoint(self, name: str, version: str) -> Dict[str, Any]:
        """加载检查点"""
        algo_version = self.get_version(name, version)
        if not algo_version or not algo_version.checkpoint_path:
            raise ValueError(f"未找到算法版本: {name} v{version}")
            
        checkpoint_path = Path(algo_version.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
            
        return torch.load(checkpoint_path, weights_only=False)
        
    def rollback(self, name: str, target_version: str) -> AlgorithmVersion:
        """回滚到指定版本"""
        target = self.get_version(name, target_version)
        if not target:
            raise ValueError(f"未找到目标版本: {name} v{target_version}")
            
        if target.status == AlgorithmStatus.DEPRECATED:
            print(f"⚠️ 警告: 回滚到已弃用的版本 {target_version}")
            
        print(f"🔄 回滚算法 {name} 到版本 {target_version}")
        return target
        
    def list_versions(self, name: str, status: Optional[AlgorithmStatus] = None) -> List[AlgorithmVersion]:
        """列出所有版本"""
        if name not in self.versions:
            return []
            
        versions = self.versions[name]
        if status:
            versions = [v for v in versions if v.status == status]
            
        return sorted(versions, key=lambda x: x.version, reverse=True)
        
    def mark_deprecated(self, name: str, version: str, reason: str):
        """标记版本为已弃用"""
        algo_version = self.get_version(name, version)
        if algo_version:
            algo_version.status = AlgorithmStatus.DEPRECATED
            algo_version.breaking_changes.append(f"DEPRECATED: {reason}")
            self.save_registry()
            print(f"⚠️ 已标记为弃用: {name} v{version}")
            
    def promote_to_production(self, name: str, version: str):
        """提升到生产环境"""
        algo_version = self.get_version(name, version)
        if not algo_version:
            raise ValueError(f"未找到版本: {name} v{version}")
            
        if algo_version.status not in [AlgorithmStatus.STABLE, AlgorithmStatus.BETA]:
            raise ValueError(f"只有 stable 或 beta 版本可以提升到生产环境")
            
        algo_version.status = AlgorithmStatus.PRODUCTION
        self.save_registry()
        print(f"🚀 已提升到生产环境: {name} v{version}")
        
    def save_registry(self):
        """保存版本注册表"""
        registry_path = self.storage_dir / "registry.json"
        
        registry_data = {
            name: [v.to_dict() for v in versions]
            for name, versions in self.versions.items()
        }
        
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
    def load_registry(self):
        """加载版本注册表"""
        registry_path = self.storage_dir / "registry.json"
        if not registry_path.exists():
            return
            
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry_data = json.load(f)
            
        self.versions = {
            name: [AlgorithmVersion.from_dict(v) for v in versions]
            for name, versions in registry_data.items()
        }
        
    def get_compatibility_matrix(self) -> Dict[str, Dict[str, str]]:
        """获取版本兼容性矩阵"""
        matrix = {}
        for name, versions in self.versions.items():
            matrix[name] = {}
            for version in versions:
                if version.status != AlgorithmStatus.DEPRECATED:
                    matrix[name][version.version] = {
                        'status': version.status.value,
                        'dependencies': version.dependencies
                    }
        return matrix
        
    def generate_version_report(self) -> str:
        """生成版本报告"""
        report = "# H2Q-Evo 算法版本报告\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for name, versions in sorted(self.versions.items()):
            report += f"## {name}\n\n"
            report += "| 版本 | 状态 | 创建时间 | 作者 | 描述 |\n"
            report += "|------|------|----------|------|------|\n"
            
            for v in sorted(versions, key=lambda x: x.version, reverse=True):
                created = datetime.fromisoformat(v.created_at).strftime('%Y-%m-%d')
                status_emoji = {
                    AlgorithmStatus.EXPERIMENTAL: "🧪",
                    AlgorithmStatus.BETA: "🔬",
                    AlgorithmStatus.STABLE: "✅",
                    AlgorithmStatus.DEPRECATED: "⚠️",
                    AlgorithmStatus.PRODUCTION: "🚀"
                }[v.status]
                
                report += f"| {v.version} | {status_emoji} {v.status.value} | {created} | {v.author} | {v.description} |\n"
                
            report += "\n"
            
        return report

# 全局版本控制实例
_version_control = None

def get_version_control() -> AlgorithmVersionControl:
    """获取全局版本控制实例"""
    global _version_control
    if _version_control is None:
        _version_control = AlgorithmVersionControl()
    return _version_control

# 核心算法版本定义
CORE_ALGORITHM_VERSIONS = {
    "DiscreteDecisionEngine": "2.1.0",
    "SpectralShiftTracker": "1.5.0",
    "QuaternionicManifold": "1.8.0",
    "ReversibleKernel": "1.3.0",
    "AutonomousSystem": "2.0.0",
    "LatentConfig": "1.0.0"
}

def verify_algorithm_compatibility(
    algorithm_name: str,
    required_version: str
) -> bool:
    """验证算法兼容性"""
    vc = get_version_control()
    current = vc.get_latest_stable(algorithm_name)
    
    if not current:
        print(f"⚠️ 警告: 未找到算法 {algorithm_name} 的稳定版本")
        return False
        
    # 简单的版本比较（实际应使用 semantic versioning）
    if current.version >= required_version:
        return True
    else:
        print(f"⚠️ 版本不兼容: {algorithm_name} 需要 >={required_version}, 当前 {current.version}")
        return False

if __name__ == "__main__":
    # 示例用法
    vc = AlgorithmVersionControl()
    
    # 注册示例算法
    from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
    
    config = LatentConfig(latent_dim=256, n_choices=64)
    dde = DiscreteDecisionEngine(config=config)
    
    vc.register_algorithm(
        name="DiscreteDecisionEngine",
        version="2.1.0",
        status=AlgorithmStatus.STABLE,
        description="核心决策引擎，支持 SU(2) 流形投影和谱移跟踪",
        module=dde,
        config=asdict(config)
    )
    
    print("\n" + "="*50)
    print("📋 版本报告:")
    print("="*50)
    print(vc.generate_version_report())
    
    print("✅ 算法版本控制系统已初始化")
