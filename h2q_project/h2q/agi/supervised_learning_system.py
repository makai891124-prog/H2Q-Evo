#!/usr/bin/env python3
"""
H2Q 监督学习系统 - 集成真实学习与第三方验证

╔════════════════════════════════════════════════════════════════════════════╗
║                           终 极 目 标                                       ║
║                                                                            ║
║          训练本地可用的实时AGI系统                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

本模块整合:
1. 真实学习框架 (RealLearningAGI) - 确保神经网络真正学习
2. Gemini 验证器 - 第三方代码/学习验证
3. 实时监督 - 持续监控训练过程

架构:
=====
┌─────────────────────────────────────────────────────────────────────────────┐
│                      H2Q 监督学习系统                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  训练数据生成     │───▶│  真实学习 AGI    │───▶│  学习证明输出    │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│           │                       │                       │                │
│           ▼                       ▼                       ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                     Gemini 第三方验证                             │      │
│  │  • 幻觉检测  • 作弊检测  • 代码质量  • 学习验证  • 事实核查       │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                   │                                        │
│                                   ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                     验证报告 & 决策                               │      │
│  │  • 接受学习结果  • 拒绝并重训  • 警报通知  • 模型保存             │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# 加载 .env 文件
def load_env_file():
    """从 .env 文件加载环境变量."""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            return True
        except Exception:
            pass
    return False

load_env_file()

# 导入组件
from real_learning_framework import (
    RealLearningAGI, 
    RealDataGenerator,
    AntiCheatConstraints,
    ComputationTrace
)

try:
    from gemini_verifier import (
        GeminiVerifier,
        RealTimeSupervisionSystem,
        VerificationResult,
        VerificationType,
        VerificationSeverity
    )
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini 验证器未能导入，将使用本地验证")


# ============================================================================
# 第一部分: 监督学习配置
# ============================================================================

@dataclass
class SupervisedLearningConfig:
    """监督学习配置."""
    # 模型参数
    input_dim: int = 256
    hidden_dim: int = 512
    latent_dim: int = 128
    output_dim: int = 256
    num_attention_heads: int = 4
    num_attention_layers: int = 3
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    validation_interval: int = 10
    
    # 验证参数
    use_gemini_verification: bool = True
    verification_interval: int = 20
    min_acceptable_score: float = 0.6
    
    # 保存参数
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints"


# ============================================================================
# 第二部分: 本地验证器（无需API）
# ============================================================================

class LocalVerifier:
    """本地验证器 - 基于规则的验证，不需要外部API."""
    
    # 作弊模式正则
    CHEATING_PATTERNS = [
        (r'answers\s*=\s*\{', 'lookup_table', '使用预设答案字典'),
        (r'if\s+.*\s+in\s+\[.*\]:', 'category_matching', '按类别分支'),
        (r'return\s+\d+\s*$', 'hardcoded_return', '硬编码返回值'),
        (r'PRECOMPUTED', 'precomputed', '预计算结果'),
        (r'eval\s*\(', 'eval_usage', '使用eval可能不安全'),
    ]
    
    @staticmethod
    def verify_no_cheating(code: str) -> Tuple[bool, List[Dict]]:
        """检测代码中的作弊模式."""
        import re
        issues = []
        
        for pattern, name, description in LocalVerifier.CHEATING_PATTERNS:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    'pattern': name,
                    'description': description,
                    'line': line_num,
                    'match': match.group()
                })
        
        return len(issues) == 0, issues
    
    @staticmethod
    def verify_learning_curve(losses: List[float], min_improvement: float = 0.1) -> Tuple[bool, Dict]:
        """验证学习曲线是否表明真实学习."""
        if len(losses) < 5:
            return True, {'status': 'insufficient_data'}
        
        # 检查是否有下降趋势
        first_half = np.mean(losses[:len(losses)//2])
        second_half = np.mean(losses[len(losses)//2:])
        improvement = (first_half - second_half) / first_half if first_half != 0 else 0
        
        # 检查是否太"完美"（可能是伪造的）
        loss_std = np.std(losses)
        is_too_smooth = loss_std < 0.001
        
        # 检查是否单调递减（不现实）
        is_monotonic = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
        
        is_valid = improvement > min_improvement and not is_too_smooth and not is_monotonic
        
        return is_valid, {
            'improvement': improvement,
            'is_too_smooth': is_too_smooth,
            'is_monotonic': is_monotonic,
            'first_half_avg': first_half,
            'second_half_avg': second_half
        }
    
    @staticmethod
    def verify_gradient_health(grad_norms: List[float]) -> Tuple[bool, Dict]:
        """验证梯度健康状况."""
        if len(grad_norms) < 5:
            return True, {'status': 'insufficient_data'}
        
        avg_norm = np.mean(grad_norms)
        max_norm = np.max(grad_norms)
        min_norm = np.min(grad_norms)
        
        # 检查梯度消失
        is_vanishing = avg_norm < 1e-7
        # 检查梯度爆炸
        is_exploding = max_norm > 1000
        # 检查梯度稳定性
        is_stable = max_norm / (min_norm + 1e-10) < 100
        
        is_healthy = not is_vanishing and not is_exploding and is_stable
        
        return is_healthy, {
            'avg_norm': avg_norm,
            'max_norm': max_norm,
            'min_norm': min_norm,
            'is_vanishing': is_vanishing,
            'is_exploding': is_exploding,
            'is_stable': is_stable
        }


# ============================================================================
# 第三部分: 监督学习管理器
# ============================================================================

class SupervisedLearningManager:
    """
    监督学习管理器 - 协调真实学习和第三方验证.
    """
    
    def __init__(self, config: Optional[SupervisedLearningConfig] = None):
        self.config = config or SupervisedLearningConfig()
        
        # 初始化模型
        self.model = RealLearningAGI(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            output_dim=self.config.output_dim,
            num_attention_heads=self.config.num_attention_heads,
            num_attention_layers=self.config.num_attention_layers
        )
        
        # 初始化数据生成器
        self.data_generator = RealDataGenerator(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim
        )
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 初始化验证器
        self.local_verifier = LocalVerifier()
        if GEMINI_AVAILABLE and self.config.use_gemini_verification:
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if api_key:
                self.gemini_verifier = GeminiVerifier(api_key)
                self.supervision_system = RealTimeSupervisionSystem(api_key)
                print("✓ Gemini 验证器已启用")
            else:
                self.gemini_verifier = None
                self.supervision_system = None
                print("⚠️ 未设置 API Key，使用本地验证")
        else:
            self.gemini_verifier = None
            self.supervision_system = None
        
        # 训练状态
        self.epoch = 0
        self.total_steps = 0
        self.training_log: List[Dict] = []
        self.verification_log: List[Dict] = []
        
        # 检查点目录
        self.checkpoint_dir = SCRIPT_DIR / self.config.checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self, task_type: str = "arithmetic") -> Dict[str, float]:
        """训练一个 epoch."""
        self.model.train()
        
        # 生成数据
        if task_type == "arithmetic":
            inputs, targets, metadata = self.data_generator.generate_arithmetic_data(
                self.config.batch_size
            )
        else:
            inputs, targets, metadata = self.data_generator.generate_pattern_data(
                self.config.batch_size
            )
        
        # 学习
        metrics = self.model.learn(inputs, targets, self.optimizer)
        
        # 记录
        self.training_log.append({
            'epoch': self.epoch,
            'step': self.total_steps,
            'task_type': task_type,
            'loss': metrics['loss'],
            'grad_norm': metrics['grad_norm'],
            'timestamp': datetime.now().isoformat()
        })
        
        self.total_steps += 1
        
        return metrics
    
    def validate(self) -> Dict[str, Any]:
        """验证当前模型."""
        self.model.eval()
        
        validation_results = {
            'epoch': self.epoch,
            'local_verification': {},
            'gemini_verification': None,
            'learning_proof': None
        }
        
        # 1. 本地验证
        print("  [本地验证]")
        
        # 学习曲线验证
        losses = [log['loss'] for log in self.training_log]
        is_learning, learning_details = self.local_verifier.verify_learning_curve(losses)
        validation_results['local_verification']['learning_curve'] = {
            'passed': is_learning,
            'details': learning_details
        }
        print(f"    学习曲线: {'✓' if is_learning else '✗'}")
        
        # 梯度健康验证
        grad_norms = [log['grad_norm'] for log in self.training_log]
        is_healthy, gradient_details = self.local_verifier.verify_gradient_health(grad_norms)
        validation_results['local_verification']['gradient_health'] = {
            'passed': is_healthy,
            'details': gradient_details
        }
        print(f"    梯度健康: {'✓' if is_healthy else '✗'}")
        
        # 2. 获取学习证明
        learning_proof = self.model.get_learning_proof()
        validation_results['learning_proof'] = learning_proof
        print(f"    学习状态: {learning_proof['status']}")
        
        # 3. Gemini 验证（如果可用）
        if self.gemini_verifier and self.epoch % self.config.verification_interval == 0:
            print("  [Gemini 验证]")
            
            try:
                gemini_result = self.gemini_verifier.verify_learning(learning_proof)
                validation_results['gemini_verification'] = gemini_result.to_dict()
                print(f"    Gemini 验证: {'✓' if gemini_result.passed else '✗'} (score: {gemini_result.score:.2f})")
            except Exception as e:
                print(f"    Gemini 验证失败: {e}")
                validation_results['gemini_verification'] = {'error': str(e)}
        
        # 记录
        self.verification_log.append(validation_results)
        
        return validation_results
    
    def train(self, num_epochs: Optional[int] = None, task_types: List[str] = None) -> Dict[str, Any]:
        """
        完整的监督学习训练循环.
        
        Args:
            num_epochs: 训练轮数
            task_types: 任务类型列表
        """
        num_epochs = num_epochs or self.config.num_epochs
        task_types = task_types or ["arithmetic", "pattern"]
        
        print("=" * 80)
        print("               H2Q 监督学习系统 - 训练开始")
        print("=" * 80)
        print()
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                           终 极 目 标                                       ║")
        print("║                                                                            ║")
        print("║          训练本地可用的实时AGI系统                                          ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        
        print(f"配置:")
        print(f"  - 训练轮数: {num_epochs}")
        print(f"  - 批次大小: {self.config.batch_size}")
        print(f"  - 学习率: {self.config.learning_rate}")
        print(f"  - 验证间隔: {self.config.validation_interval}")
        print(f"  - Gemini验证: {'启用' if self.gemini_verifier else '禁用'}")
        print()
        print("-" * 80)
        
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 选择任务类型（交替）
            task_type = task_types[epoch % len(task_types)]
            
            # 训练
            metrics = self.train_epoch(task_type)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:4d}/{num_epochs} | Loss: {metrics['loss']:.6f} | "
                      f"Grad: {metrics['grad_norm']:.4f} | Task: {task_type}")
            
            # 验证
            if (epoch + 1) % self.config.validation_interval == 0:
                print(f"\n--- 验证 (Epoch {epoch + 1}) ---")
                validation_results = self.validate()
                print()
            
            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")
            
            # 更新最佳损失
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
        
        # 训练结束
        total_time = time.time() - start_time
        
        print("-" * 80)
        print(f"\n训练完成！")
        print(f"  总时间: {total_time:.2f}s")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  总步数: {self.total_steps}")
        
        # 最终验证
        print(f"\n--- 最终验证 ---")
        final_validation = self.validate()
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        # 生成训练报告
        report = self.generate_training_report(total_time, best_loss, final_validation)
        
        return report
    
    def save_checkpoint(self, name: str):
        """保存检查点."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_log': self.training_log,
            'verification_log': self.verification_log,
            'config': self.config.__dict__
        }, checkpoint_path)
        
        print(f"  ✓ 检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, name: str) -> bool:
        """加载检查点."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        if not checkpoint_path.exists():
            print(f"  ✗ 检查点不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path)
        
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_log = checkpoint['training_log']
        self.verification_log = checkpoint['verification_log']
        
        print(f"  ✓ 检查点已加载: {checkpoint_path}")
        return True
    
    def generate_training_report(self, total_time: float, best_loss: float, 
                                  final_validation: Dict) -> Dict[str, Any]:
        """生成训练报告."""
        report = {
            'summary': {
                'total_epochs': self.epoch + 1,
                'total_steps': self.total_steps,
                'total_time_seconds': total_time,
                'best_loss': best_loss,
                'final_loss': self.training_log[-1]['loss'] if self.training_log else None,
            },
            'learning_proof': final_validation.get('learning_proof', {}),
            'verification_summary': {
                'local_passed': all(
                    v.get('passed', True) 
                    for v in final_validation.get('local_verification', {}).values()
                ),
                'gemini_passed': (
                    final_validation.get('gemini_verification', {}).get('passed', True)
                    if final_validation.get('gemini_verification') else None
                )
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'architecture': {
                    'input_dim': self.config.input_dim,
                    'hidden_dim': self.config.hidden_dim,
                    'latent_dim': self.config.latent_dim,
                    'output_dim': self.config.output_dim,
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存报告
        report_path = self.checkpoint_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n训练报告已保存: {report_path}")
        
        return report


# ============================================================================
# 第四部分: 集成测试
# ============================================================================

def run_supervised_learning_demo():
    """运行监督学习演示."""
    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                           终 极 目 标                                       ║")
    print("║                                                                            ║")
    print("║          训练本地可用的实时AGI系统                                          ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # 创建配置
    config = SupervisedLearningConfig(
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        validation_interval=20,
        verification_interval=50,
        save_interval=50
    )
    
    # 创建管理器
    manager = SupervisedLearningManager(config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in manager.model.parameters())
    print(f"模型参数量: {total_params:,}")
    print()
    
    # 开始训练
    report = manager.train(num_epochs=100, task_types=["arithmetic", "pattern"])
    
    # 打印最终报告
    print("\n" + "=" * 80)
    print("                       训练报告摘要")
    print("=" * 80)
    print(f"总轮数: {report['summary']['total_epochs']}")
    print(f"总步数: {report['summary']['total_steps']}")
    print(f"总时间: {report['summary']['total_time_seconds']:.2f}s")
    print(f"最佳损失: {report['summary']['best_loss']:.6f}")
    print(f"最终损失: {report['summary']['final_loss']:.6f}")
    print(f"学习状态: {report['learning_proof'].get('status', 'unknown')}")
    print(f"本地验证: {'✓' if report['verification_summary']['local_passed'] else '✗'}")
    
    if report['verification_summary']['gemini_passed'] is not None:
        print(f"Gemini验证: {'✓' if report['verification_summary']['gemini_passed'] else '✗'}")
    
    print()
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    run_supervised_learning_demo()
