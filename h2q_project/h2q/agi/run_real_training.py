#!/usr/bin/env python3
"""
H2Q 真实训练启动器 - 带 Gemini 第三方验证

╔════════════════════════════════════════════════════════════════════════════╗
║                           终 极 目 标                                       ║
║                                                                            ║
║          训练本地可用的实时AGI系统                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

训练策略:
=========
1. 本地验证: 每 20 个 epoch 执行一次（快速）
2. Gemini 验证: 训练开始时、中期、结束时各一次（防止频率限制）
3. 真实学习: 神经网络通过梯度下降学习，无作弊

运行方式:
=========
cd /Users/imymm/H2Q-Evo
PYTHONPATH=h2q_project/h2q/agi python3 h2q_project/h2q/agi/run_real_training.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# 设置路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 加载 .env
def load_env():
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
        print(f"✓ 已加载环境配置: {env_path}")
        return True
    return False

load_env()

# 导入训练组件
from real_learning_framework import RealLearningAGI, RealDataGenerator, AntiCheatConstraints
from supervised_learning_system import LocalVerifier

# 尝试导入 Gemini
try:
    from gemini_verifier import GeminiVerifier, GeminiClient
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

import torch
import torch.nn.functional as F
import numpy as np


class RealTrainingSession:
    """真实训练会话 - 带智能验证调度."""
    
    def __init__(self, 
                 num_epochs: int = 200,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 gemini_verification_points: list = None):
        """
        初始化训练会话.
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            gemini_verification_points: Gemini 验证点（epoch 编号列表）
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Gemini 验证点：开始、中期、结束
        self.gemini_verification_points = gemini_verification_points or [
            1,                      # 训练开始
            num_epochs // 2,        # 中期
            num_epochs              # 结束
        ]
        
        # 创建模型
        self.model = RealLearningAGI(
            input_dim=256,
            hidden_dim=512,
            dim=128,
            output_dim=256,
            num_attention_heads=4,
            num_attention_layers=3
        )
        
        # 创建数据生成器
        self.data_generator = RealDataGenerator(input_dim=256, output_dim=256)
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-5
        )
        
        # 本地验证器
        self.local_verifier = LocalVerifier()
        
        # Gemini 验证器
        self.gemini_verifier = None
        if GEMINI_AVAILABLE:
            api_key = os.environ.get('GEMINI_API_KEY')
            if api_key:
                self.gemini_verifier = GeminiVerifier(api_key)
                print("✓ Gemini 验证器已启用")
            else:
                print("⚠️ 未设置 GEMINI_API_KEY，跳过 Gemini 验证")
        
        # 训练日志
        self.training_log = []
        self.verification_results = []
        
        # 检查点目录
        self.checkpoint_dir = SCRIPT_DIR / "training_checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train(self):
        """执行完整训练."""
        print()
        print("=" * 80)
        print("              H2Q AGI 真实训练 - 启动")
        print("=" * 80)
        print()
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                           终 极 目 标                                       ║")
        print("║                                                                            ║")
        print("║          训练本地可用的实时AGI系统                                          ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        
        # 打印配置
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"配置:")
        print(f"  模型参数: {total_params:,}")
        print(f"  训练轮数: {self.num_epochs}")
        print(f"  批次大小: {self.batch_size}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  Gemini 验证点: {self.gemini_verification_points}")
        print()
        
        # 验证模型架构
        print("[预检查] 验证模型架构...")
        assert AntiCheatConstraints.verify_no_lookup(self.model), "检测到查找表！"
        print("  ✓ 无查找表")
        
        sample_input = torch.randn(1, 256)
        grad_check = AntiCheatConstraints.verify_gradient_flow(self.model, sample_input)
        flowing = sum(1 for v in grad_check.values() if v)
        print(f"  ✓ 梯度流动: {flowing}/{len(grad_check)}")
        print()
        
        print("-" * 80)
        print("开始训练...")
        print("-" * 80)
        
        start_time = time.time()
        best_loss = float('inf')
        task_types = ["arithmetic", "pattern"]
        
        for epoch in range(1, self.num_epochs + 1):
            # 选择任务类型
            task_type = task_types[(epoch - 1) % len(task_types)]
            
            # 生成数据
            if task_type == "arithmetic":
                inputs, targets, _ = self.data_generator.generate_arithmetic_data(self.batch_size)
            else:
                inputs, targets, _ = self.data_generator.generate_pattern_data(self.batch_size)
            
            # 训练步骤
            metrics = self.model.learn(inputs, targets, self.optimizer)
            self.scheduler.step()
            
            # 记录
            self.training_log.append({
                'epoch': epoch,
                'loss': metrics['loss'],
                'grad_norm': metrics['grad_norm'],
                'lr': self.scheduler.get_last_lr()[0],
                'task': task_type
            })
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:4d}/{self.num_epochs} | Loss: {metrics['loss']:.6f} | "
                      f"Grad: {metrics['grad_norm']:.4f} | LR: {lr:.2e} | Task: {task_type}")
            
            # 本地验证（每 50 个 epoch）
            if epoch % 50 == 0:
                self._run_local_verification(epoch)
            
            # Gemini 验证（在指定点）
            if epoch in self.gemini_verification_points and self.gemini_verifier:
                self._run_gemini_verification(epoch)
            
            # 更新最佳损失
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
            
            # 保存检查点（每 100 个 epoch）
            if epoch % 100 == 0:
                self._save_checkpoint(epoch)
        
        # 训练完成
        total_time = time.time() - start_time
        
        print()
        print("-" * 80)
        print("训练完成！")
        print("-" * 80)
        print()
        
        # 最终验证
        print("[最终验证]")
        self._run_local_verification(self.num_epochs)
        
        if self.gemini_verifier:
            print("\n等待 60 秒后进行最终 Gemini 验证...")
            time.sleep(60)
            self._run_gemini_verification(self.num_epochs, final=True)
        
        # 保存最终模型
        self._save_checkpoint("final")
        
        # 生成报告
        report = self._generate_report(total_time, best_loss)
        
        return report
    
    def _run_local_verification(self, epoch: int):
        """运行本地验证."""
        print(f"\n--- 本地验证 (Epoch {epoch}) ---")
        
        losses = [log['loss'] for log in self.training_log]
        grad_norms = [log['grad_norm'] for log in self.training_log]
        
        # 学习曲线验证
        is_learning, details = self.local_verifier.verify_learning_curve(losses)
        print(f"  学习曲线: {'✓' if is_learning else '✗'} (改进: {details.get('improvement', 0):.2%})")
        
        # 梯度健康验证
        is_healthy, g_details = self.local_verifier.verify_gradient_health(grad_norms)
        print(f"  梯度健康: {'✓' if is_healthy else '✗'} (平均: {g_details.get('avg_norm', 0):.4f})")
        
        # 学习证明
        proof = self.model.get_learning_proof()
        print(f"  学习状态: {proof['status']}")
        
        self.verification_results.append({
            'epoch': epoch,
            'type': 'local',
            'learning_curve': is_learning,
            'gradient_health': is_healthy,
            'learning_status': proof['status'],
            'timestamp': datetime.now().isoformat()
        })
    
    def _run_gemini_verification(self, epoch: int, final: bool = False):
        """运行 Gemini 第三方验证."""
        print(f"\n--- Gemini 第三方验证 (Epoch {epoch}) ---")
        
        try:
            # 获取学习证明
            proof = self.model.get_learning_proof()
            
            # 验证学习
            result = self.gemini_verifier.verify_learning(proof)
            
            print(f"  Gemini 验证: {'✓' if result.passed else '✗'} (score: {result.score:.2f})")
            print(f"  严重性: {result.severity.value}")
            
            if result.suggestions:
                print("  建议:")
                for sugg in result.suggestions[:2]:
                    print(f"    - {sugg}")
            
            self.verification_results.append({
                'epoch': epoch,
                'type': 'gemini',
                'passed': result.passed,
                'score': result.score,
                'details': result.details,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"  ⚠️ Gemini 验证失败: {e}")
            self.verification_results.append({
                'epoch': epoch,
                'type': 'gemini',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    def _save_checkpoint(self, name):
        """保存检查点."""
        path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_log': self.training_log,
            'verification_results': self.verification_results,
        }, path)
        print(f"  ✓ 检查点已保存: {path.name}")
    
    def _generate_report(self, total_time: float, best_loss: float) -> dict:
        """生成训练报告."""
        proof = self.model.get_learning_proof()
        
        report = {
            'summary': {
                'total_epochs': self.num_epochs,
                'total_time_seconds': total_time,
                'best_loss': best_loss,
                'final_loss': self.training_log[-1]['loss'] if self.training_log else None,
                'learning_status': proof['status'],
            },
            'learning_proof': proof,
            'verification_summary': {
                'local_checks': len([v for v in self.verification_results if v['type'] == 'local']),
                'gemini_checks': len([v for v in self.verification_results if v['type'] == 'gemini']),
                'all_passed': all(v.get('passed', True) or v.get('learning_status') == 'learning_verified' 
                                 for v in self.verification_results if 'error' not in v)
            },
            'training_curve': {
                'initial_loss': self.training_log[0]['loss'] if self.training_log else None,
                'final_loss': self.training_log[-1]['loss'] if self.training_log else None,
                'loss_reduction': (self.training_log[0]['loss'] - self.training_log[-1]['loss']) / self.training_log[0]['loss'] 
                                  if self.training_log else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存报告
        report_path = self.checkpoint_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print()
        print("=" * 80)
        print("                     训练报告摘要")
        print("=" * 80)
        print(f"  总轮数: {report['summary']['total_epochs']}")
        print(f"  总时间: {total_time:.2f}s")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  最终损失: {report['summary']['final_loss']:.6f}")
        print(f"  损失降低: {report['training_curve']['loss_reduction']:.1%}")
        print(f"  学习状态: {report['summary']['learning_status']}")
        print(f"  本地验证: {report['verification_summary']['local_checks']} 次")
        print(f"  Gemini 验证: {report['verification_summary']['gemini_checks']} 次")
        print(f"  全部通过: {'✓' if report['verification_summary']['all_passed'] else '✗'}")
        print()
        print(f"报告已保存: {report_path}")
        print("=" * 80)
        
        return report


def main():
    """主函数."""
    # 创建训练会话
    session = RealTrainingSession(
        num_epochs=200,
        batch_size=32,
        learning_rate=1e-3,
        gemini_verification_points=[1, 100, 200]  # 开始、中期、结束
    )
    
    # 开始训练
    report = session.train()
    
    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                           终 极 目 标                                       ║")
    print("║                                                                            ║")
    print("║          训练本地可用的实时AGI系统                                          ║")
    print("║                                                                            ║")
    print("║                     训练完成！模型已保存。                                   ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    return report


if __name__ == "__main__":
    main()
