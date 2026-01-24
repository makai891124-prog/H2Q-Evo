#!/usr/bin/env python3
"""
H2Q-Evo AGI训练监控和管理工具
提供训练状态监控、控制和分析功能
"""

import os
import sys
import json
import time
import signal
import psutil
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional

# 配置日志
logger = logging.getLogger('AGITrainingMonitor')

class AGITrainingMonitor:
    """AGI训练监控器"""

    def __init__(self, project_root: str = "./agi_persistent_training"):
        self.project_root = Path(project_root)
        self.state_file = self.project_root / "evolution_state.json"
        self.log_file = self.project_root / "logs" / "training.log"
        self.checkpoint_dir = self.project_root / "checkpoints"

        # 监控状态
        self.is_monitoring = False
        self.last_update = None

    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_running': False,
            'process_info': None,
            'memory_usage': None,
            'gpu_usage': None,
            'evolution_state': {},
            'recent_logs': [],
            'checkpoints': []
        }

        # 检查训练进程
        training_processes = self._find_training_processes()
        if training_processes:
            status['is_running'] = True
            status['process_info'] = {
                'pid': training_processes[0].pid,
                'cpu_percent': training_processes[0].cpu_percent(),
                'memory_mb': training_processes[0].memory_info().rss / (1024**2),
                'create_time': datetime.fromtimestamp(training_processes[0].create_time()).isoformat()
            }

        # 获取内存使用情况
        memory = psutil.virtual_memory()
        status['memory_usage'] = {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'usage_percent': memory.percent
        }

        # 获取GPU使用情况 (如果可用)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                status['gpu_usage'] = {
                    'total_gb': gpu_memory / (1024**3),
                    'used_gb': gpu_used / (1024**3),
                    'usage_percent': (gpu_used / gpu_memory) * 100
                }
        except:
            pass

        # 加载进化状态
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    status['evolution_state'] = json.load(f)
            except Exception as e:
                status['evolution_state'] = {'error': str(e)}

        # 获取最近日志
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-20:]  # 最近20行
                    status['recent_logs'] = [line.strip() for line in lines]
            except Exception as e:
                status['recent_logs'] = [f'Error reading logs: {e}']

        # 获取检查点列表
        if self.checkpoint_dir.exists():
            checkpoints = []
            for cp_dir in self.checkpoint_dir.iterdir():
                if cp_dir.is_dir():
                    checkpoints.append({
                        'name': cp_dir.name,
                        'path': str(cp_dir),
                        'size_mb': sum(f.stat().st_size for f in cp_dir.rglob('*') if f.is_file()) / (1024**2),
                        'modified': datetime.fromtimestamp(cp_dir.stat().st_mtime).isoformat()
                    })
            status['checkpoints'] = sorted(checkpoints, key=lambda x: x['modified'], reverse=True)

        return status

    def _find_training_processes(self) -> list:
        """查找训练进程"""
        training_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python3' or proc.info['name'] == 'python':
                    cmdline = proc.info['cmdline']
                    if cmdline and 'agi_persistent_evolution.py' in ' '.join(cmdline):
                        training_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return training_processes

    def stop_training(self) -> bool:
        """停止训练"""
        training_processes = self._find_training_processes()
        if not training_processes:
            print("❌ 未找到正在运行的训练进程")
            return False

        for proc in training_processes:
            try:
                proc.terminate()
                print(f"✅ 发送终止信号到进程 {proc.pid}")
                # 等待进程结束
                proc.wait(timeout=10)
                print(f"✅ 进程 {proc.pid} 已停止")
            except psutil.TimeoutExpired:
                print(f"⚠️ 进程 {proc.pid} 未在预期时间内停止，强制终止")
                proc.kill()
            except Exception as e:
                print(f"❌ 停止进程 {proc.pid} 时出错: {e}")
                return False

        return True

    def show_training_stats(self):
        """显示训练统计"""
        status = self.get_training_status()

        print("🚀 H2Q-Evo AGI训练状态")
        print("=" * 50)

        # 运行状态
        if status['is_running']:
            print("✅ 状态: 运行中")
            proc_info = status['process_info']
            print(f"   PID: {proc_info['pid']}")
            print(f"   CPU使用: {proc_info['cpu_percent']:.1f}%")
            print(f"   内存使用: {proc_info['memory_mb']:.1f} MB")
            print(f"   启动时间: {proc_info['create_time']}")
        else:
            print("❌ 状态: 未运行")
        print()

        # 内存使用
        mem = status['memory_usage']
        print("💾 内存使用:")
        print(f"   已用: {mem['used_gb']:.1f} GB")
        print(f"   可用: {mem['available_gb']:.1f} GB")
        print(f"   使用率: {mem['percent']:.1f}%")
        print(f"   进程数: {mem['process_count']}")
        print()

        # GPU使用 (如果可用)
        if status['gpu_usage']:
            gpu = status['gpu_usage']
            print("🎮 GPU使用:")
            print(f"   GPU内存使用: {gpu['gpu_memory_used']:.1f} MB")
            print(f"   GPU内存总量: {gpu['gpu_memory_total']:.1f} MB")
            print(f"   GPU利用率: {gpu['gpu_utilization']:.1f}%")
            print()

        # 进化状态
        evo_state = status['evolution_state']
        if evo_state and 'generation' in evo_state:
            print("🧬 进化状态:")
            print(f"   当前代数: {evo_state['generation']}")
            print(f"   最佳适应度: {evo_state.get('best_fitness', 0):.4f}")
            print(f"   当前适应度: {evo_state.get('current_fitness', 0):.4f}")
            print(f"   平均损失: {evo_state.get('average_loss', 0):.4f}")
            print(f"   总训练步数: {evo_state.get('total_training_steps', 0)}")
            print(f"   模型版本数: {len(evo_state.get('model_versions', []))}")
        else:
            print("🧬 进化状态: 未找到状态文件")
        print()

        # 最近检查点
        checkpoints = status['checkpoints']
        if checkpoints:
            print("💾 最近检查点:")
            for i, cp in enumerate(checkpoints[:3]):  # 显示前3个
                print(f"   {i+1}. {cp['name']} ({cp['size_mb']:.1f} MB) - {cp['modified']}")
        else:
            print("💾 检查点: 无")
        print()

        # 最近日志
        logs = status['recent_logs']
        if logs:
            print("📝 最近日志:")
            for log in logs[-5:]:  # 显示最后5条
                print(f"   {log}")
        else:
            print("📝 日志: 无")

    def plot_training_progress(self, save_path: Optional[str] = None):
        """绘制训练进度图"""
        status = self.get_training_status()
        evo_state = status['evolution_state']

        if not evo_state or 'learning_curve' not in evo_state:
            print("❌ 未找到学习曲线数据")
            return

        learning_curve = evo_state['learning_curve']
        if not learning_curve:
            print("❌ 学习曲线数据为空")
            return

        # 提取数据
        steps = [point['step'] for point in learning_curve]
        losses = [point['loss'] for point in learning_curve]
        generations = [point['generation'] for point in learning_curve]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 损失曲线
        ax1.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('AGI Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 代数分布
        unique_gens = list(set(generations))
        gen_counts = [generations.count(gen) for gen in unique_gens]
        ax2.bar(unique_gens, gen_counts, alpha=0.7, color='green')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Number of Steps')
        ax2.set_title('Training Steps per Generation')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图表已保存到: {save_path}")
        else:
            plt.show()

    def export_training_report(self, output_file: str):
        """导出训练报告"""
        status = self.get_training_status()

        report = {
            'generated_at': datetime.now().isoformat(),
            'training_status': status,
            'summary': {
                'is_running': status['is_running'],
                'current_generation': status['evolution_state'].get('generation', 0),
                'best_fitness': status['evolution_state'].get('best_fitness', 0),
                'total_training_steps': status['evolution_state'].get('total_training_steps', 0),
                'memory_usage_percent': status['memory_usage']['usage_percent'],
                'num_checkpoints': len(status['checkpoints'])
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ 训练报告已导出到: {output_file}")

    def start_monitoring(self):
        """开始监控训练过程"""
        if self.is_monitoring:
            logger.warning("监控已在运行")
            return False

        self.is_monitoring = True
        self.last_update = datetime.now()
        logger.info("AGI训练监控器已启动")
        return True

    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            logger.warning("监控未在运行")
            return False

        self.is_monitoring = False
        logger.info("AGI训练监控器已停止")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='H2Q-Evo AGI训练监控和管理工具')
    parser.add_argument('action', choices=['status', 'stop', 'plot', 'report'],
                       help='执行的操作')
    parser.add_argument('--project-root', default='./agi_persistent_training',
                       help='项目根目录')
    parser.add_argument('--output', help='输出文件路径')

    args = parser.parse_args()

    monitor = AGITrainingMonitor(args.project_root)

    if args.action == 'status':
        monitor.show_training_stats()

    elif args.action == 'stop':
        if monitor.stop_training():
            print("✅ 训练已停止")
        else:
            print("❌ 停止训练失败")

    elif args.action == 'plot':
        output_path = args.output or './training_progress.png'
        monitor.plot_training_progress(output_path)

    elif args.action == 'report':
        output_path = args.output or './training_report.json'
        monitor.export_training_report(output_path)

if __name__ == "__main__":
    main()