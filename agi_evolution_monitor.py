#!/usr/bin/env python3
"""
H2Q-Evo AGI进化监控和可视化系统
实时监控AGI系统的进化过程和性能指标
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import psutil
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('AGI-EvolutionMonitor')

class AGIEvolutionMonitor:
    """AGI进化监控器"""

    def __init__(self, config_path: str = "./agi_training_config.ini"):
        self.config = self._load_config(config_path)

        # 监控数据存储
        self.metrics_history = {
            'generation': [],
            'loss': [],
            'accuracy': [],
            'compression_ratio': [],
            'training_time': [],
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_memory': [],
            'fitness_score': [],
            'diversity_score': [],
            'timestamp': []
        }

        # 实时数据缓冲区
        self.realtime_buffer = deque(maxlen=1000)

        # 可视化设置
        self.fig_size = (15, 10)
        self.update_interval = 5  # 秒

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None

        # 数据文件路径
        self.metrics_file = Path("./agi_persistent_training/metrics/evolution_metrics.jsonl")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("AGI进化监控器初始化完成")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        config = {}

        if os.path.exists(config_path):
            try:
                import configparser
                parser = configparser.ConfigParser()
                parser.read(config_path)

                # 读取监控相关配置
                if 'monitoring' in parser:
                    config.update(dict(parser['monitoring']))

            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")

        # 默认配置
        config.setdefault('metrics_update_interval', 5)
        config.setdefault('max_history_points', 1000)
        config.setdefault('alert_thresholds', {
            'loss': 10.0,
            'memory_usage': 90.0,
            'cpu_usage': 95.0
        })

        return config

    def start_monitoring(self, background: bool = True):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行")
            return

        self.is_monitoring = True
        logger.info("开始AGI进化监控...")

        if background:
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
        else:
            self._monitoring_loop()

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("AGI进化监控已停止")

    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()

                # 收集训练指标
                training_metrics = self._collect_training_metrics()

                # 合并指标
                metrics = {**system_metrics, **training_metrics}
                metrics['timestamp'] = datetime.now().isoformat()

                # 添加到历史记录
                self._add_metrics_to_history(metrics)

                # 保存到文件
                self._save_metrics(metrics)

                # 检查告警
                self._check_alerts(metrics)

                # 等待下次更新
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(10)  # 出错时等待更长时间

    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        metrics = {}

        try:
            # CPU使用率
            metrics['cpu_usage'] = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            metrics['memory_usage'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            metrics['disk_usage'] = disk.percent

            # 网络I/O (可选)
            try:
                net = psutil.net_io_counters()
                metrics['network_bytes_sent'] = net.bytes_sent
                metrics['network_bytes_recv'] = net.bytes_recv
            except:
                pass

            # GPU信息 (如果可用)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_used = torch.cuda.memory_allocated(0)
                    metrics['gpu_memory_usage'] = (gpu_used / gpu_memory) * 100
                    metrics['gpu_memory_used_gb'] = gpu_used / (1024**3)
            except:
                pass

        except Exception as e:
            logger.warning(f"收集系统指标失败: {e}")

        return metrics

    def _collect_training_metrics(self) -> Dict[str, Any]:
        """收集训练指标"""
        metrics = {}

        try:
            # 尝试从训练状态文件读取
            state_files = [
                "./evo_state.json",
                "./agi_persistent_training/training_state.json",
                "./evolution_24h_state.json"
            ]

            for state_file in state_files:
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        state = json.load(f)

                    # 提取训练指标
                    if 'generation' in state:
                        metrics['generation'] = state['generation']
                    if 'current_loss' in state:
                        metrics['loss'] = state['current_loss']
                    if 'fitness_score' in state:
                        metrics['fitness_score'] = state['fitness_score']
                    if 'compression_ratio' in state:
                        metrics['compression_ratio'] = state['compression_ratio']

                    break  # 只读取第一个找到的文件

            # 如果没有状态文件，生成模拟数据用于演示
            if not metrics:
                metrics.update(self._generate_demo_metrics())

        except Exception as e:
            logger.warning(f"收集训练指标失败: {e}")
            metrics.update(self._generate_demo_metrics())

        return metrics

    def _generate_demo_metrics(self) -> Dict[str, Any]:
        """生成演示指标 (用于测试)"""
        base_generation = len(self.metrics_history['generation']) + 1

        return {
            'generation': base_generation,
            'loss': max(0.1, 2.0 * np.exp(-base_generation / 50) + np.random.normal(0, 0.1)),
            'accuracy': min(0.95, 0.5 + base_generation / 200 + np.random.normal(0, 0.02)),
            'compression_ratio': 0.85 + np.random.normal(0, 0.05),
            'fitness_score': 0.1 + base_generation / 100 + np.random.normal(0, 0.05),
            'diversity_score': 0.3 + np.random.normal(0, 0.1)
        }

    def _add_metrics_to_history(self, metrics: Dict[str, Any]):
        """添加到历史记录"""
        for key in self.metrics_history.keys():
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
            elif key == 'timestamp':
                self.metrics_history[key].append(metrics.get('timestamp', datetime.now().isoformat()))

        # 限制历史记录长度
        max_points = self.config.get('max_history_points', 1000)
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_points:
                self.metrics_history[key] = self.metrics_history[key][-max_points:]

    def _save_metrics(self, metrics: Dict[str, Any]):
        """保存指标到文件"""
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"保存指标失败: {e}")

    def _check_alerts(self, metrics: Dict[str, Any]):
        """检查告警条件"""
        thresholds = self.config.get('alert_thresholds', {})

        alerts = []

        # 检查损失阈值
        if 'loss' in metrics and metrics['loss'] > thresholds.get('loss', 10.0):
            alerts.append(f"损失过高: {metrics['loss']:.3f}")

        # 检查内存使用率
        if 'memory_usage' in metrics and metrics['memory_usage'] > thresholds.get('memory_usage', 90.0):
            alerts.append(f"内存使用率过高: {metrics['memory_usage']:.1f}%")

        # 检查CPU使用率
        if 'cpu_usage' in metrics and metrics['cpu_usage'] > thresholds.get('cpu_usage', 95.0):
            alerts.append(f"CPU使用率过高: {metrics['cpu_usage']:.1f}%")

        if alerts:
            logger.warning("🚨 监控告警: " + "; ".join(alerts))

    def create_dashboard(self, save_path: str = "./agi_persistent_training/metrics/dashboard.png") -> str:
        """创建监控仪表板"""
        logger.info("生成进化监控仪表板...")

        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 创建大图
        fig, axes = plt.subplots(3, 3, figsize=self.fig_size)
        fig.suptitle('H2Q-Evo AGI进化监控仪表板', fontsize=16, fontweight='bold')

        # 转换数据为DataFrame
        df = pd.DataFrame(self.metrics_history)

        # 绘制各个指标
        self._plot_training_metrics(axes[0, 0], df, 'loss', '训练损失', 'red')
        self._plot_training_metrics(axes[0, 1], df, 'accuracy', '准确率', 'green')
        self._plot_training_metrics(axes[0, 2], df, 'compression_ratio', '压缩率', 'blue')

        self._plot_system_metrics(axes[1, 0], df, 'cpu_usage', 'CPU使用率 (%)', 'orange')
        self._plot_system_metrics(axes[1, 1], df, 'memory_usage', '内存使用率 (%)', 'purple')
        self._plot_training_metrics(axes[1, 2], df, 'fitness_score', '适应度分数', 'cyan')

        self._plot_training_metrics(axes[2, 0], df, 'diversity_score', '多样性分数', 'magenta')
        self._plot_generation_progress(axes[2, 1], df)
        self._plot_correlation_matrix(axes[2, 2], df)

        plt.tight_layout()

        # 保存图像
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"仪表板已保存: {save_path}")
        return str(save_path)

    def _plot_training_metrics(self, ax: Axes, df: pd.DataFrame, column: str, title: str, color: str):
        """绘制训练指标"""
        if column in df.columns and not df[column].empty:
            ax.plot(df['generation'], df[column], color=color, linewidth=2, marker='o', markersize=3)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('进化代数')
            ax.grid(True, alpha=0.3)

            # 添加最新值标注
            if len(df[column]) > 0:
                latest_val = df[column].iloc[-1]
                ax.annotate(f'{latest_val:.3f}', xy=(df['generation'].iloc[-1], latest_val),
                           xytext=(5, 5), textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    def _plot_system_metrics(self, ax: Axes, df: pd.DataFrame, column: str, title: str, color: str):
        """绘制系统指标"""
        if column in df.columns and not df[column].empty:
            ax.plot(df.index, df[column], color=color, linewidth=2)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('时间点')
            ax.grid(True, alpha=0.3)

            # 添加最新值标注
            if len(df[column]) > 0:
                latest_val = df[column].iloc[-1]
                ax.annotate(f'{latest_val:.1f}', xy=(len(df)-1, latest_val),
                           xytext=(5, 5), textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    def _plot_generation_progress(self, ax: Axes, df: pd.DataFrame):
        """绘制进化进度"""
        if 'generation' in df.columns and not df['generation'].empty:
            generations = df['generation'].values
            progress = np.arange(len(generations)) / max(1, len(generations) - 1)

            ax.plot(generations, progress, 'g-', linewidth=3, marker='s', markersize=5)
            ax.set_title('进化进度', fontsize=12, fontweight='bold')
            ax.set_xlabel('当前代数')
            ax.set_ylabel('完成百分比')
            ax.grid(True, alpha=0.3)

            # 添加进度标注
            current_gen = generations[-1] if len(generations) > 0 else 0
            ax.text(0.7, 0.3, f'当前代数: {current_gen}', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    def _plot_correlation_matrix(self, ax: Axes, df: pd.DataFrame):
        """绘制相关性矩阵"""
        numeric_cols = ['loss', 'accuracy', 'compression_ratio', 'fitness_score', 'cpu_usage', 'memory_usage']
        available_cols = [col for col in numeric_cols if col in df.columns and not df[col].empty]

        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()

            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('指标相关性矩阵', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '数据不足\n无法计算相关性', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('相关性矩阵', fontsize=12, fontweight='bold')

    def create_realtime_animation(self, save_path: str = "./agi_persistent_training/metrics/realtime_animation.gif") -> str:
        """创建实时动画"""
        logger.info("生成实时进化动画...")

        # 设置动画
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('H2Q-Evo AGI实时进化监控', fontsize=14, fontweight='bold')

        def animate(frame):
            # 清除之前的绘图
            for ax in axes.flat:
                ax.clear()

            # 获取最新数据
            df = pd.DataFrame(self.metrics_history)

            if not df.empty:
                # 绘制实时指标
                self._plot_training_metrics(axes[0, 0], df, 'loss', '训练损失', 'red')
                self._plot_training_metrics(axes[0, 1], df, 'accuracy', '准确率', 'green')
                self._plot_system_metrics(axes[1, 0], df, 'cpu_usage', 'CPU使用率 (%)', 'orange')
                self._plot_system_metrics(axes[1, 1], df, 'memory_usage', '内存使用率 (%)', 'purple')

            return axes.flat

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=50, interval=1000, blit=False)

        # 保存动画
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer='pillow', fps=2)

        plt.close()
        logger.info(f"实时动画已保存: {save_path}")
        return str(save_path)

    def generate_report(self, output_file: str = "./agi_persistent_training/metrics/evolution_report.md") -> str:
        """生成进化报告"""
        logger.info("生成进化报告...")

        df = pd.DataFrame(self.metrics_history)

        report = f"""# H2Q-Evo AGI进化监控报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概述

- 监控总时长: {len(df)} 个数据点
- 当前进化代数: {df['generation'].iloc[-1] if not df.empty else 0}
- 系统运行状态: {'正常' if self.is_monitoring else '已停止'}

## 性能指标总结

"""

        if not df.empty:
            # 计算统计信息
            numeric_cols = ['loss', 'accuracy', 'compression_ratio', 'fitness_score', 'cpu_usage', 'memory_usage']
            for col in numeric_cols:
                if col in df.columns and not df[col].empty:
                    latest = df[col].iloc[-1]
                    mean_val = df[col].mean()
                    min_val = df[col].min()
                    max_val = df[col].max()

                    report += f"### {col.replace('_', ' ').title()}\n"
                    report += f"- 当前值: {latest:.3f}\n"
                    report += f"- 平均值: {mean_val:.3f}\n"
                    report += f"- 最小值: {min_val:.3f}\n"
                    report += f"- 最大值: {max_val:.3f}\n\n"

        # 保存报告
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"进化报告已保存: {output_path}")
        return str(output_path)

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        df = pd.DataFrame(self.metrics_history)

        status = {
            'is_monitoring': self.is_monitoring,
            'total_data_points': len(df),
            'current_generation': df['generation'].iloc[-1] if not df.empty else 0,
            'latest_metrics': {}
        }

        if not df.empty:
            latest = df.iloc[-1]
            for col in ['loss', 'accuracy', 'compression_ratio', 'fitness_score', 'cpu_usage', 'memory_usage']:
                if col in latest.index:
                    status['latest_metrics'][col] = latest[col]

        return status

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='H2Q-Evo AGI进化监控工具')
    parser.add_argument('--mode', choices=['monitor', 'dashboard', 'animation', 'report', 'status'],
                       default='monitor', help='运行模式')
    parser.add_argument('--background', action='store_true', help='后台运行监控')
    parser.add_argument('--output-dir', default='./agi_persistent_training/metrics',
                       help='输出目录')

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # 创建监控器
    monitor = AGIEvolutionMonitor()

    try:
        if args.mode == 'monitor':
            print("🚀 启动AGI进化监控...")
            monitor.start_monitoring(background=args.background)

            if not args.background:
                # 前台运行，等待用户中断
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 停止监控...")
                    monitor.stop_monitoring()

        elif args.mode == 'dashboard':
            dashboard_path = monitor.create_dashboard(f"{args.output_dir}/dashboard.png")
            print(f"✅ 仪表板已生成: {dashboard_path}")

        elif args.mode == 'animation':
            animation_path = monitor.create_realtime_animation(f"{args.output_dir}/realtime_animation.gif")
            print(f"✅ 实时动画已生成: {animation_path}")

        elif args.mode == 'report':
            report_path = monitor.generate_report(f"{args.output_dir}/evolution_report.md")
            print(f"✅ 进化报告已生成: {report_path}")

        elif args.mode == 'status':
            status = monitor.get_current_status()
            print("📊 当前状态:")
            print(json.dumps(status, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"监控工具运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()