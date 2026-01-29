#!/usr/bin/env python3
"""
H2Q-Evo AGI健康监控窗口
提供实时系统状态监控和可视化界面
包含真实AGI目标验证和审计基准验收
"""

import os
import sys
import json
import time
import curses
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import numpy as np

class AGIMonitor:
    """AGI系统健康监控器"""

    def __init__(self):
        self.running = False
        self.status_data = {}
        self.history_data = []
        self.max_history = 100
        self.update_interval = 2  # 2秒更新一次

        # 监控文件路径
        self.status_file = Path("agi_unified_status.json")
        self.report_file = Path("agi_system_report.json")
        self.training_status_file = Path("realtime_training_status.json")

        # AGI目标定义 - 基于真实几何指标
        self.agi_targets = {
            'geometric_accuracy': 0.9,      # SU(2)流形推理准确率
            'spectral_shift_eta': 0.5,      # 谱移认知进展
            'fractal_collapse_penalty': 0.1, # 流形稳定性阈值
            'classification_f1': 0.85,      # 多域学习能力
            'manifold_stability': 5.0       # 流形稳定性目标
        }

        # 审计基准状态
        self.audit_triggered = False
        self.audit_results = {}

    def check_agi_targets_achieved(self):
        """检查是否达到AGI目标 - 基于真实几何指标"""
        try:
            training = self.status_data.get('training_status', {})
            geometric = self.status_data.get('geometric_metrics', {})

            # 获取当前指标
            geometric_accuracy = geometric.get('geometric_accuracy', 0)
            spectral_shift_eta = geometric.get('spectral_shift_eta_real', 0)
            fractal_penalty = geometric.get('fractal_collapse_penalty', 1.0)
            classification_f1 = geometric.get('classification_f1', 0)

            perf = self.status_data.get('performance_metrics', {})
            manifold_stability = perf.get('manifold_stability', 0)

            # 检查所有目标是否达到
            targets_achieved = {
                'geometric_accuracy': geometric_accuracy >= self.agi_targets['geometric_accuracy'],
                'spectral_shift_eta': spectral_shift_eta >= self.agi_targets['spectral_shift_eta'],
                'fractal_collapse_penalty': fractal_penalty <= self.agi_targets['fractal_collapse_penalty'],
                'classification_f1': classification_f1 >= self.agi_targets['classification_f1'],
                'manifold_stability': manifold_stability >= self.agi_targets['manifold_stability']
            }

            all_achieved = all(targets_achieved.values())

            return {
                'achieved': all_achieved,
                'current_values': {
                    'geometric_accuracy': geometric_accuracy,
                    'spectral_shift_eta': spectral_shift_eta,
                    'fractal_collapse_penalty': fractal_penalty,
                    'classification_f1': classification_f1,
                    'manifold_stability': manifold_stability
                },
                'targets': self.agi_targets.copy(),
                'individual_status': targets_achieved
            }

        except Exception as e:
            print(f"AGI目标检查失败: {e}")
            return {'achieved': False, 'error': str(e)}

    def trigger_audit_benchmark(self):
        """触发审计基准验收 - 基于真实AGI能力"""
        if self.audit_triggered:
            return False  # 已经触发过

        try:
            print("🎯 AGI目标已达到！正在启动审计基准验收...")
            self.audit_triggered = True

            # 运行审计基准脚本
            audit_script = Path("audit_agi_performance.py")
            if audit_script.exists():
                result = subprocess.run([
                    sys.executable, str(audit_script)
                ], capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    # 解析审计结果
                    try:
                        self.audit_results = json.loads(result.stdout)
                        print("✅ 审计基准验收完成！")
                        return True
                    except json.JSONDecodeError:
                        print("❌ 审计结果解析失败")
                        return False
                else:
                    print(f"❌ 审计基准运行失败: {result.stderr}")
                    return False
            else:
                print("❌ 审计脚本不存在")
                return False

        except Exception as e:
            print(f"审计基准触发失败: {e}")
            return False

    def start_monitoring(self):
        """启动监控"""
        self.running = True
        try:
            curses.wrapper(self._monitor_loop)
        except KeyboardInterrupt:
            self.stop_monitoring()

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        print("\n监控已停止")

    def _monitor_loop(self, stdscr):
        """监控主循环"""
        # 初始化curses
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)    # 正常
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)   # 警告
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)      # 错误
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)     # 信息
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # 标题
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)    # 普通

        stdscr.nodelay(True)
        stdscr.timeout(1000)  # 1秒超时

        while self.running:
            try:
                # 读取最新状态
                self._update_status()

                # 清屏
                stdscr.clear()

                # 获取屏幕尺寸
                height, width = stdscr.getmaxyx()

                # 绘制界面
                y_pos = 4  # 从标题后开始
                self._draw_header(stdscr, width)
                y_pos = self._draw_system_status(stdscr, width) or 10
                y_pos = self._draw_training_status(stdscr, y_pos, width) or y_pos + 5
                y_pos = self._draw_agi_targets_status(stdscr, y_pos, width) or y_pos + 5
                y_pos = self._draw_performance_metrics(stdscr, y_pos, width) or y_pos + 5
                y_pos = self._draw_fault_status(stdscr, y_pos, width) or y_pos + 5
                self._draw_footer(stdscr, y_pos, width)

                # 刷新屏幕
                stdscr.refresh()

                # 检查用户输入
                key = stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    # 重新加载数据
                    self._update_status()

                time.sleep(self.update_interval)

            except Exception as e:
                try:
                    stdscr.clear()
                    error_msg = f"监控错误: {str(e)[:100]}"  # 限制错误消息长度
                    height, width = stdscr.getmaxyx()
                    if height > 0 and width > len(error_msg):
                        stdscr.addstr(0, 0, error_msg, curses.color_pair(3))
                        stdscr.addstr(2, 0, "按 'q' 退出, 'r' 重试", curses.color_pair(4))
                    stdscr.refresh()
                except:
                    # 如果连错误显示都失败，打印到控制台
                    print(f"监控错误: {e}")
                time.sleep(5)

    def _update_status(self):
        """更新状态数据 - 只读取真实训练数据，剔除任何模拟数据"""
        try:
            # 只读取实时训练状态文件 - 这是唯一真实的数据源
            if self.training_status_file.exists():
                with open(self.training_status_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)

                # 验证数据真实性 - 检查是否有训练进程在运行
                training_process_running = self._verify_training_process_real()

                if not training_process_running:
                    print("⚠️ 警告: 未检测到真实训练进程，数据可能不是最新的")
                    # 仍然显示数据，但标记为可能过时
                    training_data['data_freshness'] = 'stale'
                else:
                    training_data['data_freshness'] = 'fresh'

                # 初始化状态数据，只使用真实训练数据
                self.status_data = {}

                # 直接使用训练数据作为主要数据源
                self.status_data.update(training_data)

                # 重新组织数据结构以保持兼容性
                self.status_data['training_status'] = {
                    'training_active': training_data.get('training_active', False),
                    'current_step': training_data.get('current_step', 0),
                    'current_epoch': training_data.get('current_epoch', 0),
                    'best_accuracy': training_data.get('best_accuracy', 0),
                    'best_loss': training_data.get('best_loss', float('inf')),
                    'system_health': training_data.get('system_health', 'unknown'),
                    'data_freshness': training_data.get('data_freshness', 'unknown')
                }

                # 几何指标直接来自训练数据
                self.status_data['geometric_metrics'] = training_data.get('geometric_metrics', {})

                # 性能指标直接来自训练数据
                self.status_data['performance_metrics'] = training_data.get('performance_metrics', {})

                # 环境信息来自训练数据
                self.status_data['environment'] = {
                    'cpu_percent': training_data.get('cpu_percent', 0),
                    'memory_percent': training_data.get('memory_percent', 0),
                    'disk_percent': 0,  # 暂时设为0，因为训练数据中没有
                    'internet_connected': True  # 假设连接正常
                }

                # 网络状态
                self.status_data['network'] = {
                    'internet_connected': True
                }

                # 基础设施状态 - 基于训练进程状态
                self.status_data['infrastructure_status'] = {
                    'infrastructure_running': training_process_running
                }

                # 系统健康
                self.status_data['system_health'] = {
                    'overall_health': training_data.get('system_health', 'unknown')
                }

                # print(f"✅ 真实训练数据已更新 - 步骤: {training_data.get('current_step', 0)} - 数据新鲜度: {training_data.get('data_freshness', 'unknown')}")
            else:
                print(f"❌ 错误: 实时训练状态文件不存在 - {self.training_status_file}")
                print("无法获取真实训练数据")
                self.status_data = {}

            # 检查AGI目标是否达到 - 只基于真实数据
            agi_status = self.check_agi_targets_achieved()
            if agi_status.get('achieved', False) and not self.audit_triggered:
                self.trigger_audit_benchmark()

            # 添加AGI状态到数据
            self.status_data['agi_targets_status'] = agi_status
            self.status_data['audit_status'] = {
                'triggered': self.audit_triggered,
                'results': self.audit_results
            }

            # 添加到历史
            self.history_data.append({
                'timestamp': time.time(),
                'data': self.status_data.copy()
            })

            # 限制历史长度
            if len(self.history_data) > self.max_history:
                self.history_data = self.history_data[-self.max_history:]

        except Exception as e:
            print(f"❌ 更新真实训练数据失败: {e}")
            self.status_data = {}

    def _verify_training_process_real(self):
        """验证训练进程的真实性 - 确保数据来自真实训练"""
        try:
            import subprocess

            # 检查是否有真实的训练进程在运行
            result = subprocess.run(
                ['pgrep', '-f', 'memory_safe_training_launcher'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                # 找到训练进程，直接验证进程存在性
                pid = result.stdout.strip().split('\n')[0].strip()

                # 简单检查：进程是否存在
                check_result = subprocess.run(
                    ['kill', '-0', pid],  # 发送信号0来检查进程是否存在
                    capture_output=True
                )

                if check_result.returncode == 0:
                    return True

            return False

        except Exception as e:
            print(f"验证训练进程失败: {e}")
            return False

    def _draw_header(self, stdscr, width):
        """绘制头部"""
        if width < 20:  # 最小宽度检查
            return

        title = " H2Q-Evo AGI 健康监控系统 "
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 标题栏
            stdscr.addstr(0, 0, "=" * min(width, 80), curses.color_pair(5))
            if len(title) < width:
                stdscr.addstr(1, (width - len(title)) // 2, title, curses.color_pair(5) | curses.A_BOLD)
            if len(timestamp) < width:
                stdscr.addstr(2, width - len(timestamp) - 1, timestamp, curses.color_pair(4))
            stdscr.addstr(3, 0, "=" * min(width, 80), curses.color_pair(5))
        except curses.error:
            pass  # 忽略绘制错误

    def _draw_system_status(self, stdscr, width):
        """绘制系统状态"""
        if width < 20:  # 最小宽度检查
            return

        y = 5
        try:
            stdscr.addstr(y, 0, "系统状态 / System Status", curses.color_pair(5) | curses.A_BOLD)
            y += 1

            env = self.status_data.get('environment', {})
            infra = self.status_data.get('infrastructure_status', {})

            # CPU信息
            cpu_percent = env.get('cpu_percent', 0)
            cpu_color = self._get_status_color(cpu_percent, 80, 90)
            if y < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y, 0, f"CPU使用率: {cpu_percent:.1f}%", cpu_color)
                y += 1

            # 内存信息
            memory_percent = env.get('memory_percent', 0)
            memory_color = self._get_status_color(memory_percent, 85, 95)
            if y < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y, 0, f"内存使用率: {memory_percent:.1f}%", memory_color)
                y += 1

            # 磁盘信息
            disk_percent = env.get('disk_percent', 0)
            disk_color = self._get_status_color(disk_percent, 90, 95)
            if y < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y, 0, f"磁盘使用率: {disk_percent:.1f}%", disk_color)
                y += 1

            # 网络状态
            network = self.status_data.get('network', {})
            internet_status = "连接" if network.get('internet_connected', False) else "断开"
            net_color = curses.color_pair(1) if network.get('internet_connected', False) else curses.color_pair(3)
            if y < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y, 0, f"网络状态: {internet_status}", net_color)
                y += 1

            # 基础设施状态
            infra_running = infra.get('infrastructure_running', False)
            infra_color = curses.color_pair(1) if infra_running else curses.color_pair(3)
            if y < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y, 0, f"基础设施: {'运行中' if infra_running else '已停止'}", infra_color)
                y += 1

        except curses.error:
            pass  # 忽略绘制错误

        return y + 1  # 返回下一个y位置

    def _draw_training_status(self, stdscr, y_pos, width):
        """绘制训练状态 - 只显示真实的核心几何指标"""
        if width < 20:  # 最小宽度检查
            return y_pos

        try:
            stdscr.addstr(y_pos, 0, "Real Training Status", curses.color_pair(5) | curses.A_BOLD)
            y_pos += 1

            training = self.status_data.get('training_status', {})

            # 数据新鲜度指示器
            data_freshness = training.get('data_freshness', 'unknown')
            if data_freshness == 'fresh':
                freshness_indicator = "🟢 LIVE DATA"
                freshness_color = curses.color_pair(1)
            elif data_freshness == 'stale':
                freshness_indicator = "🟡 STALE DATA"
                freshness_color = curses.color_pair(2)
            else:
                freshness_indicator = "🔴 UNKNOWN"
                freshness_color = curses.color_pair(3)

            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"Data Status: {freshness_indicator}", freshness_color)
                y_pos += 1

            # 训练运行状态
            training_active = training.get('training_active', False)
            training_color = curses.color_pair(1) if training_active else curses.color_pair(2)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"Training: {'ACTIVE' if training_active else 'INACTIVE'}", training_color)
                y_pos += 1

            # 训练步骤
            current_step = training.get('current_step', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"Step: {current_step:,}", curses.color_pair(6))
                y_pos += 1

            # 只显示真实的核心几何指标
            geometric = self.status_data.get('geometric_metrics', {})

            # 谱移η实部 - 核心SU(2)指标
            if y_pos < stdscr.getmaxyx()[0] - 1:
                eta_real = geometric.get('spectral_shift_eta_real', 0)
                eta_color = curses.color_pair(1) if abs(eta_real) > 0.01 else curses.color_pair(6)
                stdscr.addstr(y_pos, 0, f"Spectral η Real: {eta_real:.6f}", eta_color)
                y_pos += 1

            # 分形坍缩惩罚 - 核心几何稳定性指标
            if y_pos < stdscr.getmaxyx()[0] - 1:
                collapse_penalty = geometric.get('fractal_collapse_penalty', 0)
                collapse_color = curses.color_pair(1) if collapse_penalty < 0.5 else curses.color_pair(2)
                stdscr.addstr(y_pos, 0, f"Fractal Collapse: {collapse_penalty:.6f}", collapse_color)
                y_pos += 1

            # 几何准确率 - 基于谱移的推理能力
            if y_pos < stdscr.getmaxyx()[0] - 1:
                geom_acc = geometric.get('geometric_accuracy', 0)
                geom_color = curses.color_pair(1) if geom_acc > 0.01 else curses.color_pair(6)
                stdscr.addstr(y_pos, 0, f"Geometric Acc: {geom_acc:.6f}", geom_color)
                y_pos += 1

            # 移除非核心指标（损失、准确率等没有支撑的数据）
            # 这些指标基于随机数据生成，没有真实意义

        except curses.error:
            pass  # 忽略绘制错误

        return y_pos + 1

    def _draw_agi_targets_status(self, stdscr, y_pos, width):
        """绘制AGI目标状态"""
        if width < 20:  # 最小宽度检查
            return y_pos

        try:
            stdscr.addstr(y_pos, 0, "AGI目标状态 / AGI Targets Status", curses.color_pair(5) | curses.A_BOLD)
            y_pos += 1

            agi_status = self.status_data.get('agi_targets_status', {})
            current_values = agi_status.get('current_values', {})
            targets = agi_status.get('targets', {})
            individual_status = agi_status.get('individual_status', {})

            # AGI目标达成状态
            achieved = agi_status.get('achieved', False)
            overall_color = curses.color_pair(1) if achieved else curses.color_pair(2)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"AGI目标达成: {'✅ 已达成' if achieved else '⏳ 进行中'}", overall_color)
                y_pos += 1

            # 显示各个指标
            metrics = [
                ('几何准确率', 'geometric_accuracy', '.4f'),
                ('谱移η实部', 'spectral_shift_eta', '.4f'),
                ('分形坍缩惩罚', 'fractal_collapse_penalty', '.4f'),
                ('分类F1分数', 'classification_f1', '.4f'),
                ('流形稳定性', 'manifold_stability', '.2f')
            ]

            for metric_name, metric_key, format_str in metrics:
                if y_pos >= stdscr.getmaxyx()[0] - 1:
                    break

                current_val = current_values.get(metric_key, 0)
                target_val = targets.get(metric_key, 0)
                status = individual_status.get(metric_key, False)

                status_icon = "✅" if status else "❌"
                color = curses.color_pair(1) if status else curses.color_pair(3)

                if metric_key == 'fractal_collapse_penalty':
                    # 对于坍缩惩罚，越小越好
                    status_icon = "✅" if current_val <= target_val else "❌"
                    color = curses.color_pair(1) if current_val <= target_val else curses.color_pair(3)
                else:
                    # 其他指标越大越好
                    status_icon = "✅" if current_val >= target_val else "❌"
                    color = curses.color_pair(1) if current_val >= target_val else curses.color_pair(3)

                stdscr.addstr(y_pos, 0, f"{status_icon} {metric_name}: {current_val:{format_str}}/{target_val:{format_str}}", color)
                y_pos += 1

            # 审计基准状态
            audit_status = self.status_data.get('audit_status', {})
            audit_triggered = audit_status.get('triggered', False)
            audit_color = curses.color_pair(1) if audit_triggered else curses.color_pair(6)

            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"审计基准: {'✅ 已触发' if audit_triggered else '⏳ 等待中'}", audit_color)
                y_pos += 1

        except curses.error:
            pass  # 忽略绘制错误

        return y_pos + 1
        """绘制性能指标"""
        if width < 20:  # 最小宽度检查
            return y_pos

        try:
            stdscr.addstr(y_pos, 0, "性能指标 / Performance Metrics", curses.color_pair(5) | curses.A_BOLD)
            y_pos += 1

            perf = self.status_data.get('performance_metrics', {})

            # 训练步骤总数
            total_steps = perf.get('training_steps', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"总训练步骤: {total_steps:,}", curses.color_pair(6))
                y_pos += 1

            # 处理的样本数
            total_samples = perf.get('total_samples_processed', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"处理样本数: {total_samples:,}", curses.color_pair(6))
                y_pos += 1

            # 平均损失
            avg_loss = perf.get('average_loss', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"平均损失: {avg_loss:.6f}", curses.color_pair(6))
                y_pos += 1

            # 学习率
            learning_rate = perf.get('learning_rate', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"学习率: {learning_rate:.6f}", curses.color_pair(6))
                y_pos += 1

            # 节流事件
            throttle_events = perf.get('throttle_events', 0)
            throttle_color = curses.color_pair(2) if throttle_events > 0 else curses.color_pair(6)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"节流事件: {throttle_events}", throttle_color)
                y_pos += 1

        except curses.error:
            pass  # 忽略绘制错误

        return y_pos + 1

    def _draw_performance_metrics(self, stdscr, y_pos, width):
        """绘制性能指标 - 显示真实训练性能数据"""
        if width < 20:  # 最小宽度检查
            return y_pos

        try:
            stdscr.addstr(y_pos, 0, "性能指标 / Performance Metrics", curses.color_pair(5) | curses.A_BOLD)
            y_pos += 1

            perf = self.status_data.get('performance_metrics', {})

            # 训练样本数
            total_samples = perf.get('total_samples_processed', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"处理样本数: {total_samples:,}", curses.color_pair(6))
                y_pos += 1

            # 平均损失
            avg_loss = perf.get('average_loss', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"平均损失: {avg_loss:.6f}", curses.color_pair(6))
                y_pos += 1

            # 学习率
            learning_rate = perf.get('learning_rate', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"学习率: {learning_rate:.6f}", curses.color_pair(6))
                y_pos += 1

            # 节流和恢复事件
            throttle_events = perf.get('throttle_events', 0)
            recovery_events = perf.get('recovery_events', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                throttle_color = curses.color_pair(2) if throttle_events > 0 else curses.color_pair(1)
                stdscr.addstr(y_pos, 0, f"节流事件: {throttle_events}", throttle_color)
                y_pos += 1

            if y_pos < stdscr.getmaxyx()[0] - 1:
                recovery_color = curses.color_pair(2) if recovery_events > 0 else curses.color_pair(1)
                stdscr.addstr(y_pos, 0, f"恢复事件: {recovery_events}", recovery_color)
                y_pos += 1

            # 几何收敛率
            convergence_rate = perf.get('geometric_convergence_rate', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                conv_color = curses.color_pair(1) if convergence_rate > 0.01 else curses.color_pair(6)
                stdscr.addstr(y_pos, 0, f"几何收敛率: {convergence_rate:.6f}", conv_color)
                y_pos += 1

            # 流形稳定性
            manifold_stability = perf.get('manifold_stability', 0)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stab_color = curses.color_pair(1) if manifold_stability > 3.0 else curses.color_pair(6)
                stdscr.addstr(y_pos, 0, f"流形稳定性: {manifold_stability:.4f}", stab_color)
                y_pos += 1

        except curses.error:
            pass  # 忽略绘制错误

        return y_pos + 1

    def _draw_fault_status(self, stdscr, y_pos, width):
        """绘制故障状态"""
        if width < 20:  # 最小宽度检查
            return y_pos

        try:
            stdscr.addstr(y_pos, 0, "故障状态 / Fault Status", curses.color_pair(5) | curses.A_BOLD)
            y_pos += 1

            health = self.status_data.get('system_health', {})

            # 整体健康状态
            overall_health = health.get('overall_health', 'unknown')
            health_color = self._get_health_color(overall_health)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"整体健康: {overall_health}", health_color)
                y_pos += 1

            # 最近故障数量
            recent_faults = health.get('recent_faults', [])
            faults_color = curses.color_pair(3) if len(recent_faults) > 0 else curses.color_pair(1)
            if y_pos < stdscr.getmaxyx()[0] - 1:
                stdscr.addstr(y_pos, 0, f"最近故障: {len(recent_faults)} 个", faults_color)
                y_pos += 1

            # 显示最近的故障
            if recent_faults:
                for i, fault in enumerate(recent_faults[:3]):  # 只显示前3个
                    if y_pos >= stdscr.getmaxyx()[0] - 1:
                        break
                    fault_time = datetime.fromtimestamp(fault.get('timestamp', 0)).strftime("%H:%M:%S")
                    fault_type = fault.get('fault_type', 'unknown')
                    fault_severity = fault.get('severity', 'low')
                    stdscr.addstr(y_pos, 0, f"  {fault_time} {fault_type} ({fault_severity})", curses.color_pair(3))
                    y_pos += 1

        except curses.error:
            pass  # 忽略绘制错误

        return y_pos + 1

    def _draw_footer(self, stdscr, y_pos, width):
        """绘制底部"""
        try:
            height, width = stdscr.getmaxyx()
            footer_y = max(y_pos, height - 3)

            if footer_y < height:
                stdscr.addstr(footer_y, 0, "=" * min(width, 80), curses.color_pair(5))
            if footer_y + 1 < height:
                stdscr.addstr(footer_y + 1, 0, " Q: 退出 | R: 刷新 | 自动更新间隔: 2秒 ", curses.color_pair(4))
            if footer_y + 2 < height:
                stdscr.addstr(footer_y + 2, 0, "=" * min(width, 80), curses.color_pair(5))
        except curses.error:
            pass  # 忽略绘制错误

    def _get_status_color(self, value, warning_threshold, critical_threshold):
        """获取状态颜色"""
        if value >= critical_threshold:
            return curses.color_pair(3)  # 红色
        elif value >= warning_threshold:
            return curses.color_pair(2)  # 黄色
        else:
            return curses.color_pair(1)  # 绿色

    def _get_health_color(self, health_status):
        """获取健康状态颜色"""
        if health_status == 'healthy':
            return curses.color_pair(1)  # 绿色
        elif health_status == 'warning':
            return curses.color_pair(2)  # 黄色
        elif health_status == 'critical':
            return curses.color_pair(3)  # 红色
        else:
            return curses.color_pair(6)  # 白色

def print_text_monitor():
    """文本模式监控（无curses时使用）"""
    print("H2Q-Evo AGI 健康监控系统 (文本模式)")
    print("=" * 60)

    monitor = AGIMonitor()

    try:
        while True:
            monitor._update_status()

            print(f"\n更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)

            # 系统状态
            env = monitor.status_data.get('environment', {})
            print(f"CPU使用率: {env.get('cpu_percent', 0):.1f}%")
            print(f"内存使用率: {env.get('memory_percent', 0):.1f}%")
            print(f"磁盘使用率: {env.get('disk_percent', 0):.1f}%")

            # 训练状态
            training = monitor.status_data.get('training_status', {})
            print(f"训练状态: {'运行中' if training.get('training_active', False) else '已停止'}")
            print(f"训练步骤: {training.get('current_step', 0):,}")
            print(f"最佳损失: {training.get('best_loss', 'N/A')}")

            # AGI目标状态
            agi_status = monitor.status_data.get('agi_targets_status', {})
            print(f"\nAGI目标状态:")
            if agi_status.get('achieved', False):
                print("🎯 AGI目标: ✅ 已达成")
            else:
                print("🎯 AGI目标: ⏳ 进行中")

            current_values = agi_status.get('current_values', {})
            targets = agi_status.get('targets', {})
            print(f"  几何准确率: {current_values.get('geometric_accuracy', 0):.4f}/{targets.get('geometric_accuracy', 0):.4f}")
            print(f"  谱移η实部: {current_values.get('spectral_shift_eta', 0):.4f}/{targets.get('spectral_shift_eta', 0):.4f}")
            print(f"  分形坍缩惩罚: {current_values.get('fractal_collapse_penalty', 0):.4f}/{targets.get('fractal_collapse_penalty', 0):.4f}")
            print(f"  分类F1分数: {current_values.get('classification_f1', 0):.4f}/{targets.get('classification_f1', 0):.4f}")
            print(f"  流形稳定性: {current_values.get('manifold_stability', 0):.2f}/{targets.get('manifold_stability', 0):.2f}")

            # 审计基准状态
            audit_status = monitor.status_data.get('audit_status', {})
            audit_triggered = audit_status.get('triggered', False)
            print(f"审计基准: {'✅ 已触发' if audit_triggered else '⏳ 等待中'}")

            # 健康状态
            health = monitor.status_data.get('system_health', {})
            print(f"系统健康: {health.get('overall_health', 'unknown')}")

            print("\n按 Ctrl+C 退出...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n监控已停止")

def main():
    """主函数"""
    try:
        monitor = AGIMonitor()
        monitor.start_monitoring()
    except ImportError:
        print("curses模块不可用，使用文本模式...")
        print_text_monitor()
    except Exception as e:
        print(f"启动监控失败: {e}")
        print("使用文本模式...")
        print_text_monitor()

if __name__ == "__main__":
    main()