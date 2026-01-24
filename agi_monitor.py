#!/usr/bin/env python3
"""
H2Q-Evo AGI健康监控窗口
提供实时系统状态监控和可视化界面
"""

import os
import sys
import json
import time
import curses
import threading
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
        self.status_file = Path("agi_system_status.json")
        self.report_file = Path("agi_system_report.json")
        self.training_status_file = Path("realtime_training_status.json")

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
                self._draw_header(stdscr, width)
                self._draw_system_status(stdscr, width)
                self._draw_training_status(stdscr, width)
                self._draw_performance_metrics(stdscr, width)
                self._draw_fault_status(stdscr, width)
                self._draw_footer(stdscr, width)

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
                stdscr.clear()
                stdscr.addstr(0, 0, f"监控错误: {e}", curses.color_pair(3))
                stdscr.refresh()
                time.sleep(5)

    def _update_status(self):
        """更新状态数据"""
        try:
            # 读取系统状态
            if self.status_file.exists():
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    self.status_data = json.load(f)

            # 读取训练状态
            training_data = {}
            if self.training_status_file.exists():
                with open(self.training_status_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)

            # 合并数据
            self.status_data.update(training_data)

            # 添加到历史
            self.history_data.append({
                'timestamp': time.time(),
                'data': self.status_data.copy()
            })

            # 限制历史长度
            if len(self.history_data) > self.max_history:
                self.history_data = self.history_data[-self.max_history:]

        except Exception as e:
            print(f"更新状态失败: {e}")

    def _draw_header(self, stdscr, width):
        """绘制头部"""
        title = " H2Q-Evo AGI 健康监控系统 "
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 标题栏
        stdscr.addstr(0, 0, "=" * width, curses.color_pair(5))
        stdscr.addstr(1, (width - len(title)) // 2, title, curses.color_pair(5) | curses.A_BOLD)
        stdscr.addstr(2, width - len(timestamp) - 1, timestamp, curses.color_pair(4))
        stdscr.addstr(3, 0, "=" * width, curses.color_pair(5))

    def _draw_system_status(self, stdscr, width):
        """绘制系统状态"""
        y = 5
        stdscr.addstr(y, 0, "系统状态 / System Status", curses.color_pair(5) | curses.A_BOLD)
        y += 1

        try:
            env = self.status_data.get('environment', {})
            infra = self.status_data.get('infrastructure_status', {})

            # CPU信息
            cpu_percent = env.get('cpu_percent', 0)
            cpu_color = self._get_status_color(cpu_percent, 80, 90)
            stdscr.addstr(y, 0, f"CPU使用率: {cpu_percent:.1f}%", cpu_color)
            y += 1

            # 内存信息
            memory_percent = env.get('memory_percent', 0)
            memory_color = self._get_status_color(memory_percent, 85, 95)
            stdscr.addstr(y, 0, f"内存使用率: {memory_percent:.1f}%", memory_color)
            y += 1

            # 磁盘信息
            disk_percent = env.get('disk_percent', 0)
            disk_color = self._get_status_color(disk_percent, 90, 95)
            stdscr.addstr(y, 0, f"磁盘使用率: {disk_percent:.1f}%", disk_color)
            y += 1

            # 网络状态
            network = self.status_data.get('network', {})
            internet_status = "连接" if network.get('internet_connected', False) else "断开"
            net_color = curses.color_pair(1) if network.get('internet_connected', False) else curses.color_pair(3)
            stdscr.addstr(y, 0, f"网络状态: {internet_status}", net_color)
            y += 1

            # 基础设施状态
            infra_running = infra.get('infrastructure_running', False)
            infra_color = curses.color_pair(1) if infra_running else curses.color_pair(3)
            stdscr.addstr(y, 0, f"基础设施: {'运行中' if infra_running else '已停止'}", infra_color)
            y += 1

        except Exception as e:
            stdscr.addstr(y, 0, f"系统状态读取失败: {e}", curses.color_pair(3))
            y += 1

        y += 1  # 空行

    def _draw_training_status(self, stdscr, width):
        """绘制训练状态"""
        y = 15
        stdscr.addstr(y, 0, "训练状态 / Training Status", curses.color_pair(5) | curses.A_BOLD)
        y += 1

        try:
            training = self.status_data.get('training_status', {})

            # 训练运行状态
            training_active = training.get('training_active', False)
            training_color = curses.color_pair(1) if training_active else curses.color_pair(2)
            stdscr.addstr(y, 0, f"训练状态: {'运行中' if training_active else '已停止'}", training_color)
            y += 1

            # 热生成状态
            hot_gen_active = training.get('hot_generation_active', False)
            hot_gen_color = curses.color_pair(1) if hot_gen_active else curses.color_pair(2)
            stdscr.addstr(y, 0, f"热生成: {'运行中' if hot_gen_active else '已停止'}", hot_gen_color)
            y += 1

            # 训练步骤
            current_step = training.get('current_step', 0)
            stdscr.addstr(y, 0, f"训练步骤: {current_step:,}", curses.color_pair(6))
            y += 1

            # 最佳损失
            best_loss = training.get('best_loss', float('inf'))
            if best_loss != float('inf'):
                stdscr.addstr(y, 0, f"最佳损失: {best_loss:.6f}", curses.color_pair(6))
            else:
                stdscr.addstr(y, 0, "最佳损失: N/A", curses.color_pair(6))
            y += 1

            # 最佳准确率
            best_accuracy = training.get('best_accuracy', 0)
            stdscr.addstr(y, 0, f"最佳准确率: {best_accuracy:.4f}", curses.color_pair(6))
            y += 1

        except Exception as e:
            stdscr.addstr(y, 0, f"训练状态读取失败: {e}", curses.color_pair(3))
            y += 1

        y += 1  # 空行

    def _draw_performance_metrics(self, stdscr, width):
        """绘制性能指标"""
        y = 25
        stdscr.addstr(y, 0, "性能指标 / Performance Metrics", curses.color_pair(5) | curses.A_BOLD)
        y += 1

        try:
            perf = self.status_data.get('performance_metrics', {})

            # 训练步骤总数
            total_steps = perf.get('training_steps', 0)
            stdscr.addstr(y, 0, f"总训练步骤: {total_steps:,}", curses.color_pair(6))
            y += 1

            # 处理的样本数
            total_samples = perf.get('total_samples_processed', 0)
            stdscr.addstr(y, 0, f"处理样本数: {total_samples:,}", curses.color_pair(6))
            y += 1

            # 平均损失
            avg_loss = perf.get('average_loss', 0)
            stdscr.addstr(y, 0, f"平均损失: {avg_loss:.6f}", curses.color_pair(6))
            y += 1

            # 学习率
            learning_rate = perf.get('learning_rate', 0)
            stdscr.addstr(y, 0, f"学习率: {learning_rate:.6f}", curses.color_pair(6))
            y += 1

            # 节流事件
            throttle_events = perf.get('throttle_events', 0)
            throttle_color = curses.color_pair(2) if throttle_events > 0 else curses.color_pair(6)
            stdscr.addstr(y, 0, f"节流事件: {throttle_events}", throttle_color)
            y += 1

        except Exception as e:
            stdscr.addstr(y, 0, f"性能指标读取失败: {e}", curses.color_pair(3))
            y += 1

        y += 1  # 空行

    def _draw_fault_status(self, stdscr, width):
        """绘制故障状态"""
        y = 35
        stdscr.addstr(y, 0, "故障状态 / Fault Status", curses.color_pair(5) | curses.A_BOLD)
        y += 1

        try:
            health = self.status_data.get('system_health', {})

            # 整体健康状态
            overall_health = health.get('overall_health', 'unknown')
            health_color = self._get_health_color(overall_health)
            stdscr.addstr(y, 0, f"整体健康: {overall_health}", health_color)
            y += 1

            # 最近故障数量
            recent_faults = health.get('recent_faults', [])
            faults_color = curses.color_pair(3) if len(recent_faults) > 0 else curses.color_pair(1)
            stdscr.addstr(y, 0, f"最近故障: {len(recent_faults)} 个", faults_color)
            y += 1

            # 显示最近的故障
            if recent_faults:
                for i, fault in enumerate(recent_faults[:3]):  # 只显示前3个
                    fault_time = datetime.fromtimestamp(fault.get('timestamp', 0)).strftime("%H:%M:%S")
                    fault_type = fault.get('fault_type', 'unknown')
                    fault_severity = fault.get('severity', 'low')
                    stdscr.addstr(y, 0, f"  {fault_time} {fault_type} ({fault_severity})", curses.color_pair(3))
                    y += 1

        except Exception as e:
            stdscr.addstr(y, 0, f"故障状态读取失败: {e}", curses.color_pair(3))
            y += 1

        y += 1  # 空行

    def _draw_footer(self, stdscr, width):
        """绘制底部"""
        height, width = stdscr.getmaxyx()
        footer_y = height - 3

        stdscr.addstr(footer_y, 0, "=" * width, curses.color_pair(5))
        stdscr.addstr(footer_y + 1, 0, " Q: 退出 | R: 刷新 | 自动更新间隔: 2秒 ", curses.color_pair(4))
        stdscr.addstr(footer_y + 2, 0, "=" * width, curses.color_pair(5))

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