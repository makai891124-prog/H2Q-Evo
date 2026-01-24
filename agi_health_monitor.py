#!/usr/bin/env python3
"""
H2Q-Evo AGI健康监控窗口
为Mac Mini M4优化的实时监控界面
"""

import os
import sys
import json
import time
import curses
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import psutil

class HealthMonitorWindow:
    """健康监控窗口"""

    def __init__(self):
        self.screen = None
        self.running = False
        self.status_data = {}
        self.update_interval = 2  # 2秒更新一次
        self.monitor_thread = None

    def start_monitoring(self):
        """启动监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        try:
            curses.wrapper(self._main_window)
        except KeyboardInterrupt:
            self.stop_monitoring()
        except Exception as e:
            print(f"监控窗口异常: {e}")
            self.stop_monitoring()

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._update_status_data()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"监控循环异常: {e}")
                time.sleep(5)

    def _update_status_data(self):
        """更新状态数据"""
        try:
            # 读取流式训练状态
            streaming_status_file = Path('mac_mini_streaming_status.json')
            if streaming_status_file.exists():
                with open(streaming_status_file, 'r', encoding='utf-8') as f:
                    streaming_data = json.load(f)
            else:
                streaming_data = {}

            # 读取系统状态
            system_status_file = Path('agi_system_status.json')
            if system_status_file.exists():
                with open(system_status_file, 'r', encoding='utf-8') as f:
                    system_data = json.load(f)
            else:
                system_data = {}

            # 读取基础设施状态
            infra_status_file = Path('agi_system_report.json')
            if infra_status_file.exists():
                with open(infra_status_file, 'r', encoding='utf-8') as f:
                    infra_data = json.load(f)
            else:
                infra_data = {}

            # 合并状态数据
            self.status_data = {
                'streaming': streaming_data,
                'system': system_data,
                'infrastructure': infra_data,
                'realtime': self._get_realtime_stats()
            }

        except Exception as e:
            self.status_data = {'error': str(e)}

    def _get_realtime_stats(self) -> Dict[str, Any]:
        """获取实时统计"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'timestamp': datetime.now().isoformat()
        }

    def _main_window(self, stdscr):
        """主窗口"""
        self.screen = stdscr
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # 正常
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # 警告
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # 错误
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # 信息
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # 标题

        self.screen.clear()
        self.screen.nodelay(True)  # 非阻塞输入

        while self.running:
            try:
                self._draw_interface()
                self.screen.refresh()

                # 检查键盘输入
                key = self.screen.getch()
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self._force_refresh()
                elif key == ord('c') or key == ord('C'):
                    self._clear_logs()

                time.sleep(0.1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.screen.clear()
                self.screen.addstr(0, 0, f"界面错误: {e}", curses.color_pair(3))
                self.screen.refresh()
                time.sleep(2)

        self.stop_monitoring()

    def _draw_interface(self):
        """绘制界面"""
        if not self.screen:
            return

        self.screen.clear()
        height, width = self.screen.getmaxyx()

        # 标题
        title = "H2Q-Evo AGI健康监控 - Mac Mini M4优化版"
        self.screen.addstr(0, 0, title, curses.color_pair(5) | curses.A_BOLD)
        self.screen.addstr(1, 0, "=" * min(len(title), width-1), curses.color_pair(5))

        # 实时状态
        self._draw_realtime_stats(3, 0)

        # 流式训练状态
        self._draw_streaming_status(10, 0)

        # 系统健康状态
        self._draw_system_health(20, 0)

        # 基础设施状态
        self._draw_infrastructure_status(30, 0)

        # 操作提示
        self._draw_controls(height-3, 0)

        # 状态栏
        self._draw_status_bar(height-1, 0)

    def _draw_realtime_stats(self, y: int, x: int):
        """绘制实时统计"""
        realtime = self.status_data.get('realtime', {})

        self.screen.addstr(y, x, "实时系统状态:", curses.color_pair(4) | curses.A_BOLD)
        self.screen.addstr(y+1, x, f"CPU使用率: {realtime.get('cpu_percent', 0):.1f}%")
        self.screen.addstr(y+2, x, f"内存使用: {realtime.get('memory_used_gb', 0):.2f}GB ({realtime.get('memory_percent', 0):.1f}%)")
        self.screen.addstr(y+3, x, f"磁盘使用: {realtime.get('disk_percent', 0):.1f}%")
        self.screen.addstr(y+4, x, f"网络连接: {realtime.get('network_connections', 0)}")

    def _draw_streaming_status(self, y: int, x: int):
        """绘制流式训练状态"""
        streaming = self.status_data.get('streaming', {})

        self.screen.addstr(y, x, "流式训练状态:", curses.color_pair(4) | curses.A_BOLD)

        if streaming.get('training_active', False):
            status_color = curses.color_pair(1)  # 绿色
            status_text = "运行中"
        else:
            status_color = curses.color_pair(2)  # 黄色
            status_text = "未运行"

        self.screen.addstr(y+1, x, f"状态: {status_text}", status_color)
        self.screen.addstr(y+2, x, f"训练步骤: {streaming.get('current_step', 0)}")
        self.screen.addstr(y+3, x, f"最佳损失: {streaming.get('best_loss', 'N/A')}")
        self.screen.addstr(y+4, x, f"内存使用: {streaming.get('memory_usage_mb', 0):.1f}MB")
        self.screen.addstr(y+5, x, f"CPU使用: {streaming.get('cpu_usage_percent', 0):.1f}%")
        self.screen.addstr(y+6, x, f"流式效率: {streaming.get('streaming_efficiency', 0):.2f}")

    def _draw_system_health(self, y: int, x: int):
        """绘制系统健康状态"""
        system = self.status_data.get('system', {})

        self.screen.addstr(y, x, "系统健康状态:", curses.color_pair(4) | curses.A_BOLD)

        overall_health = system.get('system_health', {}).get('overall_health', 'unknown')

        if overall_health == 'healthy':
            health_color = curses.color_pair(1)
        elif overall_health == 'warning':
            health_color = curses.color_pair(2)
        elif overall_health == 'critical':
            health_color = curses.color_pair(3)
        else:
            health_color = curses.color_pair(2)

        self.screen.addstr(y+1, x, f"整体健康: {overall_health}", health_color)
        self.screen.addstr(y+2, x, f"最近故障: {len(system.get('issues', []))}")

    def _draw_infrastructure_status(self, y: int, x: int):
        """绘制基础设施状态"""
        infra = self.status_data.get('infrastructure', {})

        self.screen.addstr(y, x, "基础设施状态:", curses.color_pair(4) | curses.A_BOLD)
        self.screen.addstr(y+1, x, f"状态: {infra.get('status', 'unknown')}")

        components = infra.get('components', {})
        if components:
            self.screen.addstr(y+2, x, f"训练运行: {components.get('training', {}).get('running', False)}")
            self.screen.addstr(y+3, x, f"检查点数量: {len(components.get('checkpoint', []))}")

    def _draw_controls(self, y: int, x: int):
        """绘制操作控制"""
        self.screen.addstr(y, x, "控制: Q-退出 | R-刷新 | C-清屏", curses.color_pair(4))

    def _draw_status_bar(self, y: int, x: int):
        """绘制状态栏"""
        height, width = self.screen.getmaxyx()
        timestamp = datetime.now().strftime("%H:%M:%S")

        status_text = f"最后更新: {timestamp} | 按Q退出监控"
        if len(status_text) < width:
            self.screen.addstr(y, width - len(status_text) - 1, status_text, curses.color_pair(4))

    def _force_refresh(self):
        """强制刷新"""
        self._update_status_data()

    def _clear_logs(self):
        """清屏"""
        if self.screen:
            self.screen.clear()

def create_monitoring_dashboard():
    """创建监控仪表板"""
    print("启动H2Q-Evo AGI健康监控窗口...")
    print("按Q退出监控，R刷新数据，C清屏")
    print("=" * 50)

    monitor = HealthMonitorWindow()
    monitor.start_monitoring()

    print("\n监控窗口已关闭")

def get_monitoring_data() -> Dict[str, Any]:
    """获取监控数据"""
    monitor = HealthMonitorWindow()
    monitor._update_status_data()
    return monitor.status_data

if __name__ == "__main__":
    create_monitoring_dashboard()