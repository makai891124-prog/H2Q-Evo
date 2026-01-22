"""H2Q AGI 生存守护进程 (Survival Daemon).

实现核心功能:
1. 进程监控与自动重启 - 防止假死
2. 心跳机制 - 定时反馈能力认证
3. 资源监控 - 内存/CPU 使用率
4. 自我恢复 - 检测异常并恢复

安全设计:
- 所有操作本地化
- 资源使用限制
- 优雅退出机制
"""

import os
import sys
import time
import json
import signal
import threading
import traceback
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import hashlib


class ProcessState(Enum):
    """进程状态."""
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    LEARNING = "learning"
    SUSPENDED = "suspended"
    RECOVERING = "recovering"
    STOPPED = "stopped"
    DEAD = "dead"


@dataclass
class HeartbeatRecord:
    """心跳记录."""
    timestamp: str
    state: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    tasks_completed: int = 0
    errors_count: int = 0
    capability_score: float = 0.0
    message: str = ""


@dataclass
class SurvivalConfig:
    """生存配置."""
    heartbeat_interval: int = 30          # 心跳间隔 (秒)
    max_no_heartbeat: int = 120           # 最大无心跳时间 (秒)
    max_restart_attempts: int = 5         # 最大重启尝试次数
    restart_cooldown: int = 60            # 重启冷却时间 (秒)
    memory_limit_mb: float = 2048         # 内存限制 (MB)
    capability_check_interval: int = 3600 # 能力检查间隔 (秒)
    state_file: str = "agi_survival_state.json"
    heartbeat_file: str = "agi_heartbeat.json"
    log_file: str = "agi_survival.log"


class SurvivalDaemon:
    """AGI 生存守护进程."""
    
    def __init__(self, config: SurvivalConfig = None, work_dir: str = None):
        self.config = config or SurvivalConfig()
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        
        # 状态
        self.state = ProcessState.STARTING
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.restart_count = 0
        self.last_restart = None
        self.tasks_completed = 0
        self.errors_count = 0
        self.capability_score = 0.0
        
        # 线程
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # 强制停止标志 - 人工干预时使用
        self._force_stop = False
        self._force_stop_file = self.work_dir / "FORCE_STOP"
        
        # 回调
        self._on_restart: Optional[Callable] = None
        self._on_capability_check: Optional[Callable] = None
        self._target_process: Optional[Callable] = None
        
        # 信号处理
        self._setup_signals()
        
        # 日志
        self._log_buffer: List[str] = []
    
    def _setup_signals(self):
        """设置信号处理."""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except:
            pass  # Windows 兼容
    
    def _signal_handler(self, signum, frame):
        """信号处理器."""
        self.log(f"收到信号 {signum}，准备退出...")
        self.stop()
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        
        self._log_buffer.append(log_line)
        print(log_line)
        
        # 写入文件
        try:
            log_path = self.work_dir / self.config.log_file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_line + "\n")
        except:
            pass
        
        # 限制缓冲区大小
        if len(self._log_buffer) > 1000:
            self._log_buffer = self._log_buffer[-500:]
    
    def get_memory_usage(self) -> float:
        """获取内存使用量 (MB)."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024 / 1024  # 转换为 MB
        except:
            try:
                # 备选方案
                import psutil
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            except:
                return 0.0
    
    def get_cpu_percent(self) -> float:
        """获取 CPU 使用率."""
        try:
            import psutil
            return psutil.Process(os.getpid()).cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def send_heartbeat(self) -> HeartbeatRecord:
        """发送心跳."""
        with self._lock:
            self.last_heartbeat = datetime.now()
            
            record = HeartbeatRecord(
                timestamp=self.last_heartbeat.isoformat(),
                state=self.state.value,
                cpu_percent=self.get_cpu_percent(),
                memory_mb=self.get_memory_usage(),
                tasks_completed=self.tasks_completed,
                errors_count=self.errors_count,
                capability_score=self.capability_score,
                message=f"运行时间: {self.get_uptime()}"
            )
            
            # 保存到文件
            try:
                hb_path = self.work_dir / self.config.heartbeat_file
                with open(hb_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(record), f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.log(f"心跳保存失败: {e}", "ERROR")
            
            return record
    
    def get_uptime(self) -> str:
        """获取运行时间."""
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _heartbeat_loop(self):
        """心跳循环."""
        while self._running:
            try:
                record = self.send_heartbeat()
                self.log(f"💓 心跳: {record.state}, 内存: {record.memory_mb:.1f}MB, 任务: {record.tasks_completed}")
                
                # 检查内存限制
                if record.memory_mb > self.config.memory_limit_mb:
                    self.log(f"⚠️ 内存超限: {record.memory_mb:.1f}MB > {self.config.memory_limit_mb}MB", "WARNING")
                    self._trigger_gc()
                
            except Exception as e:
                self.log(f"心跳错误: {e}", "ERROR")
                self.errors_count += 1
            
            time.sleep(self.config.heartbeat_interval)
    
    def _monitor_loop(self):
        """监控循环."""
        last_capability_check = datetime.now()
        
        while self._running:
            try:
                # 首先检查强制停止标志
                if self.check_force_stop():
                    self.log("🛑 监控循环检测到强制停止，退出监控", "WARNING")
                    self._running = False
                    self.state = ProcessState.STOPPED
                    break
                
                # 检查心跳超时
                elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
                
                if elapsed > self.config.max_no_heartbeat:
                    self.log(f"⚠️ 心跳超时: {elapsed:.0f}秒", "WARNING")
                    # 再次检查强制停止（防止在心跳超时后仍尝试恢复）
                    if not self.check_force_stop():
                        self._attempt_recovery()
                
                # 定期能力检查
                if (datetime.now() - last_capability_check).total_seconds() > self.config.capability_check_interval:
                    self._perform_capability_check()
                    last_capability_check = datetime.now()
                
            except Exception as e:
                self.log(f"监控错误: {e}", "ERROR")
            
            time.sleep(10)  # 监控间隔
    
    def _trigger_gc(self):
        """触发垃圾回收."""
        import gc
        gc.collect()
        self.log("🗑️ 触发垃圾回收")
    
    def _attempt_recovery(self):
        """尝试恢复."""
        with self._lock:
            # 检查是否被强制停止（人工干预）
            if self._force_stop:
                self.log("🛑 检测到强制停止标志，跳过自动恢复", "WARNING")
                self.state = ProcessState.STOPPED
                return
            
            if self.restart_count >= self.config.max_restart_attempts:
                self.log("❌ 达到最大重启次数，停止恢复", "ERROR")
                self.state = ProcessState.DEAD
                return
            
            # 检查冷却时间
            if self.last_restart:
                cooldown = (datetime.now() - self.last_restart).total_seconds()
                if cooldown < self.config.restart_cooldown:
                    self.log(f"⏳ 等待冷却: {self.config.restart_cooldown - cooldown:.0f}秒")
                    return
            
            self.state = ProcessState.RECOVERING
            self.restart_count += 1
            self.last_restart = datetime.now()
            
            self.log(f"🔄 尝试恢复 ({self.restart_count}/{self.config.max_restart_attempts})")
            
            # 调用重启回调
            if self._on_restart:
                try:
                    self._on_restart()
                    self.state = ProcessState.RUNNING
                    self.log("✅ 恢复成功")
                except Exception as e:
                    self.log(f"恢复失败: {e}", "ERROR")
                    self.state = ProcessState.DEAD
            else:
                # 默认恢复：重置状态
                self._trigger_gc()
                self.state = ProcessState.RUNNING
                self.log("✅ 默认恢复完成")
    
    def _perform_capability_check(self):
        """执行能力检查."""
        self.log("🧪 执行能力认证...")
        
        if self._on_capability_check:
            try:
                score = self._on_capability_check()
                self.capability_score = float(score)
                self.log(f"📊 能力评分: {self.capability_score:.1f}%")
            except Exception as e:
                self.log(f"能力检查失败: {e}", "ERROR")
        else:
            # 默认能力检查
            self.capability_score = self._default_capability_check()
            self.log(f"📊 能力评分 (默认): {self.capability_score:.1f}%")
    
    def _default_capability_check(self) -> float:
        """默认能力检查."""
        score = 0.0
        
        # 检查系统响应性
        start = time.time()
        _ = sum(range(10000))
        latency = time.time() - start
        if latency < 0.01:
            score += 25
        elif latency < 0.1:
            score += 15
        
        # 检查内存状态
        mem = self.get_memory_usage()
        if mem < self.config.memory_limit_mb * 0.5:
            score += 25
        elif mem < self.config.memory_limit_mb * 0.8:
            score += 15
        
        # 检查错误率
        if self.tasks_completed > 0:
            error_rate = self.errors_count / self.tasks_completed
            if error_rate < 0.01:
                score += 25
            elif error_rate < 0.1:
                score += 15
        else:
            score += 25
        
        # 检查运行稳定性
        if self.restart_count == 0:
            score += 25
        elif self.restart_count < 3:
            score += 15
        
        return score
    
    def save_state(self):
        """保存状态."""
        state_data = {
            "state": self.state.value,
            "start_time": self.start_time.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "restart_count": self.restart_count,
            "tasks_completed": self.tasks_completed,
            "errors_count": self.errors_count,
            "capability_score": self.capability_score,
            "uptime": self.get_uptime(),
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            state_path = self.work_dir / self.config.state_file
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"状态保存失败: {e}", "ERROR")
    
    def load_state(self) -> bool:
        """加载状态."""
        try:
            state_path = self.work_dir / self.config.state_file
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.restart_count = data.get("restart_count", 0)
                self.tasks_completed = data.get("tasks_completed", 0)
                self.errors_count = data.get("errors_count", 0)
                self.capability_score = data.get("capability_score", 0.0)
                
                self.log(f"📂 加载状态: 任务={self.tasks_completed}, 重启={self.restart_count}")
                return True
        except Exception as e:
            self.log(f"状态加载失败: {e}", "WARNING")
        return False
    
    def set_restart_callback(self, callback: Callable):
        """设置重启回调."""
        self._on_restart = callback
    
    def set_capability_callback(self, callback: Callable[[], float]):
        """设置能力检查回调."""
        self._on_capability_check = callback
    
    def report_task_complete(self):
        """报告任务完成."""
        with self._lock:
            self.tasks_completed += 1
    
    def report_error(self):
        """报告错误."""
        with self._lock:
            self.errors_count += 1
    
    def start(self):
        """启动守护进程."""
        self.log("🚀 启动生存守护进程...")
        
        # 加载之前的状态
        self.load_state()
        
        self._running = True
        self.state = ProcessState.RUNNING
        
        # 启动心跳线程
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        
        # 启动监控线程
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.log("✅ 守护进程已启动")
        self.send_heartbeat()
    
    def stop(self):
        """停止守护进程."""
        self.log("🛑 停止生存守护进程...")
        
        self._running = False
        self.state = ProcessState.STOPPED
        
        # 保存状态
        self.save_state()
        
        # 等待线程结束
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.log("✅ 守护进程已停止")
    
    def force_stop(self):
        """强制停止 - 完全阻止自动重启（人工干预使用）."""
        self.log("🛑 执行强制停止，禁用自动重启...", "WARNING")
        
        # 设置强制停止标志
        self._force_stop = True
        
        # 创建强制停止文件作为持久标记
        try:
            with open(self._force_stop_file, 'w') as f:
                f.write(f"FORCE_STOP at {datetime.now().isoformat()}\n")
                f.write("删除此文件以允许系统重新启动自动恢复功能\n")
        except Exception as e:
            self.log(f"创建强制停止文件失败: {e}", "ERROR")
        
        # 停止守护进程
        self.stop()
        self.state = ProcessState.DEAD  # 标记为死亡，防止任何恢复尝试
        
        self.log("✅ 强制停止完成，系统已完全停止")
        self.log("   提示: 删除 FORCE_STOP 文件以恢复自动重启功能")
    
    def check_force_stop(self) -> bool:
        """检查是否存在强制停止标志."""
        # 检查内存标志
        if self._force_stop:
            return True
        # 检查文件标志
        if self._force_stop_file.exists():
            self._force_stop = True
            self.log("检测到 FORCE_STOP 文件，已禁用自动重启", "WARNING")
            return True
        return False
    
    def clear_force_stop(self):
        """清除强制停止标志（允许恢复自动重启）."""
        self._force_stop = False
        if self._force_stop_file.exists():
            try:
                self._force_stop_file.unlink()
                self.log("已清除强制停止标志，自动重启已恢复")
            except Exception as e:
                self.log(f"清除强制停止文件失败: {e}", "ERROR")
    
    def run_with_protection(self, target: Callable, *args, **kwargs):
        """在保护下运行目标函数."""
        self._target_process = target
        
        try:
            self.start()
            self.state = ProcessState.LEARNING
            
            result = target(*args, **kwargs)
            
            self.report_task_complete()
            return result
            
        except Exception as e:
            self.log(f"目标进程错误: {e}", "ERROR")
            self.report_error()
            traceback.print_exc()
            raise
        finally:
            self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态摘要."""
        return {
            "state": self.state.value,
            "uptime": self.get_uptime(),
            "start_time": self.start_time.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "restart_count": self.restart_count,
            "tasks_completed": self.tasks_completed,
            "errors_count": self.errors_count,
            "capability_score": self.capability_score,
            "memory_mb": self.get_memory_usage(),
            "is_healthy": self.state in [ProcessState.RUNNING, ProcessState.LEARNING, ProcessState.IDLE]
        }


# 工厂函数
def create_survival_daemon(work_dir: str = None, 
                           config: SurvivalConfig = None) -> SurvivalDaemon:
    """创建生存守护进程."""
    return SurvivalDaemon(config, work_dir)


if __name__ == "__main__":
    # 演示
    daemon = create_survival_daemon()
    
    def demo_task():
        print("执行演示任务...")
        time.sleep(5)
        return "完成"
    
    result = daemon.run_with_protection(demo_task)
    print(f"结果: {result}")
