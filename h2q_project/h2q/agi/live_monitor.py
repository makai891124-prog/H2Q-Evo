#!/usr/bin/env python3
"""
AGI训练实时终端监控 - 美化版
Rich Terminal Live Monitor for AGI Training

使用rich库在终端实时显示训练进度（可选）
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.text import Text
    from rich import box
except ImportError:
    print("正在安装 rich 库...")
    os.system("pip3 install rich -q")
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.text import Text
    from rich import box

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / 'optimized_training.log'
CHECKPOINT_DIR = SCRIPT_DIR / 'optimized_checkpoints'

console = Console()


def get_process_info():
    """获取进程信息"""
    result = subprocess.run(
        ['pgrep', '-f', 'optimized_5h_training.py'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        pid = result.stdout.strip().split('\n')[0]
        ps = subprocess.run(
            ['ps', '-p', pid, '-o', 'etime=,pcpu=,pmem='],
            capture_output=True, text=True
        )
        if ps.returncode == 0:
            parts = ps.stdout.strip().split()
            return {
                'running': True,
                'pid': pid,
                'elapsed': parts[0] if len(parts) > 0 else '-',
                'cpu': parts[1] if len(parts) > 1 else '-',
                'mem': parts[2] if len(parts) > 2 else '-'
            }
    return {'running': False, 'pid': '-', 'elapsed': '-', 'cpu': '-', 'mem': '-'}


def parse_training_log():
    """解析训练日志获取最新状态"""
    if not LOG_FILE.exists():
        return None
    
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    result = {
        'epoch': 0,
        'train_loss': 0.0,
        'train_acc': 0.0,
        'val_acc': 0.0,
        'best_acc': 0.0,
        'speed': 0,
        'progress': 0.0,
        'eta': '--:--:--',
        'recent_logs': []
    }
    
    # 获取最近的日志
    for line in lines[-20:]:
        stripped = line.strip()
        if stripped and ('Batch' in stripped or 'Epoch' in stripped or '验证' in stripped):
            result['recent_logs'].append(stripped[-80:])  # 截取最后80字符
    
    result['recent_logs'] = result['recent_logs'][-8:]  # 保留最近8行
    
    # 从后向前解析找关键信息
    for line in reversed(lines):
        if 'Epoch' in line and '完成' in line:
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'Epoch' and i + 1 < len(parts):
                        result['epoch'] = int(parts[i + 1])
                        break
            except:
                pass
        
        if '训练 Loss:' in line:
            try:
                parts = line.split('|')
                for p in parts:
                    if 'Loss:' in p:
                        result['train_loss'] = float(p.split(':')[1].strip())
                    if 'Acc:' in p and '训练' in p:
                        result['train_acc'] = float(p.split(':')[1].strip().replace('%', ''))
            except:
                pass
        
        if '验证 Acc:' in line:
            try:
                parts = line.split('|')
                for p in parts:
                    if '验证 Acc:' in p:
                        result['val_acc'] = float(p.split(':')[1].strip().replace('%', ''))
                    if '最佳:' in p:
                        result['best_acc'] = float(p.split(':')[1].strip().replace('%', ''))
            except:
                pass
        
        if '速度:' in line:
            try:
                parts = line.split('|')
                for p in parts:
                    if '速度:' in p:
                        speed_str = p.split(':')[1].strip().split()[0]
                        result['speed'] = int(float(speed_str))
            except:
                pass
        
        if '进度:' in line:
            try:
                parts = line.split('|')
                for p in parts:
                    if '进度:' in p:
                        result['progress'] = float(p.split(':')[1].strip().replace('%', ''))
            except:
                pass
        
        if '预计完成:' in line:
            try:
                result['eta'] = line.split('预计完成:')[1].strip()
            except:
                pass
        
        # 找到足够信息后退出
        if result['epoch'] > 0 and result['train_loss'] > 0:
            break
    
    return result


def create_status_display():
    """创建状态显示"""
    layout = Layout()
    
    # 获取数据
    proc = get_process_info()
    train = parse_training_log() or {
        'epoch': 0, 'train_loss': 0, 'train_acc': 0, 
        'val_acc': 0, 'best_acc': 0, 'speed': 0, 
        'progress': 0, 'eta': '--:--:--', 'recent_logs': []
    }
    
    # 标题
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status_emoji = "🟢" if proc['running'] else "🔴"
    status_text = "训练运行中" if proc['running'] else "训练已停止"
    
    title = Text()
    title.append("🤖 AGI训练监控面板\n", style="bold cyan")
    title.append(f"{status_emoji} {status_text}", style="bold green" if proc['running'] else "bold red")
    title.append(f"    📅 {now}", style="dim")
    
    # 进程信息表
    proc_table = Table(box=box.ROUNDED, expand=True)
    proc_table.add_column("项目", style="cyan")
    proc_table.add_column("值", style="green")
    proc_table.add_row("进程 PID", str(proc['pid']))
    proc_table.add_row("运行时间", proc['elapsed'])
    proc_table.add_row("CPU 占用", f"{proc['cpu']}%")
    proc_table.add_row("内存占用", f"{proc['mem']}%")
    
    # 训练指标表
    train_table = Table(box=box.ROUNDED, expand=True)
    train_table.add_column("指标", style="cyan")
    train_table.add_column("值", style="yellow")
    train_table.add_row("当前 Epoch", str(train['epoch']))
    train_table.add_row("训练 Loss", f"{train['train_loss']:.4f}")
    train_table.add_row("训练准确率", f"{train['train_acc']:.2f}%")
    train_table.add_row("验证准确率", f"[bold green]{train['val_acc']:.2f}%[/]")
    train_table.add_row("最佳准确率", f"[bold magenta]{train['best_acc']:.2f}%[/]")
    train_table.add_row("处理速度", f"{train['speed']} samples/s")
    
    # 进度条
    progress = train['progress']
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    progress_text = Text()
    progress_text.append("训练进度: ", style="bold")
    progress_text.append(f"[{bar}] ", style="cyan")
    progress_text.append(f"{progress:.1f}%\n", style="bold green")
    progress_text.append(f"预计完成时间: {train['eta']}", style="dim")
    
    # 最近日志
    log_text = Text()
    log_text.append("📝 实时日志:\n", style="bold cyan")
    for log in train['recent_logs']:
        log_text.append(f"  {log}\n", style="dim green")
    
    # 组合显示
    output = Table.grid(expand=True)
    output.add_row(Panel(title, box=box.DOUBLE))
    
    info_table = Table.grid(expand=True)
    info_table.add_column(ratio=1)
    info_table.add_column(ratio=1)
    info_table.add_row(
        Panel(proc_table, title="⚙️ 进程信息"),
        Panel(train_table, title="📊 训练指标")
    )
    output.add_row(info_table)
    
    output.add_row(Panel(progress_text, title="📈 进度"))
    output.add_row(Panel(log_text, title="📋 日志", height=12))
    
    return Panel(output, box=box.HEAVY, border_style="blue")


def main():
    """主函数 - 实时监控"""
    console.clear()
    
    print("\n" + "=" * 60)
    print("   🤖 AGI训练终端监控 (按 Ctrl+C 退出)")
    print("=" * 60 + "\n")
    
    try:
        with Live(create_status_display(), refresh_per_second=0.5, console=console) as live:
            while True:
                time.sleep(2)
                live.update(create_status_display())
    except KeyboardInterrupt:
        console.print("\n[yellow]监控已停止[/yellow]\n")


if __name__ == "__main__":
    main()
