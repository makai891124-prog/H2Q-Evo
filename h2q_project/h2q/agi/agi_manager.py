#!/usr/bin/env python3
"""
AGI服务统一管理器
Unified AGI Service Manager

提供训练、监控、查询的统一入口
像一个持续进化的AI服务一样运行在你身边
"""

import os
import sys
import subprocess
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# 服务配置
SERVICES = {
    'train': {
        'name': 'AGI训练服务',
        'script': SCRIPT_DIR / 'optimized_5h_training.py',
        'log': SCRIPT_DIR / 'optimized_training.log',
        'pidfile': SCRIPT_DIR / '.train.pid',
        'desc': '核心训练进程，后台运行模型训练'
    },
    'web': {
        'name': 'Web监控面板',
        'script': SCRIPT_DIR / 'web_monitor.py',
        'port': 5000,
        'pidfile': SCRIPT_DIR / '.web.pid',
        'desc': '浏览器图形化监控界面'
    },
    'terminal': {
        'name': '终端监控',
        'script': SCRIPT_DIR / 'live_monitor.py',
        'desc': '终端实时监控（前台运行）'
    }
}


def print_banner():
    """打印Banner"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██╗  ██╗██████╗  ██████╗       █████╗  ██████╗ ██╗             ║
║   ██║  ██║╚════██╗██╔═══██╗     ██╔══██╗██╔════╝ ██║             ║
║   ███████║ █████╔╝██║   ██║     ███████║██║  ███╗██║             ║
║   ██╔══██║██╔═══╝ ██║▄▄ ██║     ██╔══██║██║   ██║██║             ║
║   ██║  ██║███████╗╚██████╔╝     ██║  ██║╚██████╔╝██║             ║
║   ╚═╝  ╚═╝╚══════╝ ╚══▀▀═╝      ╚═╝  ╚═╝ ╚═════╝ ╚═╝             ║
║                                                                  ║
║           🤖 自主进化人工通用智能系统                              ║
║              Autonomous Evolving AGI System                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def get_service_status(service_name):
    """获取服务状态"""
    if service_name == 'train':
        result = subprocess.run(
            ['pgrep', '-f', 'optimized_5h_training.py'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pid = result.stdout.strip().split('\n')[0]
            return {'running': True, 'pid': pid}
    elif service_name == 'web':
        result = subprocess.run(
            ['pgrep', '-f', 'web_monitor.py'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pid = result.stdout.strip().split('\n')[0]
            return {'running': True, 'pid': pid}
    
    return {'running': False, 'pid': None}


def start_training(duration_hours=5):
    """启动训练服务"""
    status = get_service_status('train')
    if status['running']:
        print(f"⚠️  训练服务已在运行中 (PID: {status['pid']})")
        return
    
    print("🚀 启动AGI训练服务...")
    
    # 后台启动训练
    log_file = SERVICES['train']['log']
    cmd = f"cd {SCRIPT_DIR} && nohup python3 optimized_5h_training.py > /dev/null 2>&1 &"
    subprocess.run(cmd, shell=True)
    
    time.sleep(2)
    status = get_service_status('train')
    if status['running']:
        print(f"✅ 训练服务已启动 (PID: {status['pid']})")
        print(f"📝 日志文件: {log_file}")
    else:
        print("❌ 训练服务启动失败")


def start_web_monitor():
    """启动Web监控"""
    status = get_service_status('web')
    if status['running']:
        print(f"⚠️  Web监控已在运行中 (PID: {status['pid']})")
        print("🌐 访问: http://localhost:5000")
        return
    
    print("🌐 启动Web监控面板...")
    
    # 后台启动
    cmd = f"cd {SCRIPT_DIR} && nohup python3 web_monitor.py > /dev/null 2>&1 &"
    subprocess.run(cmd, shell=True)
    
    time.sleep(2)
    status = get_service_status('web')
    if status['running']:
        print(f"✅ Web监控已启动 (PID: {status['pid']})")
        print("🌐 请在浏览器中打开: http://localhost:5000")
    else:
        print("❌ Web监控启动失败")


def start_terminal_monitor():
    """启动终端监控"""
    print("📺 启动终端实时监控...")
    script = SERVICES['terminal']['script']
    subprocess.run(['python3', str(script)])


def stop_service(service_name):
    """停止服务"""
    status = get_service_status(service_name)
    if not status['running']:
        print(f"⚠️  {SERVICES[service_name]['name']} 未在运行")
        return
    
    pid = status['pid']
    print(f"🛑 停止 {SERVICES[service_name]['name']} (PID: {pid})...")
    
    subprocess.run(['kill', pid])
    time.sleep(1)
    
    if not get_service_status(service_name)['running']:
        print(f"✅ {SERVICES[service_name]['name']} 已停止")
    else:
        subprocess.run(['kill', '-9', pid])
        print(f"✅ {SERVICES[service_name]['name']} 已强制停止")


def show_status():
    """显示所有服务状态"""
    print("\n📊 服务状态概览")
    print("=" * 60)
    
    for name, config in SERVICES.items():
        status = get_service_status(name)
        icon = "🟢" if status['running'] else "🔴"
        pid_info = f"PID: {status['pid']}" if status['running'] else "未运行"
        print(f"  {icon} {config['name']:15} | {pid_info:15} | {config['desc']}")
    
    print("=" * 60)
    
    # 训练进度快照
    log_file = SERVICES['train']['log']
    if log_file.exists():
        print("\n📈 训练进度快照")
        print("-" * 60)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # 找最近的epoch完成信息
        for line in reversed(lines):
            if 'Epoch' in line and '完成' in line:
                print(f"  {line.strip()}")
                break
            if '验证' in line and 'Acc' in line:
                print(f"  {line.strip()}")
            if '进度' in line:
                print(f"  {line.strip()}")
                break


def interactive_mode():
    """交互式模式"""
    print_banner()
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    while True:
        show_status()
        print("\n🎮 可用命令:")
        print("  [1] 启动训练      [2] 启动Web监控    [3] 终端监控")
        print("  [4] 停止训练      [5] 停止Web监控    [6] 查看日志")
        print("  [7] 刷新状态      [q] 退出")
        print()
        
        try:
            choice = input("请选择 > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见!")
            break
        
        if choice == '1':
            start_training()
        elif choice == '2':
            start_web_monitor()
        elif choice == '3':
            start_terminal_monitor()
        elif choice == '4':
            stop_service('train')
        elif choice == '5':
            stop_service('web')
        elif choice == '6':
            log_file = SERVICES['train']['log']
            if log_file.exists():
                subprocess.run(['tail', '-50', str(log_file)])
            else:
                print("❌ 日志文件不存在")
        elif choice == '7':
            continue
        elif choice == 'q':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择")
        
        input("\n按回车继续...")


def main():
    parser = argparse.ArgumentParser(
        description='H2Q AGI服务管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python agi_manager.py                    # 交互式模式
  python agi_manager.py --start-train      # 启动训练
  python agi_manager.py --start-web        # 启动Web监控
  python agi_manager.py --monitor          # 终端监控
  python agi_manager.py --status           # 查看状态
  python agi_manager.py --stop-all         # 停止所有服务
        """
    )
    
    parser.add_argument('--start-train', action='store_true', help='启动训练服务')
    parser.add_argument('--start-web', action='store_true', help='启动Web监控')
    parser.add_argument('--monitor', action='store_true', help='启动终端监控')
    parser.add_argument('--status', action='store_true', help='显示服务状态')
    parser.add_argument('--stop-train', action='store_true', help='停止训练')
    parser.add_argument('--stop-web', action='store_true', help='停止Web监控')
    parser.add_argument('--stop-all', action='store_true', help='停止所有服务')
    
    args = parser.parse_args()
    
    if args.start_train:
        start_training()
    elif args.start_web:
        start_web_monitor()
    elif args.monitor:
        start_terminal_monitor()
    elif args.status:
        print_banner()
        show_status()
    elif args.stop_train:
        stop_service('train')
    elif args.stop_web:
        stop_service('web')
    elif args.stop_all:
        stop_service('train')
        stop_service('web')
    else:
        # 默认进入交互式模式
        interactive_mode()


if __name__ == "__main__":
    main()
