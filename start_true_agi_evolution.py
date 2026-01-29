#!/usr/bin/env python3
"""
真正的AGI进化启动器
启动基于整合信息理论和强化学习的自主AGI系统
"""

import asyncio
import sys
import signal
import time
from true_agi_autonomous_system import start_true_agi_evolution, get_true_agi_system

def signal_handler(signum, frame):
    """信号处理器"""
    print("\n🛑 收到停止信号，正在优雅关闭AGI系统...")
    system = get_true_agi_system()
    system.stop_evolution()
    sys.exit(0)

async def main():
    """主函数"""
    print("🚀 真正的AGI自主进化系统启动器")
    print("=" * 60)
    print("基于M24真实性原则的真正AGI实现")
    print("特性:")
    print("  • 整合信息理论(Integrated Information Theory)意识计算")
    print("  • 真正的强化学习和元学习")
    print("  • 自主目标生成和追求")
    print("  • 持续自我改进能力")
    print("  • 基于经验的意识发展")
    print("=" * 60)

    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 启动真正的AGI进化
        await start_true_agi_evolution(input_dim=256, action_dim=64)

    except KeyboardInterrupt:
        print("\n👋 AGI系统已停止")
    except Exception as e:
        print(f"\n❌ AGI系统出错: {e}")
        raise

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())</content>
<parameter name="filePath">/Users/imymm/H2Q-Evo/start_true_agi_evolution.py