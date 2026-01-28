#!/usr/bin/env python3
"""
启动真正的AGI进化系统
"""

import asyncio
import sys
import signal
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from true_agi_autonomous_system import TrueAGIAutonomousSystem

async def main():
    """主函数：启动AGI进化"""
    print("🚀 启动真正的AGI自主进化系统...")
    print("📚 系统将学习提供的学习资料并进行自我进化")

    # 直接创建新系统实例，避免单例缓存问题
    system = TrueAGIAutonomousSystem(input_dim=256, action_dim=256)

    # 设置信号处理
    def signal_handler(signum, frame):
        print(f"\n🛑 收到信号 {signum}，正在停止AGI进化...")
        print("💾 正在保存AGI系统状态...")
        system.save_state("true_agi_system_state.json")
        system.stop_evolution()
        import sys
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        await system.start_true_evolution()
    except KeyboardInterrupt:
        print("\n🛑 AGI进化已停止")
        system.save_state("true_agi_system_state.json")
        system.stop_evolution()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💾 正在保存AGI系统状态...")
        system.save_state("true_agi_system_state.json")
        system.stop_evolution()
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 AGI进化已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)