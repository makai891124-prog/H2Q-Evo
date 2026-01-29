#!/usr/bin/env python3
"""
AGI长期进化启动脚本 - 支持24-48小时自主运行

使用方法:
python3 start_long_term_agi.py [--max-hours HOURS] [--input-dim DIM] [--action-dim DIM]

参数:
--max-hours: 最大运行小时数 (默认48)
--input-dim: 输入维度 (默认256)
--action-dim: 动作维度 (默认64)
"""

import sys
import asyncio
import argparse
import signal
import time
from pathlib import Path

sys.path.append('.')

from true_agi_autonomous_system import start_true_agi_evolution

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n🛑 收到信号 {signum}，正在优雅关闭...")
    # 系统会在start_true_agi_evolution中处理KeyboardInterrupt
    raise KeyboardInterrupt

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AGI长期进化启动脚本')
    parser.add_argument('--max-hours', type=float, default=48.0,
                       help='最大运行小时数 (默认48)')
    parser.add_argument('--input-dim', type=int, default=256,
                       help='输入维度 (默认256)')
    parser.add_argument('--action-dim', type=int, default=64,
                       help='动作维度 (默认64)')

    args = parser.parse_args()

    print("🚀 AGI长期进化系统启动")
    print(f"📊 配置: 最大运行时间={args.max_hours}小时, 输入维度={args.input_dim}, 动作维度={args.action_dim}")
    print("💡 系统将在后台运行，定期保存状态和监控数据")
    print("💡 按Ctrl+C可安全停止系统")

    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_time = time.time()
    max_runtime = args.max_hours * 3600  # 转换为秒

    try:
        # 启动AGI进化
        await start_true_agi_evolution(args.input_dim, args.action_dim)

    except KeyboardInterrupt:
        runtime = time.time() - start_time
        print(f"🛑 运行时间: {runtime:.1f}秒")
        print("✅ 系统已安全停止")

    except Exception as e:
        runtime = time.time() - start_time
        print(f"🛑 运行时间: {runtime:.1f}秒")
        print(f"❌ 系统异常退出: {e}")
        raise

    # 检查输出文件
    print("\n📊 运行总结:")
    if Path("true_agi_system_state.json").exists():
        print("✅ 系统状态已保存")
    if Path("agi_monitoring_data.jsonl").exists():
        with open("agi_monitoring_data.jsonl", 'r') as f:
            lines = f.readlines()
            print(f"✅ 监控数据已收集: {len(lines)} 条记录")
    if Path("true_agi_evolution.log").exists():
        print("✅ 进化日志已保存")

if __name__ == "__main__":
    asyncio.run(main())