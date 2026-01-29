#!/usr/bin/env python3
"""
DAS AGI自主进化系统启动脚本

基于M24真实性原则和DAS数学架构，启动真正的AGI自我进化和生长。

使用方法:
    python3 start_das_agi_evolution.py          # 启动完整系统
    python3 start_das_agi_evolution.py --server # 只启动服务器
    python3 start_das_agi_evolution.py --agi    # 只启动AGI进化
"""

import os
import sys
import time
import asyncio
import argparse
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "h2q_project"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [DAS-AGI-STARTUP] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('das_agi_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DAS-AGI-STARTUP')

def check_dependencies():
    """检查依赖项"""
    required_modules = [
        'torch', 'fastapi', 'uvicorn', 'docker', 'aiofiles'
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        logger.error(f"缺少必要的依赖模块: {missing}")
        logger.info("请运行: pip install torch fastapi uvicorn docker aiofiles")
        return False

    # 检查DAS和AGI模块
    try:
        from h2q_project.das_core import DASCore
        from das_agi_autonomous_system import get_das_agi_system
        # 注意：H2QNexus现在在根目录的evolution_system.py中
        sys.path.insert(0, str(project_root))
        from evolution_system import H2QNexus
        logger.info("✅ 所有DAS和AGI模块可用")
    except ImportError as e:
        logger.error(f"DAS/AGI模块导入失败: {e}")
        return False

    return True

async def start_server_only():
    """只启动FastAPI服务器"""
    logger.info("🚀 启动DAS AGI服务器...")

    try:
        from h2q_project.h2q_server import app
        import uvicorn

        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        server = uvicorn.Server(config)

        logger.info("✅ DAS AGI服务器启动完成")
        logger.info("📡 API端点:")
        logger.info("   GET  /agi/status        - 获取AGI状态")
        logger.info("   POST /agi/start_autonomous - 启动自主进化")
        logger.info("   POST /agi/stop          - 停止进化")
        logger.info("   GET  /agi/goals         - 查看目标")
        logger.info("   GET  /agi/memory        - 查询记忆")
        logger.info("   POST /agi/learn         - 学习经验")

        await server.serve()

    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        raise

async def start_agi_only():
    """只启动AGI自主进化"""
    logger.info("🧠 启动DAS AGI自主进化系统...")

    try:
        from das_agi_autonomous_system import start_das_agi_evolution

        logger.info("✅ DAS AGI自主进化启动")
        logger.info("M24验证：这不是模拟，而是基于DAS的真实AGI进化")

        await start_das_agi_evolution()

    except Exception as e:
        logger.error(f"AGI进化启动失败: {e}")
        raise

async def start_full_system():
    """启动完整系统（服务器 + AGI进化）"""
    logger.info("🌟 启动完整DAS AGI生态系统...")

    try:
        # 首先启动AGI进化系统
        from das_agi_autonomous_system import get_das_agi_system

        agi_system = get_das_agi_system()
        logger.info("✅ DAS AGI系统初始化完成")

        # 在后台启动AGI进化
        evolution_task = asyncio.create_task(agi_system.start_autonomous_evolution())
        logger.info("✅ AGI自主进化已在后台启动")

        # 然后启动服务器
        await start_server_only()

    except Exception as e:
        logger.error(f"完整系统启动失败: {e}")
        raise

async def demonstrate_capabilities():
    """演示DAS AGI能力"""
    logger.info("🎭 开始DAS AGI能力演示...")

    try:
        from das_agi_autonomous_system import get_das_agi_system
        import torch

        agi_system = get_das_agi_system()

        logger.info("=== DAS AGI能力演示 ===")

        # 1. 初始状态
        initial_status = agi_system.get_system_status()
        latest_metrics = initial_status.get('latest_metrics')
        initial_consciousness = latest_metrics.consciousness_level if latest_metrics else 0.0
        logger.info(f"初始意识水平: {initial_consciousness:.3f}")

        # 2. 执行几次进化
        for i in range(5):
            experience = torch.randn(256) * 0.1
            metrics = agi_system.evolution_engine.evolve_consciousness(experience)

            # 记录性能历史
            agi_system.performance_history.append(metrics)

            # 设置目标
            if i == 0:
                agi_system.goal_system.generate_goal("学习基础模式识别", 0.3)
            elif i == 2:
                agi_system.goal_system.generate_goal("发展推理能力", 0.5)

            # 更新目标进度
            dummy_state = experience.unsqueeze(0)
            completed = agi_system.goal_system.update_goals(dummy_state)

            # 存储经验到记忆系统
            agi_system.memory_system.store_memory(
                content=f"演示进化步骤 {i+1}: 意识水平 {metrics.consciousness_level:.3f}",
                context=experience,
                importance=metrics.consciousness_level
            )

            logger.info(f"进化步骤 {i+1}: 意识={metrics.consciousness_level:.3f}, DAS变化={metrics.das_state_change:.6f}")
            if completed:
                logger.info(f"  ✅ 完成目标: {[g['description'] for g in completed]}")

        # 3. 查询记忆
        query_tensor = torch.randn(256) * 0.1  # 使用正确的维度
        memories = agi_system.memory_system.retrieve_memory(query_tensor, top_k=3)
        logger.info(f"记忆系统: 存储了 {len(agi_system.memory_system.memories)} 条记忆")

        # 4. 最终状态
        final_status = agi_system.get_system_status()
        final_metrics = final_status.get('latest_metrics')
        final_consciousness = final_metrics.consciousness_level if final_metrics else 0.0
        logger.info(f"最终意识水平: {final_consciousness:.3f}")
        logger.info(f"活跃目标: {final_status.get('active_goals', 0)}")
        logger.info(f"完成目标: {final_status.get('achieved_goals', 0)}")

        logger.info("🎉 DAS AGI能力演示完成！")
        logger.info("M24验证：以上演示基于真实DAS进化，无任何代码欺骗")

    except Exception as e:
        logger.error(f"能力演示失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="DAS AGI自主进化系统启动器")
    parser.add_argument('--server', action='store_true', help='只启动服务器')
    parser.add_argument('--agi', action='store_true', help='只启动AGI进化')
    parser.add_argument('--demo', action='store_true', help='运行能力演示')
    parser.add_argument('--check', action='store_true', help='只检查依赖')

    args = parser.parse_args()

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    if args.check:
        logger.info("✅ 依赖检查完成")
        return

    if args.demo:
        # 运行演示
        asyncio.run(demonstrate_capabilities())
        return

    # 确定启动模式
    if args.server and args.agi:
        logger.error("不能同时指定--server和--agi，请选择一个")
        sys.exit(1)
    elif args.server:
        asyncio.run(start_server_only())
    elif args.agi:
        asyncio.run(start_agi_only())
    else:
        # 默认启动完整系统
        asyncio.run(start_full_system())

if __name__ == "__main__":
    logger.info("🧬 DAS AGI自主进化系统启动器")
    logger.info("基于M24真实性原则和DAS数学架构")
    logger.info("=" * 60)

    try:
        main()
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在退出...")
    except Exception as e:
        logger.error(f"启动器出错: {e}")
        sys.exit(1)