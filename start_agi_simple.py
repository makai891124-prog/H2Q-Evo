#!/usr/bin/env python3
"""
H2Q-Evo 自动进化AGI系统 - 简化启动器
不依赖Docker的本地模式启动
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

# 设置环境变量
os.environ['INFERENCE_MODE'] = 'local'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agi_evolution_startup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AGI-Starter")

def check_environment():
    """检查运行环境"""
    logger.info("🔍 检查AGI系统运行环境...")

    # 检查Python版本
    python_version = sys.version_info
    logger.info(f"   Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 检查关键文件
    required_files = [
        'evolution_system.py',
        'h2q_project/h2q_server.py',
        'simple_agi_training.py'
    ]

    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"   ✅ {file_path}")
        else:
            logger.error(f"   ❌ 缺少文件: {file_path}")
            return False

    # 检查训练检查点
    if Path('checkpoints').exists():
        checkpoints = list(Path('checkpoints').glob('*.pth'))
        logger.info(f"   ✅ 发现 {len(checkpoints)} 个模型检查点")
    else:
        logger.warning("   ⚠️  未发现检查点目录")

    return True

def start_evolution_system():
    """启动进化系统"""
    logger.info("🚀 启动H2Q-Evo自动进化AGI系统...")

    try:
        # 导入并初始化系统
        from evolution_system import H2QNexus

        logger.info("   初始化H2Q-Evo系统...")
        nexus = H2QNexus()

        logger.info("   系统初始化完成")
        logger.info("   AGI自动进化循环已启动")
        logger.info("   按Ctrl+C停止系统")

        # 保持运行
        while True:
            time.sleep(10)
            logger.info("   AGI系统运行中... (心跳检测)")

    except KeyboardInterrupt:
        logger.info("   收到停止信号，正在关闭AGI系统...")
    except Exception as e:
        logger.error(f"   AGI系统启动失败: {e}")
        return False

    return True

def start_inference_server():
    """启动推理服务器"""
    logger.info("🌐 启动AGI推理服务器...")

    try:
        import subprocess

        # 启动FastAPI服务器
        cmd = [
            sys.executable, "-m", "uvicorn",
            "h2q_project.h2q_server:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]

        logger.info("   启动服务器: http://localhost:8000")
        logger.info("   健康检查: http://localhost:8000/health")

        # 后台运行服务器
        process = subprocess.Popen(cmd)
        logger.info(f"   服务器进程ID: {process.pid}")

        return process

    except Exception as e:
        logger.error(f"   服务器启动失败: {e}")
        return None

def main():
    """主函数"""
    print("🔥 H2Q-Evo 自动进化AGI系统启动器")
    print("=" * 50)

    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，无法启动AGI系统")
        return 1

    # 启动推理服务器
    server_process = start_inference_server()

    # 启动进化系统
    success = start_evolution_system()

    # 清理服务器进程
    if server_process:
        logger.info("关闭推理服务器...")
        server_process.terminate()
        server_process.wait()

    if success:
        logger.info("✅ AGI系统运行完成")
        return 0
    else:
        logger.error("❌ AGI系统运行失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())