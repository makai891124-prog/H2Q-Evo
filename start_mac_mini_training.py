#!/usr/bin/env python3
"""
H2Q-Evo AGI Mac Mini M4流式训练系统启动器
专为16GB内存Mac Mini M4设计，实现4GB内存限制的流式AGI训练
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('agi_mac_mini_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AGI-MacMiniTrainingSystem')

def check_mac_mini_requirements():
    """检查Mac Mini M4系统要求"""
    logger.info("检查Mac Mini M4系统要求...")

    requirements_met = True

    # 检查Python版本
    if sys.version_info < (3, 8):
        logger.error("需要Python 3.8或更高版本")
        requirements_met = False

    # 检查内存（Mac Mini M4 16GB）
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 8:  # 至少8GB内存
        logger.warning(f"检测到{memory_gb:.1f}GB内存，建议使用16GB内存的Mac Mini M4以获得最佳性能")

    # 检查CPU核心数
    cpu_count = psutil.cpu_count()
    if cpu_count < 4:
        logger.warning(f"检测到{cpu_count}个CPU核心，建议使用多核心CPU")

    # 检查必要模块
    required_modules = ['torch', 'numpy', 'psutil', 'curses']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            logger.error(f"缺少必要模块: {module}")
            requirements_met = False

    # 检查目录权限
    dirs_to_check = ['checkpoints', 'agi_backups', 'data', 'streaming_cache']
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        try:
            dir_path.mkdir(exist_ok=True)
            # 测试写入权限
            test_file = dir_path / '.test_write'
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            logger.error(f"目录权限检查失败 {dir_name}: {e}")
            requirements_met = False

    if not requirements_met:
        logger.error("Mac Mini M4系统要求检查失败，请解决上述问题后重试")
        return False

    logger.info("Mac Mini M4系统要求检查通过")
    return True

def initialize_mac_mini_streaming():
    """初始化Mac Mini流式训练"""
    logger.info("初始化Mac Mini流式训练组件...")

    try:
        # 导入Mac Mini专用组件
        from agi_mac_mini_streaming import start_mac_mini_streaming_training

        # 启动Mac Mini流式训练
        logger.info("启动Mac Mini流式AGI训练...")
        trainer = start_mac_mini_streaming_training()

        logger.info("Mac Mini流式训练组件初始化完成")
        return trainer

    except Exception as e:
        logger.error(f"Mac Mini流式训练初始化失败: {e}")
        raise

def create_mac_mini_system_report(trainer=None):
    """创建Mac Mini系统报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'platform': 'Mac Mini M4',
        'memory_config': '16GB Unified Memory',
        'optimization': 'Streaming Training with 4GB Memory Limit',
        'components': {},
        'status': 'initializing'
    }

    try:
        from agi_mac_mini_streaming import get_streaming_status
        from agi_health_monitor import get_monitoring_data

        # 流式训练状态
        if trainer:
            streaming_status = trainer.get_streaming_status()
        else:
            streaming_status = get_streaming_status()

        report['components']['streaming_trainer'] = {
            'running': streaming_status.get('running', False),
            'training_active': streaming_status.get('training_active', False),
            'current_step': streaming_status.get('current_step', 0),
            'best_loss': streaming_status.get('best_loss', float('inf')),
            'memory_limit_gb': streaming_status.get('memory_config', {}).get('max_memory_gb', 4.0)
        }

        # 监控数据
        monitor_data = get_monitoring_data()
        report['components']['health_monitor'] = {
            'realtime_stats': monitor_data.get('realtime', {}),
            'streaming_status': monitor_data.get('streaming', {}),
            'system_health': monitor_data.get('system', {})
        }

        report['status'] = 'running'

    except Exception as e:
        logger.error(f"创建Mac Mini系统报告失败: {e}")
        report['status'] = 'error'
        report['error'] = str(e)

    # 保存报告
    with open('agi_mac_mini_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

def monitor_mac_mini_system(trainer):
    """监控Mac Mini系统运行状态"""
    logger.info("开始Mac Mini系统监控...")

    while trainer.running:
        try:
            # 获取状态
            streaming_status = trainer.get_streaming_status()

            # 检查关键指标
            issues = []

            # 检查内存使用
            memory_usage = streaming_status.get('performance_metrics', {}).get('memory_usage_mb', 0)
            if memory_usage > 4 * 1024:  # 超过4GB
                issues.append(f"内存使用过高: {memory_usage:.1f}MB")
            if not streaming_status.get('training_active', False):
                issues.append("流式训练未激活")

            # 检查CPU使用率
            cpu_usage = streaming_status.get('performance_metrics', {}).get('cpu_usage_percent', 0)
            if cpu_usage > 90:
                issues.append(f"CPU使用率过高: {cpu_usage:.1f}%")
            status_summary = {
                'timestamp': datetime.now().isoformat(),
                'streaming_running': streaming_status.get('running', False),
                'training_active': streaming_status.get('training_active', False),
                'current_step': streaming_status.get('current_step', 0),
                'best_loss': streaming_status.get('best_loss', float('inf')),
                'memory_usage_mb': memory_usage,
                'cpu_usage_percent': cpu_usage,
                'streaming_efficiency': streaming_status.get('performance_metrics', {}).get('streaming_efficiency', 0),
                'issues': issues,
                'platform': 'Mac Mini M4',
                'memory_limit': '4GB'
            }

            # 写入状态文件
            with open('agi_mac_mini_status.json', 'w', encoding='utf-8') as f:
                json.dump(status_summary, f, indent=2, ensure_ascii=False)

            # 控制台输出
            if issues:
                logger.warning(f"Mac Mini系统问题检测: {', '.join(issues)}")
            else:
                logger.info(f"Mac Mini系统运行正常 - 步骤: {streaming_status.get('current_step', 0)}, "
                           f"内存: {memory_usage:.1f}MB, 效率: {streaming_status.get('performance_metrics', {}).get('streaming_efficiency', 0):.2f}")

            # 更新系统报告
            create_mac_mini_system_report(trainer)

            time.sleep(30)  # 30秒检查一次

        except Exception as e:
            logger.error(f"Mac Mini系统监控异常: {e}")
            time.sleep(30)

def graceful_shutdown_mac_mini(trainer):
    """优雅关闭Mac Mini系统"""
    logger.info("开始Mac Mini系统优雅关闭...")

    try:
        # 停止流式训练
        logger.info("停止Mac Mini流式训练...")
        from agi_mac_mini_streaming import stop_mac_mini_streaming_training
        stop_mac_mini_streaming_training()

        # 创建最终报告
        logger.info("创建Mac Mini最终系统报告...")
        final_report = create_mac_mini_system_report(trainer)
        final_report['status'] = 'shutdown'
        final_report['shutdown_time'] = datetime.now().isoformat()

        with open('agi_mac_mini_final_report.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        logger.info("Mac Mini系统已优雅关闭")

    except Exception as e:
        logger.error(f"Mac Mini关闭过程中发生错误: {e}")

def start_health_monitor_background():
    """在后台启动健康监控窗口"""
    def monitor_thread():
        try:
            from agi_health_monitor import create_monitoring_dashboard
            logger.info("启动后台健康监控窗口...")
            create_monitoring_dashboard()
        except Exception as e:
            logger.error(f"后台健康监控窗口异常: {e}")

    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    return monitor

def main():
    """主函数 - Mac Mini M4流式训练版"""
    parser = argparse.ArgumentParser(description='H2Q-Evo AGI Mac Mini M4流式训练系统')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查系统要求，不启动训练')
    parser.add_argument('--monitor-only', action='store_true',
                       help='仅启动监控窗口，不启动训练')
    parser.add_argument('--no-monitor', action='store_true',
                       help='不启动监控窗口')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=== H2Q-Evo AGI Mac Mini M4流式训练系统启动器 ===")
    logger.info(f"启动时间: {datetime.now().isoformat()}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info(f"平台: Mac Mini M4 (16GB Unified Memory)")
    logger.info(f"优化目标: 4GB内存限制，流式训练，边缘计算")
    logger.info(f"Python版本: {sys.version}")

    # 检查系统要求
    if not check_mac_mini_requirements():
        sys.exit(1)

    if args.check_only:
        logger.info("Mac Mini系统检查完成，退出")
        return

    if args.monitor_only:
        logger.info("启动监控窗口模式...")
        from agi_health_monitor import create_monitoring_dashboard
        create_monitoring_dashboard()
        return

    trainer = None
    monitor_thread = None

    try:
        # 初始化Mac Mini流式训练
        trainer = initialize_mac_mini_streaming()

        # 启动后台监控窗口（除非被禁用）
        if not args.no_monitor:
            monitor_thread = start_health_monitor_background()
            logger.info("后台健康监控窗口已启动")

        # 创建初始系统报告
        create_mac_mini_system_report(trainer)

        # 启动监控
        monitor_mac_mini_system(trainer)

    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as e:
        logger.error(f"Mac Mini流式训练系统运行异常: {e}")
        raise
    finally:
        if trainer:
            graceful_shutdown_mac_mini(trainer)
        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.join(timeout=5)

if __name__ == "__main__":
    main()