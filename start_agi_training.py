#!/usr/bin/env python3
"""
H2Q-Evo AGI完整训练系统启动器
集成所有训练前置组件，实现完整的实时在线训练生态系统
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('agi_training_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AGI-TrainingSystem')

def check_system_requirements():
    """检查系统要求"""
    logger.info("检查系统要求...")

    requirements_met = True

    # 检查Python版本
    if sys.version_info < (3, 8):
        logger.error("需要Python 3.8或更高版本")
        requirements_met = False

    # 检查必要模块
    required_modules = ['torch', 'numpy', 'psutil', 'asyncio']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            logger.error(f"缺少必要模块: {module}")
            requirements_met = False

    # 检查目录权限
    dirs_to_check = ['checkpoints', 'agi_backups', 'data']
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
        logger.error("系统要求检查失败，请解决上述问题后重试")
        return False

    logger.info("系统要求检查通过")
    return True

def initialize_components():
    """初始化所有组件"""
    logger.info("初始化训练组件...")

    try:
        # 导入组件
        from agi_training_infrastructure import start_agi_infrastructure
        from agi_realtime_training import start_realtime_agi_training
        from agi_fault_tolerance import get_fault_tolerance_manager, get_process_supervisor

        # 启动基础设施
        logger.info("启动AGI基础设施...")
        infrastructure = start_agi_infrastructure()

        # 启动容错系统
        logger.info("启动容错系统...")
        fault_manager = get_fault_tolerance_manager()
        process_supervisor = get_process_supervisor()
        process_supervisor.start_supervision()

        # 启动实时训练
        logger.info("启动实时训练系统...")
        trainer = start_realtime_agi_training()

        logger.info("所有组件初始化完成")
        return infrastructure, trainer, fault_manager

    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        raise

def create_system_report():
    """创建系统报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        },
        'components': {},
        'status': 'initializing'
    }

    try:
        from agi_training_infrastructure import get_infrastructure_status
        from agi_realtime_training import get_training_status
        from agi_fault_tolerance import get_fault_tolerance_manager

        # 基础设施状态
        infra_status = get_infrastructure_status()
        report['components']['infrastructure'] = {
            'running': infra_status.get('infrastructure_running', False),
            'environment': infra_status.get('environment', {}),
            'data_sources': len(infra_status.get('data_sources', {}))
        }

        # 训练状态
        training_status = get_training_status()
        report['components']['training'] = {
            'running': training_status.get('running', False),
            'training_active': training_status.get('training_active', False),
            'current_step': training_status.get('current_step', 0),
            'best_loss': training_status.get('best_loss', float('inf'))
        }

        # 容错状态
        fault_manager = get_fault_tolerance_manager()
        health = fault_manager.get_system_health()
        report['components']['fault_tolerance'] = {
            'overall_health': health.get('overall_health', 'unknown'),
            'recent_faults': len(health.get('recent_faults', []))
        }

        report['status'] = 'running'

    except Exception as e:
        logger.error(f"创建系统报告失败: {e}")
        report['status'] = 'error'
        report['error'] = str(e)

    # 保存报告
    with open('agi_system_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

def monitor_system(components):
    """监控系统运行状态"""
    infrastructure, trainer, fault_manager = components

    logger.info("开始系统监控...")

    while True:
        try:
            # 获取状态
            infra_status = infrastructure.get_system_status()
            training_status = trainer.get_training_status()
            health_status = fault_manager.get_system_health()

            # 检查关键指标
            issues = []

            # 检查基础设施
            if not infra_status.get('infrastructure_running', False):
                issues.append("基础设施未运行")

            # 检查训练
            if not training_status.get('running', False):
                issues.append("训练系统未运行")

            # 检查健康状态
            if health_status.get('overall_health') == 'critical':
                issues.append("系统健康状态严重")

            # 记录状态
            status_summary = {
                'timestamp': datetime.now().isoformat(),
                'infrastructure_running': infra_status.get('infrastructure_running', False),
                'training_running': training_status.get('running', False),
                'training_active': training_status.get('training_active', False),
                'current_step': training_status.get('current_step', 0),
                'best_loss': training_status.get('best_loss', float('inf')),
                'system_health': health_status.get('overall_health'),
                'issues': issues
            }

            # 写入状态文件
            with open('agi_system_status.json', 'w', encoding='utf-8') as f:
                json.dump(status_summary, f, indent=2, ensure_ascii=False)

            # 控制台输出
            if issues:
                logger.warning(f"系统问题检测: {', '.join(issues)}")
            else:
                logger.info(f"系统运行正常 - 训练步骤: {training_status.get('current_step', 0)}, "
                           f"最佳损失: {training_status.get('best_loss', 'N/A')}")

            # 更新系统报告
            create_system_report()

            time.sleep(60)  # 每分钟检查一次

        except Exception as e:
            logger.error(f"系统监控异常: {e}")
            time.sleep(30)

def graceful_shutdown(components):
    """优雅关闭系统"""
    logger.info("开始优雅关闭...")

    try:
        infrastructure, trainer, fault_manager = components

        # 停止训练
        logger.info("停止实时训练...")
        from agi_realtime_training import stop_realtime_agi_training
        stop_realtime_agi_training()

        # 停止基础设施
        logger.info("停止基础设施...")
        infrastructure.stop_infrastructure()

        # 创建最终报告
        logger.info("创建最终系统报告...")
        final_report = create_system_report()
        final_report['status'] = 'shutdown'
        final_report['shutdown_time'] = datetime.now().isoformat()

        with open('agi_final_report.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        logger.info("系统已优雅关闭")

    except Exception as e:
        logger.error(f"关闭过程中发生错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='H2Q-Evo AGI完整训练系统')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查系统要求，不启动训练')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=== H2Q-Evo AGI训练系统启动器 ===")
    logger.info(f"启动时间: {datetime.now().isoformat()}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info(f"Python版本: {sys.version}")

    # 检查系统要求
    if not check_system_requirements():
        sys.exit(1)

    if args.check_only:
        logger.info("系统检查完成，退出")
        return

    # 加载配置
    if args.config and os.path.exists(args.config):
        logger.info(f"加载配置文件: {args.config}")
        # 这里可以实现配置加载逻辑

    components = None
    try:
        # 初始化组件
        components = initialize_components()

        # 创建初始系统报告
        create_system_report()

        # 启动监控
        monitor_system(components)

    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        raise
    finally:
        if components:
            graceful_shutdown(components)

if __name__ == "__main__":
    main()