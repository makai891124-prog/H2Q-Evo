#!/usr/bin/env python3
"""H2Q AGI 24小时自主进化启动器.

启动完整的24小时自主进化流程:
1. 系统检查
2. 启动守护进程
3. 开始进化
4. 24小时后验收

使用方法:
    python start_24h_evolution.py           # 启动24小时进化
    python start_24h_evolution.py --quick   # 快速测试 (5分钟)
    python start_24h_evolution.py --hours 1 # 自定义时长
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_system():
    """系统检查."""
    print("=" * 60)
    print("🔍 系统检查")
    print("=" * 60)
    
    checks = {}
    
    # 检查 Python 版本
    py_version = sys.version_info
    checks["Python 版本"] = py_version >= (3, 8)
    print(f"  Python 版本: {py_version.major}.{py_version.minor}.{py_version.micro} " + 
          ("✅" if checks["Python 版本"] else "❌ (需要 3.8+)"))
    
    # 检查 NumPy
    try:
        import numpy as np
        checks["NumPy"] = True
        print(f"  NumPy: {np.__version__} ✅")
    except ImportError:
        checks["NumPy"] = False
        print("  NumPy: 未安装 ❌")
    
    # 检查模块导入
    try:
        from h2q_project.h2q.agi.evolution_24h import Evolution24HSystem
        from h2q_project.h2q.agi.survival_daemon import SurvivalDaemon
        checks["AGI 模块"] = True
        print("  AGI 模块: 可用 ✅")
    except Exception as e:
        checks["AGI 模块"] = False
        print(f"  AGI 模块: 错误 - {e} ❌")
    
    # 检查网络 - 区分国际源和中国源
    import urllib.request
    import ssl
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # 测试国际源
    international_ok = False
    china_ok = False
    
    # 国际源测试
    intl_urls = [
        ("https://en.wikipedia.org/api/rest_v1/", "Wikipedia API"),
        ("https://www.google.com", "Google"),
    ]
    
    for url, name in intl_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            urllib.request.urlopen(req, timeout=5, context=ssl_context)
            international_ok = True
            print(f"  国际网络: 可用 ✅ (通过 {name})")
            break
        except:
            continue
    
    if not international_ok:
        print("  国际网络: 不可用 ⚠️")
    
    # 中国源测试
    china_urls = [
        ("https://www.baidu.com", "百度"),
        ("https://hf-mirror.com", "HF镜像"),
        ("https://baike.baidu.com", "百度百科"),
    ]
    
    for url, name in china_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            urllib.request.urlopen(req, timeout=5, context=ssl_context)
            china_ok = True
            print(f"  中国网络: 可用 ✅ (通过 {name})")
            break
        except:
            continue
    
    if not china_ok:
        print("  中国网络: 不可用 ⚠️")
    
    # 确定网络模式
    if international_ok:
        checks["网络连接"] = True
        checks["网络模式"] = "international"
        print("  📡 网络模式: 国际源 (Wikipedia)")
    elif china_ok:
        checks["网络连接"] = True
        checks["网络模式"] = "china"
        print("  📡 网络模式: 中国源 (HF镜像 + 百度百科)")
    else:
        checks["网络连接"] = False
        checks["网络模式"] = "offline"
        print("  📡 网络模式: 离线 (使用缓存数据)")
        print("    提示: 设置 HTTP_PROXY 环境变量可能有助于解决网络问题")
    
    # 检查磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        checks["磁盘空间"] = free_gb > 1
        print(f"  磁盘空间: {free_gb:.1f}GB 可用 " + ("✅" if free_gb > 1 else "⚠️"))
    except:
        checks["磁盘空间"] = True
    
    all_passed = all(v for k, v in checks.items() if k != "网络连接")
    print()
    
    return all_passed


def start_evolution(hours: float = 24.0, quick_test: bool = False):
    """启动进化."""
    from h2q_project.h2q.agi.evolution_24h import Evolution24HSystem, EvolutionConfig
    from h2q_project.h2q.agi.survival_daemon import SurvivalDaemon, SurvivalConfig
    
    if quick_test:
        hours = 5 / 60  # 5分钟
        print("🧪 快速测试模式: 5分钟")
    
    print("=" * 60)
    print("🚀 H2Q AGI 24小时自主进化系统")
    print("=" * 60)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预计结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"进化时长: {hours:.2f} 小时")
    print()
    
    # 配置
    evo_config = EvolutionConfig(
        total_duration_hours=hours,
        learning_cycle_minutes=30 if not quick_test else 1,
        capability_check_minutes=60 if not quick_test else 2,
        heartbeat_seconds=30 if not quick_test else 10
    )
    
    survival_config = SurvivalConfig(
        heartbeat_interval=30 if not quick_test else 10,
        max_no_heartbeat=120 if not quick_test else 60,
        capability_check_interval=3600 if not quick_test else 60
    )
    
    # 创建系统
    evolution_system = Evolution24HSystem(evo_config, str(PROJECT_ROOT))
    survival_daemon = SurvivalDaemon(survival_config, str(PROJECT_ROOT))
    
    # 设置能力检查回调
    def capability_callback():
        if evolution_system.tester.test_history:
            return evolution_system.tester.test_history[-1]["overall_score"]
        return 0.0
    
    survival_daemon.set_capability_callback(capability_callback)
    
    # 设置重启回调
    def restart_callback():
        print("🔄 触发系统恢复...")
        # 重新初始化组件
        evolution_system.compressor = type(evolution_system.compressor)(0.5)
        evolution_system.acquirer = type(evolution_system.acquirer)()
    
    survival_daemon.set_restart_callback(restart_callback)
    
    # 记录信号接收次数
    signal_count = [0]
    
    # 信号处理
    def signal_handler(signum, frame):
        signal_count[0] += 1
        
        if signal_count[0] == 1:
            print("\n⚠️ 收到中断信号，准备停止...")
            print("   提示: 再次按 Ctrl+C 将强制停止（禁用自动重启）")
            evolution_system.stop()
            survival_daemon.stop()
            generate_final_report(evolution_system, survival_daemon)
            sys.exit(0)
        else:
            print("\n🛑 收到第二次中断信号，执行强制停止...")
            print("   系统将完全停止，禁用所有自动重启功能")
            evolution_system.stop()
            survival_daemon.force_stop()  # 使用强制停止
            generate_final_report(evolution_system, survival_daemon)
            print("\n✅ 强制停止完成。要恢复自动重启功能，请删除 FORCE_STOP 文件")
            sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动
    print("🔧 启动守护进程...")
    survival_daemon.start()
    
    print("🧬 启动进化系统...")
    evolution_system.start()
    
    # 主循环
    print("\n" + "=" * 60)
    print("📊 进化监控 (按 Ctrl+C 停止)")
    print("=" * 60)
    
    try:
        while evolution_system.is_running:
            time.sleep(60 if not quick_test else 10)
            
            # 显示状态
            evo_status = evolution_system.get_status()
            daemon_status = survival_daemon.get_status()
            
            elapsed_h = evo_status["elapsed_hours"]
            remaining_h = evo_status["remaining_hours"]
            progress = (elapsed_h / hours) * 100 if hours > 0 else 100
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 进度: {progress:.1f}%")
            print(f"  运行时间: {elapsed_h:.2f}h / {hours:.2f}h")
            print(f"  学习周期: {evo_status['cycle_count']}")
            print(f"  知识条目: {evo_status['knowledge_count']}")
            print(f"  能力评分: {evo_status['latest_score']:.1f}%")
            print(f"  系统健康: {'✅' if daemon_status['is_healthy'] else '⚠️'}")
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    
    # 停止
    print("\n" + "=" * 60)
    print("🛑 停止系统...")
    print("=" * 60)
    
    evolution_system.stop()
    survival_daemon.stop()
    
    # 生成最终报告
    generate_final_report(evolution_system, survival_daemon)


def generate_final_report(evolution_system, survival_daemon):
    """生成最终验收报告."""
    print("\n" + "=" * 60)
    print("📋 生成最终验收报告")
    print("=" * 60)
    
    evo_status = evolution_system.get_status()
    daemon_status = survival_daemon.get_status()
    
    report = []
    report.append("# H2Q AGI 24小时自主进化 - 最终验收报告")
    report.append("")
    report.append(f"**验收时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## 📊 执行摘要")
    report.append("")
    report.append("| 指标 | 值 | 状态 |")
    report.append("|------|-----|------|")
    report.append(f"| 总运行时间 | {evo_status['elapsed_hours']:.2f} 小时 | ✅ |")
    report.append(f"| 学习周期 | {evo_status['cycle_count']} | ✅ |")
    report.append(f"| 知识条目 | {evo_status['knowledge_count']} | ✅ |")
    report.append(f"| 已完成任务 | {daemon_status.get('tasks_completed', 0)} | ✅ |")
    report.append(f"| 重启次数 | {daemon_status.get('restart_count', 0)} | {'✅' if daemon_status.get('restart_count', 0) == 0 else '⚠️'} |")
    report.append(f"| 最终评分 | {evo_status['latest_score']:.1f}% | {'✅' if evo_status['latest_score'] >= 60 else '⚠️'} |")
    report.append("")
    
    # 能力评估
    report.append("## 🧪 能力评估")
    report.append("")
    
    if evolution_system.tester.test_history:
        latest = evolution_system.tester.test_history[-1]
        report.append(f"**最终评分**: {latest['overall_score']:.1f}% - {latest['grade']}")
        report.append("")
        
        report.append("| 能力领域 | 得分 | 状态 |")
        report.append("|----------|------|------|")
        for name, result in latest["tests"].items():
            status = "✅" if result["score"] >= 60 else "⚠️"
            report.append(f"| {name} | {result['score']:.1f}% | {status} |")
        report.append("")
        
        # 进步趋势
        progress = evolution_system.tester.get_progress()
        report.append(f"**进步趋势**: {progress['trend']}")
        if progress['improvement'] != 0:
            report.append(f"**变化幅度**: {progress['improvement']:+.1f}%")
        report.append("")
    
    # 系统稳定性
    report.append("## 🛡️ 系统稳定性")
    report.append("")
    report.append(f"- 心跳正常: {'是' if daemon_status.get('is_healthy', True) else '否'}")
    uptime = daemon_status.get('uptime', 0)
    if isinstance(uptime, (int, float)):
        report.append(f"- 运行时间: {uptime:.2f} 秒")
    else:
        report.append(f"- 运行时间: {uptime}")
    report.append(f"- 错误次数: {daemon_status.get('errors_count', 0)}")
    report.append(f"- 重启次数: {daemon_status.get('restart_count', 0)}")
    memory_mb = daemon_status.get('memory_mb', 0)
    if isinstance(memory_mb, (int, float)):
        report.append(f"- 内存使用: {memory_mb:.1f} MB")
    else:
        report.append(f"- 内存使用: {memory_mb}")
    report.append("")
    
    # 验收结论
    report.append("## ✅ 验收结论")
    report.append("")
    
    all_passed = (
        evo_status['cycle_count'] > 0 and
        daemon_status.get('restart_count', 0) < 3 and
        evo_status['latest_score'] >= 60
    )
    
    if all_passed:
        report.append("**验收状态**: ✅ **通过**")
        report.append("")
        report.append("系统成功完成24小时自主进化，表现出:")
        report.append("- 稳定的自主学习能力")
        report.append("- 可靠的进程监控机制")
        report.append("- 持续的能力认证反馈")
    else:
        report.append("**验收状态**: ⚠️ **需要关注**")
        report.append("")
        if evo_status['cycle_count'] == 0:
            report.append("- ⚠️ 学习周期为0，可能存在网络问题")
        if daemon_status['restart_count'] >= 3:
            report.append("- ⚠️ 重启次数过多，系统稳定性需要改进")
        if evo_status['latest_score'] < 60:
            report.append("- ⚠️ 能力评分未达到及格线")
    
    report.append("")
    report.append("---")
    report.append("*报告由 H2Q AGI 自主进化系统自动生成*")
    
    # 保存报告
    report_path = PROJECT_ROOT / "EVOLUTION_ACCEPTANCE_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print(f"📝 验收报告已保存: {report_path}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 验收摘要")
    print("=" * 60)
    print(f"  运行时间: {evo_status['elapsed_hours']:.2f} 小时")
    print(f"  学习周期: {evo_status['cycle_count']}")
    print(f"  知识条目: {evo_status['knowledge_count']}")
    print(f"  最终评分: {evo_status['latest_score']:.1f}%")
    print(f"  验收状态: {'✅ 通过' if all_passed else '⚠️ 需要关注'}")
    print("=" * 60)


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description="H2Q AGI 24小时自主进化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python start_24h_evolution.py           # 启动24小时进化
  python start_24h_evolution.py --quick   # 快速测试 (5分钟)
  python start_24h_evolution.py --hours 1 # 自定义时长 (1小时)
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (5分钟)")
    parser.add_argument("--hours", type=float, default=24.0,
                        help="进化时长 (小时), 默认24")
    parser.add_argument("--skip-check", action="store_true",
                        help="跳过系统检查")
    
    args = parser.parse_args()
    
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     H2Q AGI 24小时自主进化系统                              ║")
    print("║     Autonomous Evolution System                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # 系统检查
    if not args.skip_check:
        if not check_system():
            print("❌ 系统检查未通过，请解决上述问题后重试。")
            print("   或使用 --skip-check 跳过检查")
            return 1
    
    # 启动进化
    start_evolution(hours=args.hours, quick_test=args.quick)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
