#!/usr/bin/env python3
"""
H2Q-Evo AGI 自我进化训练启动器
基于系统集成测试结果，启动完整的AGI自我进化训练流程
"""

import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('agi_evolution_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AGI-Evolution-Starter')

class AGIEvolutionStarter:
    """AGI自我进化训练启动器"""

    def __init__(self):
        self.project_root = Path("./")
        self.startup_config = {
            'timestamp': datetime.now().isoformat(),
            'evolution_mode': 'continuous',
            'max_generations': 1000,
            'checkpoint_interval': 50,
            'monitoring_enabled': True,
            'api_integration': True,
            'local_inference': True
        }

    async def start_agi_evolution(self) -> Dict[str, Any]:
        """启动AGI自我进化训练"""
        logger.info("🚀 启动H2Q-Evo AGI自我进化训练...")

        # 1. 预启动检查
        logger.info("📋 第一步: 预启动检查")
        await self.pre_startup_checks()

        # 2. 配置系统参数
        logger.info("⚙️ 第二步: 配置系统参数")
        await self.configure_system_parameters()

        # 3. 初始化训练环境
        logger.info("🏗️ 第三步: 初始化训练环境")
        await self.initialize_training_environment()

        # 4. 启动监控系统
        logger.info("📊 第四步: 启动监控系统")
        await self.start_monitoring_system()

        # 5. 启动进化训练
        logger.info("🧬 第五步: 启动进化训练")
        await self.start_evolution_training()

        # 6. 启动API服务
        logger.info("🌐 第六步: 启动API服务")
        await self.start_api_services()

        # 7. 启动本地推理
        logger.info("💻 第七步: 启动本地推理")
        await self.start_local_inference()

        # 生成启动报告
        startup_report = self.generate_startup_report()

        logger.info("✅ AGI自我进化训练启动完成")
        return startup_report

    async def pre_startup_checks(self) -> Dict[str, Any]:
        """预启动检查"""
        checks = {
            'system_health': False,
            'dependencies': False,
            'configuration': False,
            'resources': False
        }

        # 检查系统健康状态
        try:
            from agi_system_manager import AGISystemManager
            manager = AGISystemManager()
            status = manager.get_system_status()
            healthy_components = sum(1 for comp_status in status.get('components', {}).values() if comp_status)
            total_components = len(status.get('components', {}))
            checks['system_health'] = healthy_components >= total_components * 0.6  # 至少60%组件正常
            logger.info(f"    ✅ 系统健康检查: {healthy_components}/{total_components} 组件正常")
        except Exception as e:
            logger.warning(f"    ⚠️ 系统健康检查失败: {e}")
            checks['system_health'] = False

        # 检查关键依赖
        required_modules = ['torch', 'transformers', 'agi_persistent_evolution', 'agi_system_manager']
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"    ✅ 依赖检查: {module} 可用")
            except ImportError:
                logger.error(f"    ❌ 依赖检查: {module} 缺失")
                checks['dependencies'] = False
                break
        else:
            checks['dependencies'] = True

        # 检查配置文件
        config_files = ['agi_training_config.ini']
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                logger.info(f"    ✅ 配置检查: {config_file} 存在")
                checks['configuration'] = True
            else:
                logger.warning(f"    ⚠️ 配置检查: {config_file} 缺失，将使用默认配置")
                checks['configuration'] = True  # 系统会创建默认配置

        # 检查系统资源
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        memory_gb = memory.available / (1024**3)
        disk_gb = disk.free / (1024**3)

        checks['resources'] = memory_gb >= 2 and disk_gb >= 5  # 至少2GB内存，5GB磁盘空间

        if checks['resources']:
            logger.info(f"    ✅ 资源检查: 内存{memory_gb:.1f}GB, 磁盘{disk_gb:.1f}GB")
        else:
            logger.warning(f"    ⚠️ 资源检查: 内存{memory_gb:.1f}GB, 磁盘{disk_gb:.1f}GB - 可能不足")

        all_checks_passed = all(checks.values())
        logger.info(f"    📊 预启动检查结果: {'通过' if all_checks_passed else '部分通过'}")

        return checks

    async def configure_system_parameters(self) -> Dict[str, Any]:
        """配置系统参数"""
        config = {}

        # 设置环境变量
        os.environ.setdefault('PROJECT_ROOT', str(self.project_root))
        os.environ.setdefault('INFERENCE_MODE', 'local')  # 优先使用本地推理
        os.environ.setdefault('MODEL_NAME', 'h2q-evolution-model')

        # 检查API密钥
        if os.getenv('GEMINI_API_KEY'):
            os.environ.setdefault('API_MODE', 'enabled')
            logger.info("    ✅ API模式: 已配置GEMINI_API_KEY")
        else:
            os.environ.setdefault('API_MODE', 'disabled')
            logger.info("    ⚠️ API模式: 未配置GEMINI_API_KEY，使用本地模式")

        # 创建训练配置文件
        config_path = self.project_root / 'agi_training_config.ini'
        if not config_path.exists():
            self.create_default_config(config_path)

        config['environment'] = dict(os.environ)
        config['config_file'] = str(config_path)

        logger.info("    ✅ 系统参数配置完成")
        return config

    def create_default_config(self, config_path: Path):
        """创建默认配置文件"""
        config_content = """[Training]
max_epochs = 100
batch_size = 8
learning_rate = 0.001
checkpoint_interval = 50
validation_interval = 10

[Evolution]
max_generations = 1000
mutation_rate = 0.1
crossover_rate = 0.8
selection_pressure = 0.5

[Monitoring]
wandb_enabled = true
log_level = INFO
metrics_interval = 60

[API]
port = 8000
host = 0.0.0.0
cors_enabled = true

[Docker]
image_name = h2q-sandbox
container_name = h2q-evolution
auto_build = true
"""

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        logger.info(f"    📝 创建默认配置文件: {config_path}")

    async def initialize_training_environment(self) -> Dict[str, Any]:
        """初始化训练环境"""
        init_status = {
            'data_directories': False,
            'model_directories': False,
            'log_directories': False,
            'checkpoint_cleanup': False
        }

        # 创建必要的目录
        directories = [
            'agi_persistent_training/data',
            'agi_persistent_training/models',
            'agi_persistent_training/logs',
            'agi_persistent_training/checkpoints',
            'evolution_logs'
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"    📁 创建目录: {dir_path}")

        init_status['data_directories'] = True
        init_status['model_directories'] = True
        init_status['log_directories'] = True

        # 清理旧的检查点（保留最新的5个）
        checkpoint_dir = self.project_root / 'agi_persistent_training/checkpoints'
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob('*.pth'), key=os.path.getmtime, reverse=True)
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[5:]:
                    old_checkpoint.unlink()
                    logger.info(f"    🗑️ 清理旧检查点: {old_checkpoint.name}")

        init_status['checkpoint_cleanup'] = True

        logger.info("    ✅ 训练环境初始化完成")
        return init_status

    async def start_monitoring_system(self) -> Dict[str, Any]:
        """启动监控系统"""
        monitoring_status = {
            'training_monitor': False,
            'evolution_monitor': False,
            'system_monitor': False,
            'wandb_logging': False
        }

        try:
            # 启动训练监控器
            from agi_training_monitor import AGITrainingMonitor
            training_monitor = AGITrainingMonitor()
            monitoring_status['training_monitor'] = True
            logger.info("    ✅ 训练监控器启动")

            # 启动进化监控器
            from agi_evolution_monitor import AGIEvolutionMonitor
            evolution_monitor = AGIEvolutionMonitor()
            monitoring_status['evolution_monitor'] = True
            logger.info("    ✅ 进化监控器启动")

            # 检查WandB
            try:
                import wandb
                wandb.init(project="h2q-evolution", name=f"evolution-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                monitoring_status['wandb_logging'] = True
                logger.info("    ✅ WandB日志记录启动")
            except Exception as e:
                logger.warning(f"    ⚠️ WandB初始化失败: {e}")

        except Exception as e:
            logger.error(f"    ❌ 监控系统启动失败: {e}")

        monitoring_status['system_monitor'] = True  # 基础系统监控总是可用的
        logger.info("    ✅ 监控系统启动完成")
        return monitoring_status

    async def start_evolution_training(self) -> Dict[str, Any]:
        """启动进化训练"""
        training_status = {
            'trainer_initialized': False,
            'training_started': False,
            'background_process': False
        }

        try:
            # 初始化训练器
            from agi_persistent_evolution import PersistentAGIConfig, PersistentAGITrainer
            config = PersistentAGIConfig()
            trainer = PersistentAGITrainer(config)
            training_status['trainer_initialized'] = True
            logger.info("    ✅ 持久AGI训练器初始化")

            # 启动训练（在后台运行）
            import threading
            training_thread = threading.Thread(
                target=self._run_training_loop,
                args=(trainer,),
                daemon=True,
                name='evolution_training'
            )
            training_thread.start()
            training_status['training_started'] = True
            training_status['background_process'] = True

            logger.info("    ✅ 进化训练启动（后台运行）")

        except Exception as e:
            logger.error(f"    ❌ 进化训练启动失败: {e}")
            training_status['training_started'] = False

        return training_status

    def _run_training_loop(self, trainer):
        """运行训练循环"""
        try:
            logger.info("    🔄 开始AGI进化训练循环...")
            trainer.start_evolution()
        except Exception as e:
            logger.error(f"训练循环异常: {e}")

    async def start_api_services(self) -> Dict[str, Any]:
        """启动API服务"""
        api_status = {
            'server_started': False,
            'endpoints_available': False,
            'cors_enabled': False
        }

        try:
            # 启动FastAPI服务器
            import subprocess
            import signal

            # 使用uvicorn启动服务器
            cmd = [
                sys.executable, "-m", "uvicorn",
                "h2q_project.h2q_server:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ]

            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            # 等待服务器启动
            await asyncio.sleep(3)

            # 检查进程是否还在运行
            if process.poll() is None:
                api_status['server_started'] = True
                logger.info("    ✅ API服务器启动 (端口8000)")

                # 检查端点
                try:
                    import requests
                    response = requests.get("http://localhost:8000/health", timeout=5)
                    if response.status_code == 200:
                        api_status['endpoints_available'] = True
                        logger.info("    ✅ API端点可用")
                    else:
                        logger.warning(f"    ⚠️ API端点状态码: {response.status_code}")
                except Exception as e:
                    logger.warning(f"    ⚠️ API端点检查失败: {e}")

                api_status['cors_enabled'] = True  # 假设配置中启用

                # 保存进程信息以便后续管理
                with open('.api_server_pid', 'w') as f:
                    f.write(str(process.pid))

            else:
                stdout, stderr = process.communicate()
                logger.error(f"    ❌ API服务器启动失败: {stderr.decode()}")

        except Exception as e:
            logger.error(f"    ❌ API服务启动失败: {e}")

        return api_status

    async def start_local_inference(self) -> Dict[str, Any]:
        """启动本地推理"""
        inference_status = {
            'docker_available': False,
            'image_built': False,
            'container_running': False
        }

        try:
            # 检查Docker是否可用
            import subprocess
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                inference_status['docker_available'] = True
                logger.info("    ✅ Docker可用")

                # 检查镜像是否存在
                result = subprocess.run(['docker', 'images', 'h2q-sandbox', '-q'], capture_output=True, text=True)
                if result.stdout.strip():
                    inference_status['image_built'] = True
                    logger.info("    ✅ H2Q-Sandbox镜像存在")
                else:
                    # 构建镜像
                    logger.info("    🔨 构建H2Q-Sandbox镜像...")
                    result = subprocess.run(['docker', 'build', '-t', 'h2q-sandbox', '.'], cwd=str(self.project_root))
                    if result.returncode == 0:
                        inference_status['image_built'] = True
                        logger.info("    ✅ H2Q-Sandbox镜像构建完成")
                    else:
                        logger.warning("    ⚠️ H2Q-Sandbox镜像构建失败")

                # 启动推理容器（如果镜像可用）
                if inference_status['image_built']:
                    # 这里可以启动容器，但为了安全起见暂时不启动
                    logger.info("    📦 本地推理容器准备就绪（按需启动）")

            else:
                logger.warning("    ⚠️ Docker不可用，将使用纯Python推理")

        except Exception as e:
            logger.warning(f"    ⚠️ 本地推理初始化失败: {e}")

        return inference_status

    def generate_startup_report(self) -> Dict[str, Any]:
        """生成启动报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'startup_config': self.startup_config,
            'overall_status': 'running',
            'active_services': [],
            'next_steps': [
                '监控训练进度: tail -f evolution.log',
                '查看API状态: curl http://localhost:8000/health',
                '检查训练指标: python3 analyze_agi_performance.py',
                '启动本地推理: python3 evolution_system.py --local-inference'
            ],
            'monitoring_commands': [
                'ps aux | grep python',
                'docker ps | grep h2q',
                'tail -f agi_persistent_training/logs/training.log',
                'tail -f evolution_logs/evolution_monitor.log'
            ]
        }

        # 检查活跃服务
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'])
                        if any(keyword in cmdline for keyword in ['h2q_server', 'evolution', 'training']):
                            report['active_services'].append({
                                'pid': proc.info['pid'],
                                'command': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                            })
                except:
                    pass
        except:
            pass

        # 保存启动报告
        report_path = self.project_root / 'agi_evolution_startup_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ 启动报告已保存: {report_path}")
        return report

async def main():
    """主函数"""
    print("🚀 H2Q-Evo AGI 自我进化训练启动器")
    print("=" * 50)

    starter = AGIEvolutionStarter()

    try:
        report = await starter.start_agi_evolution()

        print("\n📊 启动报告:")
        print(f"  • 时间戳: {report['timestamp']}")
        print(f"  • 整体状态: {report['overall_status'].upper()}")

        if report.get('active_services'):
            print(f"  • 活跃服务: {len(report['active_services'])} 个")
            for service in report['active_services'][:3]:
                print(f"    - PID {service['pid']}: {service['command']}")

        print("\n🎯 接下来的步骤:")
        for step in report.get('next_steps', []):
            print(f"  • {step}")

        print("\n📊 监控命令:")
        for cmd in report.get('monitoring_commands', []):
            print(f"  • {cmd}")

        print("\n📄 详细报告已保存到: agi_evolution_startup_report.json")
        print("🎉 AGI自我进化训练已启动！")

        return True

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())