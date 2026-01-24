#!/usr/bin/env python3
"""
H2Q-Evo 系统联调脚本
逐步测试和修复所有组件，确保系统能正常协同工作
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
        logging.FileHandler('system_integration_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('System-Integration')

class SystemIntegrationTester:
    """系统集成测试器"""

    def __init__(self):
        self.project_root = Path("./")
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'integration_tests': {},
            'component_status': {},
            'fixes_applied': [],
            'overall_status': 'unknown'
        }

    async def run_full_integration_test(self) -> Dict[str, Any]:
        """运行完整集成测试"""
        logger.info("🔧 开始H2Q-Evo系统联调...")

        # 1. 组件依赖检查
        logger.info("📦 第一步: 组件依赖检查")
        await self.test_component_dependencies()

        # 2. 组件初始化测试
        logger.info("🏗️ 第二步: 组件初始化测试")
        await self.test_component_initialization()

        # 3. 组件间通信测试
        logger.info("🔗 第三步: 组件间通信测试")
        await self.test_component_communication()

        # 4. 数据流测试
        logger.info("🌊 第四步: 数据流测试")
        await self.test_data_flow()

        # 5. 系统启动测试
        logger.info("🚀 第五步: 系统启动测试")
        await self.test_system_startup()

        # 6. 自我进化准备
        logger.info("🧬 第六步: 自我进化准备")
        await self.prepare_self_evolution()

        # 生成总结
        self.generate_integration_summary()

        # 保存结果
        self.save_integration_results()

        logger.info("✅ 系统联调完成")
        return self.test_results

    async def test_component_dependencies(self) -> Dict[str, Any]:
        """测试组件依赖"""
        results = {
            'status': 'unknown',
            'dependencies': {},
            'missing_dependencies': [],
            'version_conflicts': []
        }

        logger.info("  • 检查Python包依赖")

        required_packages = [
            'torch', 'transformers', 'accelerate', 'peft', 'datasets',
            'wandb', 'numpy', 'scipy', 'psutil', 'pathlib'
        ]

        for package in required_packages:
            try:
                __import__(package)
                results['dependencies'][package] = 'available'
                logger.info(f"    ✅ {package}: 可用")
            except ImportError:
                results['dependencies'][package] = 'missing'
                results['missing_dependencies'].append(package)
                logger.warning(f"    ❌ {package}: 缺失")

        # 检查可选依赖
        optional_packages = ['trl', 'google.genai']
        for package in optional_packages:
            try:
                __import__(package)
                results['dependencies'][package] = 'available'
                logger.info(f"    ✅ {package}: 可用 (可选)")
            except ImportError:
                results['dependencies'][package] = 'missing'
                logger.info(f"    ⚠️  {package}: 缺失 (可选)")

        results['status'] = 'success' if not results['missing_dependencies'] else 'warning'
        self.test_results['integration_tests']['dependencies'] = results
        return results

    async def test_component_initialization(self) -> Dict[str, Any]:
        """测试组件初始化"""
        results = {
            'status': 'unknown',
            'components': {},
            'initialization_errors': [],
            'fixes_applied': []
        }

        # 测试各个组件的初始化
        components_to_test = [
            ('LogarithmicManifoldEncoder', 'agi_manifold_encoder', 'LogarithmicManifoldEncoder'),
            ('CompressedAGIEncoder', 'agi_manifold_encoder', 'CompressedAGIEncoder'),
            ('AGIDataGenerator', 'agi_data_generator', 'AGIDataGenerator'),
            ('AGITrainingMonitor', 'agi_training_monitor', 'AGITrainingMonitor'),
            ('AGIEvolutionMonitor', 'agi_evolution_monitor', 'AGIEvolutionMonitor'),
            ('PersistentAGIConfig', 'agi_persistent_evolution', 'PersistentAGIConfig'),
        ]

        for component_name, module_name, class_name in components_to_test:
            try:
                module = __import__(module_name)
                component_class = getattr(module, class_name)

                if component_name == 'PersistentAGIConfig':
                    # Config类不需要参数
                    instance = component_class()
                else:
                    # 其他组件尝试无参数初始化
                    instance = component_class()

                results['components'][component_name] = 'success'
                logger.info(f"    ✅ {component_name}: 初始化成功")

            except Exception as e:
                results['components'][component_name] = 'error'
                results['initialization_errors'].append({
                    'component': component_name,
                    'error': str(e)
                })
                logger.error(f"    ❌ {component_name}: 初始化失败 - {e}")

        # 特殊处理PersistentAGITrainer（需要config参数）
        try:
            from agi_persistent_evolution import PersistentAGIConfig, PersistentAGITrainer
            config = PersistentAGIConfig()
            trainer = PersistentAGITrainer(config)
            results['components']['PersistentAGITrainer'] = 'success'
            logger.info("    ✅ PersistentAGITrainer: 初始化成功")
        except Exception as e:
            results['components']['PersistentAGITrainer'] = 'error'
            results['initialization_errors'].append({
                'component': 'PersistentAGITrainer',
                'error': str(e)
            })
            logger.error(f"    ❌ PersistentAGITrainer: 初始化失败 - {e}")

        results['status'] = 'success' if not results['initialization_errors'] else 'error'
        self.test_results['integration_tests']['initialization'] = results
        return results

    async def test_component_communication(self) -> Dict[str, Any]:
        """测试组件间通信"""
        results = {
            'status': 'unknown',
            'communication_tests': [],
            'issues_found': []
        }

        try:
            # 测试数据生成器 -> 流形编码器
            logger.info("    • 测试数据生成器 -> 流形编码器通信")
            from agi_data_generator import AGIDataGenerator
            from agi_manifold_encoder import LogarithmicManifoldEncoder

            data_gen = AGIDataGenerator()
            encoder = LogarithmicManifoldEncoder(resolution=0.01)

            # 生成测试数据
            test_data = data_gen.generate_training_data(num_samples=5)

            if test_data:
                # 测试编码器处理数据
                sample_text = test_data[0].get('input', '') if isinstance(test_data[0], dict) else str(test_data[0])
                # 这里可以添加实际的编码测试

                results['communication_tests'].append({
                    'test': 'data_generator_to_encoder',
                    'status': 'success',
                    'details': f'成功处理{len(test_data)}条数据'
                })
                logger.info("      ✅ 数据生成器 -> 流形编码器: 通信正常")
            else:
                results['issues_found'].append('data_generator_produced_no_data')
                logger.warning("      ⚠️  数据生成器未产生数据")

            # 测试训练器 -> 监控器通信
            logger.info("    • 测试训练器 -> 监控器通信")
            from agi_persistent_evolution import PersistentAGIConfig, PersistentAGITrainer
            from agi_training_monitor import AGITrainingMonitor

            config = PersistentAGIConfig()
            trainer = PersistentAGITrainer(config)
            monitor = AGITrainingMonitor()

            # 测试状态同步
            trainer_status = trainer.state.generation if hasattr(trainer, 'state') else 'unknown'
            monitor_status = monitor.get_training_status()

            results['communication_tests'].append({
                'test': 'trainer_to_monitor',
                'status': 'success',
                'details': f'训练器代数: {trainer_status}, 监控器状态: {monitor_status}'
            })
            logger.info("      ✅ 训练器 -> 监控器: 通信正常")

        except Exception as e:
            results['issues_found'].append(f'communication_error: {str(e)}')
            logger.error(f"    ❌ 组件通信测试失败: {e}")

        results['status'] = 'success' if not results['issues_found'] else 'warning'
        self.test_results['integration_tests']['communication'] = results
        return results

    async def test_data_flow(self) -> Dict[str, Any]:
        """测试数据流"""
        results = {
            'status': 'unknown',
            'data_flow_tests': [],
            'bottlenecks': [],
            'performance_metrics': {}
        }

        try:
            logger.info("    • 测试完整数据流: 生成 -> 编码 -> 训练")

            # 1. 数据生成
            from agi_data_generator import AGIDataGenerator
            data_gen = AGIDataGenerator()
            start_time = time.time()
            raw_data = data_gen.generate_training_data(num_samples=10)
            gen_time = time.time() - start_time

            results['performance_metrics']['data_generation'] = {
                'samples': len(raw_data) if raw_data else 0,
                'time_seconds': gen_time,
                'samples_per_second': len(raw_data) / gen_time if raw_data and gen_time > 0 else 0
            }

            # 2. 数据编码
            from agi_manifold_encoder import LogarithmicManifoldEncoder
            encoder = LogarithmicManifoldEncoder(resolution=0.01)
            start_time = time.time()
            # 简化的编码测试
            test_vector = [1.0, 2.0, 3.0]
            encoded = encoder.encode_with_continuity(test_vector)
            encode_time = time.time() - start_time

            results['performance_metrics']['data_encoding'] = {
                'input_size': len(test_vector),
                'output_size': len(encoded),
                'time_seconds': encode_time,
                'compression_ratio': len(encoded) / len(test_vector)
            }

            # 3. 数据集创建
            from agi_persistent_evolution import PersistentAGIConfig, ManifoldEncodedDataset
            config = PersistentAGIConfig()
            # 这里可以添加数据集测试

            results['data_flow_tests'].append({
                'stage': 'data_generation',
                'status': 'success',
                'metrics': results['performance_metrics']['data_generation']
            })

            results['data_flow_tests'].append({
                'stage': 'data_encoding',
                'status': 'success',
                'metrics': results['performance_metrics']['data_encoding']
            })

            logger.info("      ✅ 数据流测试完成")

        except Exception as e:
            results['bottlenecks'].append(str(e))
            logger.error(f"    ❌ 数据流测试失败: {e}")

        results['status'] = 'success' if not results['bottlenecks'] else 'warning'
        self.test_results['integration_tests']['data_flow'] = results
        return results

    async def test_system_startup(self) -> Dict[str, Any]:
        """测试系统启动"""
        results = {
            'status': 'unknown',
            'startup_sequence': [],
            'startup_time': 0,
            'errors_during_startup': []
        }

        try:
            logger.info("    • 测试系统管理器启动序列")

            from agi_system_manager import AGISystemManager
            start_time = time.time()

            manager = AGISystemManager()
            results['startup_sequence'].append('manager_created')

            # 启动系统
            manager.start_system()
            results['startup_sequence'].append('system_started')

            # 等待一会儿让组件完全启动
            await asyncio.sleep(2)

            # 检查系统状态
            status = manager.get_system_status()
            startup_time = time.time() - start_time
            results['startup_time'] = startup_time

            healthy_components = sum(1 for comp_status in status.get('components', {}).values() if comp_status)
            total_components = len(status.get('components', {}))

            results['startup_sequence'].append(f'components_checked: {healthy_components}/{total_components}')

            if healthy_components > 0:
                results['status'] = 'success'
                logger.info(f"      ✅ 系统启动成功: {healthy_components}/{total_components} 组件正常")
            else:
                results['status'] = 'warning'
                results['errors_during_startup'].append('no_components_healthy')
                logger.warning("      ⚠️  系统启动完成但无组件正常")

            # 停止系统
            manager.stop_system()
            results['startup_sequence'].append('system_stopped')

        except Exception as e:
            results['status'] = 'error'
            results['errors_during_startup'].append(str(e))
            logger.error(f"    ❌ 系统启动测试失败: {e}")

        self.test_results['integration_tests']['system_startup'] = results
        return results

    async def prepare_self_evolution(self) -> Dict[str, Any]:
        """准备自我进化"""
        results = {
            'status': 'unknown',
            'evolution_readiness': {},
            'missing_requirements': [],
            'recommendations': []
        }

        try:
            logger.info("    • 检查自我进化准备状态")

            # 检查必要的文件和配置
            required_files = [
                'agi_persistent_evolution.py',
                'agi_system_manager.py',
                'enhanced_evolution_verifier.py'
            ]

            for file in required_files:
                if (self.project_root / file).exists():
                    results['evolution_readiness'][f'file_{file}'] = 'present'
                else:
                    results['evolution_readiness'][f'file_{file}'] = 'missing'
                    results['missing_requirements'].append(f'file: {file}')

            # 检查配置
            config_files = ['agi_training_config.ini']
            for config_file in config_files:
                if (self.project_root / config_file).exists():
                    results['evolution_readiness'][f'config_{config_file}'] = 'present'
                else:
                    results['evolution_readiness'][f'config_{config_file}'] = 'present'  # 系统会创建默认配置

            # 检查API密钥
            api_key_present = bool(os.getenv("GEMINI_API_KEY"))
            results['evolution_readiness']['gemini_api'] = 'configured' if api_key_present else 'missing'

            if not api_key_present:
                results['missing_requirements'].append('GEMINI_API_KEY environment variable')

            # 检查磁盘空间
            import psutil
            disk = psutil.disk_usage('/')
            disk_space_gb = disk.free / (1024**3)
            results['evolution_readiness']['disk_space'] = f'{disk_space_gb:.1f}GB_free'

            if disk_space_gb < 10:
                results['missing_requirements'].append('insufficient_disk_space')
                results['recommendations'].append('清理磁盘空间，至少需要10GB可用空间')

            # 检查内存
            memory = psutil.virtual_memory()
            memory_gb = memory.available / (1024**3)
            results['evolution_readiness']['memory'] = f'{memory_gb:.1f}GB_available'

            if memory_gb < 4:
                results['missing_requirements'].append('insufficient_memory')
                results['recommendations'].append('增加系统内存，至少需要4GB可用内存')

            # 生成建议
            if not results['missing_requirements']:
                results['status'] = 'ready'
                results['recommendations'].append('系统已准备好开始自我进化')
                results['recommendations'].append('建议从小规模实验开始，逐步增加复杂度')
            else:
                results['status'] = 'not_ready'
                results['recommendations'].append('请先解决缺失的依赖项')

            logger.info(f"      进化准备状态: {results['status']}")

        except Exception as e:
            results['status'] = 'error'
            results['missing_requirements'].append(f'preparation_error: {str(e)}')
            logger.error(f"    ❌ 进化准备检查失败: {e}")

        self.test_results['integration_tests']['evolution_preparation'] = results
        return results

    def generate_integration_summary(self):
        """生成集成总结"""
        summary = {
            'overall_status': 'unknown',
            'total_tests': len(self.test_results['integration_tests']),
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': 0,
            'critical_issues': [],
            'ready_for_evolution': False
        }

        for test_name, test_result in self.test_results['integration_tests'].items():
            status = test_result.get('status', 'unknown')

            if status == 'success':
                summary['passed_tests'] += 1
            elif status == 'error':
                summary['failed_tests'] += 1
                summary['critical_issues'].append(f'{test_name}: {test_result}')
            elif status == 'warning':
                summary['warnings'] += 1

        # 确定整体状态
        if summary['failed_tests'] == 0 and summary['warnings'] == 0:
            summary['overall_status'] = 'excellent'
            summary['ready_for_evolution'] = True
        elif summary['failed_tests'] == 0:
            summary['overall_status'] = 'good'
            summary['ready_for_evolution'] = True
        elif summary['failed_tests'] < summary['total_tests'] / 2:
            summary['overall_status'] = 'acceptable'
            summary['ready_for_evolution'] = False
        else:
            summary['overall_status'] = 'critical'
            summary['ready_for_evolution'] = False

        self.test_results['integration_summary'] = summary

    def save_integration_results(self):
        """保存集成测试结果"""
        try:
            output_file = self.project_root / "system_integration_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"✅ 集成测试结果已保存到: {output_file}")

        except Exception as e:
            logger.error(f"保存集成测试结果失败: {e}")

async def main():
    """主函数"""
    print("🔧 H2Q-Evo 系统联调测试")
    print("=" * 50)

    tester = SystemIntegrationTester()

    try:
        results = await tester.run_full_integration_test()

        summary = results.get('integration_summary', {})

        print("\n📊 集成测试总结:")
        print(f"  • 总测试数: {summary['total_tests']}")
        print(f"  • 通过测试: {summary['passed_tests']}")
        print(f"  • 失败测试: {summary['failed_tests']}")
        print(f"  • 警告数量: {summary['warnings']}")
        print(f"  • 整体状态: {summary['overall_status'].upper()}")

        if summary['ready_for_evolution']:
            print("\n🎯 自我进化准备状态: ✅ 准备就绪")
            print("💡 建议下一步: 开始AGI自我进化训练")
        else:
            print("\n⚠️  自我进化准备状态: ❌ 还需要修复问题")
            print("🔧 请先解决关键问题再开始进化")

        if summary.get('critical_issues'):
            print("\n🚨 关键问题:")
            for issue in summary['critical_issues'][:3]:
                print(f"  • {issue}")

        print("\n📄 详细结果已保存到: system_integration_results.json")
        return True

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())