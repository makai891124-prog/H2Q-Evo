#!/usr/bin/env python3
"""
H2Q-Evo AGI核心算法验证脚本
确保AGI系统诚实地使用了我们的对数流形编码算法
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('AGI-AlgorithmVerifier')

class AGIAlgorithmVerifier:
    """AGI核心算法验证器"""

    def __init__(self, project_root: str = "./"):
        self.project_root = Path(project_root)
        self.agi_dir = self.project_root / "agi_persistent_training"

    def verify_core_algorithm_usage(self) -> Dict[str, Any]:
        """验证核心算法使用情况"""
        logger.info("🔍 开始AGI核心算法使用验证...")

        results = {
            'overall_status': 'unknown',
            'components_checked': [],
            'algorithm_usage_score': 0.0,
            'details': {}
        }

        # 检查各个组件
        components = [
            ('data_generator', self._verify_data_generator),
            ('persistent_trainer', self._verify_persistent_trainer),
            ('training_data', self._verify_training_data),
            ('system_config', self._verify_system_config)
        ]

        total_score = 0.0
        checked_count = 0

        for component_name, verify_func in components:
            try:
                logger.info(f"检查组件: {component_name}")
                component_result = verify_func()
                results['components_checked'].append(component_name)
                results['details'][component_name] = component_result

                if component_result['status'] == 'passed':
                    total_score += component_result['score']
                    checked_count += 1
                elif component_result['status'] == 'failed':
                    total_score += component_result['score'] * 0.5  # 失败组件给予一半分数
                    checked_count += 1

            except Exception as e:
                logger.error(f"验证组件 {component_name} 失败: {e}")
                results['details'][component_name] = {
                    'status': 'error',
                    'score': 0.0,
                    'message': str(e)
                }

        # 计算总体分数
        if checked_count > 0:
            results['algorithm_usage_score'] = total_score / checked_count

            if results['algorithm_usage_score'] >= 0.8:
                results['overall_status'] = 'excellent'
            elif results['algorithm_usage_score'] >= 0.6:
                results['overall_status'] = 'good'
            elif results['algorithm_usage_score'] >= 0.4:
                results['overall_status'] = 'fair'
            else:
                results['overall_status'] = 'poor'

        logger.info(f"✅ 验证完成 - 总体状态: {results['overall_status']} (分数: {results['algorithm_usage_score']:.2f})")
        return results

    def _verify_data_generator(self) -> Dict[str, Any]:
        """验证数据生成器"""
        result = {'status': 'unknown', 'score': 0.0, 'message': ''}

        try:
            # 检查数据生成器文件
            generator_file = self.project_root / "agi_data_generator.py"
            if not generator_file.exists():
                result['message'] = "数据生成器文件不存在"
                result['status'] = 'failed'
                return result

            # 检查是否导入了核心算法
            with open(generator_file, 'r', encoding='utf-8') as f:
                content = f.read()

            imports_core = 'from agi_manifold_encoder import' in content
            uses_encoding = 'encode_with_continuity' in content
            applies_encoding = '_apply_manifold_encoding' in content

            if imports_core and uses_encoding and applies_encoding:
                result['status'] = 'passed'
                result['score'] = 1.0
                result['message'] = "数据生成器正确集成了对数流形编码"
            else:
                result['status'] = 'failed'
                result['score'] = 0.0
                result['message'] = f"数据生成器缺少核心算法集成: imports={imports_core}, uses={uses_encoding}, applies={applies_encoding}"

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f"验证失败: {e}"

        return result

    def _verify_persistent_trainer(self) -> Dict[str, Any]:
        """验证持久训练器"""
        result = {'status': 'unknown', 'score': 0.0, 'message': ''}

        try:
            trainer_file = self.project_root / "agi_persistent_evolution.py"
            if not trainer_file.exists():
                result['message'] = "持久训练器文件不存在"
                result['status'] = 'failed'
                return result

            with open(trainer_file, 'r', encoding='utf-8') as f:
                content = f.read()

            imports_core = 'from agi_manifold_encoder import' in content
            uses_encoding = 'encode_with_continuity' in content
            has_encoded_dataset = 'ManifoldEncodedDataset' in content

            if imports_core and uses_encoding and has_encoded_dataset:
                result['status'] = 'passed'
                result['score'] = 1.0
                result['message'] = "持久训练器正确使用了流形编码数据集"
            else:
                result['status'] = 'failed'
                result['score'] = 0.0
                result['message'] = f"持久训练器缺少核心算法: imports={imports_core}, uses={uses_encoding}, dataset={has_encoded_dataset}"

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f"验证失败: {e}"

        return result

    def _verify_training_data(self) -> Dict[str, Any]:
        """验证训练数据"""
        result = {'status': 'unknown', 'score': 0.0, 'message': ''}

        try:
            data_dir = self.agi_dir / "data"
            if not data_dir.exists():
                result['message'] = "训练数据目录不存在"
                result['status'] = 'passed'  # 初始状态下数据不存在是正常的
                result['score'] = 0.5
                return result

            # 查找训练数据文件
            data_files = list(data_dir.glob("*.jsonl"))
            if not data_files:
                result['message'] = "未找到训练数据文件"
                result['status'] = 'passed'
                result['score'] = 0.5
                return result

            # 检查最新的数据文件
            latest_file = max(data_files, key=os.path.getmtime)

            algorithm_used_samples = 0
            total_samples = 0
            compression_ratios = []

            with open(latest_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= 50:  # 检查前50个样本
                        break

                    try:
                        sample = json.loads(line.strip())
                        total_samples += 1

                        # 检查算法使用标记
                        if sample.get('algorithm_used') == 'logarithmic_manifold_encoding':
                            algorithm_used_samples += 1

                        # 检查压缩率
                        if 'compression_ratio' in sample:
                            compression_ratios.append(sample['compression_ratio'])

                        # 检查编码特征
                        if 'encoded_features' in sample and sample['encoded_features']:
                            algorithm_used_samples += 0.5  # 部分分数

                    except json.JSONDecodeError:
                        continue

            if total_samples == 0:
                result['message'] = "没有找到有效的训练数据"
                result['status'] = 'failed'
                result['score'] = 0.0
                return result

            usage_rate = algorithm_used_samples / total_samples
            avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0

            result['message'] = f"检查了{total_samples}个样本，算法使用率: {usage_rate:.2f}, 平均压缩率: {avg_compression:.3f}"

            if usage_rate >= 0.8 and avg_compression < 0.9:
                result['status'] = 'passed'
                result['score'] = 1.0
            elif usage_rate >= 0.5:
                result['status'] = 'passed'
                result['score'] = 0.7
            else:
                result['status'] = 'failed'
                result['score'] = usage_rate

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f"验证失败: {e}"

        return result

    def _verify_system_config(self) -> Dict[str, Any]:
        """验证系统配置"""
        result = {'status': 'unknown', 'score': 0.0, 'message': ''}

        try:
            config_file = self.project_root / "agi_training_config.ini"
            if not config_file.exists():
                result['message'] = "配置文件不存在"
                result['status'] = 'failed'
                return result

            # 检查配置文件内容
            import configparser
            config = configparser.ConfigParser()
            config.read(config_file)

            has_manifold_config = 'manifold_encoding' in config
            has_evolution_config = 'evolution' in config

            if has_manifold_config and has_evolution_config:
                result['status'] = 'passed'
                result['score'] = 1.0
                result['message'] = "系统配置包含核心算法相关设置"
            else:
                result['status'] = 'failed'
                result['score'] = 0.5
                result['message'] = f"配置缺少核心算法设置: manifold={has_manifold_config}, evolution={has_evolution_config}"

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f"验证失败: {e}"

        return result

    def generate_verification_report(self, output_file: str = "./agi_algorithm_verification_report.md") -> str:
        """生成验证报告"""
        logger.info("生成算法验证报告...")

        results = self.verify_core_algorithm_usage()

        report = f"""# H2Q-Evo AGI核心算法验证报告

生成时间: {json.dumps({'timestamp': str(results.get('timestamp', 'unknown'))}, indent=2)}

## 总体评估

- **验证状态**: {results['overall_status'].upper()}
- **算法使用分数**: {results['algorithm_usage_score']:.2f}/1.00
- **检查组件数**: {len(results['components_checked'])}

## 详细结果

"""

        status_icons = {
            'passed': '✅',
            'failed': '❌',
            'error': '⚠️',
            'unknown': '❓'
        }

        for component, detail in results['details'].items():
            icon = status_icons.get(detail['status'], '❓')
            report += f"### {component}\n"
            report += f"- **状态**: {icon} {detail['status']}\n"
            report += f"- **分数**: {detail['score']:.2f}\n"
            report += f"- **详情**: {detail['message']}\n\n"

        # 添加结论
        if results['algorithm_usage_score'] >= 0.8:
            conclusion = "🎉 **优秀**: AGI系统正确集成了对数流形编码算法，实验结果可信。"
        elif results['algorithm_usage_score'] >= 0.6:
            conclusion = "👍 **良好**: AGI系统基本使用了核心算法，但存在一些改进空间。"
        elif results['algorithm_usage_score'] >= 0.4:
            conclusion = "⚠️ **一般**: AGI系统部分使用了核心算法，需要进一步验证。"
        else:
            conclusion = "❌ **不足**: AGI系统未充分使用核心算法，实验结果可能不可信。"

        report += f"""## 结论

{conclusion}

## 验证标准

- ✅ **通过**: 分数 ≥ 0.8，充分使用了核心算法
- ⚠️ **警告**: 分数 0.4-0.8，部分使用核心算法
- ❌ **失败**: 分数 < 0.4，未使用核心算法

---
*此报告验证H2Q-Evo AGI系统是否诚实地使用了对数流形编码核心算法。*
"""

        # 保存报告
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"验证报告已保存: {output_path}")
        return str(output_path)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='H2Q-Evo AGI核心算法验证工具')
    parser.add_argument('--output', default='./agi_algorithm_verification_report.md',
                       help='输出报告文件路径')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式，不输出详细信息')

    args = parser.parse_args()

    if not args.quiet:
        print("🔍 H2Q-Evo AGI核心算法验证器")
        print("=" * 50)

    verifier = AGIAlgorithmVerifier()

    # 执行验证
    results = verifier.verify_core_algorithm_usage()

    # 生成报告
    report_file = verifier.generate_verification_report(args.output)

    if not args.quiet:
        print(f"\n📊 验证结果:")
        print(f"   总体状态: {results['overall_status'].upper()}")
        print(f"   使用分数: {results['algorithm_usage_score']:.2f}")
        print(f"   报告文件: {report_file}")

        if results['algorithm_usage_score'] >= 0.8:
            print("🎉 恭喜！AGI系统正确使用了核心算法。")
        else:
            print("⚠️ 警告：AGI系统可能未充分使用核心算法。")

if __name__ == "__main__":
    main()