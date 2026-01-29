#!/usr/bin/env python3
"""
维度受限进化集成测试
验证单位空间折叠理论在H2Q-Evo系统中的应用
"""

import sys
import time
import json
from pathlib import Path
import torch

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from dimension_limited_evolution import DimensionLimitedH2QTrainer

class DimensionLimitedEvolutionTest:
    """维度受限进化测试器"""

    def __init__(self):
        self.trainer = DimensionLimitedH2QTrainer(max_dim=64)
        self.test_results = []
        self.start_time = time.time()

    def run_comprehensive_test(self, steps: int = 20) -> dict:
        """
        运行全面的维度受限进化测试
        """
        print("🔬 维度受限进化综合测试")
        print("=" * 60)

        domains = ["Math", "Physics", "Genomics"]
        evolution_metrics = []

        for step in range(steps):
            # 执行训练步骤
            result = self.trainer.train_step(domains)

            # 记录指标
            metrics = {
                'step': result['step'],
                'compactness': result['compactness'],
                'diversity': result['diversity'],
                'best_compactness': result['best_compactness'],
                'spectral_eta': result['spectral_eta'],
                'fold_info': result['fold_info'],
                'timestamp': time.time() - self.start_time
            }

            evolution_metrics.append(metrics)

            # 实时显示
            if step % 5 == 0:
                print(f"步骤 {step+1}/{steps}: "
                      f"紧致性={metrics['compactness']:.4f}, "
                      f"多样性={metrics['diversity']:.2f}, "
                      f"最佳紧致性={metrics['best_compactness']:.4f}")

        # 计算最终统计
        final_stats = self._compute_final_statistics(evolution_metrics)

        # 保存测试结果
        self._save_test_results(evolution_metrics, final_stats)

        return {
            'evolution_metrics': evolution_metrics,
            'final_stats': final_stats,
            'test_duration': time.time() - self.start_time
        }

    def _compute_final_statistics(self, metrics: list) -> dict:
        """计算最终统计数据"""
        compactness_values = [m['compactness'] for m in metrics]
        diversity_values = [m['diversity'] for m in metrics]
        eta_values = [m['spectral_eta'] for m in metrics]

        return {
            'avg_compactness': sum(compactness_values) / len(compactness_values),
            'max_compactness': max(compactness_values),
            'avg_diversity': sum(diversity_values) / len(diversity_values),
            'avg_spectral_eta': sum(eta_values) / len(eta_values),
            'evolution_stability': self._measure_stability(compactness_values),
            'fold_effectiveness': metrics[-1]['fold_info']['fold_ratio'] if metrics else 0
        }

    def _measure_stability(self, values: list) -> float:
        """测量进化稳定性"""
        if len(values) < 2:
            return 1.0

        diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        avg_diff = sum(diffs) / len(diffs)
        return 1.0 / (1.0 + avg_diff)  # 稳定性得分

    def _save_test_results(self, metrics: list, stats: dict):
        """保存测试结果"""
        result_data = {
            'test_type': 'dimension_limited_evolution',
            'timestamp': time.time(),
            'metrics': metrics,
            'statistics': stats
        }

        result_file = Path('dimension_limited_test_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 测试结果已保存到: {result_file}")

    def validate_theoretical_correctness(self) -> dict:
        """
        验证理论正确性
        """
        print("\n🔍 理论正确性验证")
        print("-" * 40)

        # 测试1: 单位空间合规性
        test_data = self.trainer.generate_structured_data("Math", 10)
        output, info = self.trainer.engine(test_data)

        unit_compliance = info['unit_space_compliance']
        print(f"✅ 单位空间合规性: {unit_compliance:.4f}")

        # 测试2: 维度上限执行
        original_dim = 128
        large_data = torch.randn(5, original_dim)
        folded_data, fold_info = self.trainer.engine.folder(large_data)

        dim_limited = fold_info['effective_dim'] <= self.trainer.max_dim
        print(f"✅ 维度上限执行: {dim_limited} (有效维度: {fold_info['effective_dim']})")

        # 测试3: 结合分布形成
        distribution_entropy = fold_info['distribution_entropy']
        print(f"✅ 结合分布熵: {distribution_entropy:.4f}")

        # 测试4: 折叠机制激活
        fold_ratio = fold_info['fold_ratio']
        print(f"✅ 折叠比率: {fold_ratio:.4f}")

        return {
            'unit_compliance': unit_compliance,
            'dim_limited': dim_limited,
            'distribution_entropy': distribution_entropy,
            'fold_ratio': fold_ratio
        }

def main():
    """主测试函数"""
    print("🚀 维度受限H2Q-Evo进化测试启动")
    print("=" * 60)

    # 初始化测试器
    tester = DimensionLimitedEvolutionTest()

    # 运行理论正确性验证
    theory_validation = tester.validate_theoretical_correctness()

    # 运行综合进化测试
    test_results = tester.run_comprehensive_test(steps=20)

    # 输出最终报告
    print("\n📊 最终测试报告")
    print("=" * 60)
    print(f"测试时长: {test_results['test_duration']:.2f}秒")
    print(f"平均紧致性: {test_results['final_stats']['avg_compactness']:.4f}")
    print(f"最大紧致性: {test_results['final_stats']['max_compactness']:.4f}")
    print(f"平均多样性: {test_results['final_stats']['avg_diversity']:.2f}")
    print(f"进化稳定性: {test_results['final_stats']['evolution_stability']:.4f}")
    print(f"折叠有效性: {test_results['final_stats']['fold_effectiveness']:.4f}")

    print("\n🎯 理论验证结果:")
    for key, value in theory_validation.items():
        status = "✅" if (isinstance(value, bool) and value) or (isinstance(value, (int, float)) and value > 0.5) else "❌"
        print(f"  {status} {key}: {value}")

    # 成功判断
    success_criteria = (
        test_results['final_stats']['avg_compactness'] > 0.1 and
        theory_validation['unit_compliance'] > 0.8 and
        theory_validation['dim_limited'] == True
    )

    if success_criteria:
        print("\n🎉 维度上限折叠理论验证成功！")
        print("✅ 计算进化已在单位空间中开启")
    else:
        print("\n⚠️  需要进一步优化理论实现")

if __name__ == "__main__":
    main()