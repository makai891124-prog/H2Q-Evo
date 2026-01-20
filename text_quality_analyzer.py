#!/usr/bin/env python3
"""
H2Q-Evo 文本质量分析与改进系统
===================================

分析不可读文本问题并提供解决方案
- 质量评估指标
- 改进策略
- 对比测试
- 自由进化建议
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import re
import math
from collections import Counter

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
H2Q_PROJECT = PROJECT_ROOT / "h2q_project"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(H2Q_PROJECT))

from local_long_text_generator import LocalLongTextGenerator


class TextQualityAnalyzer:
    """文本质量分析器"""

    def __init__(self):
        self.quality_metrics = {}

    def analyze_text_quality(self, text: str) -> Dict[str, float]:
        """分析文本质量"""
        metrics = {}

        # 1. 字符多样性
        unique_chars = len(set(text))
        total_chars = len(text)
        metrics['char_diversity'] = unique_chars / total_chars if total_chars > 0 else 0

        # 2. 词汇多样性（简单估计）
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = len(set(words))
        total_words = len(words)
        metrics['word_diversity'] = unique_words / total_words if total_words > 0 else 0

        # 3. 重复模式检测
        repeated_patterns = self._detect_repeated_patterns(text)
        metrics['repetition_score'] = min(1.0, repeated_patterns / 10.0)  # 归一化

        # 4. 可读性评分（基于字符分布）
        readable_chars = sum(1 for c in text if c.isalnum() or c in ' \n\t.,!?;:"\'')
        metrics['readability'] = readable_chars / total_chars if total_chars > 0 else 0

        # 5. 结构完整性（句子完整性）
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
        metrics['structural_integrity'] = complete_sentences / len(sentences) if sentences else 0

        # 6. 整体质量评分
        metrics['overall_quality'] = (
            metrics['char_diversity'] * 0.2 +
            metrics['word_diversity'] * 0.3 +
            (1 - metrics['repetition_score']) * 0.2 +
            metrics['readability'] * 0.2 +
            metrics['structural_integrity'] * 0.1
        )

        return metrics

    def _detect_repeated_patterns(self, text: str, min_length: int = 3) -> int:
        """检测重复模式"""
        patterns = {}
        text_lower = text.lower()

        for i in range(len(text_lower) - min_length + 1):
            pattern = text_lower[i:i+min_length]
            if pattern in patterns:
                patterns[pattern] += 1
            else:
                patterns[pattern] = 1

        # 计算重复模式的严重程度
        repeated_count = sum(count - 1 for count in patterns.values() if count > 1)
        return repeated_count


class TextGenerationComparator:
    """文本生成对比器"""

    def __init__(self):
        self.analyzer = TextQualityAnalyzer()
        self.generators = {}

    def add_generator(self, name: str, generator_func):
        """添加生成器"""
        self.generators[name] = generator_func

    def compare_generators(self, prompts: List[str], max_length: int = 200) -> Dict[str, Any]:
        """对比不同生成器的性能"""
        results = {}

        for prompt in prompts:
            print(f"\n🎯 测试提示: {prompt}")
            print("-" * 50)

            prompt_results = {}

            for gen_name, generator in self.generators.items():
                try:
                    generated_text = generator(prompt, max_length)
                    quality_metrics = self.analyzer.analyze_text_quality(generated_text)

                    prompt_results[gen_name] = {
                        'text': generated_text,
                        'metrics': quality_metrics,
                        'length': len(generated_text)
                    }

                    print(f"\n🤖 {gen_name}:")
                    print(f"  📏 长度: {len(generated_text)} 字符")
                    print(f"  🎯 质量评分: {quality_metrics['overall_quality']:.3f}")
                    print(f"  📝 字符多样性: {quality_metrics['char_diversity']:.3f}")
                    print(f"  🔄 重复度: {quality_metrics['repetition_score']:.3f}")
                    print(f"  📖 可读性: {quality_metrics['readability']:.3f}")
                    print(f"  🏗️ 结构完整性: {quality_metrics['structural_integrity']:.3f}")
                    print(f"  💬 生成内容: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")

                except Exception as e:
                    print(f"❌ {gen_name} 生成失败: {e}")
                    prompt_results[gen_name] = {'error': str(e)}

            results[prompt] = prompt_results

        return results

    def generate_improvement_report(self, comparison_results: Dict[str, Any]) -> str:
        """生成改进报告"""
        report = []
        report.append("# H2Q-Evo 文本生成质量改进报告")
        report.append("=" * 50)

        # 汇总统计
        generator_stats = {}
        for prompt, gen_results in comparison_results.items():
            for gen_name, result in gen_results.items():
                if 'error' not in result:
                    if gen_name not in generator_stats:
                        generator_stats[gen_name] = []
                    generator_stats[gen_name].append(result['metrics']['overall_quality'])

        report.append("\n## 📊 生成器性能汇总")
        for gen_name, scores in generator_stats.items():
            avg_score = sum(scores) / len(scores)
            report.append(f"- **{gen_name}**: 平均质量评分 {avg_score:.3f}")

        # 问题分析
        report.append("\n## 🔍 质量问题分析")
        report.append("基于测试结果，发现的主要问题：")

        # 分析第一个生成器的结果作为基准
        first_gen = list(generator_stats.keys())[0]
        if generator_stats[first_gen]:
            avg_metrics = {}
            for prompt, gen_results in comparison_results.items():
                if first_gen in gen_results and 'error' not in gen_results[first_gen]:
                    metrics = gen_results[first_gen]['metrics']
                    for key, value in metrics.items():
                        if key not in avg_metrics:
                            avg_metrics[key] = []
                        avg_metrics[key].append(value)

            for key, values in avg_metrics.items():
                avg_value = sum(values) / len(values)
                if key == 'char_diversity' and avg_value < 0.1:
                    report.append(f"- **字符多样性不足** ({avg_value:.3f}): 文本中重复字符过多")
                elif key == 'repetition_score' and avg_value > 0.5:
                    report.append(f"- **重复模式严重** ({avg_value:.3f}): 存在大量重复的文本模式")
                elif key == 'readability' and avg_value < 0.7:
                    report.append(f"- **可读性差** ({avg_value:.3f}): 包含太多不可读字符")
                elif key == 'structural_integrity' and avg_value < 0.3:
                    report.append(f"- **结构不完整** ({avg_value:.3f}): 句子结构残缺")

        # 改进建议
        report.append("\n## 💡 改进建议")
        report.append("### 1. 模型架构改进")
        report.append("- 使用更大的词汇表（从256扩展到50,000+）")
        report.append("- 实现BPE或WordPiece分词")
        report.append("- 增加模型参数和层数")
        report.append("- 使用预训练权重初始化")

        report.append("\n### 2. 训练数据优化")
        report.append("- 使用更高质量、多样化的训练数据")
        report.append("- 增加数据量（从几KB扩展到GB级别）")
        report.append("- 实现数据增强技术")
        report.append("- 平衡不同领域的文本分布")

        report.append("\n### 3. 解码策略改进")
        report.append("- 实现Top-k和Top-p采样")
        report.append("- 添加温度控制")
        report.append("- 使用重复惩罚机制")
        report.append("- 实现长度惩罚")

        report.append("\n### 4. 量子增强集成")
        report.append("- 集成H2Q的量子推理能力")
        report.append("- 使用全纯流中间件进行推理增强")
        report.append("- 实现量子决策引擎辅助生成")
        report.append("- 利用拓扑学原理优化生成过程")

        report.append("\n### 5. 后处理技术")
        report.append("- 实现文本后处理和清理")
        report.append("- 添加语法检查和修正")
        report.append("- 使用语言模型进行重排序")
        report.append("- 实现多样性增强技术")

        return "\n".join(report)


def create_baseline_generators():
    """创建基准生成器进行对比"""
    comparator = TextGenerationComparator()

    # 1. 原始本地生成器
    original_generator = LocalLongTextGenerator()
    comparator.add_generator("原始本地生成器", lambda prompt, length: original_generator.generate_long_text(prompt, max_tokens=length))

    # 2. 随机字符生成器（作为基准）
    def random_char_generator(prompt: str, max_length: int) -> str:
        import random
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:"
        result = prompt
        for _ in range(max_length - len(prompt)):
            result += random.choice(chars)
        return result

    comparator.add_generator("随机字符生成器", random_char_generator)

    # 3. 简单模式重复生成器
    def pattern_generator(prompt: str, max_length: int) -> str:
        base_patterns = ["人工智能", "机器学习", "深度学习", "量子计算"]
        result = prompt
        while len(result) < max_length:
            pattern = base_patterns[len(result) % len(base_patterns)]
            result += " " + pattern
        return result[:max_length]

    comparator.add_generator("模式重复生成器", pattern_generator)

    return comparator


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🔍 H2Q-Evo 文本质量分析与改进系统")
    print("="*60)
    print("🎯 目标：分析不可读文本问题，提供改进方案")
    print("🛡️ 安全：完全离线，无联网")
    print("="*60 + "\n")

    # 创建生成器对比器
    comparator = create_baseline_generators()

    # 测试提示
    test_prompts = [
        "人工智能的发展",
        "量子计算原理",
        "机器学习算法",
        "深度学习模型"
    ]

    # 运行对比测试
    print("🧪 运行生成质量对比测试...")
    comparison_results = comparator.compare_generators(test_prompts, max_length=150)

    # 生成改进报告
    print("\n📋 生成改进报告...")
    improvement_report = comparator.generate_improvement_report(comparison_results)

    # 保存报告
    report_path = PROJECT_ROOT / "text_quality_improvement_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(improvement_report)

    print(f"💾 改进报告已保存: {report_path}")

    # 显示关键发现
    print("\n🎯 关键发现:")
    print("1. **字符级模型限制**: 当前使用256字符词汇表，无法生成有意义的词汇")
    print("2. **训练数据不足**: 数据量小，质量低，导致模型无法学习语言模式")
    print("3. **缺少预训练**: 从随机初始化开始，收敛困难")
    print("4. **解码策略简单**: 没有使用先进的采样技术")
    print("5. **量子增强未集成**: 没有利用H2Q的核心量子推理能力")

    print("\n🚀 解决方案:")
    print("1. **实现高级分词**: 从字符级升级到子词级（BPE）")
    print("2. **扩大训练数据**: 创建高质量、多样化的训练数据集")
    print("3. **集成预训练**: 利用现有H2Q模型权重")
    print("4. **改进解码**: 实现Top-k、Top-p采样和重复惩罚")
    print("5. **量子推理增强**: 集成H2Q的量子决策引擎")

    print("\n🧬 自由进化路径:")
    print("1. **阶段1**: 实现BPE分词和更大的词汇表")
    print("2. **阶段2**: 创建大规模高质量训练数据集")
    print("3. **阶段3**: 集成H2Q预训练模型和量子推理")
    print("4. **阶段4**: 实现先进的解码策略和后处理")
    print("5. **阶段5**: 自动化质量评估和持续改进")

    print(f"\n📖 详细改进方案请查看: {report_path}")


if __name__ == "__main__":
    main()