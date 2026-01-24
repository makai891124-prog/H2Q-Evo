#!/usr/bin/env python3
"""
H2Q-Evo AGI数据增强和生成系统
在进化过程中动态生成多样化的训练数据
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from agi_manifold_encoder import LogarithmicManifoldEncoder, CompressedAGIEncoder

logger = logging.getLogger('AGI-DataGenerator')

class AGIDataGenerator:
    """AGI数据生成器"""

    def __init__(self, config_path: str = "./agi_training_config.ini"):
        self.config = self._load_config(config_path)

        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.manifold_encoder = LogarithmicManifoldEncoder(resolution=self.config.get('encoding_resolution', 0.01))
        self.compressed_encoder = CompressedAGIEncoder()

        # 数据生成统计
        self.generated_samples = 0
        self.data_quality_scores = []

        logger.info("AGI数据生成器初始化完成")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        config = {}

        if os.path.exists(config_path):
            try:
                import configparser
                parser = configparser.ConfigParser()
                parser.read(config_path)

                # 读取配置
                if 'data' in parser:
                    config.update(dict(parser['data']))

                if 'manifold_encoding' in parser:
                    config.update(dict(parser['manifold_encoding']))

            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")

        # 默认配置
        config.setdefault('data_sources', ['mathematical_reasoning', 'code_generation', 'conversation', 'creative_writing'])
        config.setdefault('samples_per_source', 1000)
        config.setdefault('encoding_resolution', 0.01)

        # 确保数值类型正确
        if 'encoding_resolution' in config:
            try:
                config['encoding_resolution'] = float(config['encoding_resolution'])
            except (ValueError, TypeError):
                config['encoding_resolution'] = 0.01

        if 'samples_per_source' in config:
            try:
                config['samples_per_source'] = int(config['samples_per_source'])
            except (ValueError, TypeError):
                config['samples_per_source'] = 1000

        # 解析data_sources，如果是字符串则转换为列表
        if 'data_sources' in config and isinstance(config['data_sources'], str):
            # 移除方括号和引号，然后分割
            data_sources_str = config['data_sources'].strip('[]')
            data_sources = [s.strip().strip('"\'') for s in data_sources_str.split(',') if s.strip()]
            config['data_sources'] = data_sources

        return config

    def initialize_model(self, model_name: str = "microsoft/DialoGPT-medium"):
        """初始化语言模型"""
        logger.info(f"加载语言模型: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 设置为评估模式
            self.model.eval()

            logger.info("语言模型加载完成")

        except Exception as e:
            logger.error(f"加载语言模型失败: {e}")
            raise

    def generate_training_data(self, num_samples: int = 1000,
                             output_file: str = "./agi_persistent_training/data/generated_data.jsonl") -> str:
        """生成训练数据 - 内存优化版本，使用流式处理"""
        logger.info(f"开始生成{num_samples}条训练数据 (内存优化模式)")

        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_generated = 0
        batch_size = 10  # 每批处理10个样本，避免内存累积

        # 计算每个数据源的样本数
        data_sources = self.config.get('data_sources', ['mathematical_reasoning', 'conversation'])
        samples_per_source = num_samples // len(data_sources)

        # 打开文件用于流式写入
        with open(output_file, 'w', encoding='utf-8') as f:
            for source in data_sources:
                logger.info(f"生成{source}数据...")

                # 分批生成数据
                remaining_samples = samples_per_source
                while remaining_samples > 0:
                    current_batch = min(batch_size, remaining_samples)

                    if source == 'mathematical_reasoning':
                        source_data = self._generate_mathematical_data(current_batch)
                    elif source == 'code_generation':
                        source_data = self._generate_code_data(current_batch)
                    elif source == 'conversation':
                        source_data = self._generate_conversation_data(current_batch)
                    elif source == 'creative_writing':
                        source_data = self._generate_creative_data(current_batch)
                    else:
                        logger.warning(f"未知数据源: {source}")
                        break

                    # 立即应用数据增强和编码，然后保存
                    enhanced_data = self._apply_data_augmentation(source_data)
                    encoded_data = self._apply_manifold_encoding(enhanced_data)

                    # 流式保存到文件
                    for item in encoded_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                    total_generated += len(encoded_data)
                    remaining_samples -= current_batch

                    # 强制垃圾回收
                    del source_data, enhanced_data, encoded_data
                    import gc
                    gc.collect()

                    logger.info(f"已生成 {total_generated} 条样本 (内存已清理)")

        logger.info(f"数据生成完成，共{total_generated}条样本，已流式保存到: {output_file}")
        return output_file

    def _generate_mathematical_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成数学推理数据"""
        data = []

        math_templates = [
            "证明{concept}",
            "解释{concept}的原理",
            "计算{expression}的结果",
            "推导{formula}的公式",
            "分析{problem}的解法"
        ]

        concepts = [
            "勾股定理", "微积分基本定理", "质数定理", "欧拉公式",
            "对数化流形编码", "不动点理论", "黎曼几何", "张量分析",
            "概率论基础", "线性代数", "群论", "拓扑学"
        ]

        expressions = [
            "∫sin(x)dx", "d/dx(x²)", "lim(x→0) sin(x)/x",
            "∑(1/n²)", "∂f/∂x", "∇·F", "∇×F"
        ]

        for i in range(num_samples):
            template = np.random.choice(math_templates)

            if "{concept}" in template:
                concept = np.random.choice(concepts)
                question = template.format(concept=concept)
            elif "{expression}" in template:
                expression = np.random.choice(expressions)
                question = template.format(expression=expression)
            else:
                question = template

            # 生成答案 (简化版)
            answer = self._generate_mathematical_answer(question)

            data.append({
                'input': question,
                'output': answer,
                'type': 'mathematical_reasoning',
                'difficulty': np.random.choice(['basic', 'intermediate', 'advanced']),
                'topic': 'mathematics'
            })

        return data

    def _generate_code_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成代码生成数据"""
        data = []

        code_tasks = [
            "实现{algorithm}算法",
            "创建{structure}数据结构",
            "编写{function}函数",
            "设计{pattern}设计模式",
            "优化{problem}的解决方案"
        ]

        algorithms = ["快速排序", "二分查找", "深度优先搜索", "广度优先搜索", "动态规划"]
        structures = ["栈", "队列", "链表", "二叉树", "哈希表"]
        functions = ["文件读取", "网络请求", "数据库连接", "图像处理", "文本分析"]
        patterns = ["单例模式", "工厂模式", "观察者模式", "策略模式", "装饰器模式"]
        problems = ["斐波那契数列", "背包问题", "最短路径", "最大子数组", "字符串匹配"]

        for i in range(num_samples):
            template = np.random.choice(code_tasks)

            if "{algorithm}" in template:
                item = np.random.choice(algorithms)
            elif "{structure}" in template:
                item = np.random.choice(structures)
            elif "{function}" in template:
                item = np.random.choice(functions)
            elif "{pattern}" in template:
                item = np.random.choice(patterns)
            elif "{problem}" in template:
                item = np.random.choice(problems)
            else:
                item = "通用"

            task = template.format(
                algorithm=item, structure=item, function=item,
                pattern=item, problem=item
            )

            # 生成代码答案
            code_answer = self._generate_code_answer(task)

            data.append({
                'input': f"请用Python{task}",
                'output': code_answer,
                'type': 'code_generation',
                'language': 'python',
                'difficulty': np.random.choice(['easy', 'medium', 'hard'])
            })

        return data

    def _generate_conversation_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成对话数据"""
        data = []

        conversation_starters = [
            "你好，介绍一下你自己",
            "什么是人工智能？",
            "如何学习编程？",
            "讨论机器学习的未来",
            "解释深度学习",
            "什么是AGI？",
            "讨论意识与智能",
            "人工智能的伦理问题",
            "机器学习的应用场景",
            "未来的工作会怎样变化"
        ]

        for i in range(num_samples):
            starter = np.random.choice(conversation_starters)

            # 生成对话响应
            response = self._generate_conversation_response(starter)

            data.append({
                'input': starter,
                'output': response,
                'type': 'conversation',
                'context': 'general_ai_discussion'
            })

        return data

    def _generate_creative_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成创意写作数据"""
        data = []

        creative_prompts = [
            "写一个科幻故事",
            "创作一首现代诗",
            "设计一个游戏情节",
            "写一个商业计划",
            "创作一个短篇小说开头",
            "设计一个未来城市",
            "写一个哲学思考",
            "创作一个音乐歌词",
            "设计一个创新产品",
            "写一个历史故事"
        ]

        for i in range(num_samples):
            prompt = np.random.choice(creative_prompts)

            # 生成创意内容
            creative_output = self._generate_creative_response(prompt)

            data.append({
                'input': f"请{prompt}",
                'output': creative_output,
                'type': 'creative_writing',
                'genre': self._classify_genre(prompt)
            })

        return data

    def _generate_mathematical_answer(self, question: str) -> str:
        """生成数学答案 (简化版)"""
        # 这里可以集成更复杂的数学推理
        if "证明" in question:
            return f"以下是{question}的证明过程：\n\n1. 假设条件...\n2. 推理过程...\n3. 得出结论..."
        elif "解释" in question:
            return f"{question}的原理是：\n\n这是基于[相关理论]的基本概念..."
        elif "计算" in question:
            return f"{question}的计算结果是：\n\n经过计算，答案是[结果]..."
        else:
            return f"对于{question}，我们可以这样理解：\n\n[详细解释]..."

    def _generate_code_answer(self, task: str) -> str:
        """生成代码答案"""
        # 简化的代码生成
        if "快速排序" in task:
            return '''def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)'''
        elif "二分查找" in task:
            return '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1'''
        else:
            return f'''# {task}
def solution():
    """
    实现{task}的功能
    """
    # TODO: 实现具体逻辑
    pass'''

    def _generate_conversation_response(self, input_text: str) -> str:
        """生成对话响应"""
        responses = {
            "你好": "你好！我是H2Q-Evo，一个正在进化的AGI系统。我可以帮助你解答问题、生成代码、进行数学推理等。",
            "什么是人工智能": "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。",
            "如何学习编程": "学习编程需要：1) 选择一门语言入门 2) 理解基本概念 3) 多练习项目 4) 持续学习新技术",
            "什么是AGI": "AGI（Artificial General Intelligence）是指具有与人类相当的通用智能的人工智能系统。",
        }

        for key, response in responses.items():
            if key in input_text:
                return response

        return "这是一个很有趣的问题。让我来分析一下..."

    def _generate_creative_response(self, prompt: str) -> str:
        """生成创意响应"""
        if "科幻故事" in prompt:
            return """在遥远的未来，人类发现了第四维度...

故事开始于一个普通的程序员，他无意中发现了一个古老的算法..."""
        elif "诗" in prompt:
            return """数字的河流在硅谷流淌，
AGI的梦想在夜空中闪耀。
从比特到意识的跃迁，
见证进化的奇迹..."""
        else:
            return f"这是一个关于{prompt}的创意内容...\n\n[详细内容]"

    def _classify_genre(self, prompt: str) -> str:
        """分类题材"""
        if "科幻" in prompt or "未来" in prompt:
            return "science_fiction"
        elif "诗" in prompt:
            return "poetry"
        elif "游戏" in prompt:
            return "gaming"
        elif "商业" in prompt:
            return "business"
        else:
            return "general"

    def _apply_data_augmentation(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用数据增强"""
        augmented_data = []

        for item in data:
            augmented_data.append(item)  # 保留原数据

            # 应用增强技术
            if np.random.random() < 0.3:  # 30%概率增强
                augmented_item = self._augment_sample(item)
                if augmented_item:
                    augmented_data.append(augmented_item)

        logger.info(f"数据增强完成: {len(data)} -> {len(augmented_data)}")
        return augmented_data

    def _augment_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """增强单个样本"""
        try:
            augmented = sample.copy()

            # 文本增强
            if sample['type'] == 'conversation':
                # 添加情感标记
                augmented['input'] = f"[友好] {sample['input']}"
            elif sample['type'] == 'code_generation':
                # 添加难度标记
                augmented['input'] = f"[进阶] {sample['input']}"
            elif sample['type'] == 'mathematical_reasoning':
                # 添加上下文
                augmented['input'] = f"在数学中，{sample['input']}"

            augmented['augmented'] = True
            return augmented

        except Exception as e:
            logger.warning(f"数据增强失败: {e}")
            return None

    def _apply_manifold_encoding(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用流形编码 - 这是我们核心算法的真实应用"""
        logger.info("应用对数流形编码到训练数据...")

        encoded_data = []

        for item in data:
            try:
                # 对输入文本进行编码
                input_text = item['input']
                if self.tokenizer:
                    input_ids = self.tokenizer.encode(input_text, max_length=512, truncation=True)
                else:
                    # 简化的编码
                    input_ids = [ord(c) % 1000 for c in input_text[:512]]

                # 转换为numpy数组并应用我们的核心流形编码算法
                input_array = np.array(input_ids, dtype=np.float32).reshape(1, -1)
                encoded_input = self.compressed_encoder.encode_with_continuity(input_array)

                # 关键：真正使用编码结果，而不是保留原始tokens
                # 这是诚实的AGI实验 - 必须使用我们的核心算法
                encoded_features = encoded_input.flatten()

                # 将编码后的连续特征转换为适合语言模型的离散tokens
                # 保持编码的连续性和压缩优势
                processed_input_ids = []

                if self.tokenizer and hasattr(self.tokenizer, 'vocab_size'):
                    vocab_size = self.tokenizer.vocab_size
                else:
                    # 使用默认词汇表大小
                    vocab_size = 50000

                for i, feature in enumerate(encoded_features[:512]):  # 限制最大长度
                    if i < len(input_ids):
                        # 结合原始token信息和编码特征
                        # 使用我们的算法优势：连续性保持和压缩
                        encoded_token = int((input_ids[i] * (1 + feature)) % vocab_size)
                        processed_input_ids.append(encoded_token)
                    else:
                        # 对于编码扩展的部分，基于特征生成新tokens
                        encoded_token = int(abs(feature * 10000) % vocab_size)
                        processed_input_ids.append(encoded_token)

                # 确保有有效的输入
                if len(processed_input_ids) < 1:
                    processed_input_ids = input_ids[:512]

                # 计算真实的压缩率（使用我们的算法）
                original_size = len(input_ids) * 4  # 假设每个token 4字节
                encoded_size = len(encoded_features) * 4  # 编码后的大小
                actual_compression = encoded_size / original_size if original_size > 0 else 1.0

                # 更新数据项 - 使用真正编码后的数据
                encoded_item = item.copy()
                encoded_item['input_ids'] = processed_input_ids
                encoded_item['attention_mask'] = [1] * len(processed_input_ids)
                encoded_item['encoded_features'] = encoded_input.tolist()
                encoded_item['compression_ratio'] = actual_compression
                encoded_item['algorithm_used'] = 'logarithmic_manifold_encoding'  # 诚实验证标记
                encoded_item['original_input_ids'] = input_ids  # 保存原始tokens用于对比

                encoded_data.append(encoded_item)

            except Exception as e:
                logger.warning(f"应用流形编码失败: {e}")
                # 保留原始样本，但标记未使用算法
                item_copy = item.copy()
                item_copy['algorithm_used'] = 'none'
                encoded_data.append(item_copy)

        logger.info(f"对数流形编码完成: {len(encoded_data)}条样本，平均压缩率: {np.mean([d.get('compression_ratio', 1.0) for d in encoded_data]):.3f}")
        return encoded_data

    def generate_incremental_data(self, evolution_generation: int,
                                output_file: str) -> str:
        """生成增量数据 (基于进化代数) - 内存优化版本"""
        logger.info(f"为第{evolution_generation}代生成增量数据")

        # 内存优化：根据进化代数调整数据复杂度，但严格控制样本数量
        complexity_factor = min(1.0, evolution_generation / 100.0)  # 0-1之间

        # 严格控制样本数量，避免内存爆炸
        base_samples = 20  # 从20个样本开始
        max_samples = 50   # 最大50个样本
        num_samples = int(base_samples * (1 + complexity_factor))
        num_samples = min(num_samples, max_samples)  # 确保不超过最大值

        logger.info(f"内存优化模式: 生成 {num_samples} 条样本 (复杂度因子: {complexity_factor:.2f})")

        # 调整数据源权重 - 只使用最基本的类型
        self.config['data_sources'] = ['mathematical_reasoning', 'conversation']

        return self.generate_training_data(num_samples, output_file)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='H2Q-Evo AGI数据生成工具')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='生成的样本数量')
    parser.add_argument('--output', default='./agi_persistent_training/data/generated_data.jsonl',
                       help='输出文件路径')
    parser.add_argument('--model', default='microsoft/DialoGPT-medium',
                       help='使用的语言模型')
    parser.add_argument('--evolution-gen', type=int, default=0,
                       help='进化代数 (用于增量生成)')

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # 创建数据生成器
    generator = AGIDataGenerator()

    try:
        # 初始化模型
        generator.initialize_model(args.model)

        # 生成数据
        if args.evolution_gen > 0:
            output_file = generator.generate_incremental_data(args.evolution_gen, args.output)
        else:
            output_file = generator.generate_training_data(args.num_samples, args.output)

        print(f"✅ 数据生成完成: {output_file}")

    except Exception as e:
        logger.error(f"数据生成失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()