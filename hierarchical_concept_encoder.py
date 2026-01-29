#!/usr/bin/env python3
"""
H2Q-Evo 分层概念编码器
基于四元数球面映射和分形结构的自动分层字符编码系统
集成开源英文字典实现自我组织的概念层
"""

import torch
import numpy as np
import json
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

sys.path.append('/Users/imymm/H2Q-Evo')

from h2q_project.src.h2q.tokenizer_simple import default_tokenizer
from final_integration_system import FinalIntegratedSystem, FinalIntegrationConfig


class HierarchicalConceptEncoder:
    """
    分层概念编码器
    实现自动分层字符编码链接和层级标志
    使用WordNet形成自我组织的概念层
    """

    def __init__(self, max_depth: int = 5, compression_ratio: float = 46.0):
        self.max_depth = max_depth
        self.compression_ratio = compression_ratio

        # 初始化组件
        self.tokenizer = default_tokenizer
        self.lemmatizer = WordNetLemmatizer()

        # 概念层级结构
        self.concept_layers: Dict[str, Dict] = {}
        self.layer_mappings: Dict[int, Dict] = {}
        self.abstraction_cache: Dict[str, Any] = {}

        # 四元数球面映射参数
        self.quaternion_basis = self._init_quaternion_basis()

        # 维度控制参数
        self.dimension_control = {
            'max_concepts_per_layer': 1000,
            'abstraction_threshold': 0.7,
            'recursion_limit': 10,
            'compression_factor': compression_ratio
        }

        # 初始化236B推理系统
        self.inference_system = self._init_inference_system()

        # 构建基础概念层
        self._build_base_concept_layers()

    def _init_quaternion_basis(self) -> torch.Tensor:
        """初始化四元数球面映射基"""
        # 四元数基: 1, i, j, k
        basis = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0, 0.0],  # i
            [0.0, 0.0, 1.0, 0.0],  # j
            [0.0, 0.0, 0.0, 1.0],  # k
        ], dtype=torch.float32)

        return basis

    def _init_inference_system(self) -> FinalIntegratedSystem:
        """初始化236B推理系统"""
        config = FinalIntegrationConfig(
            model_compression_ratio=self.compression_ratio,
            enable_mathematical_core=True,
            device="cpu"
        )

        system = FinalIntegratedSystem(config)

        # 尝试加载权重
        weight_paths = [
            "/Users/imymm/H2Q-Evo/h2q_project/h2q_full_l1.pth",
            "/Users/imymm/H2Q-Evo/h2q_project/h2q_qwen_crystal.pt",
            "/Users/imymm/H2Q-Evo/h2q_project/h2q_model_hierarchy.pth"
        ]

        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                if system.initialize_from_236b_weights(weight_path):
                    break

        return system

    def _build_base_concept_layers(self):
        """构建基础概念层"""
        print("🏗️ 构建基础概念层...")

        # 层级0: 原始字符
        self.layer_mappings[0] = {
            'type': 'character',
            'vocabulary': {chr(i): i for i in range(32, 127)},
            'encoding_dim': 1
        }

        # 层级1: 词素/词根
        self.layer_mappings[1] = {
            'type': 'morpheme',
            'vocabulary': {},
            'encoding_dim': 4  # 四元数维度
        }

        # 层级2: 单词
        self.layer_mappings[2] = {
            'type': 'word',
            'vocabulary': {},
            'encoding_dim': 16
        }

        # 层级3: 短语/概念
        self.layer_mappings[3] = {
            'type': 'phrase',
            'vocabulary': {},
            'encoding_dim': 64
        }

        # 层级4: 句子/抽象概念
        self.layer_mappings[4] = {
            'type': 'sentence',
            'vocabulary': {},
            'encoding_dim': 256
        }

        # 层级5: 文档/元概念
        self.layer_mappings[5] = {
            'type': 'document',
            'vocabulary': {},
            'encoding_dim': 1024
        }

        print("✅ 基础概念层构建完成")

    def quaternion_sphere_mapping(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """四元数球面映射"""
        # 将输入映射到四元数球面上
        # 使用球面坐标系进行映射

        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        batch_size, seq_len = input_tensor.shape

        # 归一化到单位球面
        norms = torch.norm(input_tensor, dim=-1, keepdim=True)
        normalized = input_tensor / (norms + 1e-8)

        # 扩展到四元数维度
        quaternion_coords = torch.zeros(batch_size, seq_len, 4, dtype=torch.float32)

        # 使用球面坐标映射
        theta = torch.acos(normalized[..., 0])  # 极角
        phi = torch.atan2(normalized[..., 1], normalized[..., 2])  # 方位角

        quaternion_coords[..., 0] = torch.cos(theta / 2)  # 实部
        quaternion_coords[..., 1] = torch.sin(theta / 2) * torch.cos(phi)  # i分量
        quaternion_coords[..., 2] = torch.sin(theta / 2) * torch.sin(phi)  # j分量
        quaternion_coords[..., 3] = torch.sin(theta / 2) * torch.cos(theta)  # k分量

        return quaternion_coords

    def encode_hierarchical(self, text: str, target_depth: int = None) -> Dict[str, Any]:
        """分层编码文本"""
        if target_depth is None:
            target_depth = self.max_depth

        result = {
            'original_text': text,
            'layers': {},
            'final_encoding': None,
            'concept_path': []
        }

        current_input = text

        for depth in range(min(target_depth + 1, self.max_depth + 1)):
            layer_result = self._encode_single_layer(current_input, depth)
            result['layers'][depth] = layer_result

            # 更新输入为下一层的抽象表示
            if depth < target_depth:
                current_input = self._abstract_to_next_layer(layer_result)

        # 生成最终编码
        result['final_encoding'] = self._generate_final_encoding(result['layers'])

        return result

    def _encode_single_layer(self, input_text: str, depth: int) -> Dict[str, Any]:
        """编码单层"""
        layer_config = self.layer_mappings.get(depth, {})

        if layer_config.get('type') == 'character':
            # 字符级编码
            tokens = [ord(c) for c in input_text if 32 <= ord(c) <= 126]
            encoding = torch.tensor(tokens, dtype=torch.long)

        elif layer_config.get('type') == 'word':
            # 单词级编码，使用WordNet概念
            words = input_text.split()
            encoding = self._encode_words_to_concepts(words)

        else:
            # 其他层级的通用编码
            encoding = self._encode_generic(input_text, depth)

        # 应用四元数球面映射
        if encoding.dtype == torch.long:
            encoding = encoding.float()

        quaternion_encoding = self.quaternion_sphere_mapping(encoding)

        return {
            'input': input_text,
            'encoding': quaternion_encoding,
            'layer_type': layer_config.get('type', 'unknown'),
            'dimension': quaternion_encoding.shape[-1]
        }

    def _encode_words_to_concepts(self, words: List[str]) -> torch.Tensor:
        """将单词编码为概念向量"""
        concept_vectors = []

        for word in words:
            # 词形还原
            lemma = self.lemmatizer.lemmatize(word.lower())

            # 获取WordNet同义词集
            synsets = wn.synsets(lemma)
            if synsets:
                # 使用第一个同义词集的定义作为概念表示
                definition = synsets[0].definition()
                # 简单编码：字符级编码定义
                concept_vec = torch.tensor([ord(c) for c in definition[:50]], dtype=torch.float32)
            else:
                # 回退到字符编码
                concept_vec = torch.tensor([ord(c) for c in lemma], dtype=torch.float32)

            concept_vectors.append(concept_vec.mean(dim=0, keepdim=True))

        if concept_vectors:
            return torch.cat(concept_vectors, dim=0)
        else:
            return torch.tensor([], dtype=torch.float32)

    def _encode_generic(self, text: str, depth: int) -> torch.Tensor:
        """通用编码方法"""
        # 简单字符级编码作为基础
        chars = [ord(c) for c in text if 32 <= ord(c) <= 126]
        encoding = torch.tensor(chars, dtype=torch.float32)

        # 根据深度应用不同级别的抽象
        if depth > 2 and len(chars) >= 4:
            # 更高层级：应用平均池化进行抽象
            # 确保可以被4整除
            remainder = len(chars) % 4
            if remainder > 0:
                # 填充到可以被4整除
                padding_size = 4 - remainder
                padding = torch.full((padding_size,), encoding.mean().item())
                encoding = torch.cat([encoding, padding])

            encoding = encoding.view(-1, 4).mean(dim=1)

        return encoding

    def _abstract_to_next_layer(self, layer_result: Dict) -> str:
        """将当前层抽象到下一层"""
        encoding = layer_result['encoding']

        # 简单策略：将编码转换为字符串表示
        # 在实际应用中，这里应该使用更复杂的抽象机制
        abstract_text = f"layer_{layer_result['layer_type']}_abstract_{encoding.shape}"

        return abstract_text

    def _generate_final_encoding(self, layers: Dict) -> torch.Tensor:
        """生成最终编码"""
        # 组合所有层的编码
        final_encodings = []

        for depth in range(len(layers)):
            layer_encoding = layers[depth]['encoding']
            # 压缩到统一维度，并确保批次维度一致
            compressed = self._compress_encoding(layer_encoding, target_dim=256)

            # 如果是2D tensor，确保第一个维度是1 (batch_size)
            if compressed.dim() == 1:
                compressed = compressed.unsqueeze(0)
            elif compressed.dim() == 2 and compressed.shape[0] != 1:
                compressed = compressed.mean(dim=0, keepdim=True)

            print(f"  层{depth}压缩后形状: {compressed.shape}")  # 调试信息
            final_encodings.append(compressed)

        # 连接所有层编码
        if final_encodings:
            # 确保所有tensor有相同的形状
            shapes = [enc.shape for enc in final_encodings]
            print(f"  各层形状: {shapes}")  # 调试信息

            if len(set(shapes)) == 1:  # 所有形状相同
                combined = torch.cat(final_encodings, dim=-1)
            else:
                # 如果形状不同，使用最大形状进行填充
                max_shape = torch.tensor(shapes).max(dim=0)[0]
                padded_encodings = []
                for enc in final_encodings:
                    if enc.shape != tuple(max_shape.tolist()):
                        padding = torch.zeros(*max_shape.tolist(), dtype=enc.dtype)
                        padding[:enc.shape[0], :enc.shape[1]] = enc
                        padded_encodings.append(padding)
                    else:
                        padded_encodings.append(enc)
                combined = torch.cat(padded_encodings, dim=-1)

            # 最终压缩
            final_encoding = self._compress_encoding(combined, target_dim=1024)
            return final_encoding

        return torch.tensor([], dtype=torch.float32)

    def _compress_encoding(self, encoding: torch.Tensor, target_dim: int) -> torch.Tensor:
        """压缩编码到目标维度"""
        if encoding.numel() == 0:
            return torch.zeros(1, target_dim, dtype=torch.float32)

        # 确保至少是2D
        if encoding.dim() == 1:
            encoding = encoding.unsqueeze(0)

        current_dim = encoding.shape[-1]

        if current_dim == target_dim:
            return encoding
        elif current_dim < target_dim:
            # 填充
            padding = torch.zeros(*encoding.shape[:-1], target_dim - current_dim, dtype=encoding.dtype)
            return torch.cat([encoding, padding], dim=-1)
        else:
            # 压缩：截断或平均
            if target_dim == 1:
                return encoding.mean(dim=-1, keepdim=True)
            else:
                # 简单截断到目标维度
                return encoding[..., :target_dim]

    def generate_code_completion(self, prompt: str, max_length: int = 1000) -> str:
        """使用236B模型进行代码补全"""
        print(f"🔧 生成代码补全: {prompt[:50]}...")

        # 首先进行分层编码
        hierarchical_encoding = self.encode_hierarchical(prompt, target_depth=3)

        # 准备输入
        final_encoding = hierarchical_encoding['final_encoding']
        if final_encoding.numel() == 0:
            # 回退到简单编码
            encoded = self.tokenizer.encode(prompt, add_specials=True, max_length=100)
            input_tensor = torch.tensor(encoded, dtype=torch.long).view(1, -1)
        else:
            # 使用分层编码
            input_tensor = final_encoding.view(1, -1).long()

        generated_tokens = []
        current_input = input_tensor.clone()

        try:
            for i in range(max_length):
                # 推理
                output = self.inference_system.perform_local_inference(current_input)

                # 获取下一个token
                if output.dim() > 1:
                    next_token_logits = output[0, -1, :]
                else:
                    next_token_logits = output[0, :]

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # 限制范围
                vocab_size = self.tokenizer.vocab_size
                if next_token >= vocab_size:
                    next_token = next_token % vocab_size

                generated_tokens.append(next_token)

                # 更新输入
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)

                # 检查停止条件
                if next_token == self.tokenizer.eos_token_id:
                    break

                if current_input.shape[1] > 2000:  # 防止过长
                    break

        except Exception as e:
            print(f"⚠️ 生成失败: {e}")
            return f"# Error: {e}"

        # 解码
        generated_text = self.tokenizer.decode(generated_tokens, skip_specials=True)

        return generated_text

    def analyze_concept_hierarchy(self, text: str) -> Dict[str, Any]:
        """分析概念层级结构"""
        encoding_result = self.encode_hierarchical(text)

        analysis = {
            'text': text,
            'layer_count': len(encoding_result['layers']),
            'total_concepts': sum(len(layer.get('vocabulary', {})) for layer in self.layer_mappings.values()),
            'encoding_shapes': {depth: layer['encoding'].shape for depth, layer in encoding_result['layers'].items()},
            'abstraction_levels': [layer['layer_type'] for layer in encoding_result['layers'].values()]
        }

        return analysis

    def save_concept_layers(self, filepath: str):
        """保存概念层"""
        data = {
            'layer_mappings': self.layer_mappings,
            'concept_layers': self.concept_layers,
            'abstraction_cache': dict(list(self.abstraction_cache.items())[:1000])  # 限制缓存大小
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"💾 概念层已保存到: {filepath}")

    def load_concept_layers(self, filepath: str):
        """加载概念层"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.layer_mappings = data.get('layer_mappings', {})
            self.concept_layers = data.get('concept_layers', {})
            self.abstraction_cache = data.get('abstraction_cache', {})

            print(f"📥 概念层已加载: {filepath}")


def test_hierarchical_encoder():
    """测试分层编码器"""
    print("🧪 测试分层概念编码器")
    print("=" * 50)

    encoder = HierarchicalConceptEncoder()

    # 测试文本 - 只用一个简单的
    test_text = "def fib(n):"

    print(f"\n📝 测试文本: {test_text}")

    # 分层编码 - 只用2层
    result = encoder.encode_hierarchical(test_text, target_depth=2)
    print(f"  层数: {len(result['layers'])}")
    print(f"  编码形状: {result['final_encoding'].shape if result['final_encoding'] is not None else 'None'}")

    # 概念分析
    analysis = encoder.analyze_concept_hierarchy(test_text)
    print(f"  概念层级: {analysis['abstraction_levels']}")

    # 代码补全测试
    completion = encoder.generate_code_completion(test_text, max_length=50)
    print(f"  补全结果: {completion[:100]}...")

    # 保存概念层
    encoder.save_concept_layers("/Users/imymm/H2Q-Evo/hierarchical_concept_layers.json")

    print("\n✅ 分层编码器测试完成")


if __name__ == "__main__":
    test_hierarchical_encoder()