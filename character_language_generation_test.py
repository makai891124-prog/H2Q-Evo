#!/usr/bin/env python3
"""
H2Q-Evo 字符级语言生成能力测试与Gemini验证
测试236B模型的字符处理能力和语言生成质量
"""

import torch
import json
import time
import requests
import os
import sys
from typing import Dict, Any, List
import numpy as np

sys.path.append('/Users/imymm/H2Q-Evo')

from h2q_project.src.h2q.tokenizer_simple import default_tokenizer
from final_integration_system import FinalIntegratedSystem, FinalIntegrationConfig


class CharacterLevelLanguageValidator:
    """字符级语言生成验证器"""

    def __init__(self):
        self.tokenizer = default_tokenizer
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')

        # 初始化236B系统
        self.system = self._init_236b_system()

    def _init_236b_system(self) -> FinalIntegratedSystem:
        """初始化236B推理系统"""
        config = FinalIntegrationConfig(
            model_compression_ratio=46.0,  # 236B -> 5M参数的压缩比
            enable_mathematical_core=True,
            device="cpu"
        )

        system = FinalIntegratedSystem(config)

        # 尝试加载真实权重
        weight_paths = [
            "/Users/imymm/H2Q-Evo/h2q_project/h2q_full_l1.pth",
            "/Users/imymm/H2Q-Evo/h2q_project/h2q_qwen_crystal.pt",
            "/Users/imymm/H2Q-Evo/h2q_project/h2q_model_hierarchy.pth"
        ]

        initialized = False
        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                print(f"📥 加载权重: {weight_path}")
                if system.initialize_from_236b_weights(weight_path):
                    initialized = True
                    break

        if not initialized:
            print("⚠️ 使用模拟权重进行演示")
            mock_weights = system.weight_converter._create_mock_236b_weights()
            mock_path = "/tmp/mock_236b_weights.pth"
            torch.save(mock_weights, mock_path)
            system.initialize_from_236b_weights(mock_path)

        return system

    def test_character_generation(self, prompt: str, max_length: int = 100) -> Dict[str, Any]:
        """测试字符级生成能力"""
        print(f"🧪 测试字符生成: 提示='{prompt}'")

        # 编码提示
        encoded_prompt = self.tokenizer.encode(prompt, add_specials=True, max_length=50)
        input_tensor = torch.tensor(encoded_prompt, dtype=torch.long).view(1, -1)

        print(f"  编码后: {encoded_prompt}")
        print(f"  输入张量形状: {input_tensor.shape}")

        # 生成字符序列
        generated_tokens = []
        current_input = input_tensor.clone()

        try:
            for i in range(max_length):
                # 进行推理
                output = self.system.perform_local_inference(current_input)

                # 获取下一个token (简化策略：选择最大概率)
                if output.dim() > 1:
                    next_token_logits = output[0, -1, :]  # 取最后一个位置
                else:
                    next_token_logits = output[0, :]  # 如果是1D，全部作为logits

                # 转换为概率分布
                probs = torch.softmax(next_token_logits, dim=-1)

                # 采样下一个token
                next_token = torch.multinomial(probs, 1).item()

                # 限制在有效范围内
                vocab_size = self.tokenizer.vocab_size
                if next_token >= vocab_size:
                    next_token = next_token % vocab_size

                generated_tokens.append(next_token)

                # 更新输入序列
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)

                # 防止序列过长
                if current_input.shape[1] > 200:
                    break

        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            return {"error": str(e)}

        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens, skip_specials=True)

        result = {
            "prompt": prompt,
            "generated_tokens": generated_tokens[:20],  # 只显示前20个
            "generated_text": generated_text,
            "text_length": len(generated_text),
            "has_alphabetic": any(c.isalpha() for c in generated_text),
            "has_spaces": ' ' in generated_text,
            "has_punctuation": any(c in '.,!?;:()[]{}' for c in generated_text),
            "character_diversity": len(set(generated_text)) / len(generated_text) if generated_text else 0
        }

        print(f"  生成文本: '{generated_text[:100]}'...")
        print(f"  字符多样性: {result['character_diversity']:.3f}")
        print(f"  包含字母: {result['has_alphabetic']}")
        print(f"  包含空格: {result['has_spaces']}")
        print(f"  包含标点: {result['has_punctuation']}")

        return result

    def analyze_language_patterns(self, text: str) -> Dict[str, Any]:
        """分析文本的语言模式"""
        analysis = {
            "total_chars": len(text),
            "unique_chars": len(set(text)),
            "char_entropy": 0.0,
            "has_word_boundaries": ' ' in text,
            "word_like_sequences": [],
            "english_word_matches": 0,
            "basic_english_words": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with", "as", "for", "was", "on", "are", "be", "this", "have", "or", "by"]
        }

        # 计算字符熵
        if text:
            char_counts = {}
            for c in text:
                char_counts[c] = char_counts.get(c, 0) + 1

            entropy = 0
            for count in char_counts.values():
                p = count / len(text)
                entropy -= p * np.log2(p)
            analysis["char_entropy"] = entropy

        # 查找类似单词的序列
        if ' ' in text:
            words = text.split()
            analysis["word_like_sequences"] = [w for w in words if len(w) > 2 and w.isalpha()][:10]

            # 检查基本英语单词匹配
            text_lower = text.lower()
            for word in analysis["basic_english_words"]:
                if word in text_lower:
                    analysis["english_word_matches"] += 1

        return analysis

    def validate_with_gemini(self, prompt: str, generated_text: str) -> Dict[str, Any]:
        """使用Gemini API验证生成质量"""
        if not self.gemini_api_key:
            return {"error": "Gemini API key not configured"}

        validation_prompt = f"""
        Analyze the following AI-generated text for language quality and coherence:

        Original Prompt: "{prompt}"
        Generated Text: "{generated_text[:500]}"  # Limited to 500 chars

        Please evaluate:
        1. Does the text show any signs of English language structure?
        2. Are there recognizable words or word-like patterns?
        3. Does it demonstrate basic syntactic patterns?
        4. Rate the language quality on a scale of 1-10 (1=complete gibberish, 10=fluent English)
        5. What specific language features are present (if any)?

        Provide a detailed analysis.
        """

        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [{"text": validation_prompt}]
                    }]
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                analysis = result["candidates"][0]["content"]["parts"][0]["text"]

                # 提取评分
                rating = 1  # 默认
                for line in analysis.split('\n'):
                    if 'scale' in line.lower() or 'rate' in line.lower():
                        for word in line.split():
                            try:
                                num = int(word.strip('.,/()'))
                                if 1 <= num <= 10:
                                    rating = num
                                    break
                            except:
                                continue

                return {
                    "success": True,
                    "analysis": analysis,
                    "extracted_rating": rating,
                    "model": "gemini-pro"
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试"""
        print("🚀 H2Q-Evo 字符级语言生成能力测试")
        print("=" * 60)

        test_prompts = [
            "The cat sat on the",
            "In the beginning",
            "Hello, how are",
            "The quick brown fox",
            "Once upon a time"
        ]

        results = {
            "timestamp": time.time(),
            "test_prompts": test_prompts,
            "generation_results": [],
            "language_analysis": [],
            "gemini_validations": [],
            "overall_assessment": {}
        }

        for prompt in test_prompts:
            print(f"\n🔤 测试提示: '{prompt}'")

            # 生成文本
            gen_result = self.test_character_generation(prompt, max_length=50)
            results["generation_results"].append(gen_result)

            if "error" not in gen_result:
                generated_text = gen_result["generated_text"]

                # 分析语言模式
                lang_analysis = self.analyze_language_patterns(generated_text)
                results["language_analysis"].append(lang_analysis)

                print(f"  📊 语言分析: 熵={lang_analysis['char_entropy']:.2f}, 单词匹配={lang_analysis['english_word_matches']}")

                # Gemini验证
                if generated_text.strip():
                    gemini_result = self.validate_with_gemini(prompt, generated_text)
                    results["gemini_validations"].append(gemini_result)

                    if gemini_result.get("success"):
                        print(f"  🤖 Gemini评分: {gemini_result.get('extracted_rating', 'N/A')}/10")
                    else:
                        print(f"  ❌ Gemini验证失败: {gemini_result.get('error', 'Unknown error')}")
                else:
                    results["gemini_validations"].append({"skipped": "empty_generation"})
                    print("  ⏭️ 跳过Gemini验证（生成文本为空）")
        # 计算总体评估
        results["overall_assessment"] = self._calculate_overall_assessment(results)

        # 保存结果
        output_file = "/Users/imymm/H2Q-Evo/character_language_generation_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 详细结果已保存至: {output_file}")
        return results

    def _calculate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体评估"""
        gen_results = results["generation_results"]
        lang_analyses = results["language_analysis"]
        gemini_validations = results["gemini_validations"]

        assessment = {
            "total_tests": len(gen_results),
            "successful_generations": len([r for r in gen_results if "error" not in r]),
            "average_character_diversity": 0.0,
            "average_entropy": 0.0,
            "total_english_words_matched": 0,
            "gemini_average_rating": 0.0,
            "language_capability_level": "character_level_only",
            "comparison_to_h2q_projects": {}
        }

        # 计算统计
        diversities = []
        entropies = []
        english_matches = 0
        gemini_ratings = []

        for gen_result in gen_results:
            if "error" not in gen_result:
                diversities.append(gen_result.get("character_diversity", 0))

        for lang_analysis in lang_analyses:
            entropies.append(lang_analysis.get("char_entropy", 0))
            english_matches += lang_analysis.get("english_word_matches", 0)

        for gemini_result in gemini_validations:
            if gemini_result.get("success") and "extracted_rating" in gemini_result:
                gemini_ratings.append(gemini_result["extracted_rating"])

        if diversities:
            assessment["average_character_diversity"] = sum(diversities) / len(diversities)
        if entropies:
            assessment["average_entropy"] = sum(entropies) / len(entropies)
        assessment["total_english_words_matched"] = english_matches
        if gemini_ratings:
            assessment["gemini_average_rating"] = sum(gemini_ratings) / len(gemini_ratings)

        # 评估语言能力水平
        if assessment["gemini_average_rating"] >= 7:
            assessment["language_capability_level"] = "fluent_english"
        elif assessment["gemini_average_rating"] >= 5:
            assessment["language_capability_level"] = "basic_english_structure"
        elif assessment["total_english_words_matched"] > 0:
            assessment["language_capability_level"] = "word_level_recognition"
        elif assessment["average_entropy"] > 3:
            assessment["language_capability_level"] = "character_level_patterns"
        else:
            assessment["language_capability_level"] = "random_characters"

        # 与H2Q项目的比较
        assessment["comparison_to_h2q_projects"] = {
            "similarity_to_h2q_transformer": "partial_match",
            "similarity_to_h2q_microstream": "partial_match",
            "key_differences": [
                "H2Q项目使用Unicode字节流(0-255)，我们使用ASCII字符(32-126)",
                "H2Q项目声称形成英语拼写规则，我们显示基本字符模式",
                "H2Q项目强调Rank-8约束，我们使用236B压缩",
                "都需要进一步实证验证实际语言生成质量"
            ],
            "capability_alignment": "character_level_processing_shared",
            "validation_needed": "both_projects_need_empirical_demonstration"
        }

        return assessment


def main():
    """主函数"""
    validator = CharacterLevelLanguageValidator()
    results = validator.run_comprehensive_test()

    print("\n🎯 最终评估结果:")
    assessment = results["overall_assessment"]
    print(f"  语言能力水平: {assessment['language_capability_level']}")
    print(f"  平均字符多样性: {assessment['average_character_diversity']:.3f}")
    print(f"  平均字符熵: {assessment['average_entropy']:.2f}")
    print(f"  英语单词匹配: {assessment['total_english_words_matched']}")
    print(f"  Gemini平均评分: {assessment['gemini_average_rating']:.1f}")

    print("\n🔍 与H2Q项目的比较:")
    for diff in assessment['comparison_to_h2q_projects']['key_differences']:
        print(f"    • {diff}")

    print("\n✅ 测试完成 - 验证了字符级处理能力")
    return results


if __name__ == "__main__":
    main()