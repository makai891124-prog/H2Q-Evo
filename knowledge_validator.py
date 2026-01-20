#!/usr/bin/env python3
"""
H2Q-Evo 知识验证和矫正系统
通过公开免费的API验证和改进知识库
"""

import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

class KnowledgeValidator:
    """知识验证器 - 连接多个免费API进行知识验证"""
    
    def __init__(self):
        self.validation_log = Path("validation_log.json")
        self.validated_count = 0
        self.corrected_count = 0
        
    def validate_with_wikipedia(self, concept: str, domain: str) -> Dict:
        """使用Wikipedia API验证知识"""
        try:
            # Wikipedia API 搜索
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "opensearch",
                "search": concept,
                "limit": 1,
                "format": "json"
            }
            
            response = requests.get(search_url, params=search_params, timeout=5)
            if response.status_code == 200:
                results = response.json()
                if len(results) > 3 and len(results[3]) > 0:
                    url = results[3][0]
                    description = results[2][0] if len(results[2]) > 0 else ""
                    
                    return {
                        "source": "wikipedia",
                        "found": True,
                        "description": description,
                        "url": url,
                        "confidence": 0.8
                    }
            
            return {"source": "wikipedia", "found": False}
            
        except Exception as e:
            return {"source": "wikipedia", "error": str(e)}
    
    def validate_with_wolfram(self, concept: str) -> Dict:
        """使用Wolfram Alpha Simple API（需要免费API key）"""
        # 注意：这需要在环境变量中设置 WOLFRAM_APP_ID
        # 免费注册: https://products.wolframalpha.com/simple-api/documentation/
        import os
        app_id = os.getenv("WOLFRAM_APP_ID")
        
        if not app_id:
            return {"source": "wolfram", "found": False, "error": "No API key"}
        
        try:
            url = f"http://api.wolframalpha.com/v1/result"
            params = {
                "i": concept,
                "appid": app_id
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return {
                    "source": "wolfram",
                    "found": True,
                    "answer": response.text,
                    "confidence": 0.9
                }
            
            return {"source": "wolfram", "found": False}
            
        except Exception as e:
            return {"source": "wolfram", "error": str(e)}
    
    def validate_with_llm_free(self, concept: str, detail: str, domain: str) -> Dict:
        """使用免费的LLM API验证（Hugging Face Inference API）"""
        try:
            # 使用Hugging Face的免费推理API
            # 可以使用各种开源模型，如 google/flan-t5-large
            api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
            
            # 可以从环境变量获取token（可选，没有token也能用但有限额）
            import os
            token = os.getenv("HUGGINGFACE_TOKEN", "")
            
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            # 构造验证提示词
            prompt = f"Verify this scientific statement about {concept} in {domain}: '{detail}'. Is this accurate? Answer with 'Correct' or 'Incorrect' and explain briefly."
            
            payload = {"inputs": prompt}
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get("generated_text", "")
                    
                    # 简单的准确性判断
                    is_correct = "correct" in answer.lower() and "incorrect" not in answer.lower()
                    
                    return {
                        "source": "huggingface_llm",
                        "found": True,
                        "answer": answer,
                        "is_correct": is_correct,
                        "confidence": 0.75
                    }
            
            return {"source": "huggingface_llm", "found": False, "status": response.status_code}
            
        except Exception as e:
            return {"source": "huggingface_llm", "error": str(e)}
    
    def validate_with_ollama_local(self, concept: str, detail: str, domain: str) -> Dict:
        """使用本地Ollama验证（如果已安装）"""
        try:
            # 检查本地是否运行了Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",  # 或其他已安装的模型
                    "prompt": f"Verify this {domain} knowledge: {concept} - {detail}. Is this accurate? Answer yes or no and explain briefly.",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                
                return {
                    "source": "ollama_local",
                    "found": True,
                    "answer": answer,
                    "confidence": 0.85
                }
            
            return {"source": "ollama_local", "found": False}
            
        except Exception as e:
            # Ollama可能未安装或未运行，这是正常的
            return {"source": "ollama_local", "available": False}
    
    def comprehensive_validation(self, concept: str, detail: str, domain: str) -> Dict:
        """综合多个来源进行验证"""
        print(f"\n🔍 验证: {concept} ({domain})")
        
        results = {
            "concept": concept,
            "domain": domain,
            "original_detail": detail,
            "validation_time": datetime.now().isoformat(),
            "sources": []
        }
        
        # 1. Wikipedia验证（快速，可靠）
        wiki_result = self.validate_with_wikipedia(concept, domain)
        results["sources"].append(wiki_result)
        if wiki_result.get("found"):
            print(f"  ✓ Wikipedia: 找到相关条目")
        
        time.sleep(0.5)  # 避免API限流
        
        # 2. 尝试Ollama本地验证（如果可用）
        ollama_result = self.validate_with_ollama_local(concept, detail, domain)
        if ollama_result.get("available", True):  # 如果不是显式不可用
            results["sources"].append(ollama_result)
            if ollama_result.get("found"):
                print(f"  ✓ Ollama: 本地验证完成")
        
        # 3. LLM验证（可选，较慢）
        # llm_result = self.validate_with_llm_free(concept, detail, domain)
        # results["sources"].append(llm_result)
        
        # 综合评分
        confidence_scores = [s.get("confidence", 0) for s in results["sources"] if s.get("found")]
        if confidence_scores:
            results["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
            results["validated"] = results["overall_confidence"] > 0.6
        else:
            results["overall_confidence"] = 0.5
            results["validated"] = False
        
        self.validated_count += 1
        
        return results
    
    def suggest_correction(self, validation_result: Dict) -> Optional[str]:
        """基于验证结果建议修正"""
        sources = validation_result.get("sources", [])
        
        # 优先使用Wikipedia的描述
        for source in sources:
            if source.get("source") == "wikipedia" and source.get("found"):
                description = source.get("description", "")
                if description and len(description) > 20:
                    self.corrected_count += 1
                    return description
        
        # 其次使用LLM的回答
        for source in sources:
            if source.get("source") in ["huggingface_llm", "ollama_local"] and source.get("found"):
                answer = source.get("answer", "")
                if answer and len(answer) > 20:
                    self.corrected_count += 1
                    return answer
        
        return None
    
    def save_validation_log(self, validations: List[Dict]):
        """保存验证日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_validated": self.validated_count,
            "total_corrected": self.corrected_count,
            "validations": validations
        }
        
        with open(self.validation_log, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 验证日志已保存: {self.validation_log}")

class LearningFeedbackLoop:
    """学习反馈循环"""
    
    def __init__(self, knowledge_base, validator):
        self.kb = knowledge_base
        self.validator = validator
        self.learning_history = []
    
    def learn_and_validate_batch(self, batch_size: int = 10, difficulty_range: Tuple[int, int] = (1, 5)):
        """批量学习和验证知识"""
        print("="*80)
        print("🎓 开始学习和验证循环")
        print("="*80)
        
        # 获取未验证的知识
        unverified = self.kb.get_unverified()
        
        # 按难度筛选
        filtered = [(d, k) for d, k in unverified if difficulty_range[0] <= k['difficulty'] <= difficulty_range[1]]
        
        if not filtered:
            print("⚠️ 没有符合条件的未验证知识")
            return
        
        # 随机选择batch
        import random
        batch = random.sample(filtered, min(batch_size, len(filtered)))
        
        print(f"\n📚 学习批次: {len(batch)} 个知识条目")
        
        validations = []
        for i, (domain, knowledge) in enumerate(batch, 1):
            print(f"\n[{i}/{len(batch)}] {knowledge['concept']}")
            
            # 验证
            validation = self.validator.comprehensive_validation(
                knowledge['concept'],
                knowledge['detail'],
                domain
            )
            
            # 建议修正
            correction = self.validator.suggest_correction(validation)
            
            if correction:
                print(f"  💡 建议更新: {correction[:100]}...")
                self.kb.update_knowledge(
                    domain,
                    knowledge['concept'],
                    correction,
                    confidence=validation.get('overall_confidence', 0.7)
                )
            
            # 标记为已验证
            if validation.get('validated'):
                self.kb.mark_verified(domain, knowledge['concept'])
                print(f"  ✅ 验证通过 (置信度: {validation.get('overall_confidence', 0)*100:.1f}%)")
            else:
                print(f"  ⚠️ 验证失败或置信度低")
            
            validations.append(validation)
            
            # 避免API限流
            time.sleep(1)
        
        # 保存结果
        self.kb.save()
        self.validator.save_validation_log(validations)
        
        # 统计
        print("\n"+"="*80)
        print("📊 学习反馈统计")
        print("="*80)
        print(f"验证总数: {self.validator.validated_count}")
        print(f"修正总数: {self.validator.corrected_count}")
        print(f"修正率: {self.validator.corrected_count/max(self.validator.validated_count, 1)*100:.1f}%")
        
        stats = self.kb.get_stats()
        print(f"知识库验证进度: {stats['verified_count']}/{stats['total_count']} ({stats['verified_count']/max(stats['total_count'], 1)*100:.1f}%)")

if __name__ == "__main__":
    from large_knowledge_base import LargeKnowledgeBase
    
    print("="*80)
    print("🚀 启动知识验证和学习反馈系统")
    print("="*80)
    
    # 初始化
    kb = LargeKnowledgeBase()
    validator = KnowledgeValidator()
    feedback_loop = LearningFeedbackLoop(kb, validator)
    
    # 运行一个学习批次
    feedback_loop.learn_and_validate_batch(
        batch_size=5,  # 先验证5个
        difficulty_range=(1, 3)  # 从简单的开始
    )
