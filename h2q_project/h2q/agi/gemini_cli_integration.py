#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini CLI集成模块：与Google Gemini进行外部大模型交互
用于在自我进化循环中获取外部矫正和智能增强
"""

import os
import subprocess
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)


class GeminiCLIIntegration:
    """
    使用Gemini CLI进行API调用，获取外部大模型的矫正和建议
    支持同步和异步调用，提供响应缓存和容错机制
    """
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        """初始化Gemini CLI集成"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.api_key = self._load_api_key_from_env_file()
        self.model = model
        self.cache_dir = Path("./gemini_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.call_history = []
        
        # 验证API密钥
        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY未设置，将在本地模式运行")
            self.api_available = False
        else:
            self.api_available = self._verify_api_key()

    def _load_api_key_from_env_file(self) -> Optional[str]:
        """从项目根目录的 .env 中加载 GEMINI_API_KEY"""
        env_path = Path(".env")
        if not env_path.exists():
            return None
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"").strip("'")
        except Exception as e:
            logger.warning(f"读取 .env 失败: {e}")
        return None
            
    def _verify_api_key(self) -> bool:
        """验证API密钥是否可用"""
        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            models = client.models.list()
            return bool(list(models))
        except:
            return False
    
    def _get_cache_key(self, prompt: str) -> str:
        """为提示生成缓存密钥"""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存加载响应"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if time.time() - data['timestamp'] < 86400:  # 24小时有效期
                        logger.info(f"✓ 从缓存加载Gemini响应")
                        return data['response']
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """保存响应到缓存"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'response': response,
                    'model': self.model
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def query(self, prompt: str, context: Optional[Dict] = None,
              use_cache: bool = True,
              response_mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        查询Gemini获取矫正建议
        
        Args:
            prompt: 查询提示词
            context: 附加上下文信息
            use_cache: 是否使用缓存
            
        Returns:
            包含响应和元数据的字典
        """
        cache_key = self._get_cache_key(prompt)
        
        # 尝试从缓存加载
        if use_cache:
            cached_response = self._load_from_cache(cache_key)
            if cached_response:
                return {
                    'status': 'success',
                    'source': 'cache',
                    'response': cached_response,
                    'timestamp': datetime.now().isoformat(),
                    'model': self.model
                }
        
        # 构建完整提示词
        full_prompt = prompt
        if context:
            full_prompt += f"\n\n【上下文信息】\n{json.dumps(context, indent=2, ensure_ascii=False)}"
        
        # 如果API可用，使用API调用
        if self.api_available:
            return self._query_via_api(full_prompt, cache_key, response_mime_type=response_mime_type)
        else:
            return self._query_via_cli(full_prompt, cache_key)
    
    def _query_via_api(self, prompt: str, cache_key: str, response_mime_type: Optional[str] = None) -> Dict[str, Any]:
        """通过Google Generative AI API查询"""
        try:
            from google import genai
            try:
                from google.genai import types
            except Exception:
                types = None

            client = genai.Client(api_key=self.api_key)
            if response_mime_type and types is not None:
                config = types.GenerateContentConfig(response_mime_type=response_mime_type)
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
            else:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
            response_text = response.text
            
            # 保存到缓存
            self._save_to_cache(cache_key, response_text)
            
            # 记录调用历史
            self.call_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': 'api',
                'model': self.model,
                'status': 'success'
            })
            
            logger.info(f"✓ Gemini API调用成功")
            
            return {
                'status': 'success',
                'source': 'api',
                'response': response_text,
                'timestamp': datetime.now().isoformat(),
                'model': self.model
            }
        except Exception as e:
            error_text = str(e)
            if "NOT_FOUND" in error_text or "not found" in error_text:
                fallback_model = "gemini-2.0-flash"
                if self.model != fallback_model:
                    try:
                        from google import genai

                        client = genai.Client(api_key=self.api_key)
                        response = client.models.generate_content(
                            model=fallback_model,
                            contents=prompt
                        )
                        response_text = response.text
                        self.model = fallback_model
                        self._save_to_cache(cache_key, response_text)
                        self.call_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'method': 'api',
                            'model': self.model,
                            'status': 'success'
                        })
                        logger.info("✓ Gemini API调用成功(回退模型)")
                        return {
                            'status': 'success',
                            'source': 'api',
                            'response': response_text,
                            'timestamp': datetime.now().isoformat(),
                            'model': self.model
                        }
                    except Exception as fallback_error:
                        logger.error(f"✗ Gemini API调用失败: {fallback_error}")
                        return {
                            'status': 'error',
                            'source': 'api',
                            'error': str(fallback_error),
                            'timestamp': datetime.now().isoformat()
                        }
            logger.error(f"✗ Gemini API调用失败: {e}")
            return {
                'status': 'error',
                'source': 'api',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _query_via_cli(self, prompt: str, cache_key: str) -> Dict[str, Any]:
        """通过CLI调用Gemini"""
        temp_prompt_file = None
        try:
            if not self.api_key:
                return {
                    'status': 'error',
                    'source': 'cli',
                    'error': 'GEMINI_API_KEY 未设置，无法调用 CLI。',
                    'timestamp': datetime.now().isoformat()
                }

            # 创建临时提示文件
            temp_prompt_file = Path(f"/tmp/gemini_prompt_{cache_key}.txt")
            temp_prompt_file.write_text(prompt, encoding='utf-8')
            
            # 使用环境变量设置API密钥
            env = os.environ.copy()
            env['GEMINI_API_KEY'] = str(self.api_key)
            
            # 调用Gemini CLI (假设已安装)
            cmd = [
                "gcloud", "ai", "generative-content", "models/generate-content",
                "--model", self.model,
                "--request-file", str(temp_prompt_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30, env=env)
            
            if result.returncode == 0:
                response_text = result.stdout.decode('utf-8')
                self._save_to_cache(cache_key, response_text)
                
                self.call_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'method': 'cli',
                    'model': self.model,
                    'status': 'success'
                })
                
                logger.info(f"✓ Gemini CLI调用成功")
                
                return {
                    'status': 'success',
                    'source': 'cli',
                    'response': response_text,
                    'timestamp': datetime.now().isoformat(),
                    'model': self.model
                }
            else:
                error_msg = result.stderr.decode('utf-8')
                logger.error(f"✗ Gemini CLI调用失败: {error_msg}")
                return {
                    'status': 'error',
                    'source': 'cli',
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"✗ CLI执行失败: {e}")
            return {
                'status': 'error',
                'source': 'cli',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            # 清理临时文件
            if temp_prompt_file and temp_prompt_file.exists():
                temp_prompt_file.unlink()
    
    def batch_query(self, prompts: List[str], max_workers: int = 3) -> List[Dict]:
        """批量查询，支持并发"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {executor.submit(self.query, p): p for p in prompts}
            for future in as_completed(future_to_prompt):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"批量查询错误: {e}")
        
        return results
    
    def analyze_decision(self, decision: Dict, reasoning: str) -> Dict[str, Any]:
        """
        分析决策的质量和正确性
        用于在进化循环中获取外部反馈
        """
        analysis_prompt = f"""
        请分析以下决策的质量和潜在问题：
        
        【决策】
        {json.dumps(decision, indent=2, ensure_ascii=False)}
        
        【推理过程】
        {reasoning}
        
        请提供以下方面的分析：
        1. 决策逻辑的严密性 (1-10分)
        2. 潜在的错误或偏差
        3. 改进建议
        4. 数学严谨性评估
        5. 是否存在欺诈或幻觉迹象
        
        请用JSON格式返回分析结果。
        """
        
        result = self.query(analysis_prompt)
        
        if result['status'] == 'success':
            try:
                # 解析JSON响应
                response_text = result['response']
                # 提取JSON部分
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    return {
                        'status': 'success',
                        'analysis': analysis,
                        'raw_response': response_text
                    }
            except json.JSONDecodeError:
                logger.warning("无法解析Gemini分析结果为JSON")
        
        return {
            'status': 'failed',
            'raw_response': result.get('response', ''),
            'error': result.get('error', '未知错误')
        }
    
    def generate_improvement_suggestions(self, system_state: Dict) -> Dict[str, Any]:
        """
        基于当前系统状态生成改进建议
        """
        suggestion_prompt = f"""
        基于以下系统状态和性能指标，请生成改进建议：
        
        【系统状态】
        {json.dumps(system_state, indent=2, ensure_ascii=False)}
        
        请提供以下方面的建议：
        1. 架构优化方向
        2. 训练策略改进
        3. 决策质量提升
        4. 诚实性增强
        5. 计算效率改进
        
        请用JSON格式返回，每个建议包含优先级、理由和实现步骤。
        """
        
        return self.query(suggestion_prompt)
    
    def verify_against_gemini(self, claim: str, expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        与Gemini验证声明的正确性
        用于M24诚实协议的交叉验证
        """
        verification_prompt = f"""
        请验证以下声明的正确性：
        
        【声明】
        {claim}
        
        {f'【预期答案】{expected_answer}' if expected_answer else ''}
        
        请评估：
        1. 声明的真实性 (True/False/Uncertain)
        2. 信心度 (0-1)
        3. 详细的验证过程
        4. 潜在的欺诈指标
        
        请用JSON格式返回结果。
        """
        
        return self.query(verification_prompt)
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """获取调用统计信息"""
        if not self.call_history:
            return {'total_calls': 0}
        
        successful = sum(1 for c in self.call_history if c.get('status') == 'success')
        total = len(self.call_history)
        
        return {
            'total_calls': total,
            'successful_calls': successful,
            'success_rate': successful / total if total > 0 else 0,
            'by_method': {
                'api': sum(1 for c in self.call_history if c.get('method') == 'api'),
                'cli': sum(1 for c in self.call_history if c.get('method') == 'cli'),
            },
            'last_call': self.call_history[-1] if self.call_history else None
        }


if __name__ == "__main__":
    # 测试Gemini集成
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    integration = GeminiCLIIntegration()
    
    # 测试查询
    test_prompt = "请解释什么是自我进化的AGI系统"
    result = integration.query(test_prompt)
    print(f"\n【测试查询结果】\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 获取统计
    stats = integration.get_call_statistics()
    print(f"\n【调用统计】\n{json.dumps(stats, indent=2, ensure_ascii=False)}")
