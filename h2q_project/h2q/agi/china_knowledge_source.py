#!/usr/bin/env python3
"""中国大陆网络环境知识源.

针对中国大陆网络环境优化的知识获取模块:
1. Hugging Face 镜像 (hf-mirror.com)
2. 百度百科 API
3. 国内开源数据集
4. 流式下载支持
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.error
import ssl
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Generator, Tuple
from dataclasses import dataclass, field
import threading
from queue import Queue

# ============================================================================
# 配置
# ============================================================================

@dataclass
class ChinaSourceConfig:
    """中国源配置."""
    # Hugging Face 镜像
    hf_mirror: str = "https://hf-mirror.com"
    hf_endpoint: str = "https://hf-mirror.com"
    
    # 国内可用源
    baidu_baike_api: str = "https://baike.baidu.com/api/openapi/BaikeLemmaCardApi"
    zhihu_search: str = "https://www.zhihu.com/api/v4/search_v3"
    
    # 数据集配置
    datasets: List[str] = field(default_factory=lambda: [
        "shibing624/alpaca-zh",      # 中文Alpaca
        "BelleGroup/train_0.5M_CN",  # Belle中文数据
        "fnlp/moss-sft-data",        # MOSS数据
        "THUDM/webglm-qa",           # WebGLM问答
    ])
    
    # 下载配置
    chunk_size: int = 8192           # 流式下载块大小
    timeout: int = 30                # 请求超时
    max_retries: int = 3             # 最大重试次数
    cache_dir: str = ".cache/knowledge"


# ============================================================================
# SSL 和代理配置
# ============================================================================

def create_ssl_context() -> ssl.SSLContext:
    """创建SSL上下文（处理证书问题）."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def get_opener() -> urllib.request.OpenerDirector:
    """获取URL打开器（支持代理）."""
    handlers = []
    
    # 检查代理环境变量
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy or https_proxy:
        proxy_dict = {}
        if http_proxy:
            proxy_dict['http'] = http_proxy
        if https_proxy:
            proxy_dict['https'] = https_proxy
        handlers.append(urllib.request.ProxyHandler(proxy_dict))
    
    # SSL处理
    handlers.append(urllib.request.HTTPSHandler(context=create_ssl_context()))
    
    return urllib.request.build_opener(*handlers)


# ============================================================================
# Hugging Face 镜像数据集访问
# ============================================================================

class HFMirrorDatasetLoader:
    """Hugging Face 镜像数据集加载器."""
    
    def __init__(self, config: ChinaSourceConfig = None):
        self.config = config or ChinaSourceConfig()
        self.opener = get_opener()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计
        self.downloaded_bytes = 0
        self.items_loaded = 0
    
    def _make_request(self, url: str, headers: Dict = None) -> Optional[bytes]:
        """发送请求."""
        default_headers = {
            'User-Agent': 'H2Q-AGI/1.0 (Educational Research)',
            'Accept': 'application/json',
        }
        if headers:
            default_headers.update(headers)
        
        req = urllib.request.Request(url, headers=default_headers)
        
        for attempt in range(self.config.max_retries):
            try:
                with self.opener.open(req, timeout=self.config.timeout) as response:
                    return response.read()
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                continue
        
        return None
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """获取数据集信息."""
        url = f"{self.config.hf_mirror}/api/datasets/{dataset_id}"
        
        data = self._make_request(url)
        if data:
            try:
                return json.loads(data.decode('utf-8'))
            except:
                pass
        return None
    
    def stream_dataset_samples(self, dataset_id: str, 
                                split: str = "train",
                                max_samples: int = 100) -> Generator[Dict, None, None]:
        """流式获取数据集样本.
        
        使用 datasets 库的流式加载或直接请求parquet/jsonl文件.
        """
        # 方案1: 尝试使用 HF datasets streaming
        try:
            # 设置镜像环境变量
            os.environ['HF_ENDPOINT'] = self.config.hf_endpoint
            
            from datasets import load_dataset
            
            dataset = load_dataset(
                dataset_id, 
                split=split, 
                streaming=True,
                trust_remote_code=True
            )
            
            count = 0
            for item in dataset:
                if count >= max_samples:
                    break
                yield item
                count += 1
                self.items_loaded += 1
            
            return
            
        except ImportError:
            pass  # datasets 库未安装
        except Exception as e:
            pass  # 其他错误，尝试备选方案
        
        # 方案2: 直接请求 README 或示例数据
        yield from self._fetch_dataset_readme(dataset_id)
    
    def _fetch_dataset_readme(self, dataset_id: str) -> Generator[Dict, None, None]:
        """获取数据集 README 作为知识."""
        url = f"{self.config.hf_mirror}/{dataset_id}/raw/main/README.md"
        
        data = self._make_request(url)
        if data:
            content = data.decode('utf-8', errors='ignore')
            
            # 分段处理 README
            sections = content.split('\n## ')
            for i, section in enumerate(sections[:5]):  # 最多5个段落
                yield {
                    "source": "hf_mirror",
                    "dataset": dataset_id,
                    "section": i,
                    "content": section[:2000],  # 限制长度
                    "timestamp": datetime.now().isoformat()
                }
                self.items_loaded += 1
    
    def download_sample_file(self, dataset_id: str, 
                             filename: str = "train.jsonl",
                             max_lines: int = 100) -> List[Dict]:
        """下载样本文件（流式）."""
        samples = []
        
        # 尝试多种文件格式
        possible_files = [
            f"{filename}",
            "data/train.jsonl",
            "train/data-00000-of-00001.parquet",
            "train.json",
        ]
        
        for file_path in possible_files:
            url = f"{self.config.hf_mirror}/{dataset_id}/resolve/main/{file_path}"
            
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'H2Q-AGI/1.0'
                })
                
                with self.opener.open(req, timeout=self.config.timeout) as response:
                    line_count = 0
                    buffer = b""
                    
                    while line_count < max_lines:
                        chunk = response.read(self.config.chunk_size)
                        if not chunk:
                            break
                        
                        buffer += chunk
                        self.downloaded_bytes += len(chunk)
                        
                        # 处理 JSONL
                        while b'\n' in buffer and line_count < max_lines:
                            line, buffer = buffer.split(b'\n', 1)
                            try:
                                item = json.loads(line.decode('utf-8'))
                                samples.append(item)
                                line_count += 1
                                self.items_loaded += 1
                            except:
                                continue
                    
                    if samples:
                        return samples
                        
            except Exception as e:
                continue
        
        return samples


# ============================================================================
# 百度百科知识获取
# ============================================================================

class BaiduBaikeAcquirer:
    """百度百科知识获取器."""
    
    def __init__(self):
        self.opener = get_opener()
        self.acquired_count = 0
    
    def fetch_lemma(self, keyword: str) -> Optional[Dict]:
        """获取词条信息."""
        # 使用百度百科 OpenAPI (需要申请 appid)
        # 这里使用简化的爬取方式
        
        url = f"https://baike.baidu.com/item/{urllib.parse.quote(keyword)}"
        
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
            })
            
            with self.opener.open(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
                # 简单提取摘要（实际使用需要更完善的解析）
                import re
                
                # 尝试提取描述
                desc_match = re.search(r'<meta name="description" content="([^"]+)"', html)
                if desc_match:
                    summary = desc_match.group(1)
                else:
                    # 备选：提取正文前几段
                    text_match = re.search(r'<div class="lemma-summary"[^>]*>(.*?)</div>', html, re.DOTALL)
                    if text_match:
                        summary = re.sub(r'<[^>]+>', '', text_match.group(1))[:500]
                    else:
                        return None
                
                self.acquired_count += 1
                
                return {
                    "title": keyword,
                    "summary": summary,
                    "source": "baidu_baike",
                    "timestamp": datetime.now().isoformat(),
                    "url": url
                }
                
        except Exception as e:
            return None
    
    def batch_fetch(self, keywords: List[str], delay: float = 1.0) -> List[Dict]:
        """批量获取（带延迟，避免限流）."""
        results = []
        
        for keyword in keywords:
            result = self.fetch_lemma(keyword)
            if result:
                results.append(result)
            time.sleep(delay)
        
        return results


# ============================================================================
# 国内开源数据集目录
# ============================================================================

class ChinaOpenDatasets:
    """国内开源数据集."""
    
    # 可用的中文数据集（Hugging Face 镜像可访问）
    DATASETS = {
        # 中文指令数据
        "instruction": [
            ("shibing624/alpaca-zh", "中文Alpaca指令数据"),
            ("BelleGroup/train_0.5M_CN", "Belle中文指令数据"),
            ("fnlp/moss-002-sft-data", "MOSS指令微调数据"),
            ("YeungNLP/firefly-train-1.1M", "流萤中文对话数据"),
        ],
        
        # 问答数据
        "qa": [
            ("THUDM/webglm-qa", "WebGLM问答数据"),
            ("suolyer/webqa", "网页问答数据"),
            ("Duxiaoman-DI/FinCorpus", "金融问答数据"),
        ],
        
        # 通用文本
        "text": [
            ("pleisto/wikipedia-cn-20230720-filtered", "中文维基百科"),
            ("liwu/MNBVC", "超大规模中文语料"),
            ("Skywork/SkyPile-150B", "中文网页语料"),
        ],
        
        # 数学推理
        "math": [
            ("TIGER-Lab/MathInstruct", "数学指令数据"),
            ("meta-math/MetaMathQA", "数学问答数据"),
        ],
        
        # 代码
        "code": [
            ("bigcode/starcoderdata", "代码数据"),
            ("codeparrot/github-code", "GitHub代码"),
        ]
    }
    
    @classmethod
    def get_datasets_by_category(cls, category: str) -> List[Tuple[str, str]]:
        """按类别获取数据集列表."""
        return cls.DATASETS.get(category, [])
    
    @classmethod
    def get_all_datasets(cls) -> List[Tuple[str, str]]:
        """获取所有数据集."""
        all_datasets = []
        for datasets in cls.DATASETS.values():
            all_datasets.extend(datasets)
        return all_datasets


# ============================================================================
# 综合知识获取器（中国优化版）
# ============================================================================

class ChinaKnowledgeAcquirer:
    """中国网络环境优化的知识获取器."""
    
    def __init__(self, config: ChinaSourceConfig = None):
        self.config = config or ChinaSourceConfig()
        
        # 组件
        self.hf_loader = HFMirrorDatasetLoader(self.config)
        self.baike_acquirer = BaiduBaikeAcquirer()
        
        # 统计
        self.total_acquired = 0
        self.source_stats: Dict[str, int] = {}
        
        # 知识缓存
        self.knowledge_cache: List[Dict] = []
        self.cache_lock = threading.Lock()
    
    def acquire_from_hf_dataset(self, dataset_id: str, 
                                 max_samples: int = 50) -> List[Dict]:
        """从 HF 镜像获取数据集样本."""
        samples = []
        
        print(f"  📥 从 HF 镜像获取: {dataset_id}")
        
        try:
            for item in self.hf_loader.stream_dataset_samples(dataset_id, max_samples=max_samples):
                samples.append({
                    "source": "hf_mirror",
                    "dataset": dataset_id,
                    "content": item,
                    "timestamp": datetime.now().isoformat()
                })
                
                if len(samples) >= max_samples:
                    break
            
            self._update_stats("hf_mirror", len(samples))
            print(f"    ✅ 获取 {len(samples)} 条样本")
            
        except Exception as e:
            print(f"    ❌ 获取失败: {e}")
        
        return samples
    
    def acquire_from_baike(self, keywords: List[str]) -> List[Dict]:
        """从百度百科获取知识."""
        print(f"  📖 从百度百科获取: {len(keywords)} 个关键词")
        
        results = self.baike_acquirer.batch_fetch(keywords, delay=0.5)
        
        self._update_stats("baidu_baike", len(results))
        print(f"    ✅ 获取 {len(results)} 条知识")
        
        return results
    
    def auto_acquire(self, categories: List[str] = None,
                     max_per_source: int = 20) -> List[Dict]:
        """自动从多个源获取知识."""
        all_knowledge = []
        
        categories = categories or ["instruction", "qa", "math"]
        
        print("🌐 开始自动知识获取...")
        
        # 1. 从 HF 镜像获取数据集样本
        for category in categories:
            datasets = ChinaOpenDatasets.get_datasets_by_category(category)
            
            for dataset_id, desc in datasets[:2]:  # 每类最多2个数据集
                samples = self.acquire_from_hf_dataset(dataset_id, max_samples=max_per_source)
                all_knowledge.extend(samples)
                
                # 避免请求过快
                time.sleep(1)
        
        # 2. 从百度百科获取关键词
        baike_keywords = [
            "人工智能", "机器学习", "深度学习", "神经网络",
            "自然语言处理", "计算机视觉", "强化学习",
            "数学", "线性代数", "概率论", "微积分"
        ]
        
        baike_results = self.acquire_from_baike(baike_keywords[:max_per_source])
        all_knowledge.extend(baike_results)
        
        # 缓存
        with self.cache_lock:
            self.knowledge_cache.extend(all_knowledge)
        
        print(f"\n📊 知识获取完成: 共 {len(all_knowledge)} 条")
        for source, count in self.source_stats.items():
            print(f"   - {source}: {count} 条")
        
        return all_knowledge
    
    def _update_stats(self, source: str, count: int):
        """更新统计."""
        self.total_acquired += count
        self.source_stats[source] = self.source_stats.get(source, 0) + count
    
    def get_random_knowledge(self, n: int = 5) -> List[Dict]:
        """随机获取缓存的知识."""
        import random
        
        with self.cache_lock:
            if not self.knowledge_cache:
                return []
            return random.sample(self.knowledge_cache, min(n, len(self.knowledge_cache)))


# ============================================================================
# 网络测试
# ============================================================================

def test_china_network() -> Dict[str, bool]:
    """测试中国网络环境连通性."""
    results = {}
    opener = get_opener()
    
    test_urls = [
        ("baidu", "https://www.baidu.com"),
        ("hf_mirror", "https://hf-mirror.com"),
        ("baike", "https://baike.baidu.com"),
        ("github_mirror", "https://ghproxy.com"),
    ]
    
    print("🔍 测试中国网络环境...")
    
    for name, url in test_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with opener.open(req, timeout=10) as response:
                results[name] = response.status == 200
                print(f"  ✅ {name}: 可用")
        except Exception as e:
            results[name] = False
            print(f"  ❌ {name}: 不可用 ({e})")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """测试运行."""
    print("=" * 60)
    print("H2Q AGI 中国网络源测试")
    print("=" * 60)
    
    # 1. 网络测试
    network_status = test_china_network()
    print()
    
    # 2. 知识获取测试
    if network_status.get("hf_mirror") or network_status.get("baike"):
        acquirer = ChinaKnowledgeAcquirer()
        
        # 自动获取
        knowledge = acquirer.auto_acquire(
            categories=["instruction", "qa"],
            max_per_source=5
        )
        
        print(f"\n获取的知识样本:")
        for i, item in enumerate(knowledge[:3]):
            print(f"\n[{i+1}] {item.get('source', 'unknown')}")
            content = str(item.get('content', item.get('summary', '')))[:200]
            print(f"    {content}...")
    else:
        print("⚠️ 网络不可用，跳过知识获取测试")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()
