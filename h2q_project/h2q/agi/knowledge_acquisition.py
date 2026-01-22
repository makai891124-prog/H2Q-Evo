"""H2Q 网络资源获取模块 (Knowledge Acquisition Module).

安全合规地从公开网络资源获取学习材料:
1. 轮询式资源发现 - 定时检查公开数据源
2. 合规过滤器 - 仅获取公开、合法资源
3. 增量下载 - 避免重复获取
4. 速率限制 - 友好的访问策略

支持的公开资源:
- Wikipedia API (公开知识)
- arXiv API (学术论文摘要)
- OpenML (机器学习数据集)
- 公开数学题库
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Generator
from enum import Enum
import time
import json
import hashlib
import re
from pathlib import Path
from collections import deque
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET


# ============================================================================
# 资源类型
# ============================================================================

class ResourceType(Enum):
    """资源类型."""
    TEXT = "text"
    MATH = "math"
    CODE = "code"
    SCIENCE = "science"
    GENERAL = "general"
    DATASET = "dataset"


class ResourceSource(Enum):
    """资源来源."""
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    OPENML = "openml"
    MATHWORLD = "mathworld"
    LOCAL = "local"


@dataclass
class KnowledgeResource:
    """知识资源."""
    id: str
    title: str
    content: str
    source: ResourceSource
    resource_type: ResourceType
    url: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.5
    relevance_score: float = 0.5


@dataclass
class AcquisitionConfig:
    """获取配置."""
    poll_interval_sec: float = 300.0  # 5 分钟
    max_resources_per_poll: int = 10
    rate_limit_requests_per_min: int = 10
    cache_dir: Path = field(default_factory=lambda: Path("./knowledge_cache"))
    user_agent: str = "H2Q-AGI-Learner/1.0 (Educational Research)"


# ============================================================================
# 合规过滤器
# ============================================================================

class ComplianceFilter:
    """合规过滤器 - 确保资源获取合法合规."""
    
    # 白名单域名
    ALLOWED_DOMAINS = {
        "wikipedia.org",
        "arxiv.org",
        "openml.org",
        "mathworld.wolfram.com",
        "projecteuler.net",
        "kaggle.com",
        "huggingface.co",
    }
    
    # 敏感词过滤
    BLOCKED_PATTERNS = [
        r'\b(password|secret|private|confidential)\b',
        r'\b(hack|crack|exploit)\b',
        r'\b(weapon|violence|illegal)\b',
    ]
    
    def __init__(self):
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS]
    
    def is_domain_allowed(self, url: str) -> bool:
        """检查域名是否在白名单."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            for allowed in self.ALLOWED_DOMAINS:
                if domain.endswith(allowed):
                    return True
            return False
        except Exception:
            return False
    
    def is_content_safe(self, content: str) -> bool:
        """检查内容是否安全."""
        for pattern in self.blocked_patterns:
            if pattern.search(content):
                return False
        return True
    
    def filter_resource(self, resource: KnowledgeResource) -> bool:
        """过滤资源.
        
        Returns:
            True if resource is allowed, False otherwise.
        """
        # 检查域名
        if not self.is_domain_allowed(resource.url):
            return False
        
        # 检查内容
        if not self.is_content_safe(resource.content):
            return False
        
        return True


# ============================================================================
# 资源获取器
# ============================================================================

class WikipediaFetcher:
    """Wikipedia API 获取器."""
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, config: AcquisitionConfig):
        self.config = config
    
    def search(self, query: str, limit: int = 5) -> List[str]:
        """搜索文章标题."""
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json",
        }
        
        try:
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data[1] if len(data) > 1 else []
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []
    
    def get_summary(self, title: str) -> Optional[KnowledgeResource]:
        """获取文章摘要."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
        }
        
        try:
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                pages = data.get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    if page_id == "-1":
                        continue
                    
                    extract = page.get("extract", "")
                    if extract:
                        return KnowledgeResource(
                            id=hashlib.md5(title.encode()).hexdigest()[:16],
                            title=title,
                            content=extract,
                            source=ResourceSource.WIKIPEDIA,
                            resource_type=ResourceType.GENERAL,
                            url=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                            timestamp=time.time(),
                            metadata={"page_id": page_id},
                        )
        except Exception as e:
            print(f"Wikipedia fetch error: {e}")
        
        return None


class ArxivFetcher:
    """arXiv API 获取器 (学术论文摘要)."""
    
    API_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, config: AcquisitionConfig):
        self.config = config
    
    def search(self, query: str, category: str = "cs.AI", 
               max_results: int = 5) -> List[KnowledgeResource]:
        """搜索论文."""
        params = {
            "search_query": f"all:{query} AND cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        
        results = []
        
        try:
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            
            with urllib.request.urlopen(req, timeout=15) as response:
                xml_data = response.read().decode()
                root = ET.fromstring(xml_data)
                
                # 解析 Atom feed
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    link = entry.find("atom:id", ns)
                    
                    if title is not None and summary is not None:
                        title_text = title.text.strip() if title.text else ""
                        summary_text = summary.text.strip() if summary.text else ""
                        link_text = link.text if link is not None else ""
                        
                        resource = KnowledgeResource(
                            id=hashlib.md5(link_text.encode()).hexdigest()[:16],
                            title=title_text,
                            content=summary_text,
                            source=ResourceSource.ARXIV,
                            resource_type=ResourceType.SCIENCE,
                            url=link_text,
                            timestamp=time.time(),
                            metadata={"category": category},
                        )
                        results.append(resource)
        
        except Exception as e:
            print(f"arXiv search error: {e}")
        
        return results


class MathProblemGenerator:
    """数学问题生成器 (本地生成)."""
    
    def __init__(self):
        np.random.seed(int(time.time()) % 10000)
    
    def generate_arithmetic(self, difficulty: int = 1) -> KnowledgeResource:
        """生成算术问题."""
        max_val = 10 ** difficulty
        a = np.random.randint(1, max_val)
        b = np.random.randint(1, max_val)
        op = np.random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            answer = a * b
        
        content = f"Problem: Calculate {a} {op} {b}\nAnswer: {answer}"
        
        return KnowledgeResource(
            id=hashlib.md5(f"math_{a}_{op}_{b}".encode()).hexdigest()[:16],
            title=f"Arithmetic: {a} {op} {b}",
            content=content,
            source=ResourceSource.LOCAL,
            resource_type=ResourceType.MATH,
            url="local://math/arithmetic",
            timestamp=time.time(),
            metadata={"difficulty": difficulty, "answer": answer},
        )
    
    def generate_algebra(self, difficulty: int = 1) -> KnowledgeResource:
        """生成代数问题."""
        a = np.random.randint(2, 10 * difficulty)
        x = np.random.randint(1, 10 * difficulty)
        b = np.random.randint(1, 20 * difficulty)
        result = a * x + b
        
        content = f"Problem: Solve {a}x + {b} = {result} for x\nAnswer: x = {x}"
        
        return KnowledgeResource(
            id=hashlib.md5(f"algebra_{a}_{x}_{b}".encode()).hexdigest()[:16],
            title=f"Algebra: {a}x + {b} = {result}",
            content=content,
            source=ResourceSource.LOCAL,
            resource_type=ResourceType.MATH,
            url="local://math/algebra",
            timestamp=time.time(),
            metadata={"difficulty": difficulty, "answer": x},
        )
    
    def generate_sequence(self, difficulty: int = 1) -> KnowledgeResource:
        """生成数列问题."""
        seq_type = np.random.choice(["arithmetic", "geometric", "fibonacci"])
        
        if seq_type == "arithmetic":
            start = np.random.randint(1, 10)
            diff = np.random.randint(1, 5 * difficulty)
            seq = [start + i * diff for i in range(5)]
            answer = seq[-1] + diff
            
        elif seq_type == "geometric":
            start = np.random.randint(1, 5)
            ratio = np.random.randint(2, 3 + difficulty)
            seq = [start * (ratio ** i) for i in range(5)]
            answer = seq[-1] * ratio
            
        else:  # fibonacci
            a, b = 1, 1
            seq = [a, b]
            for _ in range(3):
                a, b = b, a + b
                seq.append(b)
            answer = seq[-1] + seq[-2]
        
        seq_str = ", ".join(map(str, seq))
        content = f"Problem: Find the next number in the sequence: {seq_str}, ?\nAnswer: {answer}"
        
        return KnowledgeResource(
            id=hashlib.md5(f"seq_{seq_str}".encode()).hexdigest()[:16],
            title=f"Sequence: {seq_str}",
            content=content,
            source=ResourceSource.LOCAL,
            resource_type=ResourceType.MATH,
            url="local://math/sequence",
            timestamp=time.time(),
            metadata={"difficulty": difficulty, "answer": answer, "type": seq_type},
        )


# ============================================================================
# 知识获取管理器
# ============================================================================

class KnowledgeAcquisitionManager:
    """知识获取管理器 - 协调资源获取."""
    
    def __init__(self, config: AcquisitionConfig = None):
        self.config = config or AcquisitionConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 过滤器
        self.compliance_filter = ComplianceFilter()
        
        # 获取器
        self.wikipedia = WikipediaFetcher(self.config)
        self.arxiv = ArxivFetcher(self.config)
        self.math_gen = MathProblemGenerator()
        
        # 速率限制
        self.request_times: deque = deque(maxlen=self.config.rate_limit_requests_per_min)
        
        # 已获取资源缓存
        self.acquired_ids: set = set()
        self._load_cache()
        
        # 兴趣队列
        self.interest_queue: deque = deque(maxlen=100)
        
        # 统计
        self.stats = {
            "total_acquired": 0,
            "filtered_out": 0,
            "from_wikipedia": 0,
            "from_arxiv": 0,
            "from_local": 0,
        }
    
    def add_interest(self, topic: str, priority: float = 1.0):
        """添加兴趣主题."""
        self.interest_queue.append({
            "topic": topic,
            "priority": priority,
            "added_at": time.time(),
        })
    
    def poll_resources(self) -> Generator[KnowledgeResource, None, None]:
        """轮询获取资源."""
        # 检查速率限制
        if not self._check_rate_limit():
            return
        
        acquired = 0
        max_per_poll = self.config.max_resources_per_poll
        
        # 1. 基于兴趣获取
        while self.interest_queue and acquired < max_per_poll // 2:
            interest = self.interest_queue.popleft()
            topic = interest["topic"]
            
            # 尝试 Wikipedia
            for resource in self._fetch_wikipedia(topic):
                if self._process_resource(resource):
                    yield resource
                    acquired += 1
                    if acquired >= max_per_poll:
                        return
            
            # 尝试 arXiv (仅科学主题)
            if any(kw in topic.lower() for kw in ["ai", "machine", "neural", "learning", "math"]):
                for resource in self._fetch_arxiv(topic):
                    if self._process_resource(resource):
                        yield resource
                        acquired += 1
                        if acquired >= max_per_poll:
                            return
        
        # 2. 生成数学问题
        while acquired < max_per_poll:
            difficulty = np.random.randint(1, 4)
            gen_type = np.random.choice(["arithmetic", "algebra", "sequence"])
            
            if gen_type == "arithmetic":
                resource = self.math_gen.generate_arithmetic(difficulty)
            elif gen_type == "algebra":
                resource = self.math_gen.generate_algebra(difficulty)
            else:
                resource = self.math_gen.generate_sequence(difficulty)
            
            if self._process_resource(resource):
                yield resource
                acquired += 1
    
    def _fetch_wikipedia(self, topic: str) -> List[KnowledgeResource]:
        """从 Wikipedia 获取."""
        results = []
        
        titles = self.wikipedia.search(topic, limit=3)
        for title in titles:
            resource = self.wikipedia.get_summary(title)
            if resource:
                results.append(resource)
                self.stats["from_wikipedia"] += 1
        
        return results
    
    def _fetch_arxiv(self, topic: str) -> List[KnowledgeResource]:
        """从 arXiv 获取."""
        results = self.arxiv.search(topic, max_results=3)
        self.stats["from_arxiv"] += len(results)
        return results
    
    def _process_resource(self, resource: KnowledgeResource) -> bool:
        """处理资源 (过滤、去重、缓存)."""
        # 去重
        if resource.id in self.acquired_ids:
            return False
        
        # 合规过滤
        if not self.compliance_filter.filter_resource(resource):
            self.stats["filtered_out"] += 1
            return False
        
        # 缓存
        self.acquired_ids.add(resource.id)
        self._cache_resource(resource)
        
        self.stats["total_acquired"] += 1
        
        # 记录请求时间
        self.request_times.append(time.time())
        
        return True
    
    def _check_rate_limit(self) -> bool:
        """检查速率限制."""
        if len(self.request_times) < self.config.rate_limit_requests_per_min:
            return True
        
        # 检查最旧请求是否超过 1 分钟
        oldest = self.request_times[0]
        return time.time() - oldest > 60
    
    def _cache_resource(self, resource: KnowledgeResource):
        """缓存资源到本地."""
        cache_path = self.config.cache_dir / f"{resource.id}.json"
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                "id": resource.id,
                "title": resource.title,
                "content": resource.content,
                "source": resource.source.value,
                "resource_type": resource.resource_type.value,
                "url": resource.url,
                "timestamp": resource.timestamp,
                "metadata": resource.metadata,
            }, f, ensure_ascii=False, indent=2)
    
    def _load_cache(self):
        """加载缓存."""
        for cache_file in self.config.cache_dir.glob("*.json"):
            self.acquired_ids.add(cache_file.stem)
    
    def load_cached_resource(self, resource_id: str) -> Optional[KnowledgeResource]:
        """加载缓存的资源."""
        cache_path = self.config.cache_dir / f"{resource_id}.json"
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return KnowledgeResource(
                id=data["id"],
                title=data["title"],
                content=data["content"],
                source=ResourceSource(data["source"]),
                resource_type=ResourceType(data["resource_type"]),
                url=data["url"],
                timestamp=data["timestamp"],
                metadata=data.get("metadata", {}),
            )
        except Exception:
            return None
    
    def get_all_cached(self) -> Generator[KnowledgeResource, None, None]:
        """获取所有缓存资源."""
        for resource_id in self.acquired_ids:
            resource = self.load_cached_resource(resource_id)
            if resource:
                yield resource
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        return {
            **self.stats,
            "cached_resources": len(self.acquired_ids),
            "pending_interests": len(self.interest_queue),
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_knowledge_acquisition_manager(
    poll_interval: float = 300.0,
    cache_dir: str = None
) -> KnowledgeAcquisitionManager:
    """创建知识获取管理器."""
    config = AcquisitionConfig(
        poll_interval_sec=poll_interval,
        cache_dir=Path(cache_dir) if cache_dir else Path("./knowledge_cache"),
    )
    return KnowledgeAcquisitionManager(config)


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("H2Q 知识获取模块 - 演示")
    print("=" * 60)
    
    # 创建管理器
    manager = create_knowledge_acquisition_manager(cache_dir="./test_knowledge_cache")
    
    # 添加兴趣
    print("\n1. 添加学习兴趣...")
    interests = ["artificial intelligence", "quaternion mathematics", "neural networks"]
    for topic in interests:
        manager.add_interest(topic)
        print(f"  添加: {topic}")
    
    # 轮询获取
    print("\n2. 获取知识资源...")
    count = 0
    for resource in manager.poll_resources():
        print(f"\n  [{resource.source.value}] {resource.title}")
        print(f"    类型: {resource.resource_type.value}")
        print(f"    内容: {resource.content[:100]}...")
        count += 1
        
        if count >= 5:
            break
    
    # 统计
    print("\n3. 获取统计:")
    stats = manager.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
