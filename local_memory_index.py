#!/usr/bin/env python3
"""
H2Q-Evo 本地记忆索引
===================================

为进化监督系统提供知识检索能力
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class OfflineMemoryIndex:
    """极简离线记忆索引（为进化系统优化）"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index = []  # [{"path": str, "size": int, "domain": str, "content": str}]

    def build(self, max_files: int = 200):
        """构建索引"""
        if not self.root.exists():
            print(f"⚠️ 离线语料目录不存在: {self.root}")
            return

        count = 0
        for path in self.root.rglob("*.txt"):
            if count >= max_files:
                break
            if path.stat().st_size > 512 * 1024 * 1024:  # 512 MB 限制
                continue

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()[:10000]  # 只索引前10000字符

                self.index.append({
                    "path": str(path),
                    "size": path.stat().st_size,
                    "domain": path.parent.name,
                    "content": content
                })
                count += 1
            except Exception as e:
                print(f"⚠️ 索引文件失败 {path}: {e}")

    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_size = sum(item["size"] for item in self.index)
        return {
            "files_indexed": len(self.index),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "domains": list(set(item["domain"] for item in self.index))
        }

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """简单关键词搜索"""
        results = []
        query_lower = query.lower()

        for item in self.index:
            content_lower = item["content"].lower()
            if query_lower in content_lower:
                # 计算相关性分数（简化版）
                score = content_lower.count(query_lower)
                results.append({
                    **item,
                    "score": score
                })

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]