#!/usr/bin/env python3
"""
Knowledge Artifacts: 证明工件与置信度可解释记录

提供统一的工件结构、序列化与简化的“纽结式编码”模拟，以便跨智能体传承。
注意：纽结编码为审计用途的结构同构模拟，不等同严格拓扑不变量。
"""

import os
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


ARTIFACT_ROOT = Path("knowledge_artifacts")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def confidence_details(base: float, knowledge_count: int, complexity: str, noise: float) -> Dict[str, Any]:
    knowledge_boost = min(knowledge_count * 0.05, 0.2)
    complexity_factor = {"low": 0.15, "medium": 0.10, "high": 0.05}.get(complexity, 0.10)
    raw = base + knowledge_boost + complexity_factor + noise
    return {
        "base": base,
        "knowledge_boost": knowledge_boost,
        "complexity_factor": complexity_factor,
        "noise": noise,
        "raw": raw,
        "final": min(max(raw, 0.0), 0.95)
    }


def encode_knot_like(text: str) -> Dict[str, Any]:
    """
    生成简化的“纽结式编码”模拟：
    - 将文本映射为哈希
    - 从哈希导出固定长度的偶数序列，构造“Dowker-like”对序列
    - 提供压缩位串摘要与长度元数据
    用于跨系统的一致性与同构审计（非严格拓扑不变量）。
    """
    h = sha256_hex(text.encode("utf-8"))
    # 取前64个十六进制字符，映射到0-255，再投影到偶数索引空间
    bytes_arr = [int(h[i:i+2], 16) for i in range(0, 64, 2)]
    n = len(bytes_arr)
    # 生成偶数序列并两两配对作为“Dowker-like”编码
    even_seq = [(b // 2) * 2 for b in bytes_arr]
    pairs = []
    for i in range(0, n - 1, 2):
        a, b = even_seq[i], even_seq[i + 1]
        if a == b:
            b = (b + 2) % 256
        pairs.append([int(a), int(b)])
    bit_summary = bin(int(h, 16))[2:][:128]
    return {
        "hash": h,
        "dowker_like_pairs": pairs,
        "bit_summary_128": bit_summary,
        "lengths": {
            "text_len": len(text),
            "pairs": len(pairs)
        }
    }


def make_proof_artifact(
    *,
    session_id: str,
    reasoning_id: int,
    query: str,
    domain: str,
    analysis: Dict[str, Any],
    knowledge_used: List[Dict[str, Any]],
    confidence_info: Dict[str, Any],
    response: str,
    system: str,
    template: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ts = time.time()
    content_concat = "\n".join([k.get("content", "") for k in knowledge_used])
    encoding = encode_knot_like(query + "\n" + content_concat)
    artifact = {
        "schema_version": "1.0",
        "metadata": {
            "session_id": session_id,
            "reasoning_id": reasoning_id,
            "timestamp": ts,
            "system": system,
        },
        "problem": {
            "query": query,
            "domain": domain,
            "analysis": analysis,
        },
        "knowledge": [
            {
                "content": k.get("content"),
                "confidence": k.get("confidence"),
                "timestamp": k.get("timestamp"),
            } for k in knowledge_used
        ],
        "reasoning": {
            "steps": [
                {
                    "type": "decomposition",
                    "text": f"将问题分解到领域 {analysis.get('detected_domain', 'general')} 的子问题",
                    "confidence": 0.9,
                },
                {
                    "type": "knowledge_application",
                    "text": "应用检索到的相关知识若干条以支撑推导",
                    "confidence": min(1.0, max(0.0, sum([k.get('confidence', 0.7) for k in knowledge_used]) / max(1, len(knowledge_used))))
                },
                {
                    "type": "inference",
                    "text": "依据知识与逻辑规则合成结论",
                    "confidence": 0.8,
                }
            ],
            "response": response,
        },
        "confidence": confidence_info,
        "encoding_isomorphism": {
            "note": "使用哈希导出的Dowker-like对序列做一致性/同构审计（非拓扑不变量）",
            "knot_like": encoding,
        },
        "template": template or {},
        "integrity": {
            "content_hash": sha256_hex((query + response).encode("utf-8")),
        }
    }
    return artifact


def write_artifact(artifact: Dict[str, Any]) -> Path:
    session = artifact["metadata"]["session_id"]
    rid = artifact["metadata"]["reasoning_id"]
    out_dir = ARTIFACT_ROOT / session
    ensure_dir(out_dir)
    out_path = out_dir / f"proof_{rid:05d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    return out_path


def export_bundle(output_path: Path) -> Dict[str, Any]:
    """
    将所有工件导出为JSONL Bundle，便于跨智能体传播。
    格式：首行为header，后续每行为单个artifact对象。
    """
    ensure_dir(output_path.parent)
    lines = []
    count = 0
    for root, _, files in os.walk(ARTIFACT_ROOT):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            p = Path(root) / fn
            with open(p, "rb") as f:
                data = f.read()
            lines.append(data)
            count += 1
    header = {
        "schema": "knowledge_artifact_bundle/1.0",
        "created_at": time.time(),
        "artifacts": count,
        "root": str(ARTIFACT_ROOT),
    }
    with open(output_path, "wb") as f:
        f.write(json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n")
        for data in lines:
            f.write(data + b"\n")
    return {"written": count, "path": str(output_path)}


if __name__ == "__main__":
    # 简单自检：导出空/现有工件为bundle
    out = export_bundle(Path("knowledge_artifacts_bundle.jsonl"))
    print(out)
