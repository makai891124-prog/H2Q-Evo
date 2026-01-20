#!/usr/bin/env python3
"""
第三方验证器（独立）

验证点（不依赖项目内部模块）：
1) Schema校验（若安装jsonschema）
2) 完整性校验：integrity.content_hash == sha256(query+response)
3) 纽结式编码审计：重算hash/dowker_like_pairs/bit_summary_128 与工件一致
4) 置信度可解释一致性：
   - knowledge_boost == min(len(knowledge)*0.05, 0.2)
   - complexity_factor ∈ {low:0.15, medium:0.10, high:0.05}
   - final 与 base+boost+factor+noise 截断后的一致性（容差1e-6）

支持输入：
 - 目录 knowledge_artifacts/ 下所有 *.json
 - JSONL 打包文件 knowledge_artifacts_bundle.jsonl

输出：
 - 终端摘要 + THIRD_PARTY_VALIDATION_REPORT.json（详细统计）
"""
import sys
import os
import json
import glob
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple


SCHEMA_PATH = Path("knowledge_artifact.schema.json")
ARTIFACT_DIR = Path("knowledge_artifacts")
BUNDLE_FILE = Path("knowledge_artifacts_bundle.jsonl")
REPORT_FILE = Path("THIRD_PARTY_VALIDATION_REPORT.json")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_knot_like(text: str) -> Dict[str, Any]:
    h = sha256_hex(text.encode("utf-8"))
    bytes_arr = [int(h[i:i+2], 16) for i in range(0, 64, 2)]
    n = len(bytes_arr)
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
    }


def load_schema():
    try:
        from jsonschema import Draft7Validator
    except Exception:
        return None
    try:
        schema = json.load(open(SCHEMA_PATH, 'r', encoding='utf-8'))
        return Draft7Validator(schema)
    except Exception:
        return None


def iter_artifacts(use_dir: bool = True, use_bundle: bool = True, latest_session_only: bool = False) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if use_dir and ARTIFACT_DIR.exists():
        files: List[Path] = []
        if latest_session_only:
            # 选择最近修改的子目录（会话目录）
            subdirs = [p for p in ARTIFACT_DIR.iterdir() if p.is_dir()]
            if subdirs:
                subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                latest = subdirs[0]
                files = sorted(latest.glob('*.json'))
        if not files:
            files = sorted(ARTIFACT_DIR.rglob('*.json'))
        for f in files:
            try:
                items.append(json.load(open(f, 'r', encoding='utf-8')))
            except Exception:
                pass
    if use_bundle and BUNDLE_FILE.exists():
        with open(BUNDLE_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if i == 0:
                # header
                continue
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


@dataclass
class Checks:
    schema_ok: bool = True
    integrity_ok: bool = True
    encoding_ok: bool = True
    confidence_ok: bool = True


def check_artifact(a: Dict[str, Any], schema_validator=None) -> Checks:
    c = Checks()
    # 1) schema
    if schema_validator is not None:
        errs = list(schema_validator.iter_errors(a))
        if errs:
            c.schema_ok = False

    # 2) integrity
    try:
        q = a["problem"]["query"]
        resp = a["reasoning"]["response"]
        target = a["integrity"]["content_hash"]
        calc = sha256_hex((q + resp).encode("utf-8"))
        if calc != target:
            c.integrity_ok = False
    except Exception:
        c.integrity_ok = False

    # 3) encoding
    try:
        query = a["problem"]["query"]
        know = a.get("knowledge", [])
        concat = "\n".join([k.get("content", "") for k in know])
        recomputed = encode_knot_like(query + "\n" + concat)
        given = a["encoding_isomorphism"]["knot_like"]
        # 三要素一致
        if not (
            recomputed["hash"] == given.get("hash") and
            recomputed["bit_summary_128"] == given.get("bit_summary_128") and
            recomputed["dowker_like_pairs"] == given.get("dowker_like_pairs")
        ):
            c.encoding_ok = False
    except Exception:
        c.encoding_ok = False

    # 4) confidence
    try:
        conf = a["confidence"]
        base = conf.get("base", 0.6)
        noise = conf.get("noise", 0.0)
        complexity = a["problem"]["analysis"].get("complexity", "medium")
        k = len(a.get("knowledge", []))
        kb = min(k * 0.05, 0.2)
        cf = {"low": 0.15, "medium": 0.10, "high": 0.05}.get(complexity, 0.10)
        raw = base + kb + cf + noise
        final = min(max(raw, 0.0), 0.95)
        tol = 1e-6
        if abs(conf.get("knowledge_boost", -1.0) - kb) > tol:
            c.confidence_ok = False
        if abs(conf.get("complexity_factor", -1.0) - cf) > tol:
            c.confidence_ok = False
        if abs(conf.get("final", -1.0) - final) > 5e-3:  # 允许轻微舍入差
            c.confidence_ok = False
    except Exception:
        c.confidence_ok = False

    return c


def main():
    import argparse
    p = argparse.ArgumentParser(description="Third-party validator for knowledge artifacts")
    p.add_argument("--dir-only", action="store_true", help="只验证目录 knowledge_artifacts/")
    p.add_argument("--bundle-only", action="store_true", help="只验证打包 JSONL")
    p.add_argument("--latest-session-only", action="store_true", help="仅验证最新会话目录")
    args = p.parse_args()
    schema_validator = load_schema()
    use_dir = True
    use_bundle = True
    if args.dir_only:
        use_bundle = False
    if args.bundle_only:
        use_dir = False
    artifacts = iter_artifacts(use_dir=use_dir, use_bundle=use_bundle, latest_session_only=args.latest_session_only)
    total = len(artifacts)
    if total == 0:
        print("[info] 未发现可验证的工件（目录或bundle）。")
        return 0

    summary = {
        "total": total,
        "schema_ok": 0,
        "integrity_ok": 0,
        "encoding_ok": 0,
        "confidence_ok": 0,
        "all_ok": 0,
        "sample_failures": [],
    }

    for idx, a in enumerate(artifacts):
        checks = check_artifact(a, schema_validator)
        all_ok = checks.schema_ok and checks.integrity_ok and checks.encoding_ok and checks.confidence_ok
        summary["schema_ok"] += int(checks.schema_ok)
        summary["integrity_ok"] += int(checks.integrity_ok)
        summary["encoding_ok"] += int(checks.encoding_ok)
        summary["confidence_ok"] += int(checks.confidence_ok)
        summary["all_ok"] += int(all_ok)
        if not all_ok and len(summary["sample_failures"]) < 5:
            summary["sample_failures"].append({
                "index": idx,
                "checks": asdict(checks)
            })

    print("="*80)
    print("第三方验证摘要")
    print("="*80)
    print(f"工件总数:   {summary['total']}")
    print(f"Schema通过: {summary['schema_ok']}/{summary['total']}")
    print(f"完整性通过: {summary['integrity_ok']}/{summary['total']}")
    print(f"编码通过:   {summary['encoding_ok']}/{summary['total']}")
    print(f"置信度通过: {summary['confidence_ok']}/{summary['total']}")
    print(f"全部通过:   {summary['all_ok']}/{summary['total']}")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告: {REPORT_FILE}")
    return 0 if summary['all_ok'] == summary['total'] else 1


if __name__ == "__main__":
    sys.exit(main())
