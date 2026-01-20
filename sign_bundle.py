#!/usr/bin/env python3
"""
生成/验证 Bundle 的 SHA-256 签名文件。

默认: 生成 knowledge_artifacts_bundle.jsonl.sha256
--verify: 校验现有 bundle 与签名一致
"""
import argparse
import hashlib
from pathlib import Path

BUNDLE = Path("knowledge_artifacts_bundle.jsonl")
SIG = Path("knowledge_artifacts_bundle.jsonl.sha256")

def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sign():
    if not BUNDLE.exists():
        raise SystemExit(f"bundle not found: {BUNDLE}")
    digest = sha256_hex(BUNDLE)
    SIG.write_text(digest + "\n", encoding="utf-8")
    print(f"signed {BUNDLE} -> {SIG}")


def verify():
    if not BUNDLE.exists():
        raise SystemExit(f"bundle not found: {BUNDLE}")
    if not SIG.exists():
        raise SystemExit(f"signature not found: {SIG}")
    recorded = SIG.read_text(encoding="utf-8").strip()
    current = sha256_hex(BUNDLE)
    if recorded == current:
        print("signature OK")
    else:
        raise SystemExit(f"signature mismatch! recorded={recorded} current={current}")


def main():
    ap = argparse.ArgumentParser(description="Sign or verify bundle sha256")
    ap.add_argument("--verify", action="store_true", help="verify instead of sign")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        sign()

if __name__ == "__main__":
    main()
