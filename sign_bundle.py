#!/usr/bin/env python3
"""
生成/验证 Bundle 签名：默认 SHA-256；如提供环境变量 BUNDLE_SIGN_KEY，则生成 HMAC-SHA256。

默认: 生成 knowledge_artifacts_bundle.jsonl.sha256
--verify: 校验现有 bundle 与签名一致
"""
import argparse
import hashlib
from pathlib import Path
import os

BUNDLE = Path("knowledge_artifacts_bundle.jsonl")
SIG = Path("knowledge_artifacts_bundle.jsonl.sha256")

def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hmac_sha256(path: Path, key: bytes) -> str:
    h = hashlib.pbkdf2_hmac("sha256", path.read_bytes(), key, 1, dklen=32)
    return h.hex()

def sign():
    if not BUNDLE.exists():
        raise SystemExit(f"bundle not found: {BUNDLE}")
    key = os.getenv("BUNDLE_SIGN_KEY")
    if key:
        digest = hmac_sha256(BUNDLE, key.encode("utf-8"))
        mode = "HMAC-SHA256"
    else:
        digest = sha256_hex(BUNDLE)
        mode = "SHA-256"
    SIG.write_text(digest + "\n", encoding="utf-8")
    print(f"signed {BUNDLE} -> {SIG} [{mode}]")


def verify():
    if not BUNDLE.exists():
        raise SystemExit(f"bundle not found: {BUNDLE}")
    if not SIG.exists():
        raise SystemExit(f"signature not found: {SIG}")
    recorded = SIG.read_text(encoding="utf-8").strip()
    key = os.getenv("BUNDLE_SIGN_KEY")
    if key:
        current = hmac_sha256(BUNDLE, key.encode("utf-8"))
        mode = "HMAC-SHA256"
    else:
        current = sha256_hex(BUNDLE)
        mode = "SHA-256"
    if recorded == current:
        print(f"signature OK [{mode}]")
    else:
        raise SystemExit(f"signature mismatch! recorded={recorded} current={current} mode={mode}")


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
