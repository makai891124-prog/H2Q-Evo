#!/usr/bin/env python3
"""
验证 knowledge_artifacts/ 下的所有证明工件是否符合 Schema。
可选依赖: jsonschema

用法:
  python3 validate_artifacts.py
"""
import sys
import json
import glob
from pathlib import Path

SCHEMA_PATH = Path("knowledge_artifact.schema.json")
ARTIFACT_GLOB = "knowledge_artifacts/**/*.json"

def main():
    try:
        from jsonschema import validate, ValidationError, Draft7Validator
    except Exception:
        print("[warn] 未安装 jsonschema，跳过严格校验，仅做基本JSON加载检查。")
        strict = False
    else:
        strict = True

    files = sorted(glob.glob(ARTIFACT_GLOB, recursive=True))
    if not files:
        print("[info] 未发现任何工件文件。路径: knowledge_artifacts/")
        return 0

    schema = None
    if strict:
        try:
            schema = json.load(open(SCHEMA_PATH, 'r', encoding='utf-8'))
            validator = Draft7Validator(schema)
        except Exception as e:
            print(f"[warn] 读取Schema失败: {e}\n改为基本JSON检查模式。")
            strict = False

    ok = 0
    fail = 0
    for f in files:
        try:
            data = json.load(open(f, 'r', encoding='utf-8'))
            if strict and schema is not None:
                errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
                if errors:
                    print(f"[FAIL] {f}")
                    for e in errors[:5]:
                        loc = "/".join([str(x) for x in e.path])
                        print(f"  - {loc}: {e.message}")
                    fail += 1
                    continue
            ok += 1
        except Exception as e:
            print(f"[FAIL] {f}: {e}")
            fail += 1

    print(f"验证完成: OK={ok}, FAIL={fail}, 总计={ok+fail}")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
