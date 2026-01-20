#!/usr/bin/env python3
"""
导出知识传承Bundle（JSONL），便于跨智能体传播与复用。
"""
from pathlib import Path
from knowledge_artifacts import export_bundle

def main():
    out = export_bundle(Path("knowledge_artifacts_bundle.jsonl"))
    print(f"导出完成: {out['path']} (artifacts={out['written']})")

if __name__ == "__main__":
    main()
