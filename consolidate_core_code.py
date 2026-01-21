#!/usr/bin/env python3
"""
精简版项目代码合并工具 - 仅包含核心源代码
排除文档、测试、临时文件等，专注于业务逻辑代码
"""

import os
import json
from pathlib import Path
from datetime import datetime

# 配置
PROJECT_ROOT = Path("/Users/imymm/H2Q-Evo")
OUTPUT_FILE = PROJECT_ROOT / "PROJECT_CORE_CODE_SUMMARY.md"

# 要忽略的目录（针对精简版）
IGNORE_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "*.egg-info",
    ".coverage",
    "dist",
    "build",
    ".vscode",
    ".idea",
    ".DS_Store",
    "test",
    "tests",
    "docs",
    "documentation",
    "examples",
    "samples",
    "__pycache__",
}

# 仅包含的主要代码目录
INCLUDE_DIRS = {"h2q_project", "src", "lib"}

# 要忽略的文件模式
IGNORE_FILES = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".exe",
    ".lock", ".lockfile", ".package-lock.json", ".yarn.lock",
    ".h5", ".pb", ".ckpt", ".bin", "evolution.log", ".log"
}

# 要包含的代码文件扩展名
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash", ".sql", ".html", ".css"
}

def should_ignore(path: Path, is_dir: bool) -> bool:
    """检查是否应该忽略该路径"""
    if is_dir:
        for ignore_pattern in IGNORE_DIRS:
            pattern = ignore_pattern.lstrip("*").rstrip("*")
            if pattern in path.name or path.name == pattern:
                return True
    
    if not is_dir:
        for ignore_pattern in IGNORE_FILES:
            if path.suffix == ignore_pattern or path.name == ignore_pattern:
                return True
        
        if path.suffix.lower() not in CODE_EXTENSIONS:
            if path.name not in {"Dockerfile", "Makefile", "LICENSE", "README"}:
                return True
    
    return False

def get_core_files(root: Path) -> list:
    """获取核心代码文件"""
    files = []
    
    try:
        # 优先从h2q_project、src等核心目录获取
        for include_dir in INCLUDE_DIRS:
            dir_path = root / include_dir
            if dir_path.exists():
                for item in sorted(dir_path.rglob("*")):
                    if should_ignore(item, item.is_dir()):
                        continue
                    if item.is_file():
                        files.append(item)
        
        # 也包括根目录的主要脚本
        for item in sorted(root.glob("*.py")):
            if not should_ignore(item, False):
                files.append(item)
    except Exception as e:
        print(f"扫描目录时出错: {e}")
    
    return sorted(list(set(files)))

def read_file_safely(file_path: Path) -> str:
    """安全地读取文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            return f"[无法读取文件: {e}]"
    except Exception as e:
        return f"[读取文件时出错: {e}]"

def get_language_for_extension(file_path: Path) -> str:
    """获取代码块语言标记"""
    ext = file_path.suffix.lower()
    language_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".jsx": "javascript", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini",
        ".sh": "bash", ".bash": "bash", ".sql": "sql", ".html": "html",
        ".css": "css",
    }
    return language_map.get(ext, "text")

def truncate_content(content: str, max_lines: int = 500) -> tuple:
    """截断内容到指定行数"""
    lines = content.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n...\n[内容已截断]", True
    return content, False

def generate_summary() -> None:
    """生成核心代码总结"""
    print(f"开始扫描项目核心代码: {PROJECT_ROOT}")
    
    files = get_core_files(PROJECT_ROOT)
    print(f"发现 {len(files)} 个核心代码文件")
    
    content_parts = []
    
    # 头部
    header = f"""# H2Q-Evo 项目核心代码总结

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**项目路径**: {PROJECT_ROOT}

> 本文档是项目核心源代码的精简合并，用于快速逻辑一致性分析。

## 📋 目录导航

"""
    content_parts.append(header)
    
    # 按文件夹分类
    files_by_dir = {}
    for file_path in files:
        rel_path = file_path.relative_to(PROJECT_ROOT)
        dir_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "根目录"
        
        if dir_name not in files_by_dir:
            files_by_dir[dir_name] = []
        files_by_dir[dir_name].append(file_path)
    
    # 生成导航
    for dir_name in sorted(files_by_dir.keys()):
        files_in_dir = files_by_dir[dir_name]
        content_parts.append(f"\n### {dir_name} ({len(files_in_dir)} 个文件)\n")
        for f in sorted(files_in_dir):
            rel_path = f.relative_to(PROJECT_ROOT)
            content_parts.append(f"- `{rel_path}`\n")
    
    # 文件详情
    content_parts.append("\n\n---\n\n## 📝 源代码详情\n\n")
    
    total_lines = 0
    total_size = 0
    
    for idx, file_path in enumerate(files, 1):
        rel_path = file_path.relative_to(PROJECT_ROOT)
        
        try:
            content = read_file_safely(file_path)
            original_lines = len(content.splitlines())
            truncated_content, is_truncated = truncate_content(content, max_lines=200)
            language = get_language_for_extension(file_path)
            size = len(content.encode("utf-8"))
            
            total_lines += original_lines
            total_size += size
            
            file_header = f"""### {idx}. {rel_path}

**信息**: {original_lines} 行 | {size / 1024:.1f} KB

```{language}
{truncated_content}
```

---

"""
            content_parts.append(file_header)
            
            if idx % 5 == 0:
                print(f"已处理 {idx}/{len(files)} 个文件...")
        
        except Exception as e:
            print(f"处理文件出错 {file_path}: {e}")
    
    # 总结
    summary = f"""
---

## 📊 代码统计

| 指标 | 值 |
|------|-----|
| 核心文件数 | {len(files)} |
| 总代码行数 | {total_lines:,} |
| 总代码大小 | {total_size / (1024*1024):.2f} MB |
| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |

---

**说明**:
- 此版本仅包含核心业务代码，排除了测试、文档和大型生成文件
- 每个文件内容被限制到前 200 行用于快速浏览
- 适合进行项目逻辑架构分析
"""
    
    content_parts.append(summary)
    
    final_content = "".join(content_parts)
    
    print(f"\n正在写入输出文件: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_content)
    
    output_size = len(final_content.encode("utf-8"))
    print(f"\n✅ 完成!")
    print(f"📊 统计:")
    print(f"   - 输出文件: {OUTPUT_FILE}")
    print(f"   - 输出大小: {output_size / (1024*1024):.2f} MB")
    print(f"   - 包含文件数: {len(files)}")
    print(f"   - 总代码行数: {total_lines:,}")

if __name__ == "__main__":
    generate_summary()
