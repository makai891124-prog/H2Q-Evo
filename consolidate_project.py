#!/usr/bin/env python3
"""
项目代码合并工具 - 将所有源代码合并到单一Markdown文件
将排除常见的无用文件和目录，并按代码块进行结构化标记
"""

import os
import json
from pathlib import Path
from datetime import datetime

# 配置
PROJECT_ROOT = Path("/Users/imymm/H2Q-Evo")
OUTPUT_FILE = PROJECT_ROOT / "PROJECT_CODE_CONSOLIDATED.md"

# 要忽略的文件和目录
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
}

# 要忽略的文件模式
IGNORE_FILES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".lock",
    ".lockfile",
    ".package-lock.json",
    ".yarn.lock",
    ".pth",  # Python path files
    ".pt",  # PyTorch model files
    ".pth",  # PyTorch files
    ".h5",  # HDF5 files
    ".pb",  # Protocol buffer
    ".ckpt",  # Checkpoint files
    ".bin",  # Binary files
    ".so",  # Shared objects
    "evolution.log",  # Log files
    ".log",
}

# 要包含的代码文件扩展名
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
    ".bash",
    ".md",
    ".txt",
    ".dockerfile",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".sass",
}

def should_ignore(path: Path, is_dir: bool) -> bool:
    """检查是否应该忽略该路径"""
    # 检查目录名
    if is_dir:
        for ignore_pattern in IGNORE_DIRS:
            if ignore_pattern.startswith("*"):
                if path.name.endswith(ignore_pattern[1:]):
                    return True
            elif path.name == ignore_pattern:
                return True
    
    # 检查文件名和扩展名
    if not is_dir:
        for ignore_pattern in IGNORE_FILES:
            if ignore_pattern.startswith("."):
                if path.suffix == ignore_pattern or path.name.endswith(ignore_pattern):
                    return True
            if path.name == ignore_pattern:
                return True
        
        # 检查是否为允许的代码文件
        if path.suffix.lower() not in CODE_EXTENSIONS:
            if not (path.suffix == "" and path.name in {"Dockerfile", "Makefile", "LICENSE", "README"}):
                return True
    
    return False

def get_files_recursively(root: Path) -> list:
    """递归获取所有代码文件"""
    files = []
    
    try:
        for item in sorted(root.rglob("*")):
            if should_ignore(item, item.is_dir()):
                continue
            
            if item.is_file():
                files.append(item)
    except Exception as e:
        print(f"扫描目录时出错 {root}: {e}")
    
    return sorted(files)

def read_file_safely(file_path: Path) -> str:
    """安全地读取文件内容"""
    try:
        # 尝试以UTF-8编码读取
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # 如果UTF-8失败，尝试其他编码
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            return f"[无法读取文件: {e}]"
    except Exception as e:
        return f"[读取文件时出错: {e}]"

def get_language_for_extension(file_path: Path) -> str:
    """根据文件扩展名获取代码块语言标记"""
    ext = file_path.suffix.lower()
    
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "text",
        ".sh": "bash",
        ".bash": "bash",
        ".dockerfile": "dockerfile",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
    }
    
    return language_map.get(ext, "text")

def count_lines(content: str) -> int:
    """计算文件行数"""
    return len(content.splitlines())

def calculate_size(content: str) -> str:
    """计算内容大小"""
    size_bytes = len(content.encode("utf-8"))
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"

def generate_markdown() -> None:
    """生成合并后的Markdown文件"""
    print(f"开始扫描项目: {PROJECT_ROOT}")
    
    files = get_files_recursively(PROJECT_ROOT)
    print(f"发现 {len(files)} 个代码文件")
    
    # 按文件类型分组
    files_by_type = {}
    total_lines = 0
    total_size = 0
    
    content_parts = []
    
    # 头部
    header = f"""# H2Q-Evo 项目代码整体汇总

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**项目路径**: {PROJECT_ROOT}  

## 目录结构概览

此文档包含了项目中所有源代码文件的内容，按照逻辑分类和文件类型组织。

### 统计信息

- **总文件数**: {len(files)}
- **生成日期**: {datetime.now().strftime('%Y-%m-%d')}

---

## 📑 文件清单与导航

"""
    
    content_parts.append(header)
    
    # 生成文件索引
    file_index = []
    for idx, file_path in enumerate(files, 1):
        rel_path = file_path.relative_to(PROJECT_ROOT)
        try:
            content = read_file_safely(file_path)
            lines = count_lines(content)
            file_index.append({
                "idx": idx,
                "path": str(rel_path),
                "lines": lines,
                "ext": file_path.suffix,
            })
        except Exception as e:
            print(f"错误: {file_path}: {e}")
    
    # 按扩展名分组显示索引
    ext_groups = {}
    for item in file_index:
        ext = item["ext"] or "other"
        if ext not in ext_groups:
            ext_groups[ext] = []
        ext_groups[ext].append(item)
    
    for ext in sorted(ext_groups.keys()):
        items = ext_groups[ext]
        content_parts.append(f"\n### {ext} 文件 ({len(items)} 个)\n")
        for item in sorted(items, key=lambda x: x["path"]):
            content_parts.append(f"- [{item['path']}](#{item['idx']}) - {item['lines']} 行\n")
    
    # 文件内容详情
    content_parts.append("\n\n---\n\n## 📄 详细代码内容\n\n")
    
    for idx, file_path in enumerate(files, 1):
        rel_path = file_path.relative_to(PROJECT_ROOT)
        
        try:
            content = read_file_safely(file_path)
            lines = count_lines(content)
            size = calculate_size(content)
            language = get_language_for_extension(file_path)
            
            total_lines += lines
            total_size += len(content.encode("utf-8"))
            
            # 获取文件信息
            stat = file_path.stat()
            
            # 构建文件头
            file_header = f"""### {idx}. {rel_path}

**元数据**:
- **路径**: `{rel_path}`
- **大小**: {size}
- **行数**: {lines}
- **类型**: {language}
- **修改时间**: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}

**代码内容**:

```{language}
{content}
```

---

"""
            content_parts.append(file_header)
            
            # 打印进度
            if idx % 10 == 0:
                print(f"已处理 {idx}/{len(files)} 个文件...")
        
        except Exception as e:
            print(f"处理文件出错 {file_path}: {e}")
            error_content = f"""### {idx}. {rel_path}

**错误**: {e}

---

"""
            content_parts.append(error_content)
    
    # 总结
    summary = f"""
---

## 📊 项目总结统计

| 指标 | 值 |
|------|-----|
| 总文件数 | {len(files)} |
| 总代码行数 | {total_lines:,} |
| 总代码大小 | {total_size / (1024*1024):.2f} MB |
| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |

### 代码类型分布

"""
    
    ext_stats = {}
    for item in file_index:
        ext = item["ext"] or "other"
        if ext not in ext_stats:
            ext_stats[ext] = {"count": 0, "lines": 0}
        ext_stats[ext]["count"] += 1
        ext_stats[ext]["lines"] += item["lines"]
    
    summary += "| 文件类型 | 数量 | 总行数 |\n|---------|------|--------|\n"
    for ext in sorted(ext_stats.keys()):
        stats = ext_stats[ext]
        summary += f"| {ext or 'other'} | {stats['count']} | {stats['lines']:,} |\n"
    
    summary += f"""

---

**注意**: 
- 此文档自动生成，用于整体逻辑一致性分析
- 已排除二进制文件、依赖、日志等无关内容
- 可用于 AI 工具进行全局分析

"""
    
    content_parts.append(summary)
    
    # 写入文件
    final_content = "".join(content_parts)
    
    print(f"\n正在写入输出文件: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_content)
    
    output_size = len(final_content.encode("utf-8"))
    print(f"\n✅ 完成!")
    print(f"📊 统计信息:")
    print(f"   - 输出文件: {OUTPUT_FILE}")
    print(f"   - 输出大小: {output_size / (1024*1024):.2f} MB")
    print(f"   - 包含文件数: {len(files)}")
    print(f"   - 总代码行数: {total_lines:,}")

if __name__ == "__main__":
    generate_markdown()
