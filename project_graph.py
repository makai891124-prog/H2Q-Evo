import os
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple

def get_import_path(file_path: Path, root: Path) -> str:
    try:
        relative = file_path.relative_to(root)
        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            parts.pop()
        else:
            parts[-1] = parts[-1].replace(".py", "")
        return ".".join(parts)
    except:
        return str(file_path)

def parse_symbols(file_path: Path, import_path: str) -> Dict[str, str]:
    """解析文件，返回 {符号名: 类型} 字典"""
    symbols = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols[node.name] = "class"
            elif isinstance(node, ast.FunctionDef):
                symbols[node.name] = "function"
            # 也可以解析赋值变量，但这可能导致噪音，暂不处理
    except:
        pass
    return symbols

def generate_interface_map(root_dir: str) -> Tuple[str, Dict[str, str]]:
    """
    返回:
    1. 可读的接口地图字符串 (供 AI 阅读)
    2. 符号反向索引字典 {symbol_name: import_path} (供系统自动纠错)
    """
    root = Path(root_dir).resolve()
    report = ["=== H2Q GLOBAL INTERFACE REGISTRY ==="]
    symbol_index = {} # 反向索引
    
    for dirpath, _, filenames in os.walk(root):
        if any(x in dirpath for x in ["venv", "__pycache__", ".git", "temp_sandbox"]):
            continue
            
        for filename in filenames:
            if filename.endswith(".py"):
                full_path = Path(dirpath) / filename
                import_path = get_import_path(full_path, root)
                
                file_symbols = parse_symbols(full_path, import_path)
                
                if file_symbols:
                    report.append(f"\nMODULE: {import_path}")
                    for name, type_ in file_symbols.items():
                        report.append(f"  - {type_} {name}")
                        # 构建反向索引，如果有重名，保留路径较短的（通常是核心库）
                        if name not in symbol_index or len(import_path) < len(symbol_index[name]):
                            symbol_index[name] = import_path
                    
    return "\n".join(report), symbol_index

if __name__ == "__main__":
    report, index = generate_interface_map("./h2q_project")
    print(report)
    print("\n--- Symbol Index Sample ---")
    print(list(index.items())[:5])