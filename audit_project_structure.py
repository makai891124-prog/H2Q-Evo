import os
import ast
import sys
from pathlib import Path
from collections import defaultdict

# 配置
PROJECT_ROOT = Path("./h2q_project").resolve()
REPORT_FILE = "PROJECT_AUDIT_REPORT.md"

def get_imports(file_path):
    """解析 Python 文件获取导入关系"""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        pass
    return imports

def extract_math_logic(file_path):
    """提取类定义和文档字符串，分析数学实现"""
    logic_summary = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or "No documentation."
                # 简单的关键词过滤，只关注数学相关的类
                keywords = ['Quaternion', 'Manifold', 'Fractal', 'Knot', 'Spectral', 'Berry', 'Topology', 'Tensor', 'Gradient']
                if any(k.lower() in node.name.lower() or k.lower() in doc.lower() for k in keywords):
                    logic_summary.append(f"- **Class `{node.name}`**\n  - *Doc*: {doc.strip().splitlines()[0]}")
                    # 检查关键方法
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    if 'forward' in methods:
                        logic_summary.append(f"  - *Methods*: {', '.join(methods[:5])}...")
    except:
        pass
    return logic_summary

def generate_audit():
    print(f"🔍 正在深度扫描项目结构: {PROJECT_ROOT} ...")
    
    structure_map = []
    dependency_graph = defaultdict(set)
    math_implementation = []
    
    # 1. 遍历文件
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # 忽略干扰项
        if any(x in root for x in ["__pycache__", ".git", "data_", "temp_sandbox", "venv"]):
            continue
            
        level = root.replace(str(PROJECT_ROOT), '').count(os.sep)
        indent = ' ' * 4 * (level)
        rel_dir = os.path.basename(root)
        structure_map.append(f"{indent}📂 **{rel_dir}/**")
        
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)
                structure_map.append(f"{indent}    📄 `{file}`")
                
                # 分析依赖
                imps = get_imports(file_path)
                module_name = file.replace('.py', '')
                for imp in imps:
                    # 只记录内部依赖 (h2q 开头)
                    if imp.startswith('h2q'):
                        dependency_graph[str(rel_path)].add(imp)
                
                # 分析数学逻辑
                math_info = extract_math_logic(file_path)
                if math_info:
                    math_implementation.append(f"### 📄 {rel_path}")
                    math_implementation.extend(math_info)
                    math_implementation.append("")

    # 2. 生成报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# H2Q 项目全景审计报告\n\n")
        
        f.write("## 1. 文件目录树 (File Structure)\n")
        f.write("\n".join(structure_map))
        f.write("\n\n")
        
        f.write("## 2. 核心数学实现 (Mathematical Core)\n")
        f.write("> 以下模块包含关键的几何/拓扑/代数逻辑实现：\n\n")
        f.write("\n".join(math_implementation))
        f.write("\n")
        
        f.write("## 3. 依赖关系图 (Dependency Graph)\n")
        f.write("```mermaid\ngraph TD\n")
        # 生成 Mermaid 图表
        for src, dests in dependency_graph.items():
            src_node = src.replace('/', '_').replace('.', '_')
            for dest in dests:
                dest_node = dest.replace('/', '_').replace('.', '_')
                if src_node != dest_node:
                    f.write(f"    {src_node} --> {dest_node}\n")
        f.write("```\n")

    print(f"✅ 审计完成！报告已生成: {os.path.abspath(REPORT_FILE)}")
    print("👉 您可以使用 VS Code 打开此文件，并安装 'Markdown Preview Mermaid Support' 插件查看架构图。")

if __name__ == "__main__":
    generate_audit()