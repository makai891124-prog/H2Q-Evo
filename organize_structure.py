import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path("./h2q_project").resolve()
TARGET_DIR = PROJECT_ROOT / "h2q"

# 定义需要收纳的文件夹名称
FOLDERS_TO_MOVE = [
    "benchmarks",
    "core",
    "kernels",
    "models",
    "diagnostics",
    "bridge",
    "ops",
    "logic",
    "decision",
    "dna_topology"
]

def organize():
    print(f"🧹 正在整理项目结构: {PROJECT_ROOT} -> {TARGET_DIR}")
    
    if not TARGET_DIR.exists():
        print("❌ h2q 主包目录不存在！")
        return

    for folder_name in FOLDERS_TO_MOVE:
        src = PROJECT_ROOT / folder_name
        dst = TARGET_DIR / folder_name
        
        if src.exists() and src.is_dir():
            print(f"   📦 移动: {folder_name} -> h2q/{folder_name}")
            try:
                # 如果目标已存在，先合并/覆盖
                if dst.exists():
                    # 简单的策略：将源文件夹里的内容移进去，然后删掉源文件夹
                    for item in src.iterdir():
                        if item.is_dir():
                            # 递归移动比较麻烦，这里简化处理：如果目标有同名文件/文件夹，跳过或覆盖
                            # 建议使用 shutil.move 的特性
                            pass 
                    # 为安全起见，这里使用 copytree + rmtree 模拟移动合并
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    shutil.rmtree(src)
                else:
                    shutil.move(str(src), str(dst))
                
                # 确保移动后的文件夹有 __init__.py
                init_file = dst / "__init__.py"
                if not init_file.exists():
                    with open(init_file, 'w') as f: f.write("")
                    
            except Exception as e:
                print(f"   ⚠️ 移动失败: {e}")
    
    print("✅ 整理完成！结构已统一。")

if __name__ == "__main__":
    organize()