import os
import shutil
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path("./h2q_project").resolve()
SRC_DIR = PROJECT_ROOT / "src"
TARGET_ROOT = PROJECT_ROOT / "h2q"

def harvest():
    print(f"🚜 开始收割 src 目录下的高价值代码...")
    
    if not SRC_DIR.exists():
        print("❌ src 目录不存在，无需操作。")
        return

    # 定义映射规则 (源文件夹名 -> 目标文件夹名)
    # 注意：目标都是相对于 h2q/ 的
    MAPPING = {
        "grounding": "grounding",
        "h2q_core": "core",       # 将 src/h2q_core 合并入 h2q/core
        "kernels": "kernels",     # 将 src/kernels 合并入 h2q/kernels
        "visualization": "visualization"
    }

    for src_name, dest_name in MAPPING.items():
        source_path = SRC_DIR / src_name
        dest_path = TARGET_ROOT / dest_name
        
        if source_path.exists():
            print(f"   📦 正在迁移: {src_name} -> h2q/{dest_name} ...")
            
            # 确保目标父目录存在
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用 copytree 进行合并 (Python 3.8+ 支持 dirs_exist_ok)
            try:
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                print(f"      ✅ 成功合并")
            except Exception as e:
                print(f"      ❌ 移动失败: {e}")

    # 再次检查是否有遗漏的 .py 或 .metal 文件直接在 src 下
    for file in SRC_DIR.glob("*"):
        if file.is_file() and file.name != "__init__.py":
            print(f"   📄 发现散落文件: {file.name} -> 移动到 h2q/core/")
            shutil.move(str(file), str(TARGET_ROOT / "core" / file.name))

    # 清理现场
    print("🧹 清理空的 src 目录...")
    shutil.rmtree(SRC_DIR)
    print("🎉 收编完成！所有代码已归位。")

if __name__ == "__main__":
    harvest()