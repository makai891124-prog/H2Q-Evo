import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path("./h2q_project").resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ROGUE_INIT = PROJECT_ROOT / "__init__.py"
GITIGNORE = PROJECT_ROOT / ".gitignore"

def fix():
    print("🧹 正在执行环境卫生清理与记忆区构建...")

    # 1. 删除根目录的 __init__.py (这是错误的，会导致 import h2q_project)
    if ROGUE_INIT.exists():
        os.remove(ROGUE_INIT)
        print(f"✅ 已删除错误的包标记: {ROGUE_INIT}")
    else:
        print(f"✅ 根目录结构正常 (无 __init__.py)")

    # 2. 创建权重文件夹 (记忆晶体仓库)
    if not CHECKPOINT_DIR.exists():
        CHECKPOINT_DIR.mkdir(parents=True)
        print(f"✅ 已创建记忆仓库: {CHECKPOINT_DIR}")
        # 创建一个空的占位文件，防止空文件夹不被注意
        (CHECKPOINT_DIR / ".keep").touch()
    else:
        print(f"✅ 记忆仓库已存在: {CHECKPOINT_DIR}")

    # 3. 配置 .gitignore (防止权重文件上传到 Git 导致仓库爆炸)
    ignore_rules = [
        "\n# --- H2Q Memory Crystals ---",
        "checkpoints/",
        "*.pt",
        "*.pth",
        "*.h2q",
        "__pycache__/",
        "*.pyc"
    ]
    
    # 读取现有规则
    current_ignore = ""
    if GITIGNORE.exists():
        current_ignore = GITIGNORE.read_text()
    
    with open(GITIGNORE, "a") as f:
        for rule in ignore_rules:
            if rule.strip() not in current_ignore:
                f.write(f"{rule}\n")
                print(f"   + 添加 Git 忽略规则: {rule.strip()}")

    print("\n🎉 物理环境重构完成！准备注入生存逻辑。")

if __name__ == "__main__":
    fix()