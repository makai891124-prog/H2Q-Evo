import sys
import os
import shutil
from pathlib import Path
import importlib.util

# 颜色库
try:
    from colorama import Fore, init
    init(autoreset=True)
except:
    class Fore: GREEN=""; RED=""; YELLOW=""; CYAN=""

PROJECT_ROOT = Path("./h2q_project").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def verify_code_writer():
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN} 🛡️  AGI 核心能力验收: 自我修改 (CodeWriter)")
    print(f"{Fore.CYAN}{'='*60}")

    target_file = PROJECT_ROOT / "tools" / "code_writer.py"
    
    if not target_file.exists():
        print(f"{Fore.RED}❌ 文件不存在！")
        return

    try:
        # 动态导入
        spec = importlib.util.spec_from_file_location("code_writer", target_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["code_writer"] = module
        spec.loader.exec_module(module)
        
        # 获取类
        WriterClass = getattr(module, "CodeWriter")
        writer = WriterClass(project_root=str(PROJECT_ROOT))
        print(f"{Fore.GREEN}✅ CodeWriter 实例化成功")

        # 准备测试数据
        test_file = "tests/agi_self_test.py"
        test_content = "print('I am H2Q, and I can write my own code.')"
        
        print(f"{Fore.YELLOW}>>> 尝试调用 write_module()...")
        
        # 调用 AI 写的方法
        success = writer.write_module(test_file, test_content, {"spectral_shift": 1.0})
        
        if success:
            full_path = PROJECT_ROOT / test_file
            if full_path.exists() and test_content in full_path.read_text():
                print(f"{Fore.GREEN}🎉🎉🎉 验证成功！")
                print(f"    AI 工具成功创建了文件: {test_file}")
                print(f"    内容验证: 通过")
                
                # 清理
                os.remove(full_path)
            else:
                print(f"{Fore.RED}❌ 方法返回 True，但文件未找到或内容不匹配。")
        else:
            print(f"{Fore.RED}❌ write_module 返回 False。")

    except Exception as e:
        print(f"{Fore.RED}❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_code_writer()