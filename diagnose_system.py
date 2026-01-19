import sys
import os
import subprocess
import importlib.util
from pathlib import Path

# 颜色代码
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{YELLOW}{'='*60}")
    print(f" 🔍 {title}")
    print(f"{'='*60}{RESET}")

def check_shadowing():
    print_header("1. 影子文件检测 (Shadowing Check)")
    # 检查当前目录下是否有与标准库重名的文件
    suspicious_names = ['torch.py', 'fastapi.py', 'flask.py', 'json.py', 'os.py', 'sys.py', 'typing.py']
    found_shadows = []
    
    cwd = os.getcwd()
    print(f"当前工作目录: {cwd}")
    
    for name in suspicious_names:
        if os.path.exists(os.path.join(cwd, name)):
            print(f"{RED}[危险] 发现影子文件: {name} (这会导致 import 失败！){RESET}")
            found_shadows.append(name)
    
    if not found_shadows:
        print(f"{GREEN}✅ 未发现常见的影子文件。{RESET}")
    else:
        print(f"{RED}>>> 建议立即删除这些文件！{RESET}")

def check_sys_path():
    print_header("2. Python 路径检查 (sys.path)")
    for p in sys.path:
        print(f" - {p}")

def check_installed_packages():
    print_header("3. 已安装的关键库 (Pip List)")
    key_packages = ['torch', 'fastapi', 'uvicorn', 'datasets', 'numpy']
    
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        installed = result.stdout
        
        for pkg in key_packages:
            if pkg in installed.lower():
                # 提取版本号
                line = [l for l in installed.split('\n') if l.lower().startswith(pkg)][0]
                print(f"{GREEN}✅ {line}{RESET}")
            else:
                print(f"{RED}❌ 缺失: {pkg}{RESET}")
    except Exception as e:
        print(f"Pip 检查失败: {e}")

def try_import_critical():
    print_header("4. 核心库导入测试 (Import Test)")
    
    modules_to_test = ['torch', 'fastapi', 'h2q']
    
    for mod_name in modules_to_test:
        try:
            module = importlib.import_module(mod_name)
            file_path = getattr(module, '__file__', 'built-in')
            print(f"{GREEN}✅ Import {mod_name} 成功{RESET}")
            print(f"   来源: {file_path}")
            
            # 如果 torch 的来源是当前目录，那就是大问题
            if os.getcwd() in str(file_path):
                print(f"{RED}   ⚠️ 警告: 此模块是从当前目录加载的，这是错误的！{RESET}")
                
        except ImportError as e:
            print(f"{RED}❌ Import {mod_name} 失败: {e}{RESET}")
        except Exception as e:
            print(f"{RED}❌ Import {mod_name} 发生异常: {e}{RESET}")

if __name__ == "__main__":
    check_shadowing()
    check_sys_path()
    check_installed_packages()
    try_import_critical()