import os
from pathlib import Path
from colorama import Fore, init

init(autoreset=True)

STREAM_LOADER_FILE = Path("./h2q_project/h2q/data/universal_stream.py")

OLD_DATASET = '"bigcode/starcoderdata"'
NEW_DATASET = '"codeparrot/codeparrot-clean"'

def fix():
    print(f"{Fore.CYAN}>>> 正在重定向数据流以实现完全自主访问...")
    
    if not STREAM_LOADER_FILE.exists():
        print(f"{Fore.RED}❌ 错误：找不到文件 {STREAM_LOADER_FILE}")
        return

    try:
        with open(STREAM_LOADER_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if OLD_DATASET in content:
            new_content = content.replace(OLD_DATASET, NEW_DATASET)
            
            with open(STREAM_LOADER_FILE, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            print(f"{Fore.GREEN}✅ 成功将数据源切换为: {NEW_DATASET}")
        else:
            print(f"{Fore.YELLOW}⚠️ 数据源似乎已经被修改，无需操作。")
            
        print(f"\n{Fore.MAGENTA}>>> 请重启 evolution_system.py 以启动生命循环。")

    except Exception as e:
        print(f"{Fore.RED}❌ 修复失败: {e}")

if __name__ == "__main__":
    fix()