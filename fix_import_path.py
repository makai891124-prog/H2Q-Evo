import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

def fix_task():
    if not os.path.exists(STATE_FILE): return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    todos = state.get('todo_list', [])
    target_id = 130
    found = False
    
    for t in todos:
        if t['id'] == target_id:
            print(f"{Fore.YELLOW}>>> 找到任务 [130]。正在注入路径提示...")
            
            # 原始任务
            original_task = t['task']
            
            # 注入强提示
            hint = " [CRITICAL HINT: The adapter class 'UniversalAdapter' is located in 'h2q/core/adapter.py'. You MUST use 'from h2q.core.adapter import UniversalAdapter'. DO NOT import from 'h2q_project' or 'middleware'.]"
            
            # 避免重复添加
            if "CRITICAL HINT" not in original_task:
                t['task'] = original_task + hint
            
            # 重置状态，让它立即重试
            t['status'] = 'pending'
            t['retry_count'] = 0
            
            found = True
            print(f"{Fore.GREEN}✅ 任务描述已更新，路径指引已添加。")
            break
    
    if found:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"{Fore.MAGENTA}>>> 请重启 evolution_system.py")
    else:
        print(f"{Fore.RED}❌ 未找到 ID 为 130 的任务。")

if __name__ == "__main__":
    fix_task()