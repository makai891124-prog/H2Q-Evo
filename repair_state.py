import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

def repair():
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}找不到状态文件。")
        return

    print(f"{Fore.CYAN}>>> 正在扫描并修复状态文件...")
    
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    except json.JSONDecodeError:
        print(f"{Fore.RED}JSON 文件格式严重损坏，无法读取！")
        return

    todos = state.get('todo_list', [])
    fixed_count = 0
    
    for i, task in enumerate(todos):
        # 检查并修复 status
        if 'status' not in task:
            print(f"{Fore.YELLOW}⚠️  发现坏点: 任务 ID {task.get('id', 'Unknown')} 缺少 'status'。已修复为 'pending'。")
            task['status'] = 'pending'
            fixed_count += 1
            
        # 检查并修复 priority
        if 'priority' not in task:
            task['priority'] = 'medium'
            fixed_count += 1
            
        # 检查并修复 task 描述
        if 'task' not in task:
            task['task'] = "Unknown Task (Auto-Repaired)"
            fixed_count += 1

    if fixed_count > 0:
        state['todo_list'] = todos
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"{Fore.GREEN}✅ 修复完成！共修复了 {fixed_count} 个数据坏点。")
    else:
        print(f"{Fore.GREEN}✅ 文件健康，未发现结构错误。")

if __name__ == "__main__":
    repair()