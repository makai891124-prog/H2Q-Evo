import json
import os
from colorama import Fore, Style, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

def revive():
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}找不到状态文件。")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    todos = state.get('todo_list', [])
    failed_tasks = [t for t in todos if t.get('status') == 'failed']
    
    if not failed_tasks:
        print(f"{Fore.GREEN}🎉 太棒了！当前没有失败的任务。系统非常健康。")
        return

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN} 🧟‍♂️ 发现 {len(failed_tasks)} 个失败任务，准备复活...")
    print(f"{Fore.CYAN}{'='*60}")

    revived_count = 0
    for t in failed_tasks:
        old_prio = t.get('priority', 'medium')
        
        # 1. 打印详情
        print(f"{Fore.YELLOW}ID {t.get('id')}: {t.get('task')[:60]}...")
        print(f"   原状态: failed | 原重试数: {t.get('retry_count')}")
        
        # 2. 执行复活手术
        t['status'] = 'pending'
        t['retry_count'] = 0
        t['priority'] = 'high' # 提权，让它们插队执行
        
        # 3. 清理可能的错误标记（如果有的话）
        if 'error_log' in t: del t['error_log']
        
        revived_count += 1
        print(f"{Fore.GREEN}   ✅ 已复活 (Priority -> High)\n")

    # 4. 保存更改
    if revived_count > 0:
        # 将复活的任务移动到列表前面（仅次于 critical）
        # 简单的排序策略：pending 的排前面
        todos.sort(key=lambda x: 0 if x['status'] == 'pending' else 1)
        
        state['todo_list'] = todos
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.MAGENTA} 🚀 成功复活 {revived_count} 个任务！")
        print(f"{Fore.MAGENTA} 请重启 evolution_system.py，AI 将立即重新尝试这些任务。")
        print(f"{Fore.MAGENTA}{'='*60}")

if __name__ == "__main__":
    revive()