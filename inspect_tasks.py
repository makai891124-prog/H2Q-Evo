import json
import os
from colorama import Fore, Style, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

def inspect():
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}找不到 {STATE_FILE}")
        return

    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}读取失败: {e}")
        return
    
    todos = state.get('todo_list', [])
    
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN} 📋 H2Q-Evo 任务队列透视 (Gen {state.get('generation', 0)})")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{'ID':<6} | {'状态':<10} | {'优先级':<10} | {'任务内容 (前50字)'}")
    print("-" * 80)

    # 使用 .get() 防止 KeyError
    pending = [t for t in todos if t.get('status') == 'pending']
    completed = [t for t in todos if t.get('status') == 'completed']
    failed = [t for t in todos if t.get('status') == 'failed']
    # 捕获那些状态未知的幽灵任务
    unknown = [t for t in todos if t.get('status') not in ['pending', 'completed', 'failed']]

    # 1. 待处理任务
    for t in pending:
        prio = t.get('priority', 'medium')
        color = Fore.YELLOW
        if prio == 'critical': color = Fore.MAGENTA
        elif prio == 'high': color = Fore.RED
        
        tid = t.get('id', '?')
        task_txt = t.get('task', 'No description')[:50]
        print(f"{color}{str(tid):<6} | pending    | {prio:<10} | {task_txt}...")

    # 2. 异常任务 (修复后应该没有了)
    for t in unknown:
        print(f"{Fore.BLUE}{str(t.get('id','?')):<6} | {t.get('status','N/A'):<10} | UNKNOWN    | {t.get('task','...')[:50]}")

    # 3. 失败任务
    for t in failed:
        print(f"{Fore.RED}{str(t.get('id','?')):<6} | failed     | {t.get('priority','low'):<10} | {t.get('task','...')[:50]}...")

    # 4. 已完成任务
    if completed:
        print(f"{Fore.GREEN}{'-'*80}")
        print(f"{Fore.GREEN}已完成 {len(completed)} 个任务 (显示最近 3 个):")
        for t in completed[-3:]:
            tid = t.get('id', '?')
            task_txt = t.get('task', 'No description')[:50]
            print(f"{Fore.GREEN}{str(tid):<6} | completed  | {t.get('priority','low'):<10} | {task_txt}...")

if __name__ == "__main__":
    inspect()