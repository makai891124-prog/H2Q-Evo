import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("./h2q_project").resolve()
REPORT_FILE = "H2Q_EVOLUTION_REPORT.md"

def get_git_logs():
    cmd = [
        "git", "log", 
        "--pretty=format:%h|%an|%ad|%s", 
        "--date=format:%Y-%m-%d %H:%M:%S",
        "--reverse" # 从最早的开始
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return [line.split("|") for line in result.stdout.strip().split("\n") if line]

def generate_markdown():
    logs = get_git_logs()
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# H2Q-Evo AGI 进化里程碑报告\n\n")
        f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 🧬 进化概览\n")
        f.write(f"- **总进化代数**: {len(logs)}\n")
        f.write(f"- **当前架构模型**: Gemini 3 Flash Preview\n")
        f.write(f"- **核心能力**: 机器梦境 (Synthetic Dreaming), 决策对齐 (DDE Alignment)\n\n")
        
        f.write("## 📈 进化时间轴\n\n")
        f.write("| 代数 (Gen) | 时间 | 修改文件/任务 | 提交哈希 |\n")
        f.write("|---|---|---|---|\n")
        
        gen_count = 0
        for log in logs:
            if len(log) < 4: continue
            hash_id, author, date, msg = log
            
            # 提取 Gen 信息
            gen_label = f"Gen {gen_count}"
            if "Evo Gen" in msg:
                try:
                    gen_label = msg.split(":")[0].replace("H2Q Evolution System: ", "").strip()
                except: pass
            
            # 提取任务描述
            task_desc = msg
            if ": " in msg:
                parts = msg.split(": ", 1)
                if len(parts) > 1:
                    task_desc = parts[1]
            
            f.write(f"| **{gen_label}** | {date} | `{task_desc}` | `{hash_id}` |\n")
            gen_count += 1
            
        f.write("\n## 🧠 核心架构快照\n")
        f.write("以下文件已被 AI 深度重构：\n")
        f.write("- `h2q/dde.py` (决策引擎)\n")
        f.write("- `h2q/data/generator.py` (合成引擎)\n")
        f.write("- `train_spacetime_vision.py` (视觉流)\n")

    print(f"✅ 报告已生成: {os.path.abspath(REPORT_FILE)}")

if __name__ == "__main__":
    generate_markdown()