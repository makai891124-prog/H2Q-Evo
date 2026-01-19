import os
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("./h2q_project").resolve()
DATASET_FILE = "h2q_evolution_dataset.jsonl"

def get_git_history():
    cmd = ["git", "log", "--pretty=format:%H|%s", "--reverse"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return [line.split("|", 1) for line in result.stdout.strip().split("\n") if "|" in line]

def get_file_content_at_commit(commit_hash, file_path):
    try:
        cmd = ["git", "show", f"{commit_hash}:{file_path}"]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        return result.stdout
    except: return ""

def harvest():
    print(">>> 正在收割进化历史，构建本地训练集...")
    commits = get_git_history()
    dataset = []

    for i in range(1, len(commits)):
        curr_hash, msg = commits[i]
        
        # 提取任务描述
        task_desc = msg.split(":", 1)[-1].split("-")[0].strip()
        if not task_desc: continue
        
        # 找到这次提交修改的 .py 文件
        cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", curr_hash]
        files = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
        py_files = [f for f in files if f.endswith(".py")]
        
        for f in py_files:
            target_code = get_file_content_at_commit(curr_hash, f)
            if not target_code: continue

            # 构建 Prompt -> Code 样本
            sample = {
                "instruction": f"Based on the task '{task_desc}', generate the full Python code for the file '{f}'.",
                "output": target_code
            }
            dataset.append(sample)
            print(f"  + 收录样本: {task_desc[:30]}... -> {f}")

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\n✅ 数据集构建完成！共 {len(dataset)} 条高质量进化样本。")
    print(f"📂 保存位置: {DATASET_FILE}")

if __name__ == "__main__":
    harvest()