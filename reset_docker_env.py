import subprocess
import os
import sys

# 颜色代码
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_cmd(cmd, ignore_error=False):
    print(f"{YELLOW}>>> 执行: {cmd}{RESET}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        if not ignore_error:
            print(f"{RED}❌ 执行失败: {e}{RESET}")
        else:
            print(f"{YELLOW}⚠️ (已忽略错误){RESET}")

def nuke_docker():
    print(f"\n{RED}{'='*60}")
    print(f" ☢️  正在执行 Docker 环境彻底清理 (Nuclear Clean)")
    print(f"{'='*60}{RESET}\n")

    # 1. 停止所有相关容器
    print("1️⃣  停止并删除相关容器...")
    # 获取所有使用 h2q-sandbox 的容器 ID
    cmd = "docker ps -a -q --filter ancestor=h2q-sandbox"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    container_ids = result.stdout.strip().split()
    
    if container_ids:
        run_cmd(f"docker stop {' '.join(container_ids)}", ignore_error=True)
        run_cmd(f"docker rm {' '.join(container_ids)}", ignore_error=True)
        print(f"{GREEN}✅ 已清理 {len(container_ids)} 个残留容器。{RESET}")
    else:
        print(f"{GREEN}✅ 没有发现残留容器。{RESET}")

    # 2. 删除镜像
    print("\n2️⃣  删除镜像 h2q-sandbox...")
    run_cmd("docker rmi -f h2q-sandbox:latest", ignore_error=True)
    # 尝试删除悬空镜像 (Dangling images)
    run_cmd("docker image prune -f", ignore_error=True)

    # 3. 清理构建缓存 (这是解决混乱的关键)
    print("\n3️⃣  清理构建缓存 (Build Cache)...")
    print(f"{YELLOW}>>> 这将释放磁盘空间并强制重新下载依赖...{RESET}")
    run_cmd("docker builder prune -f --all")

    print(f"\n{GREEN}✅ 清理完成！环境已归零。{RESET}")

def regenerate_dockerfile():
    print("\n4️⃣  生成最稳健的 Dockerfile (阿里云源 + 无Syntax)...")
    
    # 使用之前验证过的最稳版本
    DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# --- 1. 换源 (阿里云 Debian 源) ---
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" > /etc/apt/sources.list && \\
    echo "deb-src https://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb https://mirrors.aliyun.com/debian-security/ bookworm-security main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb-src https://mirrors.aliyun.com/debian-security/ bookworm-security main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb-src https://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list

# --- 2. 安装系统依赖 ---
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    cmake \\
    libgl1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# --- 3. 升级 pip (阿里云 PyPI 源) ---
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

# --- 4. 安装基础依赖 ---
COPY requirements.txt .
# 剔除 torch 相关，先装其他的
RUN grep -v "torch" requirements.txt > requirements_base.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements_base.txt -i https://mirrors.aliyun.com/pypi/simple/ || true

# --- 5. 强制单独安装 PyTorch (阿里云源) ---
RUN pip install --no-cache-dir --default-timeout=1000 \\
    torch torchvision torchaudio \\
    -i https://mirrors.aliyun.com/pypi/simple/

# --- 6. 环境收尾 ---
RUN mkdir -p /app/h2q_project
ENV PYTHONPATH="/app"

CMD ["python3", "--version"]
"""
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT.strip())
    print(f"{GREEN}✅ Dockerfile 已重置。{RESET}")

def rebuild():
    print("\n5️⃣  开始全新构建 (这可能需要几分钟)...")
    # 使用 --no-cache 确保完全重新下载
    cmd = "docker build --no-cache -t h2q-sandbox ."
    
    try:
        # 实时打印输出
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        
        if process.returncode == 0:
            print(f"\n{GREEN}🎉🎉🎉 重建成功！Docker 环境已复活。{RESET}")
        else:
            print(f"\n{RED}❌ 构建失败，请检查上方日志。{RESET}")
            
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️ 用户取消构建。{RESET}")

if __name__ == "__main__":
    nuke_docker()
    regenerate_dockerfile()
    rebuild()