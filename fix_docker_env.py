import os

# 1. 确保 requirements.txt 里没有 torch (我们要单独装它)
# 我们读取现有的，过滤掉 torch 相关，然后写回去
try:
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
    
    clean_lines = [l for l in lines if "torch" not in l and l.strip()]
    
    with open("requirements_base.txt", "w") as f:
        f.writelines(clean_lines)
    print("✅ 已分离基础依赖到 requirements_base.txt")
except:
    print("⚠️ 未找到 requirements.txt，将使用默认值")

# 2. 生成“防弹版” Dockerfile
DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# 设置环境变量：强制 Python 将项目目录加入搜索路径
# 这样无论在哪里运行 python，都能找到 h2q 包
ENV PYTHONPATH="${PYTHONPATH}:/app/h2q_project"

# 1. 安装系统基础库 (使用官方源，虽然慢但最稳，避免镜像源连接失败)
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    cmake \\
    libgl1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 第一阶段：安装基础依赖 (不含 PyTorch)
COPY requirements_base.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements_base.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 第二阶段：单独安装 PyTorch (CPU版)
# 单独列出以确保它被正确安装，不会被其他依赖掩盖
RUN pip install --no-cache-dir --default-timeout=2000 \\
    torch torchvision torchaudio \\
    --index-url https://download.pytorch.org/whl/cpu

# 5. 预创建目录
RUN mkdir -p /app/h2q_project

VOLUME /app/h2q_project
CMD ["python3", "--version"]
"""

def fix():
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT.strip())
    
    print("✅ Dockerfile 已重写：")
    print("   1. 增加了 ENV PYTHONPATH (修复 h2q 导入错误)")
    print("   2. 独立了 PyTorch 安装步骤 (修复 torch 缺失)")
    print("\n🚀 请立即执行以下命令进行重建：")
    print("\033[92m    docker build --no-cache -t h2q-sandbox .\033[0m")

if __name__ == "__main__":
    fix()