import os

# 1. 定义超级通用的 requirements.txt
# 包含 AGI 进化可能用到的绝大多数领域库
REQUIREMENTS_CONTENT = """
# --- 宿主机控制 ---
google-genai
docker
colorama

# --- 核心计算与 AI ---
torch
torchvision
torchaudio
numpy
scipy
scikit-learn
pandas
einops
sympy

# --- 数据流与存储 ---
datasets
zstandard
pyarrow
tqdm

# --- Web 服务与网络 ---
fastapi
uvicorn
requests
httpx
pydantic
python-multipart
websockets

# --- 可视化与图像 ---
matplotlib
seaborn
pillow
plotly

# --- 生物与科学 ---
biopython
genomic-benchmarks

# --- 工程与测试 ---
pytest
pytest-asyncio
black
psutil
pyyaml
"""

# 2. 定义增强版 Dockerfile
# 增加了系统级依赖 (gcc, g++, make, cmake) 以支持某些科学库的编译
DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# 1. 换源 (清华源) 加速系统包安装
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources || true

# 2. 安装系统级依赖 (这是"全能环境"的基础)
# build-essential: 编译 C/C++ 扩展
# libgl1: OpenCV/图像处理需要
# git: 下载代码
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    cmake \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# 3. 升级 pip
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 安装所有 Python 库 (增加超时时间，防止大文件下载失败)
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 预创建项目目录结构 (防止权限问题)
RUN mkdir -p /app/h2q_project/h2q && touch /app/h2q_project/h2q/__init__.py

VOLUME /app/h2q_project
CMD ["python3", "--version"]
"""

def upgrade():
    print(">>> 正在生成超级通用 requirements.txt ...")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS_CONTENT.strip())
    
    print(">>> 正在生成增强版 Dockerfile ...")
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT.strip())
        
    print("\n✅ 配置文件已更新！")
    print(">>> 请立即执行以下命令进行构建（可能需要 5-10 分钟）：")
    print("\033[92m    docker build --no-cache -t h2q-sandbox .\033[0m")

if __name__ == "__main__":
    upgrade()