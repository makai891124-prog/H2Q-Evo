import os

# 1. 保持 requirements.txt 不变 (它没问题)
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

# 2. 修复 Dockerfile (替换废弃包名)
DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# 1. 换源 (清华源) 加速系统包安装
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources || true

# 2. 安装系统级依赖
# 修复: libgl1-mesa-glx -> libgl1 (适配新版 Debian)
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    cmake \\
    libgl1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# 3. 升级 pip
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 安装所有 Python 库
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 预创建项目目录结构
RUN mkdir -p /app/h2q_project/h2q && touch /app/h2q_project/h2q/__init__.py

VOLUME /app/h2q_project
CMD ["python3", "--version"]
"""

def upgrade():
    print(">>> 正在生成修复版 requirements.txt ...")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS_CONTENT.strip())
    
    print(">>> 正在生成修复版 Dockerfile (已替换 libgl1) ...")
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT.strip())
        
    print("\n✅ 配置文件已修复！")
    print(">>> 请再次执行构建命令：")
    print("\033[92m    docker build --no-cache -t h2q-sandbox .\033[0m")

if __name__ == "__main__":
    upgrade()