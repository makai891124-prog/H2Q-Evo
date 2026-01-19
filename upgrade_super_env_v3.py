import os

REQUIREMENTS_CONTENT = """
google-genai
docker
colorama
requests
datasets
zstandard
pyarrow
tqdm
fastapi
uvicorn
httpx
pydantic
python-multipart
websockets
matplotlib
seaborn
pillow
plotly
biopython
genomic-benchmarks
pytest
pytest-asyncio
black
psutil
pyyaml
"""

# 使用阿里云源，通常更稳定
DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# 1. 换源 (阿里云)
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" > /etc/apt/sources.list && \\
    echo "deb-src https://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb https://mirrors.aliyun.com/debian-security/ bookworm-security main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb-src https://mirrors.aliyun.com/debian-security/ bookworm-security main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \\
    echo "deb-src https://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list

# 2. 安装系统级依赖
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    cmake \\
    libgl1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# 3. 升级 pip (使用阿里云源)
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 安装所有 Python 库
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 6. 预创建项目目录结构
RUN mkdir -p /app/h2q_project/h2q && touch /app/h2q_project/h2q/__init__.py

VOLUME /app/h2q_project
CMD ["python3", "--version"]
"""

def upgrade():
    print(">>> 正在生成 v3 版 Dockerfile (阿里云源)...")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS_CONTENT.strip())
    
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT.strip())
        
    print("\n✅ 配置文件已更新！")
    print(">>> 请再次执行构建命令：")
    print("\033[92m    docker build --no-cache -t h2q-sandbox .\033[0m")

if __name__ == "__main__":
    upgrade()