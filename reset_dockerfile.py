import os

DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# 使用官方源，不进行任何镜像替换
# 增加重试和超时设置
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries
RUN echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    cmake \\
    libgl1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# 允许 pip 安装失败 (|| true)，防止构建中断
# 真正的依赖检查交给 evolution_system.py 的运行时逻辑
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt || true

RUN mkdir -p /app/h2q_project && touch /app/h2q_project/__init__.py
ENV PYTHONPATH="${PYTHONPATH}:/app/h2q_project"

VOLUME /app/h2q_project
CMD ["python3", "--version"]
"""

def reset():
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT.strip())
    print("✅ Dockerfile 已重置为官方源模式。")
    print("🚀 请运行：docker build --no-cache -t h2q-sandbox .")

if __name__ == "__main__":
    reset()