import os

DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# 1. 基础环境配置
ENV PYTHONPATH="${PYTHONPATH}:/app/h2q_project"
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    cmake \\
    libgl1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 复制依赖文件并安装基础库 (忽略错误，防止个别包卡住整个构建)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || true

# 4. 【核心修复】强制单独安装 Transformers 和 Accelerate
# 这一步是独立的，Docker 必须执行它
RUN pip install --no-cache-dir \\
    transformers \\
    accelerate \\
    tokenizers \\
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 预创建目录
RUN mkdir -p /app/h2q_project

VOLUME /app/h2q_project
CMD ["python3", "--version"]
"""

def fix():
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT.strip())
    print("✅ Dockerfile 已更新：增加了强制安装 Transformers 的独立指令。")

if __name__ == "__main__":
    fix()