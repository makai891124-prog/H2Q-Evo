FROM python:3.11-slim

# Set environment variables for optimal streaming and Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_DATASETS_OFFLINE=0

WORKDIR /app

# Install system dependencies for zstandard and build-essential for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libzstd-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core H2Q dependencies
# [STABLE] datasets: For Hugging Face stream processing
# [STABLE] zstandard: For high-ratio decompression of petabyte-scale streams
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch \
    numpy \
    quaternion \
    datasets \
    zstandard \
    tqdm

# Copy project files
COPY . .

# Labeling the build for the M24-CW context
LABEL project="H2Q" \
      engine="M24-Cognitive-Weaver" \
      architecture="SU(2)-Geodesic-Flow"

CMD ["python", "main.py"]