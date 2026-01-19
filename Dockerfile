# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked     --mount=type=cache,target=/var/lib/apt,sharing=locked     apt-get update && apt-get install -y git build-essential cmake libgl1 libglib2.0-0
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt || true
RUN mkdir -p /app/h2q
CMD ["python3", "--version"]