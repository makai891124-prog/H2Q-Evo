FROM python:3.10-slim
WORKDIR /app
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000
RUN apt-get update \
	&& (apt-get install -y --no-install-recommends -o Acquire::Retries=5 git build-essential cmake \
		|| (apt-get update && apt-get install -y --fix-missing --no-install-recommends git build-essential cmake)) \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt || true
RUN mkdir -p /app/h2q
CMD ["python3", "--version"]