FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8000
ENV MODELS_DIR=models
ENV LOG_DIR=logs

RUN mkdir -p /app/logs

CMD sh -c "uvicorn fx_api_sniper_CLperpair:app --host 0.0.0.0 --port ${PORT}"