# Road Accident Detection System - Multi-stage Slim Dockerfile for Render Free Tier (<512MB RAM)

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Minimal runtime image
FROM python:3.11-slim

WORKDIR /app

# Only runtime library needed for headless opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Security: Non-root user
RUN useradd -m -u 1001 appuser

# Bake model directly into image (~9.9 MB) and copy application code
COPY backend/app/ ./app/
COPY backend/model.onnx ./model.onnx

RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PORT=10000 \
    HOST=0.0.0.0 \
    MODEL_PATH=/app/model.onnx

EXPOSE 10000

CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]