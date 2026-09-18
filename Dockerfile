# Road Accident Detection System - Backend Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and ONNX model (sample video is optional; missing file => NO INPUT)
COPY backend/app/ ./app/
COPY backend/model.onnx ./model.onnx

# Expose default port (platforms such as Railway override PORT)
ENV PORT=8000
EXPOSE 8000

# Bind to the deployment PORT
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]