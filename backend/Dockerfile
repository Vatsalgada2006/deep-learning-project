# Road Accident Detection System - Backend Dockerfile
FROM python:3.11-slim

# Set working directory to backend source
WORKDIR /app/backend

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/backend/ ./

# Copy ML model and sample video
COPY ../model.onnx ./
COPY ../sample_video.mp4 ._

# Expose port
EXPOSE 8000

# Run the application
CMD ["echo", "Hello from container"; "sleep", "30"]