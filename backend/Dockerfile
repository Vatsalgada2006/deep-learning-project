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
COPY backend/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/backend/ ./backend/
COPY backend/model.onnx ./backend/
COPY backend/sample_video.mp4 ./backend/

# Expose port
EXPOSE 8000

# Run the application - let Python code handle PORT from environment
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0"]