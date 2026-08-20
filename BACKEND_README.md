# Road Accident Detection System - Backend

This directory contains the backend implementation for the real-time road accident detection system.

## Features

- **ONNX Model Serving**: Loads and serves the CNN-LSTM model in ONNX format for fast CPU inference
- **Real-time Video Processing**: Simulates CCTV feed by processing video frames sequentially
- **WebSocket Alerts**: Sends real-time alerts to connected frontend clients
- **MongoDB Integration**: Logs all detection events for historical analysis
- **REST API**: Health checks, model information, and control endpoints
- **Docker Support**: Easy deployment with Docker Compose

## Architecture

```
Backend Structure:
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Containerization configuration
├── .env.template           # Environment variables template
├── models/                 # ONNX model storage (will contain model.onnx)
├── utils/                  # Utility functions
├── routes/                 # API route definitions
�└── services/               # Business logic services
```

## Key Components

### Main Application (`main.py`)
- FastAPI app with CORS middleware
- Startup/shutdown event handlers for resource management
- ONNX model loading and inference pipeline
- Video frame processing loop (background task)
- WebSocket connection manager for real-time alerts
- MongoDB connection and logging
- REST endpoints:
  - `GET /` - Health check
  - `GET /health` - Detailed health status
  - `GET /model-info` - Model architecture information
  - `POST /reset-video` - Reset video to beginning
  - `WS /ws/alerts` - Real-time alert WebSocket

### Video Processing
- Processes video frame-by-frame at ~30 FPS
- Maintains frame buffer for sequence creation (SEQ_LENGTH frames)
- Preprocesses frames to match model input requirements
- Runs inference when sufficient frames are available
- Triggers alerts when accident confidence > threshold
- Includes cooldown period to prevent alert spam

### Alert System
- Sends JSON alerts via WebSocket to all connected clients
- Alert format: `{type, timestamp, confidence, camera_id, message}`
- Integrated with MongoDB for persistence
- Ready for Discord webhook integration

## Setup Instructions

### Prerequisites
- Docker and Docker Compose (for containerized deployment)
- OR Python 3.11+ with pip (for local development)
- Trained ONNX model (`model.onnx`) from the model training phase

### Local Development
1. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Copy trained model:
   ```bash
   # After training, copy model.onnx to backend directory
   cp /path/to/trained/model.onnx ./model.onnx
   ```

3. Configure environment:
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

4. Run the application:
   ```bash
   python main.py
   # Or with reload for development:
   uvicorn main:app --reload
   ```

### Docker Deployment
1. Ensure you have the trained model available:
   ```bash
   # Place model.onnx in backend directory or update MODEL_PATH in docker-compose.yml
   ```

2. Start services:
   ```bash
   docker-compose up -d
   ```

3. Access the API:
   - Health check: http://localhost:8000/
   - Interactive docs: http://localhost:8000/docs

## API Endpoints

### Health Checks
- `GET /` - Basic health status
- `GET /health` - Detailed health with model loading status

### Model Information
- `GET /model-info` - Returns model architecture details

### Control
- `POST /reset-video` - Resets video playback to beginning

### WebSocket
- `WS /ws/alerts` - Real-time accident detection alerts
  - Sends JSON messages when accidents are detected
  - Format: `{"type":"accident_detected","timestamp":"...","confidence":0.95,"camera_id":"cam_001","message":"Accident detected with 95% confidence"}`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to ONNX model file | `./model.onnx` |
| `VIDEO_PATH` | Path to sample video file | `./sample_video.mp4` |
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `DISCORD_WEBHOOK_URL` | Discord webhook for alerts | (optional) |

## Implementation Notes

### Model Input/Output
- **Input**: `(batch_size, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)` float32 tensor
- **Output**: `(batch_size, 1)` float32 tensor with sigmoid activation (0-1 confidence)
- Matches the CNN-LSTM architecture from training: TimeDistributed(MobileNetV2) → LSTM → Dense(sigmoid)

### Performance Considerations
- Frame processing runs in background task to avoid blocking API requests
- WebSocket connections are managed efficiently with connection cleanup
- MongoDB operations are fire-and-forget for minimum latency
- Model inference timing is logged for performance monitoring

## Next Steps

1. **Model Integration**: After training, place `model.onnx` in the backend directory
2. **Frontend Connection**: Connect React frontend to WebSocket endpoint `/ws/alerts`
3. **Alert Enhancement**: Add Discord webhook notifications
4. **Performance Optimization**: Add model inference timing metrics
5. **Production Hardening**: Add authentication, rate limiting, and SSL