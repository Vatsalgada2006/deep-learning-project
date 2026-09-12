"""
Road Accident Detection System - Backend API
FastAPI server for serving ONNX model, handling WebSocket alerts,
and simulating CCTV feed processing.
"""


import os
import asyncio
import json
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
import base64
import aiohttp
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Road Accident Detection API",
    description="Real-time road accident detection using CNN-LSTM ONNX model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and services
model_session: Optional[ort.InferenceSession] = None
model_input_name: Optional[str] = None
model_output_name: Optional[str] = None
is_model_loaded = False

# YOLOv8 object detection model
yolo_model: Optional[YOLO] = None
is_yolo_loaded = False
latest_yolo_detections: List[Dict[str, Any]] = []  # Store latest detections for accident alerts

# MongoDB connection
mongo_client: Optional[MongoClient] = None
db = None
detection_collection = None

# Configuration
ACCIDENT_CONFIDENCE_THRESHOLD = float(os.getenv("ACCIDENT_CONFIDENCE_THRESHOLD", "0.5"))

# Video source (never fabricate CCTV frames)
video_capture: Optional[cv2.VideoCapture] = None
video_path = os.getenv("VIDEO_PATH", "sample_video.mp4")  # Path to sample video
has_video_input = False
frame_buffer: List[np.ndarray] = []
SEQ_LENGTH = 10  # Must match training sequence length
IMG_SIZE = 128   # Must match training image size

# YOLOv8 object detection configuration
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.05"))

# Video streaming globals
latest_jpeg_frame: Optional[bytes] = None
placeholder_jpeg: Optional[bytes] = None
frame_width: int = 640
frame_height: int = 480

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic models for request/response
from pydantic import BaseModel

class DetectionResult(BaseModel):
    timestamp: str
    confidence: float
    is_accident: bool
    camera_id: str = "cam_001"
    bbox: Optional[List[float]] = None  # [x, y, width, height] if applicable

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    monitoring_status: str = "no_input"
    has_video_input: bool = False
    # Additional health fields (optional for backward compatibility)
    yolo_loaded: Optional[bool] = None
    mongodb_connected: Optional[bool] = None
    websocket_connections: Optional[int] = None

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global model_session, model_input_name, model_output_name, is_model_loaded
    global mongo_client, db, detection_collection, video_capture, has_video_input
    global yolo_model, is_yolo_loaded
    global latest_jpeg_frame, placeholder_jpeg, frame_width, frame_height

    logger.info("Starting Road Accident Detection System...")

    # Load ONNX model
    try:
        model_path = os.getenv("MODEL_PATH", "./model.onnx")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Looking for model at: {model_path}")
        if os.path.exists(model_path):
            model_session = ort.InferenceSession(model_path)
            model_input_name = model_session.get_inputs()[0].name
            model_output_name = model_session.get_outputs()[0].name
            is_model_loaded = True
            logger.info(f"ONNX model loaded successfully from {model_path}")
            logger.info(f"Model input: {model_input_name}, output: {model_output_name}")
        else:
            logger.warning(f"Model file not found at {model_path}. Please ensure model.onnx is available.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")

    # Connect to MongoDB
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client.accident_detection
        detection_collection = db.detections
        # Test connection
        mongo_client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")

        # Create index on processed_at for efficient history queries
        try:
            detection_collection.create_index("processed_at")
            logger.info("Created index on processed_at field")
        except Exception as e:
            # Index might already exist, which is fine
            logger.info(f"Index on processed_at already exists or creation failed: {e}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")

    # Load YOLOv8 model for object detection
    try:
        logger.info("Loading YOLOv8n model...")
        yolo_model = YOLO("yolov8n.pt")  # Will download automatically if not present
        is_yolo_loaded = True
        logger.info("YOLOv8n model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}")
        is_yolo_loaded = False

    # Initialize video capture — do not invent frames if the file is missing
    has_video_input = False
    try:
        if os.path.exists(video_path):
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                logger.warning(f"Could not open video file {video_path}. Monitoring status: NO INPUT.")
                video_capture = None
            else:
                has_video_input = True
                # Get video dimensions for placeholder
                frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Video capture initialized from {video_path} at {frame_width}x{frame_height}")
        else:
            logger.warning(
                f"Video file not found at {video_path}. "
                "Monitoring status: NO INPUT. Accident detection is paused until a real video source is available."
            )
    except Exception as e:
        logger.error(f"Failed to initialize video capture: {e}")
        video_capture = None
        has_video_input = False

    # Generate placeholder JPEG for when no video is available
    try:
        placeholder_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        success, placeholder_buffer = cv2.imencode('.jpg', placeholder_img)
        if success:
            placeholder_jpeg = placeholder_buffer.tobytes()
            logger.info(f"Generated placeholder JPEG: {frame_width}x{frame_height}")
        else:
            logger.error("Failed to generate placeholder JPEG")
            placeholder_jpeg = b''
    except Exception as e:
        logger.error(f"Error generating placeholder JPEG: {e}")
        placeholder_jpeg = b''

    # Start background task for video processing
    asyncio.create_task(process_video_frames())
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global video_capture, mongo_client
    logger.info("Shutting down Road Accident Detection System...")

    if video_capture:
        video_capture.release()

    if mongo_client:
        mongo_client.close()

    logger.info("Shutdown complete")

# Helper functions
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess a single frame for model input."""
    # Resize
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    return frame_normalized

def create_sequence(frames: List[np.ndarray]) -> np.ndarray:
    """Create a sequence of frames for LSTM input."""
    if len(frames) < SEQ_LENGTH:
        # Pad with zeros if we don't have enough frames
        padding = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                  for _ in range(SEQ_LENGTH - len(frames))]
        frames = padding + frames
    else:
        # Take the most recent SEQ_LENGTH frames
        frames = frames[-SEQ_LENGTH:]

    # Stack frames to create sequence: (SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
    sequence = np.stack(frames, axis=0)
    # Add batch dimension: (1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
    sequence = np.expand_dims(sequence, axis=0)
    return sequence


async def detect_objects_yolo(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect objects in a frame using YOLOv8 model.

    Returns:
        List of dictionaries containing detection info:
        [{label, confidence, box: [x1, y1, x2, y2], track_id}]
    """
    global yolo_model, is_yolo_loaded

    if not is_yolo_loaded or yolo_model is None:
        return []

    try:
        # Run YOLOv8 detection with confidence and IoU thresholds in thread to prevent blocking
        results = await asyncio.to_thread(
            yolo_model,
            frame,
            verbose=False,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get class name
                    label = yolo_model.names[class_id] if hasattr(yolo_model, 'names') else str(class_id)

                    # Get track ID if available (from tracking)
                    track_id = None
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())

                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "track_id": track_id
                    })

        return detections
    except Exception as e:
        logger.error(f"Error in YOLO detection: {e}")
        return []


def _monitoring_status() -> str:
    return "live" if has_video_input else "no_input"


async def process_video_frames():
    """Background task to process video frames and detect accidents."""
    global frame_buffer, video_capture, is_model_loaded, has_video_input
    global latest_jpeg_frame

    logger.info("Starting video frame processing task...")

    frame_count = 0
    last_detection_time = 0
    detection_cooldown = 5.0  # Seconds between detections to prevent spam
    logged_no_input = False

    while True:
        try:
            # Get frame from video source. Never synthesize CCTV/noise frames.
            if not (video_capture and video_capture.isOpened()):
                has_video_input = False
                if not logged_no_input:
                    logger.warning(
                        "NO INPUT: no CCTV/video source. Skipping inference and alerts."
                    )
                    logged_no_input = True
                await asyncio.sleep(1)
                continue

            logged_no_input = False
            has_video_input = True
            ret, frame = video_capture.read()
            if not ret:
                # Loop video back to start
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = video_capture.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue

            # Encode frame for MJPEG streaming
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                latest_jpeg_frame = buffer.tobytes()
            else:
                logger.warning("Failed to encode frame for streaming")

            # Run YOLOv8 object detection on every frame (if model is loaded)
            if is_yolo_loaded:
                latest_yolo_detections = await detect_objects_yolo(frame)

            # Preprocess frame
            processed_frame = preprocess_frame(frame)
            frame_buffer.append(processed_frame)

            # Keep only recent frames to prevent memory buildup
            if len(frame_buffer) > SEQ_LENGTH * 2:
                frame_buffer = frame_buffer[-SEQ_LENGTH * 2:]

            # Run inference if we have enough frames and model is loaded
            if is_model_loaded and len(frame_buffer) >= SEQ_LENGTH:
                current_time = time.time()

                # Check cooldown to prevent excessive detections
                if current_time - last_detection_time < detection_cooldown:
                    await asyncio.sleep(0.033)  # ~30 FPS
                    continue

                # Create sequence for model input
                sequence = create_sequence(frame_buffer)

                # Run inference
                start_time = time.time()
                try:
                    # Run ONNX inference in thread to prevent blocking the event loop
                    outputs = await asyncio.to_thread(
                        model_session.run,
                        [model_output_name],
                        {model_input_name: sequence}
                    )
                    inference_time = time.time() - start_time

                    # Get prediction
                    confidence = float(outputs[0][0][0])  # Sigmoid output
                    is_accident = confidence > ACCIDENT_CONFIDENCE_THRESHOLD

                    logger.debug(f"Frame {frame_count}: Confidence={confidence:.4f}, "
                               f"Accident={is_accident}, Inference={inference_time*1000:.2f}ms")

                    # If accident detected, send alert
                    if is_accident:
                        await handle_accident_detection(confidence, frame_count)
                        last_detection_time = current_time

                except Exception as e:
                    logger.error(f"Error during inference: {e}")

            frame_count += 1
            await asyncio.sleep(0.033)  # ~30 FPS

        except Exception as e:
            logger.error(f"Error in video processing loop: {e}")
            await asyncio.sleep(1)


async def handle_accident_detection(confidence: float, frame_count: int):
    """Handle accident detection: log to DB, send WebSocket alert, trigger Discord webhook."""
    timestamp = datetime.now().isoformat()

    # Create detection record
    detection_record = {
        "timestamp": timestamp,
        "frame_count": frame_count,
        "confidence": confidence,
        "is_accident": True,
        "camera_id": "cam_001",
        "processed_at": datetime.utcnow()
    }

    # Save to MongoDB
    try:
        if detection_collection is not None:
            # Run MongoDB insert in thread to prevent blocking the event loop
            result = await asyncio.to_thread(
                detection_collection.insert_one,
                detection_record
            )
            logger.info(f"Accident detection saved to DB with ID: {result.inserted_id}")
    except Exception as e:
        logger.error(f"Failed to save detection to MongoDB: {e}")

    # Prepare alert message with object detection information
    # Count objects by label
    object_counts = {}
    for det in latest_yolo_detections:
        label = det["label"]
        object_counts[label] = object_counts.get(label, 0) + 1

    # Build object description
    if object_counts:
        object_descriptions = []
        for label, count in object_counts.items():
            object_descriptions.append(f"{count} {label}{'s' if count != 1 else ''}")
        object_summary = ", ".join(object_descriptions)
        message = f"Accident detected - {object_summary} in frame"
    else:
        message = f"Accident detected with {confidence:.2%} confidence (no objects detected)"

    alert_data = {
        "type": "accident_detected",
        "timestamp": timestamp,
        "confidence": confidence,
        "camera_id": "cam_001",
        "message": message,
        "detections": latest_yolo_detections.copy()  # Include the raw detections
    }

    # Send to all connected WebSocket clients
    await manager.broadcast(json.dumps(alert_data))
    logger.info(f"Accident alert broadcasted to {len(manager.active_connections)} clients")

    # Send Discord webhook notification
    await send_discord_alert(alert_data)


async def send_discord_alert(alert_data: dict):
    """Send alert to Discord webhook."""
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not discord_webhook_url:
        logger.warning("Discord webhook URL not configured")
        return

    try:
        # Create a Discord embed message
        embed = {
            "title": "🚨 Road Accident Detected",
            "description": alert_data.get("message", "Accident detected"),
            "color": 0xFF0000,  # Red
            "fields": [
                {
                    "name": "Camera ID",
                    "value": alert_data.get("camera_id", "Unknown"),
                    "inline": True
                },
                {
                    "name": "Confidence",
                    "value": f"{alert_data.get('confidence', 0):.2%}",
                    "inline": True
                },
                {
                    "name": "Timestamp",
                    "value": alert_data.get("timestamp", "Unknown"),
                    "inline": False
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {
            "embeds": [embed],
            "username": "Road Accident Detector"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(discord_webhook_url, json=payload) as response:
                if response.status == 204:
                    logger.info("Discord alert sent successfully")
                else:
                    logger.error(f"Failed to send Discord alert. Status: {response.status}, Response: {await response.text()}")
    except Exception as e:
        logger.error(f"Error sending Discord alert: {e}")

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    # Check MongoDB connectivity with a short timeout
    mongodb_connected = False
    if mongo_client is not None and db is not None:
        try:
            # Use asyncio.to_thread to prevent blocking the event loop
            await asyncio.wait_for(
                asyncio.to_thread(mongo_client.admin.command, 'ping'),
                timeout=2.0  # 2 second timeout
            )
            mongodb_connected = True
        except Exception as e:
            mongodb_connected = False

    return HealthResponse(
        status="healthy",
        model_loaded=is_model_loaded,
        timestamp=datetime.now().isoformat(),
        monitoring_status=_monitoring_status(),
        has_video_input=has_video_input,
        yolo_loaded=is_yolo_loaded,
        mongodb_connected=mongodb_connected,
        websocket_connections=len(manager.active_connections),
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    # Check MongoDB connectivity with a short timeout
    mongodb_connected = False
    if mongo_client is not None and db is not None:
        try:
            # Use asyncio.to_thread to prevent blocking the event loop
            await asyncio.wait_for(
                asyncio.to_thread(mongo_client.admin.command, 'ping'),
                timeout=2.0  # 2 second timeout
            )
            mongodb_connected = True
        except Exception as e:
            mongodb_connected = False
    return HealthResponse(
        status="healthy" if is_model_loaded else "degraded",
        model_loaded=is_model_loaded,
        timestamp=datetime.now().isoformat(),
        monitoring_status=_monitoring_status(),
        has_video_input=has_video_input,
        yolo_loaded=is_yolo_loaded,
        mongodb_connected=mongodb_connected,
        websocket_connections=len(manager.active_connections),
    )

@app.post("/reset-video")
async def reset_video():
    """Reset video to beginning."""
    global video_capture
    if video_capture and video_capture.isOpened():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return {"status": "video reset"}
    raise HTTPException(status_code=400, detail="Video capture not available")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if not is_model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_loaded": is_model_loaded,
        "input_name": model_input_name,
        "output_name": model_output_name,
        "input_shape": model_session.get_inputs()[0].shape if model_session else None,
        "output_shape": model_session.get_outputs()[0].shape if model_session else None
    }

# Model metrics endpoint
@app.get("/metrics")
async def get_model_metrics():
    """Get model performance metrics for both CNN and CNN-LSTM."""
    try:
        # Get base directory (three levels up from this file) to reach project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Read baseline CNN results
        baseline_path = os.path.join(base_dir, "accident_model_output", "Baseline_CNN_results.json")
        with open(baseline_path, "r") as f:
            baseline_results = json.load(f)
        # Read CNN-LSTM results
        cnn_lstm_path = os.path.join(base_dir, "accident_model_output", "CNN_LSTM_results.json")
        with open(cnn_lstm_path, "r") as f:
            cnn_lstm_results = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model results not found")

    # Function to compute metrics from confusion matrix
    def compute_metrics(cm):
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
        }

    # Since the confusion matrix in the JSON does not contain roc_auc, we need to get it from the results.
    # We'll adjust: the results JSON has roc_auc at the top level.
    baseline_cm = baseline_results["confusion_matrix"]
    cnn_lstm_cm = cnn_lstm_results["confusion_matrix"]

    baseline_metrics = compute_metrics(baseline_cm)
    baseline_metrics["roc_auc"] = baseline_results.get("roc_auc", 0.0)

    cnn_lstm_metrics = compute_metrics(cnn_lstm_cm)
    cnn_lstm_metrics["roc_auc"] = cnn_lstm_results.get("roc_auc", 0.0)

    return {
        "baseline_cnn": baseline_metrics,
        "cnn_lstm": cnn_lstm_metrics
    }


# History endpoint
@app.get("/history")
async def get_history(limit: int = 100):
    """Get detection history from MongoDB."""
    if detection_collection is None:
        # If MongoDB is not connected, return empty list
        return []
    try:
        history = list(detection_collection.find().sort("processed_at", -1).limit(limit))
        # Convert ObjectId to string for JSON serialization
        for record in history:
            record["_id"] = str(record["_id"])
            # Convert datetime to string
            if "processed_at" in record and isinstance(record["processed_at"], datetime):
                record["processed_at"] = record["processed_at"].isoformat()
        return history
    except Exception as e:
        logger.error(f"Failed to fetch history from MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")


# Video streaming endpoint
@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream."""
    async def video_stream_generator():
        while True:
            frame_to_send = latest_jpeg_frame
            if frame_to_send is None:
                frame_to_send = placeholder_jpeg or b''
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            await asyncio.sleep(0.033)  # ~30 FPS
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# Test endpoint
@app.get("/test")
async def test_endpoint():
    return {"message": "test"}


# WebSocket endpoint for real-time alerts
@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for sending real-time alerts to frontend and receiving latency reports."""
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "system_status",
            "monitoring_status": _monitoring_status(),
            "has_video_input": has_video_input,
            "model_loaded": is_model_loaded,
        }))
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            try:
                # Parse the incoming message
                message = json.loads(data)
                # Handle latency report from frontend
                if message.get("type") == "latency_report":
                    latency_ms = message.get("latency_ms")
                    timestamp = message.get("timestamp")
                    logger.info(f"Latency report: {latency_ms}ms for detection at {timestamp}")
                else:
                    # For other message types, echo back (optional)
                    await websocket.send_text(f"Message received: {data}")
            except json.JSONDecodeError:
                # If not JSON, treat as plain text and echo back
                await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/ping")
def ping():
    return "pong"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=True
    )