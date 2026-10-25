"""
Road Accident Detection System - Backend API
FastAPI server for serving ONNX model via video upload,
handling WebSocket alerts, and logging detections to MongoDB.

Designed for Render free tier (512 MB RAM, tiny CPU share).
"""

import os
import io
import asyncio
import json
import time
import logging
import tempfile
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import cv2
import onnxruntime as ort
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException,
    UploadFile, File, Query, Request
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration & validation
# ---------------------------------------------------------------------------

def _require_env(name: str, default: Optional[str] = None) -> str:
    """Return env var or raise with a clear message."""
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return val


# Model
MODEL_PATH = os.getenv("MODEL_PATH", "./model.onnx")
SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", "10"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
ACCIDENT_CONFIDENCE_THRESHOLD = float(os.getenv("ACCIDENT_CONFIDENCE_THRESHOLD", "0.5"))

# Server
PORT = int(os.getenv("PORT", "8001"))
HOST = os.getenv("HOST", "0.0.0.0")

# MongoDB (optional – app starts without it)
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "accident_detection")

# Discord webhook (optional)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# CORS
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

# Demo mode – returns precomputed result, no model/DB needed
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# Upload limits
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
MAX_VIDEO_DURATION_S = int(os.getenv("MAX_VIDEO_DURATION_S", "15"))
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/x-msvideo", "video/quicktime", "video/webm"}

# Frame sampling – process every Nth frame to save CPU
FRAME_SAMPLE_INTERVAL = int(os.getenv("FRAME_SAMPLE_INTERVAL", "5"))
# Resize frames early to this width for processing
FRAME_RESIZE_WIDTH = int(os.getenv("FRAME_RESIZE_WIDTH", "320"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("accident_detection")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

model_session: Optional[ort.InferenceSession] = None
model_input_name: Optional[str] = None
model_output_name: Optional[str] = None
is_model_loaded = False

# MongoDB (motor async driver)
mongo_client = None
db = None
detection_collection = None
is_mongo_connected = False

# Concurrency guard – only 1 inference at a time
_inference_lock = asyncio.Lock()
_inference_busy = False

# ---------------------------------------------------------------------------
# Precomputed demo result
# ---------------------------------------------------------------------------

DEMO_RESULT = {
    "status": "demo",
    "message": "DEMO_MODE is active – returning precomputed example result.",
    "predictions": [
        {
            "window_index": 0,
            "start_frame": 0,
            "end_frame": 10,
            "confidence": 0.87,
            "is_accident": True,
        },
        {
            "window_index": 1,
            "start_frame": 10,
            "end_frame": 20,
            "confidence": 0.12,
            "is_accident": False,
        },
    ],
    "inference_time_ms": 42.0,
    "frames_processed": 20,
}

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected. Total: %d", len(self.active_connections))

    async def broadcast(self, message: str):
        broken = []
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                broken.append(conn)
        for conn in broken:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mongodb_connected: bool
    demo_mode: bool
    timestamp: str
    websocket_connections: int


class PredictionWindow(BaseModel):
    window_index: int
    start_frame: int
    end_frame: int
    confidence: float
    is_accident: bool


class PredictResponse(BaseModel):
    status: str
    message: str
    predictions: List[PredictionWindow]
    inference_time_ms: float
    frames_processed: int
    peak_memory_mb: Optional[float] = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global model_session, model_input_name, model_output_name, is_model_loaded
    global mongo_client, db, detection_collection, is_mongo_connected

    logger.info("Starting Road Accident Detection System...")
    logger.info("DEMO_MODE=%s", DEMO_MODE)

    # ---- Load ONNX model ----
    if not DEMO_MODE:
        try:
            logger.info("Looking for model at: %s (cwd=%s)", MODEL_PATH, os.getcwd())
            if os.path.exists(MODEL_PATH):
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                model_session = ort.InferenceSession(
                    MODEL_PATH, sess_options=sess_options,
                    providers=["CPUExecutionProvider"]
                )
                model_input_name = model_session.get_inputs()[0].name
                model_output_name = model_session.get_outputs()[0].name
                is_model_loaded = True
                logger.info("ONNX model loaded: input=%s, output=%s",
                            model_input_name, model_output_name)
            else:
                logger.error("Model file not found at %s — inference will be unavailable", MODEL_PATH)
        except Exception:
            logger.exception("Failed to load ONNX model")
    else:
        logger.info("DEMO_MODE active — skipping model load")

    # ---- Connect to MongoDB ----
    if MONGO_URI:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            mongo_client = AsyncIOMotorClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            # Test the connection
            await mongo_client.admin.command("ping")
            db = mongo_client[MONGO_DB_NAME]
            detection_collection = db.detections
            is_mongo_connected = True
            logger.info("Connected to MongoDB (%s)", MONGO_DB_NAME)

            # Create TTL index – auto-delete documents after 30 days
            try:
                await detection_collection.create_index(
                    "created_at", expireAfterSeconds=30 * 24 * 3600
                )
                logger.info("TTL index on created_at (30 days) ensured")
            except Exception:
                logger.warning("TTL index creation skipped (may already exist)")

            # Create regular index for sorting
            try:
                await detection_collection.create_index([("created_at", -1)])
            except Exception:
                pass

        except Exception:
            logger.warning("MongoDB unavailable — detections will not be persisted", exc_info=True)
            is_mongo_connected = False
    else:
        logger.info("MONGO_URI not set — running without database")

    logger.info("Startup complete")
    yield

    # ---- Shutdown ----
    logger.info("Shutting down...")
    if mongo_client:
        mongo_client.close()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Road Accident Detection API",
    description="Real-time road accident detection using CNN-LSTM ONNX model",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize, BGR→RGB, normalise to [0,1]."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def create_sequence(frames: List[np.ndarray]) -> np.ndarray:
    """Stack frames into (1, SEQ_LENGTH, H, W, 3)."""
    if len(frames) < SEQ_LENGTH:
        padding = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                   for _ in range(SEQ_LENGTH - len(frames))]
        frames = padding + list(frames)
    else:
        frames = list(frames[-SEQ_LENGTH:])
    seq = np.stack(frames, axis=0)
    return np.expand_dims(seq, axis=0)


def _get_rss_mb() -> float:
    """Get current process RSS in MB (best-effort)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_inference_on_video(video_bytes: bytes) -> dict:
    """
    Synchronous: decode video, sample every Nth frame, run sliding-window
    inference, return results dict.  Runs in a thread executor.
    """
    # Write to temp file (ephemerally)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Could not open uploaded video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0

        if duration_s > MAX_VIDEO_DURATION_S + 1:
            cap.release()
            raise ValueError(
                f"Video is {duration_s:.1f}s — max allowed is {MAX_VIDEO_DURATION_S}s"
            )

        # Collect frames with sampling
        sampled_frames: List[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
                # Resize early to save memory
                h, w = frame.shape[:2]
                if w > FRAME_RESIZE_WIDTH:
                    scale = FRAME_RESIZE_WIDTH / w
                    frame = cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(h * scale)))
                sampled_frames.append(frame)
            frame_idx += 1
        cap.release()

        if not sampled_frames:
            raise ValueError("No frames could be read from the video")

        # Preprocess all sampled frames
        processed = [preprocess_frame(f) for f in sampled_frames]
        # Free raw frames
        del sampled_frames

        # Sliding window inference
        predictions = []
        start_time = time.perf_counter()
        window_idx = 0

        for start in range(0, len(processed), SEQ_LENGTH):
            window = processed[start:start + SEQ_LENGTH]
            if len(window) < SEQ_LENGTH // 2:
                break  # skip very short tail windows

            seq = create_sequence(window)
            outputs = model_session.run(
                [model_output_name], {model_input_name: seq}
            )
            raw = float(outputs[0][0][0])
            # The training script enumerates ['Accident', 'NonAccident'],
            # so label 0 = Accident, label 1 = NonAccident.
            # The sigmoid output is P(NonAccident); invert to get P(Accident).
            confidence = round(1.0 - raw, 4)
            predictions.append({
                "window_index": window_idx,
                "start_frame": start * FRAME_SAMPLE_INTERVAL,
                "end_frame": (start + len(window)) * FRAME_SAMPLE_INTERVAL,
                "confidence": round(confidence, 4),
                "is_accident": confidence > ACCIDENT_CONFIDENCE_THRESHOLD,
            })
            window_idx += 1

        inference_time = (time.perf_counter() - start_time) * 1000
        peak_mem = _get_rss_mb()

        return {
            "status": "success",
            "message": f"Processed {len(processed)} frames in {window_idx} windows",
            "predictions": predictions,
            "inference_time_ms": round(inference_time, 2),
            "frames_processed": len(processed),
            "peak_memory_mb": round(peak_mem, 1) if peak_mem else None,
        }
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Discord webhook
# ---------------------------------------------------------------------------

async def send_discord_alert(alert_data: dict):
    """Fire-and-forget Discord notification."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        import aiohttp
        embed = {
            "title": "🚨 Road Accident Detected",
            "description": alert_data.get("message", "Accident detected"),
            "color": 0xFF0000,
            "fields": [
                {"name": "Confidence", "value": f"{alert_data.get('confidence', 0):.2%}", "inline": True},
                {"name": "Camera", "value": alert_data.get("camera_id", "upload"), "inline": True},
                {"name": "Timestamp", "value": alert_data.get("timestamp", ""), "inline": False},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"embeds": [embed], "username": "Accident Detector"}
        async with aiohttp.ClientSession() as session:
            async with session.post(DISCORD_WEBHOOK_URL, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status in (200, 204):
                    logger.info("Discord alert sent")
                else:
                    logger.warning("Discord alert failed: %s", resp.status)
    except Exception:
        logger.warning("Discord alert error", exc_info=True)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — used by Render, uptime pingers, and frontend wake-up polling."""
    mongo_ok = False
    if mongo_client and is_mongo_connected:
        try:
            await asyncio.wait_for(
                mongo_client.admin.command("ping"),
                timeout=2.0,
            )
            mongo_ok = True
        except Exception:
            mongo_ok = False

    return HealthResponse(
        status="healthy" if (is_model_loaded or DEMO_MODE) else "degraded",
        model_loaded=is_model_loaded,
        mongodb_connected=mongo_ok,
        demo_mode=DEMO_MODE,
        timestamp=datetime.now(timezone.utc).isoformat(),
        websocket_connections=len(manager.active_connections),
    )


@app.get("/")
async def root():
    """Basic health (redirects/aliases /health)."""
    return await health_check()


@app.get("/ping")
async def ping():
    """Ultra-light liveness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a short video clip (≤10 MB, ≤15 s) and get accident predictions.
    Only 1 inference runs at a time; returns 503 if busy.
    """
    global _inference_busy

    # Validate content type
    ct = file.content_type or ""
    if ct not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            415,
            f"Unsupported file type '{ct}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_TYPES))}"
        )

    # Demo mode — instant response once content type is verified
    if DEMO_MODE:
        return PredictResponse(**DEMO_RESULT)

    # Concurrency guard
    if _inference_busy:
        raise HTTPException(
            503,
            "Server is busy processing another video. Please try again in a few seconds."
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large ({len(contents) / 1024 / 1024:.1f} MB). Max is {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB."
        )

    # Run inference in thread (non-blocking)
    _inference_busy = True
    try:
        result = await asyncio.to_thread(run_inference_on_video, contents)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(500, "Inference failed — see server logs")
    finally:
        _inference_busy = False
        # Free memory
        del contents

    # Broadcast accidents via WebSocket and save to DB
    for pred in result.get("predictions", []):
        if pred["is_accident"]:
            alert_data = {
                "type": "accident_detected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": pred["confidence"],
                "camera_id": "upload",
                "message": f"Accident detected in window {pred['window_index']} "
                           f"(frames {pred['start_frame']}–{pred['end_frame']}) "
                           f"with {pred['confidence']:.0%} confidence",
                "window_index": pred["window_index"],
            }
            await manager.broadcast(json.dumps(alert_data))
            await _save_detection(alert_data)
            await send_discord_alert(alert_data)

    return PredictResponse(**result)


@app.get("/history")
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
):
    """Get detection history with pagination."""
    if not is_mongo_connected or detection_collection is None:
        return {"detections": [], "total": 0, "limit": limit, "offset": offset}

    try:
        sort_dir = -1 if sort == "desc" else 1
        cursor = detection_collection.find(
            {}, {"_id": 0}
        ).sort("created_at", sort_dir).skip(offset).limit(limit)

        detections = []
        async for doc in cursor:
            # Serialise datetimes
            for key in ("created_at", "timestamp"):
                if key in doc and hasattr(doc[key], "isoformat"):
                    doc[key] = doc[key].isoformat()
            detections.append(doc)

        total = await detection_collection.count_documents({})
        return {"detections": detections, "total": total, "limit": limit, "offset": offset}
    except Exception:
        logger.exception("Failed to fetch history")
        raise HTTPException(500, "Failed to fetch history")


@app.get("/model-info")
async def get_model_info():
    """Model architecture info."""
    if not is_model_loaded:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_loaded": True,
        "input_name": model_input_name,
        "output_name": model_output_name,
        "input_shape": list(model_session.get_inputs()[0].shape) if model_session else None,
        "output_shape": list(model_session.get_outputs()[0].shape) if model_session else None,
        "seq_length": SEQ_LENGTH,
        "img_size": IMG_SIZE,
    }


@app.get("/metrics")
async def get_metrics():
    """Return hardcoded model comparison metrics (from evaluation)."""
    return {
        "baseline_cnn": {
            "accuracy": 0.6234, "precision": 0.8783, "recall": 0.5479,
            "f1_score": 0.6748, "roc_auc": 0.8207,
        },
        "cnn_lstm": {
            "accuracy": 0.7998, "precision": 0.8074, "recall": 0.9445,
            "f1_score": 0.8706, "roc_auc": 0.8220,
        },
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time alert WebSocket with heartbeat."""
    await manager.connect(websocket)
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "system_status",
            "model_loaded": is_model_loaded,
            "demo_mode": DEMO_MODE,
            "mongodb_connected": is_mongo_connected,
        }))

        while True:
            try:
                # Wait for client messages with a timeout (heartbeat)
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif msg.get("type") == "latency_report":
                    logger.debug("Latency report: %sms", msg.get("latency_ms"))
            except asyncio.TimeoutError:
                # Send heartbeat ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
            except json.JSONDecodeError:
                continue
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("WebSocket error", exc_info=True)
    finally:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _save_detection(alert_data: dict):
    """Persist a detection to MongoDB (fire-and-forget)."""
    if not is_mongo_connected or detection_collection is None:
        return
    try:
        doc = {
            "timestamp": alert_data.get("timestamp"),
            "confidence": alert_data.get("confidence"),
            "camera_id": alert_data.get("camera_id", "upload"),
            "message": alert_data.get("message", ""),
            "is_accident": True,
            "created_at": datetime.now(timezone.utc),
        }
        # Cap document size (~4 KB max)
        doc_size = len(json.dumps(doc, default=str).encode())
        if doc_size > 4096:
            doc["message"] = doc["message"][:200]
        await detection_collection.insert_one(doc)
    except Exception:
        logger.warning("Failed to save detection to MongoDB", exc_info=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        workers=1,
        log_level=LOG_LEVEL.lower(),
    )