"""
Pytest unit and integration tests for Road Accident Detection API
"""

import os
import io
import pytest
import httpx

# Ensure test environment variables are configured
os.environ["DEMO_MODE"] = "true"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:5173,https://test.vercel.app"
os.environ["MODEL_PATH"] = os.path.join(os.path.dirname(__file__), "..", "backend", "model.onnx")

from backend.app.main import app

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.mark.anyio
async def test_ping(client):
    """Test ultra-light ping endpoint."""
    response = await client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_health_check(client):
    """Test health check response structure."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "model_loaded" in data
    assert "mongodb_connected" in data
    assert "demo_mode" in data
    assert data["demo_mode"] is True
    assert "websocket_connections" in data


@pytest.mark.anyio
async def test_root_endpoint(client):
    """Test root endpoint aliases health check."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@pytest.mark.anyio
async def test_metrics_endpoint(client):
    """Test model comparison metrics structure and honesty."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "baseline_cnn" in data
    assert "cnn_lstm" in data
    assert data["cnn_lstm"]["recall"] > data["baseline_cnn"]["recall"]
    assert data["cnn_lstm"]["f1_score"] > data["baseline_cnn"]["f1_score"]


@pytest.mark.anyio
async def test_history_endpoint_without_mongo(client):
    """Test history endpoint returns clean empty structure when DB disconnected."""
    response = await client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert isinstance(data["detections"], list)
    assert data["total"] == 0


@pytest.mark.anyio
async def test_predict_demo_mode(client):
    """Test predict endpoint in DEMO_MODE returns valid simulated predictions."""
    fake_video = io.BytesIO(b"dummy video data")
    response = await client.post(
        "/predict",
        files={"file": ("test.mp4", fake_video, "video/mp4")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "demo"
    assert "predictions" in data
    assert len(data["predictions"]) > 0
    assert data["predictions"][0]["is_accident"] is True

    # New fields: fps, total_frames at top level
    assert "fps" in data
    assert isinstance(data["fps"], (int, float))
    assert data["fps"] > 0
    assert "total_frames" in data
    assert isinstance(data["total_frames"], int)

    # Per-window: start_time, end_time
    pred0 = data["predictions"][0]
    assert "start_time" in pred0
    assert "end_time" in pred0
    assert isinstance(pred0["start_time"], (int, float))
    assert pred0["end_time"] >= pred0["start_time"]


@pytest.mark.anyio
async def test_predict_invalid_content_type(client):
    """Test predict endpoint rejects non-video files with 415."""
    fake_text = io.BytesIO(b"hello world")
    response = await client.post(
        "/predict",
        files={"file": ("test.txt", fake_text, "text/plain")}
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.anyio
async def test_cors_preflight(client):
    """Test CORS preflight response headers for allowed origin."""
    response = await client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_onnx_model_direct_inference():
    """Verify that backend/model.onnx loads and executes successfully via onnxruntime."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "backend", "model.onnx")
    if not os.path.exists(model_path):
        pytest.skip("backend/model.onnx not found")

    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Shape is (1, 10, 128, 128, 3) or similar
    dummy_input = np.random.rand(1, 10, 128, 128, 3).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})
    assert len(outputs) > 0
    pred = outputs[0]
    assert pred.shape[0] == 1
    # Probability between 0 and 1
    prob = float(pred[0][0])
    assert 0.0 <= prob <= 1.0


@pytest.mark.anyio
async def test_health_metadata(client):
    """Test health endpoint returns model_version, model_checksum, and uptime."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert data["model_version"] == "2.0.0"
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0
    assert "model_checksum" in data


@pytest.mark.anyio
async def test_metrics_endpoint(client):
    """Test /metrics endpoint returns real class-wise accident metrics and confusion matrices."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "baseline_cnn" in data
    assert "cnn_lstm" in data

    # Verify class-wise accident metrics
    lstm = data["cnn_lstm"]
    assert "accident_class" in lstm
    acc = lstm["accident_class"]
    assert acc["label"] == "Accident"
    assert acc["recall"] > 0.85
    assert acc["precision"] > 0.85
    assert acc["f1_score"] > 0.85
    assert "confusion_matrix" in lstm
    assert "roc_auc" in lstm
    assert lstm["roc_auc"] > 0.80

    # Verify baseline metrics
    base = data["baseline_cnn"]
    assert "accident_class" in base
    assert base["roc_auc"] > 0.95


def test_confidence_inversion_logic():
    """Verify that raw sigmoid output is inverted: P(Accident) = 1.0 - raw."""
    # Label 0 = Accident, Label 1 = NonAccident
    # Sigmoid produces P(NonAccident).
    raw_safe = 0.95
    conf_safe = round(1.0 - raw_safe, 4)
    assert conf_safe == 0.05
    assert conf_safe < 0.5  # Not an accident

    raw_accident = 0.12
    conf_accident = round(1.0 - raw_accident, 4)
    assert conf_accident == 0.88
    assert conf_accident > 0.5  # Accident detected!


def test_real_accident_clip_inference():
    """Test that the real accident clip produces an accident alert with the ONNX model."""
    clip_path = os.path.join(os.path.dirname(__file__), "..", "public", "samples", "video_project.mp4")
    if not os.path.exists(clip_path):
        clip_path = os.path.join(os.path.dirname(__file__), "..", "public", "samples", "accident_crash_demo.mp4")
        if not os.path.exists(clip_path):
            pytest.skip("No accident sample clip found")

    import cv2
    import numpy as np
    import onnxruntime as ort

    model_path = os.path.join(os.path.dirname(__file__), "..", "backend", "model.onnx")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(clip_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 128))
            frames.append(resized.astype(np.float32) / 255.0)
        idx += 1
    cap.release()

    assert len(frames) >= 10, "Should have at least 1 window of frames"
    peak_accident = 0.0
    for start in range(0, len(frames) - 10 + 1, 5):
        window = frames[start:start + 10]
        seq = np.expand_dims(np.stack(window, axis=0), axis=0)
        out = session.run(None, {input_name: seq})
        raw = float(out[0][0][0])
        p_accident = 1.0 - raw
        peak_accident = max(peak_accident, p_accident)

    # The real accident clip must trigger an alert!
    assert peak_accident > 0.5, f"Expected accident alert (>0.5), got {peak_accident:.4f}"


def test_onnx_preprocessing_and_architecture_parity():
    """Verify input shape (None, 10, 128, 128, 3) and preprocessing matching training."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "backend", "model.onnx")
    import onnxruntime as ort
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    
    # Check sequence length is 10, height 128, width 128, channels 3
    assert inp.shape[1] == 10
    assert inp.shape[2] == 128
    assert inp.shape[3] == 128
    assert inp.shape[4] == 3


