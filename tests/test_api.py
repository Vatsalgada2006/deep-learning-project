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

