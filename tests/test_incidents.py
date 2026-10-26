"""
Pytest unit and integration tests for Incident clustering, persistence, and endpoints.
"""

import pytest
import httpx
from backend.app.main import app
from backend.app.incidents import cluster_windows_into_incidents, send_discord_notification, Incident


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def test_incident_clustering_persistence_and_cooldown():
    """Verify that consecutive flagged windows merge into one incident with persistence and cooldown."""
    windows = [
        {"window_index": 0, "start_time": 0.0, "end_time": 0.5, "confidence": 0.82, "is_accident": True},
        {"window_index": 1, "start_time": 0.25, "end_time": 0.75, "confidence": 0.94, "is_accident": True},
        {"window_index": 2, "start_time": 0.5, "end_time": 1.0, "confidence": 0.30, "is_accident": False},
        {"window_index": 3, "start_time": 0.75, "end_time": 1.25, "confidence": 0.88, "is_accident": True},  # within cooldown of 2
        {"window_index": 4, "start_time": 1.0, "end_time": 1.5, "confidence": 0.20, "is_accident": False},
        {"window_index": 5, "start_time": 1.25, "end_time": 1.75, "confidence": 0.15, "is_accident": False},
        {"window_index": 6, "start_time": 1.5, "end_time": 2.0, "confidence": 0.10, "is_accident": False},
        {"window_index": 7, "start_time": 1.75, "end_time": 2.25, "confidence": 0.75, "is_accident": True},  # new incident after cooldown
    ]

    # With persistence k=1, cooldown=2
    incidents = cluster_windows_into_incidents(
        windows,
        camera_id="cam-01",
        camera_name="Camera 01",
        persistence_k=1,
        cooldown_windows=2,
    )

    assert len(incidents) == 2
    inc1 = incidents[0]
    assert inc1.window_indices == [0, 1, 3]
    assert inc1.peak_confidence == 0.94
    assert round(inc1.mean_confidence, 2) == round((0.82 + 0.94 + 0.88) / 3, 2)
    assert inc1.video_start_time == 0.0
    assert inc1.video_end_time == 1.25
    assert inc1.status == "new"

    inc2 = incidents[1]
    assert inc2.window_indices == [7]
    assert inc2.peak_confidence == 0.75


def test_incident_clustering_persistence_k_filter():
    """Verify that isolated single spikes are ignored when persistence k=2."""
    windows = [
        {"window_index": 0, "start_time": 0.0, "end_time": 0.5, "confidence": 0.82, "is_accident": True},
        {"window_index": 1, "start_time": 0.25, "end_time": 0.75, "confidence": 0.20, "is_accident": False},
        {"window_index": 2, "start_time": 0.5, "end_time": 1.0, "confidence": 0.30, "is_accident": False},
        {"window_index": 3, "start_time": 0.75, "end_time": 1.25, "confidence": 0.85, "is_accident": True},
        {"window_index": 4, "start_time": 1.0, "end_time": 1.5, "confidence": 0.90, "is_accident": True},
    ]

    # k=2: window 0 has length 1, so ignored; windows 3,4 have length 2, so accepted
    incidents = cluster_windows_into_incidents(
        windows,
        camera_id="cam-02",
        persistence_k=2,
        cooldown_windows=1,
    )

    assert len(incidents) == 1
    assert incidents[0].window_indices == [3, 4]
    assert incidents[0].peak_confidence == 0.90


@pytest.mark.anyio
async def test_get_cameras(client):
    """Test camera registry endpoint returns simulated demo cameras without real invented locations."""
    response = await client.get("/cameras")
    assert response.status_code == 200
    cameras = response.json()
    assert isinstance(cameras, list)
    assert len(cameras) >= 3
    cam1 = cameras[0]
    assert "id" in cam1
    assert "name" in cam1
    assert "location_label" in cam1
    assert "not available in dataset" in cam1["location_label"].lower()


@pytest.mark.anyio
async def test_get_incidents_list(client):
    """Test GET /incidents returns paginated incidents list with filters."""
    response = await client.get("/incidents")
    assert response.status_code == 200
    data = response.json()
    assert "incidents" in data
    assert "total" in data
    assert "page" in data
    assert len(data["incidents"]) > 0

    # Filter by status
    res_confirmed = await client.get("/incidents?status=confirmed")
    assert res_confirmed.status_code == 200
    data_conf = res_confirmed.json()
    for inc in data_conf["incidents"]:
        assert inc["status"] == "confirmed"


@pytest.mark.anyio
async def test_get_and_patch_incident(client):
    """Test GET /incidents/{id} and PATCH /incidents/{id} for operator status and note."""
    list_res = await client.get("/incidents")
    inc_id = list_res.json()["incidents"][0]["id"]

    # Fetch detail
    detail_res = await client.get(f"/incidents/{inc_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == inc_id

    # Patch status and note
    patch_res = await client.patch(
        f"/incidents/{inc_id}",
        json={"status": "confirmed", "operator_note": "Operator manually verified in unit test."},
    )
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["status"] == "confirmed"
    assert updated["operator_note"] == "Operator manually verified in unit test."

    # Test invalid status validation
    bad_patch = await client.patch(f"/incidents/{inc_id}", json={"status": "invalid_status"})
    assert bad_patch.status_code == 400


@pytest.mark.anyio
async def test_get_nonexistent_incident(client):
    """Test 404 for unknown incident id."""
    res = await client.get("/incidents/nonexistent-id-999")
    assert res.status_code == 404
