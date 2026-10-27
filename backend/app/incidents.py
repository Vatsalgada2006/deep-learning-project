"""
Incident clustering, persistence, schema and notification helpers.
"""

import os
import uuid
from datetime import datetime, timezone
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger("accident_detection.incidents")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# In-memory store for fallback / demo mode
IN_MEMORY_INCIDENTS: List[Dict[str, Any]] = [
    {
        "id": "inc-demo-001",
        "camera_id": "cam-01",
        "camera_name": "Camera 01 - Highway Sector 4",
        "detected_at": "2026-10-27T01:30:00Z",
        "video_start_time": 0.0,
        "video_end_time": 1.67,
        "peak_confidence": 0.8724,
        "mean_confidence": 0.2671,
        "window_indices": [0, 1],
        "peak_frame_snapshot": "/samples/accident_crash_demo.jpg",
        "model_version": "2.0.0",
        "status": "confirmed",
        "operator_note": "Verified vehicle collision on lane 2. Dispatched roadside unit.",
        "location_label": "Location: not available in dataset (simulated demo camera)",
    },
    {
        "id": "inc-demo-002",
        "camera_id": "cam-02",
        "camera_name": "Camera 02 - Downtown Junction",
        "detected_at": "2026-10-27T02:05:00Z",
        "video_start_time": 0.0,
        "video_end_time": 1.67,
        "peak_confidence": 0.8977,
        "mean_confidence": 0.7336,
        "window_indices": [0, 1],
        "peak_frame_snapshot": "/samples/video_project.jpg",
        "model_version": "2.0.0",
        "status": "new",
        "operator_note": "",
        "location_label": "Location: not available in dataset (simulated demo camera)",
    }
]


class Incident(BaseModel):
    id: str
    camera_id: str = "cam-01"
    camera_name: str = "Camera 01 - Highway Sector 4"
    detected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    video_start_time: float
    video_end_time: float
    peak_confidence: float
    mean_confidence: float
    window_indices: List[int]
    peak_frame_snapshot: Optional[str] = None
    model_version: str = "2.0.0"
    status: str = "new"  # "new" | "confirmed" | "false_alarm"
    operator_note: str = ""
    location_label: str = "Location: not available in dataset"


class IncidentPatchRequest(BaseModel):
    status: Optional[str] = None
    operator_note: Optional[str] = None


class IncidentListResponse(BaseModel):
    incidents: List[Incident]
    total: int
    page: int = 1
    page_size: int = 20


def cluster_windows_into_incidents(
    predictions: List[Dict[str, Any]],
    camera_id: str = "cam-01",
    camera_name: str = "Camera 01 - Highway Sector 4",
    location_label: str = "Location: not available in dataset",
    persistence_k: int = 1,
    cooldown_windows: int = 2,
    peak_frame_snapshot: Optional[str] = None,
) -> List[Incident]:
    """
    Merges consecutive flagged windows into single incidents based on:
      - persistence_k: minimum number of consecutive windows exceeding threshold
      - cooldown_windows: windows tolerance before concluding an incident
    """
    if not predictions:
        return []

    incidents: List[Incident] = []
    current_run: List[Dict[str, Any]] = []
    last_flagged_idx = -1000

    for w in predictions:
        is_flagged = bool(w.get("is_accident", False))
        idx = int(w.get("window_index", 0))

        if is_flagged:
            if current_run:
                # Check if within cooldown
                if idx - last_flagged_idx <= cooldown_windows + 1:
                    current_run.append(w)
                    last_flagged_idx = idx
                else:
                    # Previous incident concluded; finalize if met persistence
                    if len(current_run) >= persistence_k:
                        incidents.append(_finalize_incident(
                            current_run, camera_id, camera_name, location_label, peak_frame_snapshot
                        ))
                    current_run = [w]
                    last_flagged_idx = idx
            else:
                current_run = [w]
                last_flagged_idx = idx

    # Finalize any pending run
    if len(current_run) >= persistence_k:
        incidents.append(_finalize_incident(
            current_run, camera_id, camera_name, location_label, peak_frame_snapshot
        ))

    return incidents


def _finalize_incident(
    run_windows: List[Dict[str, Any]],
    camera_id: str,
    camera_name: str,
    location_label: str,
    peak_frame_snapshot: Optional[str],
) -> Incident:
    confs = [float(w.get("confidence", 0.0)) for w in run_windows]
    peak_conf = max(confs) if confs else 0.0
    mean_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
    start_t = min(float(w.get("start_time", 0.0)) for w in run_windows)
    end_t = max(float(w.get("end_time", 0.0)) for w in run_windows)
    window_idxs = [int(w.get("window_index", 0)) for w in run_windows]

    return Incident(
        id=f"inc-{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        camera_name=camera_name,
        detected_at=datetime.now(timezone.utc).isoformat(),
        video_start_time=round(start_t, 2),
        video_end_time=round(end_t, 2),
        peak_confidence=round(peak_conf, 4),
        mean_confidence=mean_conf,
        window_indices=window_idxs,
        peak_frame_snapshot=peak_frame_snapshot,
        model_version="2.0.0",
        status="new",
        operator_note="",
        location_label=location_label,
    )


async def send_discord_notification(incident: Incident):
    """
    Sends an incident alert embed to Discord webhook if DISCORD_WEBHOOK_URL is set.
    """
    url = DISCORD_WEBHOOK_URL or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return

    payload = {
        "embeds": [
            {
                "title": f"🚨 Accident Alert: {incident.camera_name}",
                "description": (
                    f"**Incident ID**: `{incident.id}`\n"
                    f"**Location**: {incident.location_label}\n"
                    f"**Peak Model Score**: `{incident.peak_confidence:.3f}`\n"
                    f"**Mean Model Score**: `{incident.mean_confidence:.3f}`\n"
                    f"**Video Range**: `{incident.video_start_time:.1f}s – {incident.video_end_time:.1f}s`\n"
                    f"**Status**: `{incident.status.upper()}`"
                ),
                "color": 15158332 if incident.peak_confidence > 0.75 else 15105570,
                "timestamp": incident.detected_at,
                "footer": {"text": "Road Accident Detection System v2.0"},
            }
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning("Discord webhook returned error status %d", resp.status_code)
    except Exception as exc:
        logger.warning("Failed to dispatch Discord webhook: %s", exc)
