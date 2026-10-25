"""
Test: the backend correctly inverts the model's sigmoid output.

The CNN-LSTM was trained with enumerate(['Accident', 'NonAccident']),
so label 0 = Accident, label 1 = NonAccident.  The sigmoid output is
P(NonAccident).  The backend must compute  confidence = 1 - raw  so
that ``is_accident`` triggers on real accidents, not safe footage.

Run:  python -m pytest test_confidence_inversion.py -v
"""

import importlib
import types
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np


# ---------------------------------------------------------------------------
# Minimal stub so we can import main.py without a real ONNX model or Mongo
# ---------------------------------------------------------------------------

def _make_stubs():
    """Create lightweight stubs for heavy dependencies."""
    # onnxruntime
    ort_stub = types.ModuleType("onnxruntime")
    ort_stub.InferenceSession = MagicMock
    ort_stub.SessionOptions = MagicMock
    ort_stub.GraphOptimizationLevel = MagicMock()
    ort_stub.GraphOptimizationLevel.ORT_ENABLE_ALL = 0
    sys.modules.setdefault("onnxruntime", ort_stub)

    # motor (never needed here)
    motor_stub = types.ModuleType("motor")
    motor_async = types.ModuleType("motor.motor_asyncio")
    motor_async.AsyncIOMotorClient = MagicMock
    motor_stub.motor_asyncio = motor_async
    sys.modules.setdefault("motor", motor_stub)
    sys.modules.setdefault("motor.motor_asyncio", motor_async)


_make_stubs()

# We import only the pure function we care about; the module-level lifespan
# never runs during import.
from app.main import run_inference_on_video, ACCIDENT_CONFIDENCE_THRESHOLD  # noqa: E402


class TestConfidenceInversion(unittest.TestCase):
    """Verify that raw sigmoid → accident confidence is inverted."""

    # -- helper -----------------------------------------------------------

    @staticmethod
    def _fake_video_bytes() -> bytes:
        """Return a tiny valid MP4 so cv2.VideoCapture can open it."""
        import cv2, tempfile, os  # noqa: E401

        path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30, (64, 64))
        # Write enough frames so that after FRAME_SAMPLE_INTERVAL=5
        # sampling we still have SEQ_LENGTH=10 frames for one window.
        # 10 * 5 = 50 frames needed.
        for _ in range(50):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        with open(path, "rb") as f:
            data = f.read()
        os.unlink(path)
        return data

    # -- tests ------------------------------------------------------------

    def test_high_raw_means_safe(self):
        """
        If the model outputs 0.95 (≈ NonAccident), the backend should
        report confidence ≈ 0.05 and is_accident = False.
        """
        raw_sigmoid = np.array([[0.95]], dtype=np.float32)

        fake_session = MagicMock()
        fake_session.run.return_value = [raw_sigmoid]

        with patch("app.main.model_session", fake_session), \
             patch("app.main.model_input_name", "input"), \
             patch("app.main.model_output_name", "output"):
            result = run_inference_on_video(self._fake_video_bytes())

        preds = result["predictions"]
        self.assertTrue(len(preds) >= 1, "Expected at least one prediction window")
        first = preds[0]
        self.assertAlmostEqual(first["confidence"], 0.05, places=2)
        self.assertFalse(first["is_accident"],
                         "Raw 0.95 (safe) should NOT trigger an accident alert")

    def test_low_raw_means_accident(self):
        """
        If the model outputs 0.10 (≈ Accident), the backend should
        report confidence ≈ 0.90 and is_accident = True.
        """
        raw_sigmoid = np.array([[0.10]], dtype=np.float32)

        fake_session = MagicMock()
        fake_session.run.return_value = [raw_sigmoid]

        with patch("app.main.model_session", fake_session), \
             patch("app.main.model_input_name", "input"), \
             patch("app.main.model_output_name", "output"):
            result = run_inference_on_video(self._fake_video_bytes())

        preds = result["predictions"]
        self.assertTrue(len(preds) >= 1)
        first = preds[0]
        self.assertAlmostEqual(first["confidence"], 0.90, places=2)
        self.assertTrue(first["is_accident"],
                        "Raw 0.10 (accident) SHOULD trigger an accident alert")

    def test_threshold_boundary(self):
        """
        Raw = 0.50 → confidence = 0.50 → equal to default threshold
        (0.5), so is_accident should be False (strict >).
        """
        raw_sigmoid = np.array([[0.50]], dtype=np.float32)

        fake_session = MagicMock()
        fake_session.run.return_value = [raw_sigmoid]

        with patch("app.main.model_session", fake_session), \
             patch("app.main.model_input_name", "input"), \
             patch("app.main.model_output_name", "output"):
            result = run_inference_on_video(self._fake_video_bytes())

        preds = result["predictions"]
        self.assertTrue(len(preds) >= 1)
        first = preds[0]
        self.assertAlmostEqual(first["confidence"], 0.50, places=2)
        # confidence == threshold → strict >, so not an accident
        self.assertFalse(first["is_accident"])


if __name__ == "__main__":
    unittest.main()
