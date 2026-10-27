"""
Test: the backend correctly inverts the model's sigmoid output.

The CNN-LSTM was trained with enumerate(['Accident', 'NonAccident']),
so label 0 = Accident, label 1 = NonAccident.  The sigmoid output is
P(NonAccident).  The backend must compute  confidence = 1 - raw  so
that ``is_accident`` triggers on real accidents, not safe footage.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.app.main import run_inference_on_video, ACCIDENT_CONFIDENCE_THRESHOLD


class TestConfidenceInversion(unittest.TestCase):
    """Verify that raw sigmoid -> accident confidence is inverted."""

    @staticmethod
    def _fake_video_bytes() -> bytes:
        import cv2
        import tempfile
        frames = []
        for i in range(50):
            val = int(255 * ((i % 10) / 10.0))
            f = np.full((128, 128, 3), val, dtype=np.uint8)
            frames.append(f)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, 30.0, (128, 128))
        for f in frames:
            writer.write(f)
        writer.release()
        with open(tmp.name, "rb") as fh:
            data = fh.read()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return data

    def test_high_raw_means_safe(self):
        """
        If the model outputs 0.95 (≈ NonAccident), the backend should
        report confidence ≈ 0.05 and is_accident = False.
        """
        raw_sigmoid = np.array([[0.95]], dtype=np.float32)
        fake_session = MagicMock()
        fake_session.run.return_value = [raw_sigmoid]

        with patch("backend.app.main.model_session", fake_session), \
             patch("backend.app.main.model_input_name", "input"), \
             patch("backend.app.main.model_output_name", "output"):
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

        with patch("backend.app.main.model_session", fake_session), \
             patch("backend.app.main.model_input_name", "input"), \
             patch("backend.app.main.model_output_name", "output"):
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

        with patch("backend.app.main.model_session", fake_session), \
             patch("backend.app.main.model_input_name", "input"), \
             patch("backend.app.main.model_output_name", "output"):
            result = run_inference_on_video(self._fake_video_bytes())

        preds = result["predictions"]
        self.assertTrue(len(preds) >= 1)
        first = preds[0]
        self.assertAlmostEqual(first["confidence"], 0.50, places=2)
        self.assertFalse(first["is_accident"])
