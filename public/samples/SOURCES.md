# Sample Video Sources & Licences

This document records the exact provenance, licensing, assembly method, and dataset split context for all media in `public/samples/`.

---

## 1. `accident_crash_demo.mp4`
- **Label**: `Highway Multi-Lane Collision`
- **Clip Category**: Real continuous traffic video (not assembled stills).
- **Temporal Characteristics**: Continuous 30 FPS video (394 frames, 13.15 seconds).
- **Source Context**: Real vehicle collision footage on a multi-lane highway.
- **Dataset Split Status**: Held-out demo clip. Excluded from training, validation, and test splits (also mirrored in `demo_clips/`).
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 1280x720, 30 FPS, 1.45 MB.
- **Poster**: `public/samples/accident_crash_demo.jpg`
- **Contact Sheet**: `public/samples/accident_crash_demo_sheet.jpg`
- **Evaluation**: `public/samples/accident_crash_demo.json` (Peak Score: **0.8724**, Mean: **0.2671**, Windows: 20).

---

## 2. `video_project.mp4`
- **Label**: `Intersection Vehicle Crash`
- **Clip Category**: Real continuous surveillance video.
- **Temporal Characteristics**: Continuous 30 FPS video (240 frames, 8.00 seconds).
- **Source Context**: Traffic collision sequence at roadway intersection.
- **Dataset Split Status**: Held-out demo clip. Excluded from training, validation, and test splits (mirrored in `demo_clips/`).
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 640x360, 30 FPS, 1.10 MB.
- **Poster**: `public/samples/video_project.jpg`
- **Contact Sheet**: `public/samples/video_project_sheet.jpg`
- **Evaluation**: `public/samples/video_project.json` (Peak Score: **0.8977**, Mean: **0.7336**, Windows: 12).

---

## 3. `non_accident.mp4`
- **Label**: `Continuous Normal Highway Flow`
- **Clip Category**: Real continuous CCTV surveillance footage.
- **Temporal Characteristics**: Continuous 30 FPS daytime highway stream (830 frames, 27.69 seconds).
- **Source Context**: Real CCTV highway traffic monitoring without incidents.
- **Dataset Split Status**: Held-out demo clip. Excluded from training, validation, and test splits (mirrored in `demo_clips/`).
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 1280x720, 30 FPS, 2.61 MB.
- **Poster**: `public/samples/non_accident.jpg`
- **Contact Sheet**: `public/samples/non_accident_sheet.jpg`
- **Evaluation**: `public/samples/non_accident.json` (Peak Score: **0.0228**, Mean: **0.0173**, Windows: 42).

---

## 4. `car-accident-dataset-example2.mp4`
- **Label**: `Dense Traffic Surveillance`
- **Clip Category**: Real continuous traffic video sequence.
- **Temporal Characteristics**: Continuous 30 FPS video (210 frames, 7.01 seconds).
- **Source Context**: Traffic sequence monitoring.
- **Dataset Split Status**: Held-out demo clip. Excluded from training, validation, and test splits (mirrored in `demo_clips/`).
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 1280x720, 30 FPS, 0.33 MB.
- **Poster**: `public/samples/car-accident-dataset-example2.jpg`
- **Contact Sheet**: `public/samples/car-accident-dataset-example2_sheet.jpg`
- **Evaluation**: `public/samples/car-accident-dataset-example2.json` (Peak Score: **0.0174**, Mean: **0.0162**, Windows: 11).
