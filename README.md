# Real-Time Road Accident Detection System

[![CI Pipeline](https://github.com/Vatsalgada2006/deep-learning-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Vatsalgada2006/deep-learning-project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev/)

An autonomous, edge-ready road accident detection system combining a temporal **CNN-LSTM** deep learning model with a reactive 3D WebGL / 2D dashboard. Deployed on **100% free-tier cloud infrastructure** (Vercel Hobby + Render Free Docker + MongoDB Atlas M0).

---

## 🏛️ System Architecture

```text
[ CCTV / Video Upload / Web Worker WASM ] 
          │
          ├── [ Optional Client-Side Inference: ONNX Runtime Web (WASM) ]
          │
          ▼
[ FastAPI Backend (Render Free Docker, <512MB RAM) ]
   ├── ONNX Runtime (CPU, Multi-Stage Slim Image, ~41ms / window)
   ├── Sliding Window Sequence Extractor (10 frames @ 128x128)
   ├── Incident Clustering Engine (k-Persistence + Cooldown Merging)
   ├── Concurrency Limiter (Single worker inference queue)
   └── TTL Auto-Cleanup Index (30-day auto-purge)
          │
          ├──► [ MongoDB Atlas M0 ] ── Incident Store (30-day TTL)
          ├──► [ WebSocket /ws/alerts ] ── Real-Time Incident Broadcast
          └──► [ Discord Webhook (Optional) ] ── Push Notifications
                    │
                    ▼
[ React 19 Frontend (Vercel Hobby) ]
   ├── Map Panel (Leaflet + OpenStreetMap) with Simulated Demo Cameras
   ├── Live Incident Alert Feed (Newest first, Status Confirm/False Alarm)
   ├── Incident Detail Drawer (Grad-CAM Attention Snapshot, Operator Notes)
   ├── Client-Side JSON & RFC-4180 CSV Export
   ├── Client vs Server Engine Parity Comparison Runner
   ├── "How to Read This" & "Where It Fails" Empirical Transparency Panels
   └── Render Cold-Start Wake-up Polling (Resolves Online/Offline ≤90s)
```

---

## 📊 Model Performance & Measured Benchmarks

The system utilizes a custom **CNN-LSTM** architecture trained on the Kaggle *Road Accidents from CCTV Footages* dataset. Because traffic surveillance is a safety-critical application, the model is specifically tuned to maximize **recall** to eliminate false negatives (unreported crashes).

| Metric | Baseline Spatial CNN (MobileNetV2) | CNN-LSTM (Ours) | Engineering Impact |
|:---|:---:|:---:|:---|
| **Accuracy** | 62.34% | **79.98%** | +17.64% overall diagnostic accuracy |
| **Precision** | **87.83%** | 80.74% | Balanced for safety over strict filtering |
| **Recall** | 54.79% | **94.45%** | **+39.66% fewer missed accidents** |
| **F1-Score** | 67.48% | **87.06** | Substantial gain in harmonic performance |
| **ROC-AUC** | 0.8207 | **0.8220** | Strong discriminative ability across thresholds |

### Real Measured Benchmark Numbers:
- **Inference Latency:** ~41.2 ms per 10-frame window on standard CPU (`onnxruntime` CPUExecutionProvider).
- **WASM Engine Parity:** Output discrepancy between Browser ONNX Runtime Web (WASM) and Python ONNX Runtime is $< 10^{-6}$ (dummy zeros raw score: `0.9840105` vs `0.9840106`).
- **Continuous Sample Video:** `sample_continuous_01.mp4` (200 frames, 10.0s @ 20 fps, 1.19 MB H.264 `avc1`, `yuv420p`, peak model score: **0.0617** [Normal/Clean]).
- **Image Sequence Pace:** Assembled stills play at 7 fps (0.35x natural pace) labeled *"Image sequence from dataset, not continuous video"*.
- **Empirical Dataset Failure Modes:**
  - *False Positive:* `NonAccident/11979.jpg` (score **0.5758**) — high-beam headlight reflections simulating wreckage glare.
  - *Borderline Normal:* `NonAccident/12013.jpg` (score **0.4793**) — sudden braking and rapid lane cut-in.
  - *False Negative:* `Accident/1424.jpg` (score **0.1606**) — distant collision obscured by heavy tree shadows.
  - *Stationary Aftermath:* `Accident/129.jpg` (score **0.1873**) — post-crash stationary vehicle lacking temporal motion.

---

## ⚡ Incident Monitoring & Clustering Rules

Rather than spamming operators with raw window detections, consecutive flagged windows are clustered into **Incidents**:
1. **$k$-Persistence:** Requires at least $k$ (configurable, default 1 or 2) consecutive/near-consecutive flagged windows before registering an incident, suppressing single-frame false spikes.
2. **Cooldown Merging:** Consecutive flagged windows within $c$ windows of each other (default 2 windows) are merged into **ONE** incident.
3. **Audit Trail & Triage:** Operators can mark incidents as `confirmed` or `false_alarm`, append operational notes, and export incident records to structured JSON and CSV files.

---

## 🛠️ API Endpoints

| Method | Path | Description |
|:---|:---|:---|
| `GET` | `/health` | Server health, model loaded status, MongoDB connection, demo mode state |
| `GET` | `/ping` | Lightweight ping probe (resolves within 90s timeout) |
| `GET` | `/cameras` | Registry of demo surveillance cameras (`id`, `name`, `label`, `lat`, `lon`) |
| `POST` | `/predict` | Upload video clip; returns window predictions and merged `incidents` |
| `GET` | `/incidents` | Paginated incident history (`limit`, `offset`, `status`, `camera_id`) |
| `GET` | `/incidents/{id}` | Retrieve individual incident record |
| `PATCH` | `/incidents/{id}` | Update incident status (`new`, `confirmed`, `false_alarm`) and operator notes |
| `GET` | `/history` | Paginated legacy detection log from MongoDB Atlas |
| `GET` | `/metrics` | Comparative benchmark metrics (Baseline CNN vs CNN-LSTM) |
| `GET` | `/model-info` | ONNX input/output tensor shapes and configuration |
| `WS` | `/ws/alerts` | Bidirectional real-time WebSocket connection for live incident broadcast |

---

## 💻 Local Development Setup

### 1. Prerequisites
- [Node.js](https://nodejs.org/) (v18+)
- [Python](https://www.python.org/) (3.11+)
- [Git](https://git-scm.com/)

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Vatsalgada2006/deep-learning-project.git
cd deep-learning-project

# Setup Python environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
pip install pytest httpx

# Configure environment
cp backend/.env.example backend/.env

# Run unit tests
python -m pytest tests/ -v

# Start the development server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Frontend Setup
```bash
# In the repository root
npm install

# Run frontend unit tests (Vitest)
npm run test

# Run frontend build
npm run build

# Start Vite dev server
npm run dev
```

Visit `http://localhost:5173` in your browser.

---

## 🚢 Production Deployment

For complete, zero-cost deployment instructions on **Vercel**, **Render**, and **MongoDB Atlas**, read [DEPLOY.md](DEPLOY.md).

---

## ⏱️ Optional Free Uptime Pinger (Prevent Render Cold Starts)

Render free-tier instances spin down automatically after **15 minutes** of inactivity. When a visitor arrives, waking the container takes 30–50 seconds. To keep your backend continuously warm and eliminate cold starts:

1. **Sign up for a free uptime monitor** (e.g., [Cron-Job.org](https://cron-job.org/), [UptimeRobot](https://uptimerobot.com/), or [Better Stack](https://betterstack.com/)).
2. **Endpoint:** Set up a `GET` monitor pointing to:
   ```text
   https://<your-render-app>.onrender.com/health
   ```
3. **Ping Interval:** Every **5 minutes** (must be < 15 minutes to prevent sleep).
4. **Expected Status:** `200 OK` (with JSON `{"status": "healthy"}`).
5. **Why it works efficiently:**
   - `/health` answers **instantly** without touching MongoDB or loading the ONNX model.
   - Consumes negligible CPU/bandwidth and zero database queries.
   - Keeps the container active while remaining 100% free-tier compliant.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
