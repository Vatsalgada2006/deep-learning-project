# Real-Time Road Accident Detection System

[![CI Pipeline](https://github.com/Vatsalgada2006/deep-learning-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Vatsalgada2006/deep-learning-project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev/)

An autonomous, edge-ready road accident detection system combining a temporal **CNN-LSTM** deep learning model with a reactive 3D WebGL / 2D dashboard. Deployed on **100% free-tier cloud infrastructure** (Vercel Hobby + Render Free Docker + MongoDB Atlas M0).

---

## 🏛️ System Architecture

```text
[ CCTV / Video Upload ] 
          │
          ▼
[ FastAPI Backend (Render Free Docker, <512MB RAM) ]
   ├── ONNX Runtime (CPU, Multi-Stage Slim Image)
   ├── Sliding Window Sequence Extractor (10 frames @ 128x128)
   ├── Concurrency Limiter (Single worker inference queue)
   └── TTL Auto-Cleanup Index (30-day auto-purge)
          │
          ├──► [ MongoDB Atlas M0 ] ── Detection History
          ├──► [ WebSocket /ws/alerts ] ── Real-Time Incident Stream
          └──► [ Discord Webhook (Optional) ] ── Push Notifications
                    │
                    ▼
[ React 19 Frontend (Vercel Hobby) ]
   ├── 3D Digital Twin (React Three Fiber / Three.js)
   ├── 2D CCTV Matrix (Low-end hardware fallback)
   ├── Video Upload & Sliding-Window Analysis UI
   └── Render Cold-Start Wake-up Polling with Exponential Backoff
```

---

## 📊 Model Performance & Evaluation

The system utilizes a custom **CNN-LSTM** architecture trained on the Kaggle *Road Accidents from CCTV Footages* dataset. Because traffic surveillance is a safety-critical application, the model is specifically tuned to maximize **recall** to eliminate false negatives (unreported crashes).

| Metric | Baseline Spatial CNN (MobileNetV2) | CNN-LSTM (Ours) | Engineering Impact |
|:---|:---:|:---:|:---|
| **Accuracy** | 62.34% | **79.98%** | +17.64% overall diagnostic accuracy |
| **Precision** | **87.83%** | 80.74% | Balanced for safety over strict filtering |
| **Recall** | 54.79% | **94.45%** | **+39.66% fewer missed accidents** |
| **F1-Score** | 67.48% | **87.06** | Substantial gain in harmonic performance |
| **ROC-AUC** | 0.8207 | **0.8220** | Strong discriminative ability across thresholds |

> **Evaluation Takeaway:** Traffic incidents are fundamentally dynamic temporal sequences. Single-frame 2D CNNs frequently misclassify stationary parked vehicles or shadows as crashes. Incorporating an LSTM sequence layer over 10 consecutive frames resolves temporal context, elevating recall from 54.8% to 94.5%.

---

## ⚡ Free-Tier Engineering & Constraints

This project is architected specifically to run on zero-cost free-tier tiers:

1. **Render Free Tier (<512 MB RAM):**
   - No PyTorch, TensorFlow, or GUI OpenCV in the production image.
   - Pure `onnxruntime` CPU execution provider with `opencv-python-headless`.
   - Multi-stage slim Docker build.
   - Strict 1-concurrent-inference guard returning HTTP 503 when busy.
   - Cold-start handling: Frontend gracefully detects cold-starts and polls `/health` with exponential backoff before establishing WebSocket connections.

2. **MongoDB Atlas M0 (512 MB Storage Limit):**
   - MongoDB documents are capped to <4 KB.
   - Built-in TTL index automatically purges incidents older than 30 days (`expireAfterSeconds: 2592000`).

3. **Vercel Hobby (Frontend):**
   - Static Vite production build with SPA rewrites (`vercel.json`).
   - Graceful 3D degradation with React Error Boundaries falling back to a responsive 2D CCTV camera matrix.

---

## 🛠️ API Endpoints

| Method | Path | Description |
|:---|:---|:---|
| `GET` | `/health` | Server health, model loaded status, MongoDB connection, demo mode state |
| `GET` | `/ping` | Lightweight ping probe |
| `POST` | `/predict` | Upload video clip (`.mp4`, `.webm` ≤10 MB, ≤15s) for sliding-window inference |
| `GET` | `/history` | Paginated incident log from MongoDB Atlas (`limit`, `offset`, `sort`) |
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

# Start Vite dev server
npm run dev
```

Visit `http://localhost:5173` in your browser.

---

## 🚢 Production Deployment

For complete, zero-cost deployment instructions on **Vercel**, **Render**, and **MongoDB Atlas**, read [DEPLOY.md](DEPLOY.md).

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
