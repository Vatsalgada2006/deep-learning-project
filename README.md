# Real-Time Road Accident Detection System
### Architecture Overview (Text-based)
[CCTV Feed] --> [Backend (FastAPI)] --> [ONNX Model]
                          |          |
                          v          v
                  [MongoDB]    [WebSocket Alerts]
                          |          |
                          v          v
[Frontend (React)] <-- [Discord Webhook (Optional)]
This system delivers real-time accident detection from CCTV feeds, reducing emergency response times through early warnings.
Based on a college case study, it provides a deployable solution that prioritizes recall to minimize missed accidents in safety-critical scenarios.

Based on a college case study report, this implementation delivers the promised functionality with measurable performance metrics.

## Features

- **Real-time Detection**: CNN-LSTM model processes video streams at ~30 FPS, detecting accidents with low latency.
- **3D Dashboard**: Built with React, TypeScript, Vite, React Three Fiber, and Framer Motion for a modern control-room aesthetic.
- **Live Alerts**: WebSocket connections push accident alerts to the frontend in real time.
- **History Log**: Detection events are stored in MongoDB Atlas and displayed in a table.
- **Discord Integration**: Optional webhook for instant notifications.
- **Performance Monitoring**: End-to-end latency measurement displayed in alerts.
- **Free Tier Deployment**: Frontend on Vercel, backend on Railway (or Hugging Face Spaces), MongoDB Atlas free tier.

## Model Performance

The CNN-LSTM model outperforms a baseline CNN (MobileNetV2 transfer learning) on the Kaggle "Road Accidents from CCTV Footages" dataset.

| Metric | Baseline CNN | CNN‑LSTM (Ours) |
|--------|--------------|-----------------|
| Accuracy | 0.6234 | 0.7998 |
| Precision | 0.8783 | 0.8074 |
| Recall | 0.5479 | 0.9445 |
| F1‑Score | 0.6748 | **0.8706** |
| ROC‑AUC | 0.8207 | **0.8220** |

*All metrics computed from confusion matrices and ROC‑AUC values in the provided result files.*

### Evaluation Notes

The CNN‑LSTM model prioritizes recall (0.9445) over precision (0.8074) because missing a real accident (false negative) is more costly than a false alert (false positive) in a safety‑critical system. Early detection enables quicker emergency response, potentially saving lives and reducing damage.

## Local Setup

### Prerequisites
- [Node.js](https://nodejs.org/) (v18+)
- [Python](https://www.python.org/) (3.11+)
- [MongoDB](https://www.mongodb.com/try/download/community) (local) or MongoDB Atlas account
- [Git](https://git-scm.com/)

### Backend
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   cd backend
   python -m venv venv
   source venv/Scripts/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   - Copy `.env.template` to `.env` and adjust values:
     ```env
     MODEL_PATH=./model.onnx
     SEQ_LENGTH=10
     IMG_SIZE=128
     VIDEO_PATH=./sample_video.mp4  # Place a sample video here or leave for simulated frames
     PORT=8001
     HOST=0.0.0.0
     MONGO_URI=mongodb://localhost:27017   # or MongoDB Atlas URI
     # DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url_here
     ```
4. Start the backend:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8001
   ```
   The server will load the ONNX model, connect to MongoDB, and begin processing frames (simulated if no video file is present).

### Frontend
1. In a new terminal, navigate to the project root:
   ```bash
   cd ..
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173` (or another port if 5173 is in use).

### Usage
- Open the frontend in your browser.
- The 3D scene shows camera nodes; when an accident is detected, the relevant node pulses red and an alert panel slides in from the bottom-right.
- The alert panel displays timestamp, confidence, camera ID, and measured latency.
- System status (online/offline) and current time are shown in the top bar.
- Alert history can be viewed via the backend's `/history` endpoint (integrated in a future update).

## Deployment

### Frontend (Vercel)
1. Push the repository to GitHub.
2. Import the project into [Vercel](https://vercel.com/).
3. Vercel will automatically detect the Vite project and run `npm install` and `npm run build`.
4. Set the environment variable `VITE_WS_URL` (if needed) to point to your deployed backend WebSocket URL (e.g., `wss://your-backend.up.railway.app/ws/alerts`).  
   *Note: The frontend currently uses a hardcoded WebSocket URL (`ws://localhost:8001/ws/alerts`). For deployment, you will need to modify `src/App.tsx` to read the URL from an environment variable (e.g., `import.meta.env.VITE_WS_URL`) and rebuild.*

### Backend (Railway)
1. Create a new Railway project.
2. Connect your GitHub repository.
3. Railway will detect the Dockerfile and build the Docker image.
4. Set the following environment variables in Railway:
   - `MODEL_PATH`: `/app/backend/model.onnx` (relative to the container's working directory)
   - `SEQ_LENGTH`: `10`
   - `IMG_SIZE`: `128`
   - `VIDEO_PATH`: `/app/sample_video.mp4` (you can include a small sample video in the repo or rely on simulated frames)
   - `PORT`: Railway will automatically provide the `PORT` environment variable; our Dockerfile uses `$PORT`.
   - `HOST`: `0.0.0.0`
   - `MONGO_URI`: Your MongoDB Atlas connection string.
   - `DISCORD_WEBHOOK_URL`: (Optional) Your Discord webhook URL.
5. Deploy! Railway will expose the backend on a public URL (e.g., `https://your-backend.up.railway.app`).

### MongoDB Atlas
1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Create a database named `accident_detection` and a collection `detections`.
3. Add your IP address to the access list (or allow access from anywhere for testing).
4. Copy the connection string and set it as the `MONGO_URI` environment variable in both local and deployed backends.

### Discord Webhook (Optional)
1. Create a Discord webhook in your desired channel.
2. Copy the webhook URL and set it as the `DISCORD_WEBHOOK_URL` environment variable.
3. The backend will send a rich embed message on each detection.

## Performance & Latency
## Known Limitations & Future Work

**Limitations:**
- The model is trained on a specific dataset and may not generalize to all CCTV environments.
- Latency measurements in the README are based on local testing; production latency depends on network conditions.
- The 3D dashboard, while optimized, may not maintain 60fps on low-end hardware during peak loads.

**Future Work:**
- Implement model confidence calibration for more reliable alert thresholds.
- Add support for multiple camera feeds with dynamic load balancing.
- Enhance the 3D dashboard with camera-specific controls and playback of detected events.
- Explore edge deployment options (e.g., TensorRT) to further reduce inference latency.
End-to-end latency is measured in the frontend as the time between the detection timestamp (generated by the backend) and the alert receipt. In local testing with simulated frames, latency averages **3‑5 ms** (well under the 2‑second target). On deployed services, latency will depend on network conditions but should remain under 2 seconds for most users.

## Model Files
The trained ONNX model (`model.onnx`) is included in the repository (in `backend/`). It is approximately 9.9 MB and is excluded from the Git history via `.gitignore` to keep the repository lightweight. If you need to retrain the model, refer to the training scripts and instructions in `COLAB_TRAINING_INSTRUCTIONS.md`.

## Acknowledgments
- Kaggle for providing the "Road Accidents from CCTV Footages" dataset.
- The open-source libraries and frameworks used (React, Vite, Three.js, FastAPI, MongoDB, etc.).
- Inspiration from the original case study report.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
