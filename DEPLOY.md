# Free-Tier Deployment Guide (100% Zero Cost)

This guide walks you through deploying the Road Accident Detection System completely free of charge (no credit card required, no trials, no paid plans):

| Component | Platform | Plan | Key Constraints |
|:---|:---|:---|:---|
| **Frontend** | [Vercel](https://vercel.com/) | Hobby | Static Vite build, SPA rewrites |
| **Backend** | [Render](https://render.com/) | Free Web Service (Docker) | 512 MB RAM, sleeps after 15m idle |
| **Database** | [MongoDB Atlas](https://www.mongodb.com/atlas) | M0 Sandbox | 512 MB storage, shared RAM |

---

## Step 1: MongoDB Atlas M0 Database Setup

1. Sign up or log in to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (Free forever, no credit card required).
2. Create a free **M0 Sandbox** cluster:
   - Select AWS or Google Cloud in your preferred region.
   - Cluster Name: `accident-cluster`.
3. Set up Database Access:
   - Go to **Security > Database Access**.
   - Click **Add New Database User**.
   - Choose **Password** authentication.
   - Username: `accident_user`.
   - Set a strong password (keep this for your URI).
   - Database User Privileges: `Read and write to any database`.
4. Set up Network Access:
   - Go to **Security > Network Access**.
   - Click **Add IP Address**.
   - Select **Allow Access From Anywhere** (`0.0.0.0/0`).
   - *Reason:* Render free tier does not provide a static IP address, so incoming backend connections originate from dynamic IPs.
5. Retrieve Connection String:
   - Go to **Deployment > Database**.
   - Click **Connect > Drivers** (Python).
   - Copy the connection URI:
     ```text
     mongodb+srv://accident_user:<PASSWORD>@accident-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
     ```
   - Replace `<PASSWORD>` with your database user password.

---

## Step 2: Render Free Docker Backend Setup

The repository includes a ready-to-deploy [`render.yaml`](render.yaml) blueprint.

### Option A: Via Render Blueprint (Fastest)
1. Sign up or log in to [Render](https://render.com/).
2. Click **New + > Blueprint**.
3. Connect your GitHub repository: `Vatsalgada2006/deep-learning-project`.
4. Render will parse [`render.yaml`](render.yaml) and configure the Docker web service.
5. In the secrets prompts:
   - Set `MONGO_URI` to your MongoDB Atlas connection string from Step 1.
   - (Optional) Set `DISCORD_WEBHOOK_URL` if you want Discord alerts.
6. Click **Apply**.

### Option B: Manual Web Service Setup
1. Click **New + > Web Service**.
2. Connect your GitHub repository.
3. Configure the service:
   - **Name:** `road-accident-detection-api`
   - **Language:** `Docker`
   - **Dockerfile Path:** `backend/Dockerfile`
   - **Docker Context:** `.` (root directory)
   - **Instance Type:** `Free` (0.1 CPU, 512 MB RAM)
   - **Health Check Path:** `/health`
4. Add Environment Variables:
   - `PORT`: `10000`
   - `HOST`: `0.0.0.0`
   - `MODEL_PATH`: `./model.onnx`
   - `DEMO_MODE`: `false`
   - `ALLOWED_ORIGINS`: `http://localhost:5173,https://<your-vercel-app>.vercel.app`
   - `MONGO_URI`: `mongodb+srv://...` (your Atlas URI)
   - `MONGO_DB_NAME`: `accident_detection`
   - `MAX_UPLOAD_BYTES`: `10485760`
   - `MAX_VIDEO_DURATION_S`: `15`
   - `FRAME_SAMPLE_INTERVAL`: `5`
   - `FRAME_RESIZE_WIDTH`: `320`
5. Click **Create Web Service**.

Render will build the multi-stage slim Docker image. Note your backend URL:
`https://road-accident-detection-api.onrender.com`

---

## Step 3: Vercel Hobby Frontend Deployment

1. Sign up or log in to [Vercel](https://vercel.com/).
2. Click **Add New... > Project**.
3. Import your GitHub repository: `Vatsalgada2006/deep-learning-project`.
4. Build settings are auto-detected via [`vercel.json`](vercel.json):
   - **Framework Preset:** `Vite`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
5. Configure Environment Variables:
   - `VITE_API_URL`: `https://road-accident-detection-api.onrender.com` (your Render backend URL, without trailing slash)
   - `VITE_WS_URL`: `wss://road-accident-detection-api.onrender.com/ws/alerts` (Render supports secure WebSockets natively)
6. Click **Deploy**.

Vercel will build and assign you a free domain (e.g., `https://deep-learning-project-xyz.vercel.app`).

---

## Step 4: Update CORS on Render

Once your Vercel URL is generated:
1. Open your Render dashboard for `road-accident-detection-api`.
2. Go to **Environment**.
3. Update `ALLOWED_ORIGINS`:
   ```text
   http://localhost:5173,https://deep-learning-project-xyz.vercel.app
   ```
4. Save changes. Render will automatically perform a zero-downtime reload.

---

## Step 5: Verification & Testing Checklist

- [ ] **Health Check:** Open `https://road-accident-detection-api.onrender.com/health` in your browser. Verify `{"status": "healthy", "model_loaded": true, "mongodb_connected": true}`.
- [ ] **Cold-Start Handling:** When accessing the frontend after idle sleep, confirm the UI displays *"Waking Server (cold-start)..."* and automatically connects once ready.
- [ ] **3D Scene & 2D Fallback:** Test dragging the 3D scene; toggle between **3D View** and **2D Grid** in the top bar.
- [ ] **Video Upload:** Upload a short MP4 clip (≤10 MB, ≤15s) in the **Analyze Video** tab. Verify sliding window confidence scores and incident alerts.
- [ ] **Database Persistence:** Check the **History** tab to verify that detected incidents appear and persist in Atlas.
- [ ] **TTL Index Verification:** Detections are automatically deleted after 30 days to stay well within the 512 MB Atlas limit.
