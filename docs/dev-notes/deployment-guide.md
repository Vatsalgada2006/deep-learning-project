# Deployment Guide: Real-Time Road Accident Detection System

This guide documents the deployment process for the Real-Time Road Accident Detection System to make it portfolio-ready.

## Overview
We will deploy:
- **Backend**: Railway (excellent WebSocket support, free tier)
- **Frontend**: Vercel (optimized for React/Vite, free tier)  
- **Database**: MongoDB Atlas (free tier M0 cluster)

## Prerequisites
1. GitHub account with access to this repository
2. Railway account (https://railway.app)
3. Vercel account (https://vercel.com) 
4. MongoDB Atlas account (https://www.mongodb.com/cloud/atlas)
5. Node.js >= 16 (for local testing if needed)
6. Git installed

## Deployment Steps

### Phase 1: Database Setup - MongoDB Atlas
1. **Create MongoDB Atlas Account**
   - Sign up at https://www.mongodb.com/cloud/atlas/register
   - Verify email and complete profile

2. **Create New Project**
   - Click "New Project"
   - Name: "road-accident-detection" or similar
   - Invite team members if needed (can skip for solo)

3. **Build Database Cluster**
   - In your project, click "Build a Database"
   - Choose "Shared" (free tier) 
   - Select cloud provider & region (AWS, closest to you)
   - Cluster Tier: M0 Sandbox (free)
   - Cluster Name: "accident-detection-cluster"
   - Click "Create Cluster" (takes 3-5 minutes)

4. **Configure Network Access**
   - Go to "Network Access" → "IP Access List"
   - Click "Add IP Address"
   - For development/testing: "Allow Access from Anywhere" (0.0.0.0/0)
   - For production: Specify specific IPs or use VPC peering
   - Confirm and add

5. **Create Database User**
   - Go to "Database Access" → "Add New Database User"
   - Authentication Method: Password
   - Username: `accident_user` (or preferred)
   - Password: Generate secure password (save this!)
   - Database User Privileges: "Read and write to any database"
   - Add User

6. **Get Connection String**
   - Go to "Database" → "Connect"
   - Choose "Connect your application"
   - Driver: Node.js, Version: 4.1 or later
   - Copy the connection string format:
     ```
     mongodb+srv://<username>:<password>@cluster0.mongodb.net/<dbname>?retryWrites=true&w=majority
     ```
   - Replace `<username>` and `<password>` with your credentials
   - Use database name: `accident_detection`
   - Final format: 
     `mongodb+srv://accident_user:<password>@cluster0.mongodb.net/accident_detection?retryWrites=true&w=majority`

### Phase 2: Backend Deployment - Railway
1. **Create Railway Account**
   - Sign up at https://railway.app (GitHub login recommended)

2. **Create New Project**
   - Click "New Project" → "Deploy from GitHub"
   - Select this repository: `Vatsalgada2006/deep-learning-project`
   - Railway will auto-detect it's a Node.js project

3. **Configure Environment Variables**
   - Go to project Settings → Variables
   - Add the following variables:
     ```
     MONGO_URI=<your_mongodb_atlas_connection_string_from_step_6_above>
     PORT=${PORT}  # Railway provides this automatically
     HOST=0.0.0.0
     MODEL_PATH=../model.onnx  # Relative path from backend/ to model.onnx in root
     VIDEO_PATH=./sample_video.mp4  # Will use simulated frames if file not found
     # Optional: DISCORD_WEBHOOK_URL=<your_discord_webhook_url>
     ```
   - Important: Railway uses `${PORT}` for the port variable - it will be replaced automatically

4. **Deploy**
   - Railway should auto-deploy when you push to main branch
   - Or click "Deploy" manually in the dashboard
   - Watch the logs for:
     - "Connected to MongoDB successfully"
     - "ONNX model loaded successfully from ../model.onnx"
     - "Starting video frame processing task..."
     - "Uvicorn running on http://0.0.0.0:<PORT>"

5. **Verify Deployment**
   - Once deployed, get your Railway domain (e.g., `https://xxxxxx.up.railway.app`)
   - Test health endpoint: `https://xxxxxx.up.railway.app/health`
   - Should return: `{"status":"healthy","model_loaded":true,"timestamp":"..."}`
   - Test model info: `https://xxxxxx.up.railway.app/model-info`

### Phase 3: Frontend Deployment - Vercel
1. **Create Vercel Account**
   - Sign up at https://vercel.com (GitHub login recommended)

2. **Import Project**
   - Click "New Project" → "Import Git Repository"
   - Select this repository: `Vatsalgada2006/deep-learning-project`
   - Vercel should auto-detect it's a Vite/React project

3. **Configure Environment Variables**
   - In project Settings → Environment Variables
   - Add:
     ```
     VITE_BACKEND_URL=<your_railway_backend_url>
     Example: VITE_BACKEND_URL=https://xxxxxx.up.railway.app
     ```
   - Important: Must be prefixed with `VITE_` for Vite to expose it to client code
   - Do NOT add trailing slash

4. **Deploy**
   - Vercel will auto-detect build command (`vite build`) and output directory (`dist`)
   - Click "Deploy"
   - Watch for successful build and deployment

5. **Verify Deployment**
   - Once deployed, get your Vercel domain (e.g., `https://deep-learning-project-xxxx.vercel.app`)
   - Visit the URL - should load the 3D dashboard
   - Check browser console for WebSocket connection logs
   - Should see: "WebSocket connected" and alert processing

### Phase 4: Post-Deployment Validation
1. **Test End-to-End Flow**
   - Wait for an accident detection (simulated frames will trigger randomly)
   - Verify alert appears in frontend panel
   - Check alert contains: timestamp, confidence %, camera ID, latency ms
   - Verify camera node pulses red in 3D scene
   - Check that latency is displayed and reasonable (<2000ms)

2. **Verify Backend Logging**
   - Check Railway logs for:
     - Detection records saved to MongoDB
     - Latency reports received from frontend
     - WebSocket connections/disconnections

3. **Test History Panel** (if implemented)
   - Should show previous detections from MongoDB
   - Should update when new detections occur

4. **Test Metrics Panel** (if implemented)
   - Should show Baseline CNN vs CNN-LSTM metrics
   - Should match values from accident_model_output/*.json files

### Phase 5: Optional Features
1. **Discord Webhook** (if desired)
   - Create Discord webhook in desired channel
   - Add `DISCORD_WEBHOOK_URL` to Railway environment variables
   - Redeploy backend
   - Verify alerts appear in Discord when accidents detected

2. **Custom Domains** (optional for professional presentation)
   - Vercel: Add custom domain in Settings → Domains
   - Railway: Add custom domain in Settings → Domains
   - Update frontend VITE_BACKEND_URL if backend domain changes

## Troubleshooting
### Common Issues:
1. **Backend can't connect to MongoDB**
   - Check MONGO_URI spelling and credentials
   - Verify IP Access List allows Railway's IP addresses
   - Check Database Access user has correct permissions
   - Test connection string locally with `mongodb://` URI in mongo shell

2. **Frontend can't connect to backend WebSocket**
   - Verify VITE_BACKEND_URL is correct (no trailing slash)
   - Check backend Railway logs for WebSocket connection attempts
   - Ensure backend is running and healthy endpoint responds
   - Check for CORS issues (backend should allow frontend origin)

3. **Model fails to load**
   - Verify MODEL_PATH is correct relative to backend/ directory
   - Check that model.onnx exists in repository root
   - Check Railway build logs for file presence

4. **High latency or timeouts**
   - Check instance sizes (both Railway and Vercel free tiers have limits)
   - Monitor logs for performance bottlenecks
   - Consider reducing model complexity if needed (though current should work)

## Maintenance
- **Logs**: Check Railway and Vercel logs regularly for errors
- **Updates**: Push to main branch to trigger redeploys on both platforms
- **Secrets**: Rotate MongoDB passwords periodically if needed
- **Usage**: Monitor free tier usage to avoid unexpected suspension

## Notes for Portfolio
- This deployment demonstrates full-stack development skills
- Shows ability to work with databases, WebSockets, deployment platforms
- Highlights performance optimization (latency measurement)
- Demonstrates full lifecycle: local dev → testing → deployment → validation
- Clean separation of concerns: backend (ML/API), frontend (UI/UX), database (storage)

## References
- Railway Docs: https://docs.railway.app/
- Vercel Docs: https://vercel.com/docs
- MongoDB Atlas: https://www.mongodb.com/docs/atlas/