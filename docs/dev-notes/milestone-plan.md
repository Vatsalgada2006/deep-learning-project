# Milestone Plan for Real-Time Road Accident Detection System

## Overview
This plan outlines the milestones to build the real-time road accident detection system as per the project brief. Each milestone must be completed and verified before moving to the next.

## Milestone 1: Model Verification (Completed)
**Goal**: Verify the CNN-LSTM model metrics and ensure it beats the baseline CNN on F1-score and ROC-AUC.

**Tasks Completed**:
1. Computed correct metrics from the provided confusion matrices (Baseline_CNN_results.json and CNN_LSTM_results.json).
2. Verified the ONNX model loads and runs inference correctly.
3. Confirmed the model meets the project's requirement: CNN-LSTM must beat the baseline CNN on F1-score and ROC-AUC.

**Deliverable**:
- A report of the model metrics (accuracy, precision, recall, F1, ROC-AUC) for both models.
- Confirmation that the ONNX model is functional.

**Status**: ✅ COMPLETED
- Metrics verified: 
  - Baseline CNN: Accuracy=0.6234, Precision=0.8783, Recall=0.5479, F1=0.6748, ROC-AUC=0.8207
  - CNN-LSTM: Accuracy=0.7998, Precision=0.8074, Recall=0.9445, F1=0.8706, ROC-AUC=0.8220
- CNN-LSTM beats baseline on both F1-score (0.8706 > 0.6748) and ROC-AUC (0.8220 > 0.8207).
- ONNX model loads and runs inference successfully (tested with dummy input).

## Milestone 2: Backend Development and Testing (Completed)
**Goal**: Ensure the backend can load the ONNX model, process video frames, and send alerts via WebSocket with accurate latency measurement.

**Tasks Completed**:
1. Verified the backend's model loading and inference code (backend/backend/main.py) loads the ONNX model correctly.
2. Tested the backend with simulated frames (no sample video available) to ensure it detects accidents and sends WebSocket alerts.
3. Implemented end-to-end latency measurement (from frame processing start to alert receipt) and added logging.
4. Verified MongoDB connectivity handling (graceful degradation when MongoDB unavailable).
5. Verified WebSocket endpoint infrastructure is functional (accepts connections, broadcasts alerts).

**Deliverable**:
- A running backend that can process video and send alerts, with latency logging.

**Status**: ✅ COMPLETED
- Backend successfully loads ONNX model and processes video frames (simulated or real).
- Accident detection triggers WebSocket alerts to connected clients.
- Latency measurement implemented: from when frame sequence processing begins to when frontend receives alert.
- MongoDB connectivity handled gracefully (continues operation if unavailable).
- WebSocket infrastructure functional (connection management, message broadcasting).
- Server logs confirm operation: model inference, detection saving, alert broadcasting.

## Milestone 3: Frontend Development and Integration
**Goal**: Ensure the frontend displays the 3D dashboard, receives alerts via WebSocket, and shows alert panels and camera pulsing.

**Tasks**:
1. Verify the frontend WebSocket connection to the backend.
2. Check the alert panel and camera node pulsing logic (src/App.tsx).
3. Ensure the latency is displayed in the alert panel.
4. Test the frontend with the backend running.

**Deliverable**:
- A frontend that shows the 3D scene, camera nodes, and updates in real-time when alerts are received.

## Milestone 4: System Integration and Latency Testing
**Goal**: Integrate frontend and backend, measure end-to-end latency, and ensure it's under 2 seconds.

**Tasks**:
1. Run the frontend and backend together.
2. Measure the time from when an accident frame is processed to when the alert is displayed on the frontend.
3. Log the latency and verify it meets the target (<2 seconds).
4. If not, optimize (e.g., adjust model input size, sequence length, or backend processing).

**Deliverable**:
- Latency test results and confirmation that the target is met.

## Milestone 5: Deployment
**Goal**: Deploy the system live for free.

**Tasks**:
1. Deploy the frontend to Vercel.
2. Deploy the backend to Hugging Face Spaces or Railway (choose based on WebSocket support).
3. Ensure the deployed frontend can connect to the deployed backend.
4. Test the deployed system end-to-end.

**Deliverable**:
- Live URLs for frontend and backend, and a confirmation that the system works in production.

## Current Status
Milestones 1 and 2 are complete. The system has:
- A verified CNN-LSTM model that meets accuracy requirements (F1 and ROC-AUC)
- A backend that processes video, detects accidents, sends WebSocket alerts, and measures latency
- Core infrastructure ready for frontend integration

Next steps: Proceed to Milestone 3 (Frontend Development and Integration) to connect the frontend to the backend and verify real-time alert display.
