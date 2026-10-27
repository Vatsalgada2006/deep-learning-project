# Step-by-Step Colab & Kaggle Training Guide for Continuous Video Accident Detection

This guide walks through training the continuous video accident detection model (Two-Stage MobileNetV2 + LSTM) on Google Colab or Kaggle GPU, and synchronizing the trained `model.onnx` and evaluation metrics back into the repository.

---

## 1. Preparing the Video Dataset
To ensure the model works on real continuous video without artificial static sequences:
1. **Download public video data**:
   - **UCF-Crime (`RoadAccidents` subset)**: Download via Kaggle CLI:
     ```bash
     kaggle datasets download -d odinsam/ucf-crime-dataset
     unzip ucf-crime-dataset.zip
     ```
   - **DoTA (Detection of Traffic Anomaly)**: Download accident clips (`Collision`, `Out_of_control`) and normal background clips from the official repository (`https://github.com/yujiang/DoTA_dataset`).
2. **Organize into folder structure**:
   ```
   dataset/
     Accident/
       video_001.mp4
       video_002.mp4
       ...
     Normal/
       normal_001.mp4
       normal_002.mp4
       ...
   ```
3. **DO NOT include demo clips**:
   Ensure `demo_clips/` (`accident_crash_demo.mp4`, `video_project.mp4`, `car-accident-dataset-example2.mp4`, `non_accident.mp4`) are strictly excluded from training, validation, and testing.

---

## 2. Running Training on Google Colab
1. Open [Google Colab](https://colab.research.google.com).
2. Click **File → Upload notebook** and select `model_training/Accident_Detection_Continuous_Video.ipynb`.
3. Switch runtime to GPU: **Runtime → Change runtime type → T4 GPU (or V100/A100)**.
4. Mount your Google Drive or upload your video dataset archive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Run the training cells:
   - **Video Group Split**: Partitions clips by video ID into Train (70%), Validation (15%), and Unseen Test (15%).
   - **Two-Stage Training**:
     - Stage 1: Frozen MobileNetV2 backbone (lr = 1e-3, early stopping on val AUC).
     - Stage 2: Fine-tune last 25 MobileNetV2 layers (lr = 5e-6, Adam).
   - **Validation Threshold Selection**: Chooses optimal threshold $\tau$ targeting accident recall $\ge 0.85$.
   - **ONNX Export**: Converts Keras model to `model.onnx` and verifies numerical parity within tolerance ($10^{-4}$).

---

## 3. Bringing Files Back Into the Repository
After the training script finishes, it generates:
- `model.onnx` (the exported ONNX model)
- `labels.json` (`{"0": "Normal", "1": "Accident"}`)
- `CNN_LSTM_results.json` and `Baseline_CNN_results.json`
- Plots: `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `training_curves.png`

### Sync Steps:
1. Download `model.onnx` and copy it to:
   - `backend/model.onnx`
   - `public/model.onnx`
2. Copy `CNN_LSTM_results.json`, `Baseline_CNN_results.json`, and `labels.json` to:
   - `backend/accident_model_output/`
3. Copy the evaluation plot PNGs to:
   - `public/plots/` (or `backend/accident_model_output/`)
4. Verify checksum and parity locally:
   ```bash
   python -m pytest tests/test_onnx_keras_parity.py -v
   python -m pytest tests/test_api.py -v
   ```
5. Commit and push:
   ```bash
   git add backend/model.onnx public/model.onnx backend/accident_model_output/
   git commit -m "Update accident detection model weights and continuous video metrics"
   git push origin main
   ```
Render and Vercel will automatically build and deploy the verified model!