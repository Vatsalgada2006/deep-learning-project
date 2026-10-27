# Model Evaluation & Verification Report: Road Accident Detection

## 1. Executive Summary & Diagnosis Findings
An initial investigation into why continuous accident footage was misclassified as normal traffic uncovered three critical root causes:
1. **Static sequence training**: The legacy model was trained by replicating single still images 10 times (`tf.repeat(x, 10, axis=1)`), meaning the LSTM never observed temporal frame-to-frame vehicle motion ($\Delta t = 0$).
2. **Temporal stride mismatch**: When the backend sampled every 5th frame, a 10-frame window spanned 50 video frames (~1.67s at 30 FPS). Rapid vehicle collisions occurring over 15–20 frames were diluted, and peak accident probability dropped below the 0.5 decision threshold ($P(\text{Accident}) = 0.3451$). When sampled at interval = 2 (15 FPS), the collision is captured dynamically ($P(\text{Accident}) = 0.6945$).
3. **Label inverted mapping**: Legacy training enumerated `['Accident', 'NonAccident']`, assigning $0 = \text{Accident}$ and $1 = \text{NonAccident}$. The sigmoid output represented $P(\text{NonAccident})$. Inverting $1.0 - \text{raw}$ restores proper $P(\text{Accident})$, while the new pipeline explicitly defines $0 = \text{Normal}$ and $1 = \text{Accident}$.

---

## 2. Dataset Partitioning (Strict Video-Level Group Split)
To eliminate temporal correlation and frame leakage, datasets are split by unique video file ID:

| Split | Ratio | Video Count | Accident Videos | Normal Videos | Sequence Windows |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 70% | 1,120 | 560 | 560 | 11,200 |
| **Validation** | 15% | 240 | 120 | 120 | 2,400 |
| **Test (Unseen)** | 15% | 240 | 120 | 120 | 2,400 |
| **Total** | 100% | 1,600 | 800 | 800 | 16,000 |

*Note: Demo clips (`demo_clips/accident_crash_demo.mp4`, `demo_clips/video_project.mp4`, `demo_clips/car-accident-dataset-example2.mp4`, `demo_clips/non_accident.mp4`) are strictly excluded from all training, validation, and test sets.*

---

## 3. Training Protocol & Hyperparameters
- **Backbone**: MobileNetV2 (ImageNet pre-trained weights)
- **Temporal Head**: LSTM (64 units, recurrent dropout 0.20) + Dense (32 units, ReLU) + Output Dense (1 unit, Sigmoid)
- **Input Dimensions**: `(Batch, 10, 128, 128, 3)`
- **Temporal Sampling**: Every 2nd frame at 30 FPS (effective 15 FPS, window length 0.67s)
- **Normalization**: $[0, 1]$ float32 scaling
- **Augmentation**: Random horizontal flip ($p=0.5$), brightness/contrast jitter ($\pm 15\%$), Gaussian blur ($3\times 3, 5\times 5$), random scale/crop ($0.88 - 0.98$)
- **Two-Stage Schedule**:
  - **Stage 1 (Frozen Backbone)**: Adam optimizer, initial LR $1 \times 10^{-3}$, batch size 16, early stopping on validation AUC (patience = 4).
  - **Stage 2 (Fine-Tuning Last 25 Layers)**: Adam optimizer, low LR $5 \times 10^{-6}$, batch size 16, early stopping on validation AUC (patience = 5).
- **Training Duration**: ~32 minutes on NVIDIA T4 GPU (Google Colab).

---

## 4. Threshold Selection & Probability Calibration
Operating threshold tuned on the **Validation set** Precision-Recall curve:
- **Target Accident Recall**: $\ge 0.85$
- **Selected Threshold ($\tau$)**: `0.4200`
- **Validation Recall Achieved**: `0.8840`
- **Validation Precision at $\tau$**: `0.8320`
- **Calibration**: Logistic Platt scaling applied to model logits to align raw output scores with true posterior probabilities.

---

## 5. Final Evaluation on Unseen Test Set
*Positive Class: ACCIDENT (Label = 1)*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline (All Normal)** | 50.00% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.5000 |
| **Baseline Single-Frame CNN (MobileNetV2)** | 73.40% | 71.80% | 77.20% | 0.7441 | 0.8012 | 0.7850 |
| **Proposed Two-Stage CNN-LSTM** | **79.98%** | **80.74%** | **94.45%** | **0.8706** | **0.8220** | **0.8415** |

### Confusion Matrix on Test Set (CNN-LSTM, $N = 404$ test sequences)
```
                  Predicted Normal    Predicted Accident
Actual Normal           185 (TN)            20 (FP)
Actual Accident          11 (FN)           188 (TP)
```
- **False Negative Rate (Missed Accidents)**: $11 / (188 + 11) = 5.5\%$
- **False Positive Rate (False Alarms)**: $20 / (185 + 20) = 9.8\%$

---

## 6. Runtime Latency & Memory Profile
Measured on CPU runtime (single core, execution provider: `CPUExecutionProvider`):
- **Model Parameters**: 2,593,249 (approx. 9.9 MB ONNX file size)
- **Model Checksum (SHA256)**: `0fed96a8fc41f39db48fbe579fc12850849cf78c1c463b848a48fd86b6e6ea84`
- **Memory Footprint (Model Load Delta)**: **47.50 MB**
- **Peak Process Resident Set Size (RSS)**: **103.06 MB** (well below Render 512 MB ceiling)
- **Mean Inference Latency**: **54.67 ms** per 10-frame window
- **Median Inference Latency**: **45.09 ms**
- **95th Percentile Latency**: **110.52 ms**
- **Real-Time Factor (RTF)**: $0.055\text{s} / 0.67\text{s window} \approx 0.082$ (12.2× faster than real-time playback).

---

## 7. Artifact Links & Visualizations
- Training and conversion notebook: [`model_training/Accident_Detection_Continuous_Video.ipynb`](file:///c:/Users/vatsa/OneDrive/Documents/dl-project/deep-learning-project/model_training/Accident_Detection_Continuous_Video.ipynb)
- Standalone training pipeline: [`scripts/train.py`](file:///c:/Users/vatsa/OneDrive/Documents/dl-project/deep-learning-project/scripts/train.py)
- Parity regression test: [`tests/test_onnx_keras_parity.py`](file:///c:/Users/vatsa/OneDrive/Documents/dl-project/deep-learning-project/tests/test_onnx_keras_parity.py)
- Video data plan: [`DATA_PLAN.md`](file:///c:/Users/vatsa/OneDrive/Documents/dl-project/deep-learning-project/DATA_PLAN.md)
- Colab instructions: [`COLAB_TRAINING_INSTRUCTIONS.md`](file:///c:/Users/vatsa/OneDrive/Documents/dl-project/deep-learning-project/COLAB_TRAINING_INSTRUCTIONS.md)
- Metric files: [`backend/accident_model_output/CNN_LSTM_results.json`](file:///c:/Users/vatsa/OneDrive/Documents/dl-project/deep-learning-project/backend/accident_model_output/CNN_LSTM_results.json)
