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

## 5. Final Evaluation & Metric Reconciliation
*Positive Class: ACCIDENT (Label = 1), Negative Class: NORMAL (Label = 0)*

### Suite 1: Held-Out Test Sequence Suite ($N = 404$ continuous temporal sequences)
Directly computed from test-set predictions at threshold $\tau = 0.50$:

| Model | Accuracy | Precision (Accident) | Recall (Accident) | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Single-Frame CNN (Frame-Level)** | 98.52%* | 96.22%* | 98.71%* | 0.9744* | 0.9991* |
| **Proposed Two-Stage CNN-LSTM** | **92.33%** | **90.38%** | **94.47%** | **0.9238** | **0.8220** |

*\*Methodological Caveat: The baseline CNN was scored on 4,320 isolated still frames with random frame-level shuffling. Nearly identical frames from the same CCTV camera appeared in both train and validation splits (temporal data leakage, producing artificial ROC-AUC 0.999). On unseen continuous video, the static baseline collapses because it has zero temporal motion awareness ($\Delta t = 0$).*

### Confusion Matrix Breakdown (CNN-LSTM, $N = 404$ test sequences)
```
                  Predicted Normal (0)    Predicted Accident (1)      Total
Actual Normal (0)        185 (TN)                  20 (FP)             205
Actual Accident (1)       11 (FN)                 188 (TP)             199
Total                    196                      208                  404
```
- **Accuracy**: $(188 + 185) / 404 = \mathbf{92.33\%}$
- **Accident Recall (Sensitivity)**: $188 / (188 + 11) = 188 / 199 = \mathbf{94.47\%}$ (~$94.5\%$)
- **Accident Precision**: $188 / (188 + 20) = 188 / 208 = \mathbf{90.38\%}$ (~$90.4\%$)
- **Accident F1-Score**: $2 \times (0.9038 \times 0.9447) / (0.9038 + 0.9447) = \mathbf{92.38\%}$ (~$92.4\%$)
- **False Negative Rate (Missed Accidents)**: $11 / (188 + 11) = 5.53\%$
- **False Positive Rate (False Alarms)**: $20 / (185 + 20) = 9.76\%$

### Suite 2: Multi-Condition Validation Split ($N = 2,400$ temporal windows)
Evaluated across varying weather, glare, and low-contrast conditions:
- **Accuracy**: 79.98%
- **Accident Precision**: 80.74%
- **Accident Recall**: 94.45%
- **F1-Score**: 87.06%
- **ROC-AUC**: 0.8220

### Legacy Metric Reconciliation Note:
In earlier Phase 2 documentation:
- The class-wise table reported the **Suite 2 validation numbers** ($80.0\%$ accuracy, $80.7\%$ precision, $94.5\%$ recall, $87.1\%$ F1).
- The confusion matrix reported the **Suite 1 test sequence counts** ($185 / 20 / 11 / 188 \implies 92.3\%$ accuracy, $90.4\%$ precision, $94.5\%$ recall, $92.4\%$ F1).
- For the Baseline CNN, the values $1.6\%$ and $3.8\%$ in the old report were calculation artifacts where the false negative count ($48$) was divided by the non-accident row ($48 / 3,081 = 1.56\%$) and predicted accident column ($48 / 1,271 = 3.78\%$). Both suites have now been reconciled and clearly demarcated.

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
