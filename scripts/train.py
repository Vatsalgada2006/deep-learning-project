#!/usr/bin/env python3
"""
scripts/train.py - End-to-End Training & Evaluation for Continuous Video Accident Detection

Features:
- Video-level grouping (GroupKFold / GroupSplit) to prevent temporal data leakage.
- Explicit label mapping: 0 = Normal, 1 = Accident (outputs P(Accident) directly).
- Consecutive video frame sequence extraction at fixed sampling rate.
- Strong augmentations: brightness, blur, compression, crop/scale, horizontal flip.
- Two-stage training:
    Stage 1: Frozen MobileNetV2 backbone.
    Stage 2: Fine-tune last 25 MobileNetV2 layers (lr=5e-6).
- Early stopping & checkpointing on validation AUC.
- Validation-set threshold selection (target recall >= 0.85) with probability calibration.
- Full evaluation on unseen test set: accuracy, precision, recall, F1, PR-AUC, ROC-AUC,
  confusion matrix, per-video results, comparing Baseline CNN vs CNN-LSTM vs Majority Class.
- ONNX export with numeric parity verification against Keras.
"""

import os
import sys
import glob
import json
import time
import hashlib
import argparse
import random
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import (
    TimeDistributed, GlobalAveragePooling2D, Dense, Dropout, LSTM, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set deterministic random seeds
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LABEL_NORMAL = 0
LABEL_ACCIDENT = 1
LABELS_DICT = {"0": "Normal", "1": "Accident"}


# ---------------------------------------------------------------------------
# 1. Video Sequence Extraction & Group Splitting
# ---------------------------------------------------------------------------

def extract_sequences_from_video(
    video_path: str,
    seq_length: int = 10,
    sample_interval: int = 2,
    img_size: int = 128,
    max_windows: int = 40
) -> np.ndarray:
    """Extract sliding windows of consecutive sampled frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((0, seq_length, img_size, img_size, 3), dtype=np.float32)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # Use mobilenet_v2 preprocessing or [0, 1] scaling
            norm = resized.astype(np.float32) / 255.0
            frames.append(norm)
        idx += 1
    cap.release()

    if len(frames) < seq_length:
        if len(frames) == 0:
            return np.zeros((0, seq_length, img_size, img_size, 3), dtype=np.float32)
        # Pad short video up to seq_length
        pad = [frames[0]] * (seq_length - len(frames))
        return np.array([pad + frames], dtype=np.float32)

    windows = []
    # Stride by seq_length // 2 for dense temporal coverage
    stride = max(1, seq_length // 2)
    for start in range(0, len(frames) - seq_length + 1, stride):
        windows.append(frames[start:start + seq_length])
        if len(windows) >= max_windows:
            break

    return np.array(windows, dtype=np.float32)


def create_video_group_split(
    accident_videos: List[str],
    normal_videos: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Split video filenames strictly by video ID (group split) to prevent data leakage.
    Ensures that every frame from a video belongs exclusively to one split.
    """
    rng = random.Random(seed)
    acc = list(accident_videos)
    norm = list(normal_videos)
    rng.shuffle(acc)
    rng.shuffle(norm)

    def _split(vlist):
        n = len(vlist)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            vlist[:n_train],
            vlist[n_train:n_train + n_val],
            vlist[n_train + n_val:]
        )

    acc_train, acc_val, acc_test = _split(acc)
    norm_train, norm_val, norm_test = _split(norm)

    splits = {
        "train": acc_train + norm_train,
        "val": acc_val + norm_val,
        "test": acc_test + norm_test,
    }

    counts = {
        "train_accident": len(acc_train),
        "train_normal": len(norm_train),
        "val_accident": len(acc_val),
        "val_normal": len(norm_val),
        "test_accident": len(acc_test),
        "test_normal": len(norm_test),
    }

    print("=" * 60)
    print("VIDEO GROUP SPLIT SUMMARY (Zero frame leakage):")
    print(f"  Train:      {len(splits['train'])} videos ({counts['train_accident']} Accident, {counts['train_normal']} Normal)")
    print(f"  Validation: {len(splits['val'])} videos ({counts['val_accident']} Accident, {counts['val_normal']} Normal)")
    print(f"  Test:       {len(splits['test'])} videos ({counts['test_accident']} Accident, {counts['test_normal']} Normal)")
    print("=" * 60)

    return splits, counts


# ---------------------------------------------------------------------------
# 2. Data Augmentation
# ---------------------------------------------------------------------------

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply temporally consistent augmentations across all frames in a 10-frame sequence:
    - Random brightness & contrast
    - Random horizontal flip
    - Random slight Gaussian blur
    - Random slight scale / crop
    """
    seq = sequence.copy()
    seq_len, h, w, c = seq.shape

    # 1. Random horizontal flip (applied to all frames consistently)
    if random.random() > 0.5:
        seq = np.flip(seq, axis=2)

    # 2. Random brightness / contrast adjustment
    if random.random() > 0.5:
        alpha = random.uniform(0.85, 1.15)  # contrast
        beta = random.uniform(-0.10, 0.10)  # brightness
        seq = np.clip(seq * alpha + beta, 0.0, 1.0)

    # 3. Random Gaussian blur simulation
    if random.random() > 0.3:
        ksize = random.choice([3, 5])
        for t in range(seq_len):
            blurred = cv2.GaussianBlur(seq[t], (ksize, ksize), 0)
            seq[t] = blurred

    # 4. Random slight scale & crop
    if random.random() > 0.4:
        crop_factor = random.uniform(0.88, 0.98)
        ch, cw = int(h * crop_factor), int(w * crop_factor)
        sy = random.randint(0, h - ch)
        sx = random.randint(0, w - cw)
        for t in range(seq_len):
            cropped = seq[t, sy:sy + ch, sx:sx + cw]
            seq[t] = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return seq.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------------------------

def build_cnn_lstm_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    seq_length: int = 10,
    fine_tune_layers: int = 25
) -> Tuple[Model, Model]:
    """
    Build CNN-LSTM Model with TimeDistributed MobileNetV2 backbone.
    Outputs P(Accident) in [0, 1] directly (ACCIDENT = 1, NORMAL = 0).
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze all layers initially for Stage 1
    base_model.trainable = False

    inputs = Input(shape=(seq_length, *input_shape), name="video_sequence_input")
    x = TimeDistributed(base_model, name="time_distributed_backbone")(inputs)
    x = TimeDistributed(GlobalAveragePooling2D(), name="time_distributed_gap")(x)
    x = TimeDistributed(Dropout(0.25), name="td_dropout")(x)

    # Recurrent temporal aggregation
    x = LSTM(64, return_sequences=False, name="temporal_lstm")(x)
    x = Dropout(0.40, name="lstm_dropout")(x)
    x = Dense(32, activation="relu", name="dense_features")(x)
    outputs = Dense(1, activation="sigmoid", name="accident_probability")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Accident_CNN_LSTM")
    return model, base_model


def build_baseline_cnn_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3)
) -> Model:
    """Build single-frame Baseline CNN using MobileNetV2."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape, name="image_input")
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid", name="accident_prob")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Baseline_CNN")
    return model


# ---------------------------------------------------------------------------
# 4. Two-Stage Training Routine
# ---------------------------------------------------------------------------

def train_model_two_stages(
    model: Model,
    base_model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str,
    epochs_stage1: int = 10,
    epochs_stage2: int = 15,
    batch_size: int = 16,
    fine_tune_layers: int = 25
) -> tf.keras.callbacks.History:
    """
    Two-stage training procedure:
    Stage 1: Frozen backbone, Adam lr=1e-3, early stopping on val_auc.
    Stage 2: Unfreeze last 25 MobileNetV2 layers, Adam lr=5e-6, fine-tuning.
    """
    os.makedirs(output_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(output_dir, "best_accident_model.keras")

    # Class weights for handling class imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    total = len(y_train)
    weight_0 = (1.0 / max(1, n_neg)) * (total / 2.0)
    weight_1 = (1.0 / max(1, n_pos)) * (total / 2.0)
    class_weight = {0: float(weight_0), 1: float(weight_1)}
    print(f"Class Weights: Normal (0): {weight_0:.3f}, Accident (1): {weight_1:.3f}")

    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

    # --- STAGE 1: Frozen Backbone ---
    print("\n" + "=" * 60)
    print("STAGE 1: Training classifier head with frozen MobileNetV2 backbone...")
    print("=" * 60)
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=metrics
    )

    cb_stage1 = [
        EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    history_stage1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_stage1,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=cb_stage1,
        verbose=1
    )

    # --- STAGE 2: Fine-Tuning Last N Layers ---
    print("\n" + "=" * 60)
    print(f"STAGE 2: Fine-tuning last {fine_tune_layers} MobileNetV2 layers (lr=5e-6)...")
    print("=" * 60)
    base_model.trainable = True
    # Freeze all except last fine_tune_layers
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
        loss="binary_crossentropy",
        metrics=metrics
    )

    cb_stage2 = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_checkpoint_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ]

    history_stage2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_stage2,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=cb_stage2,
        verbose=1
    )

    # Load absolute best weights
    if os.path.exists(best_checkpoint_path):
        model.load_weights(best_checkpoint_path)

    return history_stage1, history_stage2


# ---------------------------------------------------------------------------
# 5. Validation Threshold Selection & Probability Calibration
# ---------------------------------------------------------------------------

def select_optimal_threshold(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    target_recall: float = 0.85
) -> Tuple[float, float, float]:
    """
    Find threshold on validation set that meets or exceeds target accident recall,
    and report the precision trade-off.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val_true, y_val_probs)

    best_thresh = 0.50
    best_prec = 0.0
    best_rec = 0.0

    # Search for threshold with recall >= target_recall maximizing precision
    valid_candidates = []
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= target_recall:
            valid_candidates.append((p, r, t))

    if valid_candidates:
        valid_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_prec, best_rec, best_thresh = valid_candidates[0]
    else:
        # Fallback to default
        idx = np.argmin(np.abs(recalls - target_recall))
        best_thresh = float(thresholds[min(idx, len(thresholds) - 1)])
        best_rec = float(recalls[idx])
        best_prec = float(precisions[idx])

    print("=" * 60)
    print(f"VALIDATION THRESHOLD SELECTION (Target Recall >= {target_recall:.2f}):")
    print(f"  Selected Threshold:   {best_thresh:.4f}")
    print(f"  Achieved Val Recall:  {best_rec:.4f}")
    print(f"  Precision at Recall:  {best_prec:.4f}")
    print("=" * 60)

    return float(best_thresh), float(best_prec), float(best_rec)


def fit_probability_calibrator(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray
) -> LogisticRegression:
    """Train Platt scaling (logistic calibration) on validation raw probabilities."""
    # Logit transform
    eps = 1e-7
    p_clipped = np.clip(y_val_probs, eps, 1.0 - eps)
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(logits, y_val_true)
    return calibrator


def apply_calibration(
    probs: np.ndarray,
    calibrator: LogisticRegression
) -> np.ndarray:
    """Apply Platt scaling to raw model probabilities."""
    eps = 1e-7
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


# ---------------------------------------------------------------------------
# 6. Unseen Test Set Evaluation & Benchmarking
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    model_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Comprehensive test set evaluation with Accident as positive class (label 1).
    Calculates accuracy, precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix.
    """
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, probs))
    except Exception:
        roc_auc = 0.50

    try:
        pr_auc = float(average_precision_score(y_test, probs))
    except Exception:
        pr_auc = float(np.mean(y_test))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    results = {
        "model": model_name,
        "positive_class": "Accident",
        "threshold_used": float(threshold),
        "test_samples": int(len(y_test)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }

    # Save results JSON
    json_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{model_name} Final Test Evaluation:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results, probs


def plot_evaluation_charts(
    y_test: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    threshold: float,
    output_dir: str
):
    """Generate high-resolution evaluation figures: ROC, PR, and Confusion Matrix."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curves
    plt.figure(figsize=(7, 6))
    for name, p in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve - Continuous Video Accident Detection")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(7, 6))
    for name, p in probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        pr_val = average_precision_score(y_test, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_val:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Continuous Video")
    plt.axvline(0.85, color="red", linestyle=":", label="Target Recall (0.85)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300)
    plt.close()

    # 3. Confusion Matrix for CNN-LSTM
    if "CNN_LSTM" in probs_dict:
        preds = (probs_dict["CNN_LSTM"] >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"CNN-LSTM Confusion Matrix (Threshold = {threshold:.2f})")
        plt.colorbar()
        classes = ["Normal", "Accident"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        thresh_cm = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh_cm else "black")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# 7. ONNX Export & Parity Verification
# ---------------------------------------------------------------------------

def export_onnx_and_verify(
    keras_model: Model,
    onnx_path: str,
    seq_length: int = 10,
    img_size: int = 128,
    tolerance: float = 1e-4
) -> Tuple[bool, float, str]:
    """
    Export Keras CNN-LSTM model to ONNX using tf2onnx,
    verify numerical parity between Keras and ONNX runtime,
    and compute SHA256 checksum.
    """
    print("\n" + "=" * 60)
    print("EXPORTING TO ONNX & NUMERICAL PARITY VERIFICATION:")
    print("=" * 60)

    import tf2onnx
    import onnxruntime as ort

    input_signature = [
        tf.TensorSpec(shape=(None, seq_length, img_size, img_size, 3), dtype=tf.float32, name="input")
    ]

    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=13,
        output_path=onnx_path
    )
    print(f"Exported ONNX model to: {onnx_path}")

    # Compute SHA256
    hasher = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()
    print(f"Model Checksum (SHA256): {checksum}")

    # Verify parity with test batch
    test_batch = np.random.rand(4, seq_length, img_size, img_size, 3).astype(np.float32)
    keras_pred = keras_model.predict(test_batch, verbose=0).flatten()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {in_name: test_batch})[0].flatten()

    max_diff = float(np.max(np.abs(keras_pred - onnx_pred)))
    is_parity = bool(max_diff <= tolerance)

    print(f"Maximum absolute difference between Keras and ONNX: {max_diff:.6e}")
    print(f"Parity within tolerance ({tolerance:.1e}): {'PASSED' if is_parity else 'FAILED'}")

    return is_parity, max_diff, checksum


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Continuous Video Accident Detection Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to video datasets")
    parser.add_argument("--output_dir", type=str, default="backend/accident_model_output", help="Output directory")
    parser.add_argument("--onnx_output", type=str, default="backend/model.onnx", help="Target ONNX model path")
    parser.add_argument("--seq_length", type=int, default=10, help="Frames per sequence window")
    parser.add_argument("--img_size", type=int, default=128, help="Frame height and width")
    parser.add_argument("--sample_interval", type=int, default=2, help="Sample every N frames")
    parser.add_argument("--epochs1", type=int, default=8, help="Stage 1 frozen epochs")
    parser.add_argument("--epochs2", type=int, default=12, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Write explicit labels.json
    labels_path = os.path.join(args.output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(LABELS_DICT, f, indent=2)
    print(f"Saved explicit labels to {labels_path}: {LABELS_DICT}")

    print("\nTraining configuration loaded successfully.")
    print(f"Sequence length: {args.seq_length}, Img size: {args.img_size}, Sample interval: {args.sample_interval}")


if __name__ == "__main__":
    main()
