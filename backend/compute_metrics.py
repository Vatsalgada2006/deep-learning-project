import json
import os

def compute_metrics_from_cm(cm):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy
    }

# Path to the accident_model_output directory
base_dir = os.path.join(os.path.dirname(__file__), "..", "accident_model_output")

# Baseline CNN
baseline_path = os.path.join(base_dir, "Baseline_CNN_results.json")
with open(baseline_path, "r") as f:
    baseline_results = json.load(f)

# CNN-LSTM
cnn_lstm_path = os.path.join(base_dir, "CNN_LSTM_results.json")
with open(cnn_lstm_path, "r") as f:
    cnn_lstm_results = json.load(f)

print("Baseline CNN:")
print("  Accuracy: ", baseline_results["accuracy"])
print("  Precision: ", baseline_results["precision"])
print("  Recall: ", baseline_results["recall"])
print("  F1-Score: ", baseline_results["f1_score"])
print("  ROC-AUC: ", baseline_results["roc_auc"])
print("  Confusion Matrix: ", baseline_results["confusion_matrix"])

print("\nCNN-LSTM:")
print("  Accuracy: ", cnn_lstm_results["accuracy"])
print("  Precision: ", cnn_lstm_results["precision"])
print("  Recall: ", cnn_lstm_results["recall"])
print("  F1-Score: ", cnn_lstm_results["f1_score"])
print("  ROC-AUC: ", cnn_lstm_results["roc_auc"])
print("  Confusion Matrix: ", cnn_lstm_results["confusion_matrix"])

# Compute metrics from confusion matrix (should match the accuracy and roc_auc, but precision, recall, f1 might be off in the JSON)
baseline_cm = baseline_results["confusion_matrix"]
cnn_lstm_cm = cnn_lstm_results["confusion_matrix"]

baseline_metrics = compute_metrics_from_cm(baseline_cm)
cnn_lstm_metrics = compute_metrics_from_cm(cnn_lstm_cm)

print("\nMetrics computed from confusion matrix:")
print("Baseline CNN:")
print("  Accuracy: ", baseline_metrics["accuracy"])
print("  Precision: ", baseline_metrics["precision"])
print("  Recall: ", baseline_metrics["recall"])
print("  F1-Score: ", baseline_metrics["f1_score"])

print("\nCNN-LSTM:")
print("  Accuracy: ", cnn_lstm_metrics["accuracy"])
print("  Precision: ", cnn_lstm_metrics["precision"])
print("  Recall: ", cnn_lstm_metrics["recall"])
print("  F1-Score: ", cnn_lstm_metrics["f1_score"])

# The ROC-AUC is not in the confusion matrix, so we take it from the JSON
print("\nROC-AUC from JSON:")
print("Baseline CNN: ", baseline_results["roc_auc"])
print("CNN-LSTM: ", cnn_lstm_results["roc_auc"])