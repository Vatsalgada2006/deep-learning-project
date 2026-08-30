import json
import os

def compute_metrics_from_cm(cm):
    tn = cm['tn']
    fp = cm['fp']
    fn = cm['fn']
    tp = cm['tp']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    # Note: ROC-AUC cannot be computed from confusion matrix alone, we'll get it from the JSON
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }

# Baseline CNN
baseline_path = r'accident_model_output\Baseline_CNN_results.json'
with open(baseline_path, 'r') as f:
    baseline_data = json.load(f)

baseline_metrics = compute_metrics_from_cm(baseline_data['confusion_matrix'])
baseline_metrics['roc_auc'] = baseline_data['roc_auc']

# CNN-LSTM
lstm_path = r'accident_model_output\CNN_LSTM_results.json'
with open(lstm_path, 'r') as f:
    lstm_data = json.load(f)

lstm_metrics = compute_metrics_from_cm(lstm_data['confusion_matrix'])
lstm_metrics['roc_auc'] = lstm_data['roc_auc']

print("=== Model Metrics Comparison ===")
print(f"{'Metric':<15} {'Baseline CNN':<15} {'CNN-LSTM':<15}")
print("-" * 45)
print(f"{'Accuracy':<15} {baseline_metrics['accuracy']:<15.4f} {lstm_metrics['accuracy']:<15.4f}")
print(f"{'Precision':<15} {baseline_metrics['precision']:<15.4f} {lstm_metrics['precision']:<15.4f}")
print(f"{'Recall':<15} {baseline_metrics['recall']:<15.4f} {lstm_metrics['recall']:<15.4f}")
print(f"{'F1-Score':<15} {baseline_metrics['f1_score']:<15.4f} {lstm_metrics['f1_score']:<15.4f}")
print(f"{'ROC-AUC':<15} {baseline_metrics['roc_auc']:<15.4f} {lstm_metrics['roc_auc']:<15.4f}")

# Check if CNN-LSTM beats baseline on F1 and ROC-AUC
if lstm_metrics['f1_score'] > baseline_metrics['f1_score'] and lstm_metrics['roc_auc'] > baseline_metrics['roc_auc']:
    print("\n[PASS] CNN-LSTM beats baseline CNN on both F1-score and ROC-AUC")
else:
    print("\n[FAIL] CNN-LSTM does not beat baseline CNN on both F1-score and ROC-AUC")
    if lstm_metrics['f1_score'] <= baseline_metrics['f1_score']:
        print(f"  F1-score: CNN-LSTM ({lstm_metrics['f1_score']:.4f}) <= Baseline ({baseline_metrics['f1_score']:.4f})")
    if lstm_metrics['roc_auc'] <= baseline_metrics['roc_auc']:
        print(f"  ROC-AUC: CNN-LSTM ({lstm_metrics['roc_auc']:.4f}) <= Baseline ({baseline_metrics['roc_auc']:.4f})")