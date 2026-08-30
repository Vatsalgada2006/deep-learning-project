# Addressing Accuracy Concern

The user has expressed concern about the model accuracy being low.

## Current Metrics
From the model evaluation:
- **Baseline CNN**: Accuracy = 62.34%, Precision = 87.83%, Recall = 54.79%, F1 = 0.6748, ROC-AUC = 0.8207
- **CNN-LSTM**: Accuracy = 79.98%, Precision = 80.74%, Recall = 94.45%, F1 = 0.8706, ROC-AUC = 0.8220

## Why Accuracy is Not the Primary Metric
The project brief explicitly states:
> "Model accuracy: CNN-LSTM must beat the baseline CNN on F1-score and ROC-AUC, not just raw accuracy (raw accuracy is misleading with class imbalance — 15,420 non-accident vs 6,191 accident images). Report all of: Accuracy, Precision, Recall, F1, ROC-AUC, and a confusion matrix, for BOTH models, side by side."

The dataset is significantly imbalanced (approximately 71% non-accident, 29% accident). In such cases:
- Accuracy can be high simply by predicting the majority class (non-accident) all the time.
- F1-score (harmonic mean of precision and recall) and ROC-AUC (area under the ROC curve) are better indicators of performance on imbalanced datasets.

## Trade-offs
The CNN-LSTM model achieves:
- Higher recall (94.45% vs 54.79%): It captures more true accident cases (reduces false negatives).
- Slightly lower precision (80.74% vs 87.83%): It has more false positives (flags non-accident as accident).
- Substantially higher F1-score (0.8706 vs 0.6748): Better balance between precision and recall.
- Higher ROC-AUC (0.8220 vs 0.8207): Better overall ranking capability.

For an accident detection system, **recall (sensitivity)** is often more important than precision because missing an accident (false negative) is worse than a false alarm (false positive). The CNN-LSTM model significantly improves recall at a moderate cost to precision.

## Options to Address Accuracy Concern
1. **Accept the current model**: It meets the project's explicit requirement (beats baseline on F1 and ROC-AUC) and is suitable for accident detection where recall is prioritized.
2. **Retrain with different class weights**: Adjust the loss function to penalize false negatives less, potentially increasing precision at the cost of recall and F1.
3. **Retrain with different architecture/hyperparameters**: Try different learning rates, sequence lengths, or model capacities.
4. **Use a different threshold**: The current model uses 0.5 for converting probability to accident/non-accident. Adjusting this threshold can trade off precision and recall.

## Recommendation
Given the project brief's focus on F1-score and ROC-AUC, and the importance of recall in accident detection, we recommend proceeding with the current CNN-LSTM model. However, if you would like to explore improving accuracy (while monitoring F1 and ROC-AUC), we can retrain the model with adjustments.

Please let us know how you'd like to proceed:
- Option A: Proceed with the current model and move to backend development (Milestone 2).
- Option B: Retrain the model to attempt to improve accuracy (we will track F1 and ROC-AUC to ensure they don't degrade significantly).
- Option C: Provide specific accuracy targets or other constraints.

Once we have your direction, we will update the milestone plan accordingly.