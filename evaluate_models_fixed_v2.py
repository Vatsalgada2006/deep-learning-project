#!/usr/bin/env python3
"""
Model evaluation script for CNN and CNN-LSTM models - Fixed version 2
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, LSTM, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import cv2
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def load_image(img_path, img_size):
    """Load and preprocess a single image."""
    # img_path might be a tensor, convert to numpy string
    if isinstance(img_path, tf.Tensor):
        img_path = img_path.numpy()
    # Convert bytes to string if needed
    if isinstance(img_path, bytes):
        img_path = img_path.decode('utf-8')

    # img_size might be a tensor, convert to numpy int
    if isinstance(img_size, tf.Tensor):
        img_size = img_size.numpy()
    # Ensure img_size is an integer
    img_size = int(img_size)

    img = cv2.imread(img_path)
    if img is None:
        # Return a black image of the correct size if loading fails
        img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
    return img

def create_dataset(dataset_path, class_name, img_size, is_training=True):
    """Create a TensorFlow dataset for a specific class."""
    # Path looks like: dataset_path/Accident/Accident/*.jpg
    class_folder = os.path.join(dataset_path, class_name, class_name)
    if not os.path.isdir(class_folder):
        # Fallback: try direct structure dataset_path/Accident/*.jpg
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            raise ValueError(f"Could not find images for class {class_name} in {dataset_path}")

    # Get all image files
    image_files = []
    for filename in os.listdir(class_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(class_folder, filename))

    print(f"Found {len(image_files)} images for class {class_name}")

    # Create dataset
    def _load_image(file_path):
        img = tf.py_function(load_image, [file_path, img_size], tf.float32)
        img.set_shape([img_size, img_size, 3])
        return img

    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    # Don't batch here - we'll batch later where needed
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(image_files)

def build_cnn_model(input_shape, fine_tune_at=100):
    """Builds a CNN model using MobileNetV2 as base."""
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    # Freeze the convolutional base
    base_model.trainable = False
    # Optionally unfreeze top layers for fine-tuning
    if fine_tune_at is not None:
        for layer in base_model.layers[fine_tune_at:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model, base_model

def build_cnn_lstm_model(input_shape, seq_length, fine_tune_at=100):
    """Builds a CNN-LSTM model using TimeDistributed MobileNetV2."""
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    if fine_tune_at is not None:
        for layer in base_model.layers[fine_tune_at:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    # Define the CNN-LSTM model
    inputs = tf.keras.Input(shape=(seq_length, *input_shape))
    # TimeDistributed applies the same CNN to each frame
    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model, base_model

def compile_model(model, learning_rate=0.0001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

def evaluate_model(model, dataset, steps, batch_size, model_name, output_dir):
    """Evaluate model and print/save metrics."""
    print(f'\nEvaluating {model_name}...')

    # Collect all predictions and labels
    y_pred_probs = []
    y_true_labels = []

    # Reset dataset iterator
    dataset_iter = dataset.as_numpy_iterator()

    # Collect predictions
    for batch_images, batch_labels in dataset_iter:
        batch_pred = model.predict_on_batch(batch_images)
        y_pred_probs.extend(batch_pred.flatten())
        y_true_labels.extend(batch_labels.flatten())

        # Break if we've collected enough batches
        if len(y_true_labels) >= steps * batch_size:
            break

    # Trim to expected size
    y_pred_probs = np.array(y_pred_probs[:steps * batch_size])
    y_true_labels = np.array(y_true_labels[:steps * batch_size])

    # Convert probabilities to predictions
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Compute metrics
    acc = np.mean(y_pred == y_true_labels)

    # Handle edge case where only one class is present
    try:
        report = classification_report(y_true_labels, y_pred, output_dict=True, zero_division=0)
        if '1' in report:
            prec = report['1']['precision']
            rec = report['1']['recall']
            f1 = report['1']['f1-score']
        else:
            # If no positive predictions, precision/recall/f1 for class 1 are 0
            prec = 0.0
            rec = 0.0
            f1 = 0.0
    except Exception as e:
        print(f"Warning: Error in classification report: {e}")
        prec = 0.0
        rec = 0.0
        f1 = 0.0

    try:
        roc_auc = roc_auc_score(y_true_labels, y_pred_probs) if len(set(y_true_labels)) > 1 else 0.5
    except Exception as e:
        print(f"Warning: Error in ROC-AUC calculation: {e}")
        roc_auc = 0.5

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true_labels, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    except Exception as e:
        print(f"Warning: Error in confusion matrix: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f'{model_name} Results:')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1-Score:  {f1:.4f}')
    print(f'  ROC-AUC:   {roc_auc:.4f}')
    print(f'  Confusion Matrix:\n{cm}')

    # Save results to JSON
    results = {
        'model': model_name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }
    with open(os.path.join(output_dir, f'{model_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot confusion matrix
    try:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_name} Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Non-Accident', 'Accident'], rotation=45)
        plt.yticks(tick_marks, ['Non-Accident', 'Accident'])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")

    return results

def main():
    # Parameters matching the training script
    IMG_SIZE = 128
    SEQ_LENGTH = 10
    BATCH_SIZE = 8  # Reduced for evaluation to save memory
    DATASET_PATH = "./datasets"
    OUTPUT_DIR = "./accident_model_output"

    print("Road Accident Detection - Model Evaluation")
    print("=" * 50)
    print(f"Image size: {IMG_SIZE}")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define class folders
    class_folders = ['Accident', 'NonAccident']
    class_names = {0: 'NonAccident', 1: 'Accident'}

    # Create datasets
    print('Creating datasets...')
    train_datasets = []
    val_datasets = []
    class_counts = []

    for idx, class_name in enumerate(class_folders):
        try:
            # Create full dataset for this class
            full_dataset, count = create_dataset(DATASET_PATH, class_name, IMG_SIZE, is_training=True)
            class_counts.append(count)

            # Split into train and validation (80/20)
            dataset_size = count
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size

            # Shuffle and split
            shuffled_dataset = full_dataset.shuffle(buffer_size=1000, seed=42)
            train_dataset = shuffled_dataset.take(train_size)
            val_dataset = shuffled_dataset.skip(train_size)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

            print(f'Class {class_name}: {count} images ({train_size} train, {val_size} val)')

        except Exception as e:
            print(f'Error creating dataset for class {class_name}: {e}')
            raise

    # Combine datasets from both classes
    full_train_dataset = tf.data.Dataset.sample_from_datasets(
        train_datasets,
        weights=[count/sum(class_counts) for count in class_counts],
        seed=42
    )

    full_val_dataset = tf.data.Dataset.sample_from_datasets(
        val_datasets,
        weights=[count/sum(class_counts) for count in class_counts],
        seed=42
    )

    # Create labels dataset (0 for NonAccident, 1 for Accident)
    def add_label(image, class_idx):
        return image, tf.cast(class_idx, tf.float32)

    # Create labeled datasets by mapping each class dataset to its label
    labeled_train_datasets = []
    labeled_val_datasets = []

    for idx, (train_ds, val_ds) in enumerate(zip(train_datasets, val_datasets)):
        labeled_train_datasets.append(train_ds.map(lambda x: add_label(x, idx)))
        labeled_val_datasets.append(val_ds.map(lambda x: add_label(x, idx)))

    # Combine labeled datasets
    train_labeled = tf.data.Dataset.sample_from_datasets(
        labeled_train_datasets,
        weights=[count/sum(class_counts) for count in class_counts],
        seed=42
    )

    val_labeled = tf.data.Dataset.sample_from_datasets(
        labeled_val_datasets,
        weights=[count/sum(class_counts) for count in class_counts],
        seed=42
    )

    # Create sequences for LSTM model - CORRECTED VERSION
    print(f'Creating sequences with length {SEQ_LENGTH}...')

    def create_sequences(img, label):
        # Repeat each image SEQ_LENGTH times along a new axis
        repeated_img = tf.repeat(img[tf.newaxis, :, :, :], SEQ_LENGTH, axis=0)
        return repeated_img, label

    train_sequential = train_labeled.map(
        create_sequences,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_sequential = val_labeled.map(
        create_sequences,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch for LSTM model
    train_batched_lstm = train_sequential.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_batched_lstm = val_sequential.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # For CNN model, we need single images (not sequences)
    train_batched_cnn = train_labeled.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_batched_cnn = val_labeled.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Calculate steps per epoch
    train_steps = int(sum(class_counts) * 0.8 // BATCH_SIZE)
    val_steps = int(sum(class_counts) * 0.2 // BATCH_SIZE)

    print(f'Train steps per epoch: {train_steps}')
    print(f'Validation steps: {val_steps}')

    # Build and evaluate Baseline CNN
    print('\n=== Building Baseline CNN ===')
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    cnn_model, base_model_cnn = build_cnn_model(input_shape)
    cnn_model = compile_model(cnn_model)

    # Load the best trained weights
    try:
        cnn_model.load_weights("./accident_model_output/best_cnn.h5")
        print("[OK] Loaded best_cnn.h5 weights")
    except Exception as e:
        print(f"[ERROR] Failed to load best_cnn.h5: {e}")
        return

    cnn_results = evaluate_model(cnn_model, val_batched_cnn, val_steps, BATCH_SIZE, 'Baseline_CNN', OUTPUT_DIR)

    # Build and evaluate CNN-LSTM model
    print('\n=== Building CNN-LSTM Model ===')
    cnn_lstm_model, base_model_lstm = build_cnn_lstm_model(input_shape, SEQ_LENGTH)
    cnn_lstm_model = compile_model(cnn_lstm_model)

    # Load the best trained weights
    try:
        cnn_lstm_model.load_weights("./accident_model_output/best_cnn_lstm.h5")
        print("[OK] Loaded best_cnn_lstm.h5 weights")
    except Exception as e:
        print(f"[ERROR] Failed to load best_cnn_lstm.h5: {e}")
        return

    lstm_results = evaluate_model(cnn_lstm_model, val_batched_lstm, val_steps, BATCH_SIZE, 'CNN_LSTM', OUTPUT_DIR)

    # Compare models
    print('\n=== Model Comparison ===')
    print(f"Baseline CNN - F1: {cnn_results['f1_score']:.4f}, ROC-AUC: {cnn_results['roc_auc']:.4f}")
    print(f"CNN-LSTM     - F1: {lstm_results['f1_score']:.4f}, ROC-AUC: {lstm_results['roc_auc']:.4f}")

    # Check if CNN-LSTM beats baseline
    f1_improved = lstm_results['f1_score'] > cnn_results['f1_score']
    roc_auc_improved = lstm_results['roc_auc'] > cnn_results['roc_auc']

    print(f'\n=== Improvement Check ===')
    print(f"F1-score improved:  {f1_improved} ({cnn_results['f1_score']:.4f} → {lstm_results['f1_score']:.4f})")
    print(f"ROC-AUC improved:   {roc_auc_improved} ({cnn_results['roc_auc']:.4f} → {lstm_results['roc_auc']:.4f})")

    if f1_improved and roc_auc_improved:
        print("[OK] CNN-LSTM beats Baseline CNN on both F1-score and ROC-AUC!")
    else:
        print("[WARNING] CNN-LSTM does not beat Baseline CNN on both metrics.")
        print("         May need to fix the model before proceeding.")

    # Save experiment summary
    summary = {
        'experiment_params': {
            'img_size': IMG_SIZE,
            'seq_length': SEQ_LENGTH,
            'batch_size': BATCH_SIZE,
            'dataset_path': DATASET_PATH
        },
        'class_distribution': {
            'non_accident': int(class_counts[0]),
            'accident': int(class_counts[1])
        },
        'total_images': sum(class_counts),
        'baseline_cnn_results': cnn_results,
        'cnn_lstm_results': lstm_results
    }
    with open(os.path.join(OUTPUT_DIR, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nEvaluation complete. Results saved to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()