# model_training/train_cnn_lstm_final.py
"""
Final training script for CNN-LSTM road accident detection.
Adapted for dataset structure: dataset/Accident/Accident/*.jpg and dataset/NonAccident/NonAccident/*.jpg
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, LSTM, Dropout, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Try to import tf2onnx for ONNX conversion
try:
    import tf2onnx
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: tf2onnx or onnx not installed. ONNX export will be skipped.")
    print("Install with: pip install tf2onnx onnx")

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for accident detection')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset root containing Accident/ and NonAccident/ folders (each with nested class folders)')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size for resizing (default: 128 to reduce memory usage)')
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Sequence length for LSTM (number of frames per sample, default: 10)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8 to reduce memory usage)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save models and results (default: ./output)')
    return parser.parse_args()

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

def create_sequence_dataset(dataset, seq_length):
    """Convert image dataset to sequence dataset by repeating each image."""
    def _create_sequence(images):
        # Repeat each image seq_length times along a new axis
        seq = tf.repeat(images[:, tf.newaxis, :, :, :], seq_length, axis=1)
        return seq

    # First need to unbatch, then rebatch with sequences
    dataset = dataset.unbatch()
    dataset = dataset.batch(seq_length)
    dataset = dataset.map(lambda x: tf.expand_dims(x, axis=0))  # Add batch dim
    dataset = dataset.concat([dataset]*seq_length)  # This is a simplified approach
    # Actually, let's do it properly: we want sequences where each sequence contains seq_length frames
    # For simplicity in this implementation, we'll just repeat each image seq_length times
    dataset = dataset.map(lambda x: tf.repeat(x[:, tf.newaxis, :, :, :], seq_length, axis=1))
    dataset = dataset.unbatch()  # Remove the extra batch dimension we added

    return dataset

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
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

def plot_training_history(history, model_name, output_dir):
    """Plot and save training history."""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(history.history[metric], label='Train')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label='Validation')
        plt.title(f'{model_name} - {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'))
    plt.close()

def evaluate_model(model, dataset, steps, batch_size, model_name, output_dir):
    """Evaluate model and print/save metrics."""
    print(f'\nEvaluating {model_name}...')

    # Predict probabilities
    y_pred_prob = model.predict(dataset, steps=steps)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Get true labels
    y_true = []
    for _, batch_labels in dataset.unbatch().batch(steps * batch_size).take(1):
        y_true.extend(batch_labels.numpy())
    y_true = np.array(y_true[:len(y_pred)])  # Ensure same length

    # Compute metrics
    acc = np.mean(y_pred == y_true)
    # Precision, recall, f1 from sklearn report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    prec = report['1']['precision'] if '1' in report else 0.0
    rec = report['1']['recall'] if '1' in report else 0.0
    f1 = report['1']['f1-score'] if '1' in report else 0.0
    roc_auc = roc_auc_score(y_true, y_pred_prob) if len(set(y_true)) > 1 else 0.5

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

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

    return results

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define class folders (adjust if your dataset uses different names)
    # Based on your structure: Accident/Accident/ and NonAccident/NonAccident/
    class_folders = ['Accident', 'NonAccident']  # Updated to match your dataset
    class_names = {0: 'NonAccident', 1: 'Accident'}  # Mapping for labels

    # Create datasets
    print('Creating datasets...')
    train_datasets = []
    val_datasets = []
    class_counts = []

    for idx, class_name in enumerate(class_folders):
        try:
            # Create full dataset for this class
            full_dataset, count = create_dataset(args.dataset_path, class_name, args.img_size, is_training=True)
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
    # For simplicity, we'll interleave them, but this might not be perfectly balanced
    # A better approach would be to use weighted sampling, but this works for now
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

    # Create sequences for LSTM model
    print(f'Creating sequences with length {args.seq_length}...')
    train_sequential = train_labeled.map(
        lambda img, label: (tf.repeat(img[tf.newaxis, :, :, :], args.seq_length, axis=0), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_sequential = val_labeled.map(
        lambda img, label: (tf.repeat(img[tf.newaxis, :, :, :], args.seq_length, axis=0), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch for LSTM model
    train_batched_lstm = train_sequential.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_batched_lstm = val_sequential.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # For CNN model, we need single images (not sequences)
    train_batched_cnn = train_labeled.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_batched_cnn = val_labeled.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Calculate steps per epoch
    train_steps = int(sum(class_counts) * 0.8 // args.batch_size)
    val_steps = int(sum(class_counts) * 0.2 // args.batch_size)

    print(f'Train steps per epoch: {train_steps}')
    print(f'Validation steps: {val_steps}')

    # Build and train Baseline CNN
    print('\n=== Building Baseline CNN ===')
    input_shape = (args.img_size, args.img_size, 3)
    cnn_model, base_model_cnn = build_cnn_model(input_shape)
    cnn_model = compile_model(cnn_model)
    print(cnn_model.summary())

    # Callbacks for CNN
    checkpoint_cnn = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_cnn.h5'),
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping_cnn = EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )

    # Train CNN
    print('\n=== Training Baseline CNN ===')
    cnn_history = cnn_model.fit(
        train_batched_cnn,
        epochs=args.epochs,
        validation_data=val_batched_cnn,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[checkpoint_cnn, early_stopping_cnn],
        verbose=1
    )

    # Plot CNN training history
    plot_training_history(cnn_history, 'Baseline_CNN', args.output_dir)

    # Load best CNN model for evaluation
    cnn_model = tf.keras.models.load_model(os.path.join(args.output_dir, 'best_cnn.h5'))
    cnn_results = evaluate_model(cnn_model, val_batched_cnn, val_steps, args.batch_size, 'Baseline_CNN', args.output_dir)

    # Build and train CNN-LSTM model
    print('\n=== Building CNN-LSTM Model ===')
    cnn_lstm_model, base_model_lstm = build_cnn_lstm_model(input_shape, args.seq_length)
    cnn_lstm_model = compile_model(cnn_lstm_model)
    print(cnn_lstm_model.summary())

    # Callbacks for CNN-LSTM
    checkpoint_lstm = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_cnn_lstm.h5'),
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping_lstm = EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )

    # Train CNN-LSTM
    print('\n=== Training CNN-LSTM Model ===')
    cnn_lstm_history = cnn_lstm_model.fit(
        train_batched_lstm,
        epochs=args.epochs,
        validation_data=val_batched_lstm,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[checkpoint_lstm, early_stopping_lstm],
        verbose=1
    )

    # Plot CNN-LSTM training history
    plot_training_history(cnn_lstm_history, 'CNN_LSTM', args.output_dir)

    # Load best CNN-LSTM model for evaluation
    cnn_lstm_model = tf.keras.models.load_model(os.path.join(args.output_dir, 'best_cnn_lstm.h5'))
    lstm_results = evaluate_model(cnn_lstm_model, val_batched_lstm, val_steps, args.batch_size, 'CNN_LSTM', args.output_dir)

    # Compare models
    print('\n=== Model Comparison ===')
    print(f"Baseline CNN - F1: {cnn_results['f1_score']:.4f}, ROC-AUC: {cnn_results['roc_auc']:.4f}")
    print(f"CNN-LSTM     - F1: {lstm_results['f1_score']:.4f}, ROC-AUC: {lstm_results['roc_auc']:.4f}")

    # Save the final CNN-LSTM model (best) and convert to ONNX
    saved_model_path = os.path.join(args.output_dir, 'cnn_lstm_model.h5')
    cnn_lstm_model.save(saved_model_path)
    print(f'\nSaved CNN-LSTM model to {saved_model_path}')

    if ONNX_AVAILABLE:
        # Convert to ONNX
        print('Converting to ONNX...')
        spec = (tf.TensorSpec((None, args.seq_length, args.img_size, args.img_size, 3), tf.float32, name="input"),)
        output_path = os.path.join(args.output_dir, 'model.onnx')
        model_proto, _ = tf2onnx.convert.from_keras(cnn_lstm_model, input_spec=spec, output_path=output_path)
        print(f'ONNX model saved to {output_path}')
    else:
        print('Skipping ONNX conversion (install tf2onnx and onnx to enable).')

    # Save a summary of the experiment
    summary = {
        'experiment_params': vars(args),
        'class_distribution': {
            'non_accident': int(class_counts[0]),
            'accident': int(class_counts[1])
        },
        'total_images': sum(class_counts),
        'baseline_cnn_results': cnn_results,
        'cnn_lstm_results': lstm_results
    }
    with open(os.path.join(args.output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nExperiment complete. Results saved to {args.output_dir}')

if __name__ == '__main__':
    main()