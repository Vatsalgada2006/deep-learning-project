# Model Training: CNN-LSTM for Road Accident Detection

This directory contains code to train a CNN-LSTM model for road accident detection using the Kaggle "Road Accidents from CCTV Footages" dataset.

## Dataset Requirements

The script expects the dataset to be organized in the following structure:
```
dataset_root/
    Accident/
        img1.jpg
        img2.jpg
        ...
    Non_Accident/
        img1.jpg
        img2.jpg
        ...
```
*(Folder names can be adjusted in the script if different)*

The dataset contains individual frames labeled as accident or non-accident. For temporal modeling with LSTM, we create sequences by repeating each image N times (sequence length) to simulate a video clip. **For real temporal modeling, you should replace this with actual frame sequences from video files.**

## Installation

1.  Clone/download this repository.
2.  Install the required Python packages. It is highly recommended to use a virtual environment or Google Colab.

```bash
pip install tensorflow numpy scikit-learn opencv-python tqdm matplotlib
# For ONNX export (optional but recommended for deployment):
pip install tf2onnx onnx
```

## Usage

### Option 1: Local Execution (if you have a GPU-enabled machine)

```bash
python train_cnn_lstm.py \
    --dataset_path /path/to/your/dataset \
    --seq_length 10 \
    --batch_size 32 \
    --epochs 20 \
    --output_dir ./model_output
```

### Option 2: Google Colab (Recommended for free GPU)

1.  Upload the `train_cnn_lstm.py` script to your Colab notebook's files or mount your Google Drive.
2.  Upload your dataset to Colab (e.g., upload a zip file and extract it, or mount Drive).
3.  Run the following commands in a Colab cell:

```python
# Install dependencies
!pip install -q tensorflow numpy scikit-learn opencv-python tqdm matplotlib tf2onnx onnx

# Run the training script (adjust paths as needed)
!python train_cnn_lstm.py \
    --dataset_path /content/dataset \
    --seq_length 10 \
    --batch_size 32 \
    --epochs 20 \
    --output_dir /content/model_output
```

## Output

After training, the output directory will contain:
-   `best_cnn.h5`: Best baseline CNN model (based on validation AUC)
-   `best_cnn_lstm.h5`: Best CNN-LSTM model (based on validation AUC)
-   `cnn_lstm_model.h5`: Final saved CNN-LSTM model (same as best_cnn_lstm.h5)
-   `model.onnx`: ONNX version of the CNN-LSTM model (if tf2onnx is installed)
-   `*_results.json`: Evaluation metrics for each model
-   `*_training_history.png`: Training curves (loss, accuracy, etc.)
-   `*_confusion_matrix.png`: Confusion matrix plot
-   `experiment_summary.json`: Summary of the entire experiment

## Model Architecture Details

-   **Baseline CNN**: MobileNetV2 (ImageNet weights) -> GlobalAveragePooling2D -> Dropout -> Dense(sigmoid)
-   **CNN-LSTM**: TimeDistributed(MobileNetV2) -> TimeDistributed(GlobalAveragePooling2D) -> LSTM(50) -> Dropout -> Dense(sigmoid)

## Notes on Temporal Modeling

The current implementation uses a placeholder for temporal data by repeating each image to form a sequence. This allows the pipeline to run and demonstrates the architecture, but it does not capture true temporal dynamics.

To use actual video frames:
1.  Extract frames from video files (e.g., using OpenCV).
2.  Group frames from the same video into sequences of fixed length.
3.  Label each sequence based on the video's label (accident/non-accident).
4.  Modify the `load_images_from_folder` and sequence creation logic accordingly.

## Troubleshooting

-   **Out of Memory (OOM)**: Reduce `--batch_size` or `--img_size`.
-   **Low Accuracy**: The dataset might be challenging. Try increasing `--epochs`, adjusting the learning rate (in the script), or using data augmentation.
-   **ONNX Conversion Errors**: Ensure compatible versions of `tensorflow`, `tf2onnx`, and `onnx` are installed.

## License

This code is provided for educational purposes as part of a road accident detection system project.