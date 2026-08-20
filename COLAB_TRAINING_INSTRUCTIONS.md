# Google Colab Setup Instructions for Road Accident Detection Model Training

## Step 1: Open Google Colab
Go to: https://colab.research.google.com

## Step 2: Create New Notebook
Click "New Notebook"

## Step 3: Mount Google Drive (Recommended for Large Dataset)
Run this in the first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Click the link, log in to your Google account, and copy the authorization code.

## Step 4: Upload Dataset and Script
You have two options:

### Option A: Upload via Google Drive (Recommended)
1. Upload your `datasets` folder to Google Drive (inside MyDrive)
2. Upload `model_training\train_cnn_lstm_final.py` to Google Drive
3. In Colab, your files will be at:
   - Dataset: `/content/drive/MyDrive/datasets/`
   - Script: `/content/drive/MyDrive/model_training/train_cnn_lstm_final.py`

### Option B: Upload Directly to Colab Session
1. Use the file upload button in Colab's left pane
2. Upload the entire `datasets` folder (may take time for large files)
3. Upload `model_training\train_cnn_lstm_final.py`

## Step 5: Install Dependencies and Set Path
Run this in a new cell:
```python
# Install required packages
!pip install -q tensorflow numpy scikit-learn opencv-python tqdm matplotlib tf2onnx onnx

# Check if GPU is available
import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())
```

## Step 6: Set Working Directory and Run Training
Run this in a new cell (ADJUST PATHS AS NEEDED):

### If using Google Drive:
```python
# Change to your Drive directory where you uploaded files
%cd /content/drive/MyDrive

# Run the training script
!python model_training/train_cnn_lstm_final.py \
    --dataset_path "/content/drive/MyDrive/datasets" \
    --seq_length 10 \
    --batch_size 32 \
    --epochs 20 \
    --output_dir "/content/drive/MyDrive/accident_model_output"
```

### If using direct Colab upload:
```python
# Change to content directory
%cd /content

# Run the training script
!python model_training/train_cnn_lstm_final.py \
    --dataset_path "/content/datasets" \
    --seq_length 10 \
    --batch_size 32 \
    --epochs 20 \
    --output_dir "/content/model_output"
```

## Step 7: Monitor Training
You'll see output showing:
- Dataset loading and image counts
- Model architecture summaries
- Training progress for both CNN and CNN-LSTM
- Validation metrics after each epoch
- Final evaluation results
- ONNX conversion confirmation

## Step 8: Retrieve Results
After training completes:
- **Best models**: `best_cnn.h5` and `best_cnn_lstm.h5`
- **Final model**: `cnn_lstm_model.h5`
- **ONNX model**: `model.onnx` (this is what you'll need for the backend)
- **Results**: `*_results.json` files with metrics
- **Plots**: Training history and confusion matrix visualizations

## 📋 What to Send Me After Training
Once training is complete, please:
1. Download the `model.onnx` file from the output directory
2. Share it with me (you can upload it to our project's `backend/model/` folder or similar)
3. Optionally share the results JSON files so I can verify the metrics

## 🚨 Troubleshooting Tips
1. **Out of Memory**: Reduce `--batch_size` to 16 or 8
2. **Slow Training**: Check that GPU is being used (look for "GPU Available: True" in Step 5 output)
3. **Dataset Path Errors**: Double-check the path - use `%ls` to see what's in your current directory
4. **Dependencies Not Installing**: Run the pip install cell again if needed

## 📊 Expected Success Criteria
After training, we'll verify:
- ✅ CNN-LSTM F1-score > Baseline CNN F1-score  
- ✅ CNN-LSTM ROC-AUC > Baseline CNN ROC-AUC
- ✅ Both models show reasonable performance (>70% accuracy would be good)
- ✅ ONNX exports correctly

## ⏱️ Time Estimates
- Package installation: 2-5 minutes
- Dataset loading: 2-10 minutes (depends on size and connection)
- Training (20 epochs): 15-40 minutes on GPU
- Total: ~25-60 minutes

**Ready to start?** Just follow the steps above and let me know when training begins or if you encounter any issues! 🚂