# Railway Deployment Troubleshooting Guide

If your Railway deployment failed during the build process, here are the most common issues and how to fix them.

## 🔍 How to View Build Logs
1. In your Railway project dashboard, click on the **"Deployments"** tab
2. Find the failed deployment (red status)
3. Click on it to see detailed logs
4. Look for error messages in the build process

## 🐳 Common Build Issues & Solutions

### Issue 1: Model File Not Found During Build
**Symptom**: 
- Error: `Model file not found at ../model.onnx`
- Or: `Failed to load ONNX model: [Errno 2] No such file or directory: '../model.onnx'`

**Solution**:
The Dockerfile may not be copying the model.onnx file correctly. Check your Dockerfile:

Current Dockerfile (as seen in repo):
```
# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "main"]
```

This should copy everything including model.onnx from root to container's /app directory. But if model.onnx is in repo root and we're working from /app/backend, the path `../model.onnx` should work.

**Fix**: Ensure model.onnx exists in the repository root (it does: 10361758 bytes). If the build is failing to find it, try:

1. **Alternative approach**: Copy model to backend/ directory during build
   Modify Dockerfile to:
   ```
   # Copy source code
   COPY . .
   
   # Ensure model is accessible from backend directory
   RUN ls -la /app/
   
   # Expose port
   EXPOSE 8000
   
   # Run the application
   CMD ["python", "-m", "main"]
   ```

### Issue 2: Missing Python Dependencies
**Symptom**:
- Error: `ModuleNotFoundError: No module named 'onnxruntime'`
- Or: `pip install` failures during build

**Solution**:
Check that requirements.txt is being processed. The Dockerfile should have:
```
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
```

If missing, add it. Current Dockerfile (from inspection) seems to have this.

### Issue 3: Port Binding Issues
**Symptom**:
- Error: `Address already in use` or `Failed to bind to port`
- Or application starts but health checks fail

**Solution**:
Ensure you're using `${PORT}` correctly:
- In backend/.env: `PORT=${PORT}` (this tells the app to use Railway's provided port)
- In main.py: `port=int(os.getenv("PORT", 8000))` (already present)

### Issue 4: Buildpack vs Dockerfile Confusion
**Symptom**:
- Railway tries to use buildpacks instead of Dockerfile
- Build fails because it expects a different structure

**Solution**:
In Railway project settings:
1. Go to **Settings** → **Buildpack**
2. Ensure it's set to **Dockerfile** (not auto-detect)
3. Or delete any existing buildpack and let it use Dockerfile

### Issue 5: Missing System Dependencies for OpenCV
**Symptom**:
- Error during `pip install opencv-python`: missing system libraries
- Or: `ImportError: libSM.so.6: cannot open shared object file`

**Solution**:
The current requirements.txt uses `opencv-python` which should work on most platforms. If you see OS-level dependency errors, we may need to add system packages to Dockerfile:
```
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
```

## 📋 Immediate Diagnostic Steps

Please check your Railway deployment logs for:

1. **Exact error message** during build
2. **Which step failed**: 
   - Dependency installation (`pip install -r requirements.txt`)
   - Docker build (`docker build`)
   - Application startup (`python -m main`)
3. **Any traceback or module not found errors**

## 🛠️ Quick Fixes to Try

### Fix 1: Explicitly Verify Model Exists in Build
Add this to your Dockerfile before RUN pip install:
```
# Verify model exists
RUN ls -la /app/model.onnx || echo "Model not found in root, checking backend/"
RUN ls -la /app/backend/model.onnx || echo "Model not found in backend either"
```

### Fix 2: Use Requirements.txt Explicitly
Ensure your Dockerfile has:
```
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

### Fix 3: Simple Workaround - Copy Model to Backend
If path issues persist, in Dockerfile after COPY . . :
```
# Ensure model is accessible from backend working directory
RUN cp /app/model.onnx /app/backend/ || echo "Model copy failed"
```

## 📞 What To Do Next

Please:
1. Check your Railway deployment logs for the exact error
2. Share the key error lines here (or describe them)
3. Let me know if you see:
   - ModuleNotFoundError for any Python package
   - FileNotFoundError for model.onnx
   - Buildpack vs Dockerfile confusion
   - OpenCV/system library errors
   - Port binding issues

Once we identify the specific error, I can give you the exact fix to apply to your Dockerfile or environment variables.

**Don't worry** - build failures are common and usually fixable with small configuration adjustments!