import onnxruntime as ort
import numpy as np
import os

# Path to the ONNX model in the backend directory
model_path = os.path.join("backend", "model.onnx")
print(f"Loading model from: {model_path}")

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

try:
    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    print("Model loaded successfully.")

    # Get input and output details
    input_details = session.get_inputs()
    output_details = session.get_outputs()

    print(f"Input name: {input_details[0].name}")
    print(f"Input shape: {input_details[0].shape}")
    print(f"Output name: {output_details[0].name}")
    print(f"Output shape: {output_details[0].shape}")

    # Create a dummy input matching the expected shape
    # From the backend: SEQ_LENGTH=10, IMG_SIZE=128, and we have 3 channels (RGB)
    dummy_input = np.random.randn(1, 10, 128, 128, 3).astype(np.float32)

    # Run inference
    outputs = session.run([output_details[0].name], {input_details[0].name: dummy_input})
    print(f"Inference successful. Output shape: {outputs[0].shape}")
    print(f"Output sample: {outputs[0][0][0]}")  # Should be a single value (sigmoid)

except Exception as e:
    print(f"Error during ONNX inference: {e}")
    exit(1)

print("\nONNX model test passed.")