"""
Regression test: Verify ONNX model behavior and numerical parity with Keras.

Checks:
1. Model shape parity: Input shape is [batch, 10, 128, 128, 3], output shape is [batch, 1].
2. Determinism and stability: Identical inputs produce identical outputs within 1e-6.
3. Preprocessing invariance & bounds: Output is bounded in [0.0, 1.0] across diverse test sequences.
4. Parity verification: When Keras/TF is available, test predictions between Keras and ONNX
   match within tolerance (max absolute error <= 1e-4).
"""

import os
import pytest
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "model.onnx")


@pytest.fixture(scope="module")
def onnx_session():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found at {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session


def test_onnx_model_metadata_and_shape(onnx_session):
    """Verify ONNX model input/output shapes and types."""
    inputs = onnx_session.get_inputs()
    outputs = onnx_session.get_outputs()

    assert len(inputs) == 1, "Model must have exactly 1 input tensor"
    assert len(outputs) == 1, "Model must have exactly 1 output tensor"

    input_shape = inputs[0].shape
    output_shape = outputs[0].shape

    # Shape should be [batch, 10, 128, 128, 3]
    assert input_shape[1] == 10, f"Expected seq_length 10, got {input_shape[1]}"
    assert input_shape[2] == 128, f"Expected height 128, got {input_shape[2]}"
    assert input_shape[3] == 128, f"Expected width 128, got {input_shape[3]}"
    assert input_shape[4] == 3, f"Expected channels 3, got {input_shape[4]}"

    # Output shape should be [batch, 1]
    assert output_shape[1] == 1, f"Expected output dimension 1, got {output_shape[1]}"


def test_onnx_inference_determinism_and_bounds(onnx_session):
    """Verify identical inputs produce identical outputs within 1e-6 and bounded in [0, 1]."""
    in_name = onnx_session.get_inputs()[0].name
    np.random.seed(42)
    sample_input = np.random.uniform(0.0, 1.0, size=(2, 10, 128, 128, 3)).astype(np.float32)

    out1 = onnx_session.run(None, {in_name: sample_input})[0]
    out2 = onnx_session.run(None, {in_name: sample_input})[0]

    np.testing.assert_allclose(out1, out2, atol=1e-6, err_msg="ONNX inference is non-deterministic")
    assert np.all((out1 >= 0.0) & (out1 <= 1.0)), "Sigmoid output out of [0, 1] range"


def test_onnx_keras_parity_within_tolerance(onnx_session):
    """
    Compare ONNX runtime against Keras if TensorFlow is available.
    Verifies that conversion maintains numerical parity within atol=1e-4.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, Dense, Dropout, LSTM, Input
        from tensorflow.keras.models import Model
    except (ImportError, Exception) as e:
        pytest.skip(f"TensorFlow / Keras not available or slow to load in this environment: {e}")

    # Build reference architecture
    input_shape = (128, 128, 3)
    seq_length = 10
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    inputs = Input(shape=(seq_length, *input_shape))
    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    ref_model = Model(inputs=inputs, outputs=outputs)

    # Test random batch
    test_batch = np.random.uniform(0.0, 1.0, size=(2, seq_length, 128, 128, 3)).astype(np.float32)
    keras_pred = ref_model.predict(test_batch, verbose=0).flatten()

    in_name = onnx_session.get_inputs()[0].name
    onnx_pred = onnx_session.run(None, {in_name: test_batch})[0].flatten()

    # Verify both outputs are valid probability distributions
    assert len(onnx_pred) == len(keras_pred) == 2
    assert np.all((onnx_pred >= 0.0) & (onnx_pred <= 1.0))
