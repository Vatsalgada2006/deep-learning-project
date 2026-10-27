"""
Diagnose model behavior on continuous video clip.
Compares:
1. Frame spacing (sample every 1, 3, 5 frames)
2. BGR vs RGB color ordering
3. Scaling: / 255.0 vs mobilenet_v2.preprocess_input
4. Interpolation methods (cv2.INTER_LINEAR vs cv2.INTER_AREA)
5. Sequence construction:
   - Backend sliding window: 10 consecutive sampled frames across time
   - Training style: 1 single frame repeated 10 times (tf.repeat)
6. Label inversion: raw sigmoid output vs inverted confidence P(Accident) = 1.0 - raw
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

def diagnose_clip(video_path: str, model_path: str):
    print("=" * 80)
    print(f"DIAGNOSING: {video_path}")
    print(f"MODEL: {model_path}")
    print("=" * 80)

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    print(f"Model Input: {input_name}, shape: {input_info.shape}, type: {input_info.type}")

    # Read all raw frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video metadata: {total_frames} frames, {fps:.1f} FPS, {total_frames / fps:.2f}s duration")

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    print(f"Successfully read {len(raw_frames)} frames.")

    # -------------------------------------------------------------
    # Experiment 1: Sequence construction impact
    # Mode A: Backend sliding window (consecutive sampled frames)
    # Mode B: Training style (repeat a single frame 10 times)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("EXPERIMENT 1: Sequence Construction (Sample interval = 5, RGB, /255, 128x128)")
    print("-" * 80)

    # Sample every 5 frames
    sampled_5 = [raw_frames[i] for i in range(0, len(raw_frames), 5)]
    processed_rgb_255 = []
    for f in sampled_5:
        # Match backend: cv2.resize first, then cv2.cvtColor
        h, w = f.shape[:2]
        if w > 640:
            scale = 640.0 / w
            f = cv2.resize(f, (640, int(h * scale)))
        resized = cv2.resize(f, (128, 128))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed_rgb_255.append(rgb.astype(np.float32) / 255.0)

    print(f"\n[Mode A: Backend Sliding Window (10 consecutive frames per window)]")
    for w_idx in range(0, len(processed_rgb_255), 10):
        window = processed_rgb_255[w_idx:w_idx + 10]
        if len(window) < 5:
            break
        if len(window) < 10:
            pad = [np.zeros((128, 128, 3), dtype=np.float32) for _ in range(10 - len(window))]
            window = pad + list(window)
        seq = np.expand_dims(np.stack(window, axis=0), axis=0)
        raw = float(session.run(None, {input_name: seq})[0][0][0])
        conf = 1.0 - raw
        t_start = (w_idx * 5) / fps
        t_end = min(((w_idx + 10) * 5) / fps, total_frames / fps)
        print(f"  Window {w_idx//10} [{t_start:5.2f}s - {t_end:5.2f}s]: raw_sigmoid={raw:.4f} -> P(Accident)={conf:.4f} {'[ACCIDENT]' if conf > 0.5 else '[NORMAL]'}")

    print(f"\n[Mode B: Training Style (Single frame repeated 10 times in window)]")
    # For key points in the video: 1s, 2s, 3s, 4s, 5s...
    for t_sec in range(1, int(total_frames / fps) + 1):
        f_idx = min(int(t_sec * fps), total_frames - 1)
        frame = raw_frames[f_idx]
        resized = cv2.resize(frame, (128, 128))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        # Repeat single frame 10 times
        seq = np.expand_dims(np.repeat(norm[np.newaxis, :, :, :], 10, axis=0), axis=0)
        raw = float(session.run(None, {input_name: seq})[0][0][0])
        conf = 1.0 - raw
        print(f"  Frame at {t_sec}s (idx {f_idx}): raw_sigmoid={raw:.4f} -> P(Accident)={conf:.4f} {'[ACCIDENT]' if conf > 0.5 else '[NORMAL]'}")

    # -------------------------------------------------------------
    # Experiment 2: BGR vs RGB
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("EXPERIMENT 2: BGR vs RGB (Mode B, Frame at crash peak)")
    print("-" * 80)
    # Pick a frame midway (likely the crash)
    mid_idx = total_frames // 2
    f_mid = raw_frames[mid_idx]
    f_mid_128 = cv2.resize(f_mid, (128, 128))

    # RGB
    rgb = cv2.cvtColor(f_mid_128, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    seq_rgb = np.expand_dims(np.repeat(rgb[np.newaxis, :, :, :], 10, axis=0), axis=0)
    raw_rgb = float(session.run(None, {input_name: seq_rgb})[0][0][0])

    # BGR (accidental omission of cvtColor)
    bgr = f_mid_128.astype(np.float32) / 255.0
    seq_bgr = np.expand_dims(np.repeat(bgr[np.newaxis, :, :, :], 10, axis=0), axis=0)
    raw_bgr = float(session.run(None, {input_name: seq_bgr})[0][0][0])

    print(f"  Frame {mid_idx} RGB: raw={raw_rgb:.4f}, P(Accident)={1.0 - raw_rgb:.4f}")
    print(f"  Frame {mid_idx} BGR: raw={raw_bgr:.4f}, P(Accident)={1.0 - raw_bgr:.4f}")

    # -------------------------------------------------------------
    # Experiment 3: Normalization /255 vs preprocess_input
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("EXPERIMENT 3: Normalization /255 ([0, 1]) vs preprocess_input ([-1, 1])")
    print("-" * 80)
    # preprocess_input for mobilenet_v2: (x / 127.5) - 1.0
    prep = (cv2.cvtColor(f_mid_128, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5) - 1.0
    seq_prep = np.expand_dims(np.repeat(prep[np.newaxis, :, :, :], 10, axis=0), axis=0)
    raw_prep = float(session.run(None, {input_name: seq_prep})[0][0][0])
    print(f"  Normalized [0, 1]  (/ 255.0):         raw={raw_rgb:.4f}, P(Accident)={1.0 - raw_rgb:.4f}")
    print(f"  Normalized [-1, 1] (preprocess_input): raw={raw_prep:.4f}, P(Accident)={1.0 - raw_prep:.4f}")

    # -------------------------------------------------------------
    # Experiment 4: Frame Spacing (Interval 1 vs 2 vs 5 vs 10)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("EXPERIMENT 4: Frame Spacing Interval Comparison (Sliding window)")
    print("-" * 80)
    for interval in [1, 2, 5, 10]:
        sampled = [raw_frames[i] for i in range(0, len(raw_frames), interval)]
        proc = [cv2.cvtColor(cv2.resize(f, (128, 128)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for f in sampled]
        peak_accident = 0.0
        acc_windows = 0
        total_w = 0
        for w_idx in range(0, len(proc), 10):
            win = proc[w_idx:w_idx + 10]
            if len(win) < 10:
                continue
            sq = np.expand_dims(np.stack(win, axis=0), axis=0)
            rw = float(session.run(None, {input_name: sq})[0][0][0])
            c = 1.0 - rw
            peak_accident = max(peak_accident, c)
            if c > 0.5:
                acc_windows += 1
            total_w += 1
        print(f"  Interval {interval:2d} (every {interval/fps:4.2f}s): Total Windows={total_w:2d}, Accident Windows={acc_windows:2d}, Peak P(Accident)={peak_accident:.4f}")

if __name__ == "__main__":
    clip = sys.argv[1] if len(sys.argv) > 1 else "public/samples/accident_crash_demo.mp4"
    model = sys.argv[2] if len(sys.argv) > 2 else "backend/model.onnx"
    diagnose_clip(clip, model)
