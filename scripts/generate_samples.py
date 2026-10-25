"""
Generate real sample clips (H.264 MP4, <3MB), poster thumbnails (OpenCV JPEG),
precomputed ONNX prediction JSONs, and a hero looping WebM video.
"""

import os
import sys
import json
import cv2
import numpy as np
import imageio
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "public", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "backend", "model.onnx")
SAMPLE_VIDEO = os.path.join(BASE_DIR, "backend", "sample_video.mp4")
ACCIDENT_DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets", "Accident", "Accident")
NON_ACCIDENT_DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets", "NonAccident", "NonAccident")

SEQ_LENGTH = 10
FRAME_SAMPLE_INTERVAL = 5
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.5


def run_onnx_inference(video_path: str, session, input_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    sampled = []
    frame_idx = 0
    first_good_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if first_good_frame is None and frame is not None and frame.size > 0:
            first_good_frame = frame.copy()
        if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
            # Preprocess: BGR -> RGB, resize 128x128, float / 255
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            sampled.append(resized.astype(np.float32) / 255.0)
        frame_idx += 1
    cap.release()

    predictions = []
    window_idx = 0
    t0 = cv2.getTickCount()

    for start in range(0, len(sampled), SEQ_LENGTH):
        window = sampled[start:start + SEQ_LENGTH]
        if len(window) < SEQ_LENGTH // 2:
            break
        # Pad if needed
        if len(window) < SEQ_LENGTH:
            pad = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) for _ in range(SEQ_LENGTH - len(window))]
            window = pad + list(window)
        
        seq = np.expand_dims(np.stack(window, axis=0), axis=0)
        out = session.run(None, {input_name: seq})
        raw = float(out[0][0][0])
        # Invert raw: P(Accident) = 1.0 - raw
        confidence = round(1.0 - raw, 4)
        sf = start * FRAME_SAMPLE_INTERVAL
        ef = min((start + len(window)) * FRAME_SAMPLE_INTERVAL, total_frames)
        predictions.append({
            "window_index": window_idx,
            "start_frame": sf,
            "end_frame": ef,
            "start_time": round(sf / fps, 4),
            "end_time": round(ef / fps, 4),
            "confidence": confidence,
            "is_accident": confidence > CONFIDENCE_THRESHOLD,
        })
        window_idx += 1

    dt_ms = ((cv2.getTickCount() - t0) / cv2.getTickFrequency()) * 1000.0

    return {
        "status": "success",
        "message": f"Processed {total_frames} frames ({len(sampled)} sampled).",
        "predictions": predictions,
        "inference_time_ms": round(dt_ms, 2),
        "frames_processed": len(sampled),
        "fps": round(fps, 2),
        "total_frames": total_frames,
    }, first_good_frame


def main():
    print(f"Loading ONNX model from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"ONNX Model loaded. Input: {input_name}")

    # 1. Normal Clip: Extract first 120 frames (6s @ 20fps) from backend/sample_video.mp4
    normal_mp4 = os.path.join(SAMPLES_DIR, "sample_normal_01.mp4")
    normal_poster = os.path.join(SAMPLES_DIR, "sample_normal_01.jpg")
    normal_json = os.path.join(SAMPLES_DIR, "sample_normal_01.json")

    print("\n--- Generating sample_normal_01.mp4 ---")
    cap = cv2.VideoCapture(SAMPLE_VIDEO)
    normal_frames = []
    for _ in range(120):
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to 480x360 for high quality yet low file size (<2MB)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (480, int(h * 480 / w)))
        normal_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()

    with imageio.get_writer(normal_mp4, fps=20, codec="libx264", pixelformat="yuv420p", ffmpeg_params=["-crf", "28", "-preset", "fast"]) as writer:
        for f in normal_frames:
            writer.append_data(f)
    print(f"Wrote {normal_mp4} ({os.path.getsize(normal_mp4) / 1024:.1f} KB)")

    # Poster from first good frame
    cv2.imwrite(normal_poster, cv2.cvtColor(normal_frames[0], cv2.COLOR_RGB2BGR))
    print(f"Wrote {normal_poster}")

    # Run inference and save JSON
    res_normal, _ = run_onnx_inference(normal_mp4, session, input_name)
    with open(normal_json, "w") as f:
        json.dump(res_normal, f, indent=2)
    print(f"Wrote {normal_json} ({len(res_normal['predictions'])} windows)")

    # 2. Accident Clip: Assemble real accident sequence frames from dataset
    # We use frames 1240 to 1320 (normal approach -> accident occurrence -> aftermath)
    accident_mp4 = os.path.join(SAMPLES_DIR, "sample_accident_01.mp4")
    accident_poster = os.path.join(SAMPLES_DIR, "sample_accident_01.jpg")
    accident_json = os.path.join(SAMPLES_DIR, "sample_accident_01.json")

    print("\n--- Generating sample_accident_01.mp4 ---")
    accident_frames = []
    # Collect available frames from 1240 to 1340
    avail_frames = []
    for i in range(1240, 1340):
        p = os.path.join(ACCIDENT_DATASET_DIR, f"{i}.jpg")
        if os.path.exists(p):
            avail_frames.append(p)

    print(f"Found {len(avail_frames)} frames in accident sequence")
    for p in avail_frames:
        img = cv2.imread(p)
        if img is not None:
            accident_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Pad or repeat slightly if needed to make ~100 frames (5 seconds @ 20fps)
    if len(accident_frames) < 100:
        # Interpolate/extend sequence
        reps = int(np.ceil(100 / len(accident_frames)))
        extended = []
        for f in accident_frames:
            extended.extend([f] * reps)
        accident_frames = extended[:100]

    with imageio.get_writer(accident_mp4, fps=20, codec="libx264", pixelformat="yuv420p") as writer:
        for f in accident_frames:
            writer.append_data(f)
    print(f"Wrote {accident_mp4} ({os.path.getsize(accident_mp4) / 1024:.1f} KB)")

    cv2.imwrite(accident_poster, cv2.cvtColor(accident_frames[0], cv2.COLOR_RGB2BGR))
    print(f"Wrote {accident_poster}")

    res_accident, _ = run_onnx_inference(accident_mp4, session, input_name)
    with open(accident_json, "w") as f:
        json.dump(res_accident, f, indent=2)
    print(f"Wrote {accident_json} ({len(res_accident['predictions'])} windows)")
    for p in res_accident['predictions']:
        print(f"  Window {p['window_index']}: Conf={p['confidence']:.3f}, Alert={p['is_accident']}")

    # 3. Hero looping clip (WebM or MP4, < 2MB)
    hero_webm = os.path.join(SAMPLES_DIR, "hero_loop.webm")
    print(f"\n--- Generating {hero_webm} ---")
    # Take a 4-second excerpt from accident clip with subtle visual HUD watermark
    hero_frames = [f.copy() for f in accident_frames[:80]]
    # Add minimal cyber HUD text overlay to hero clip
    for i, hf in enumerate(hero_frames):
        # cv2 uses BGR
        bgr = cv2.cvtColor(hf, cv2.COLOR_RGB2BGR)
        t_sec = i / 20.0
        cv2.putText(bgr, f"REC [LIVE] 00:0{int(t_sec)}:{int((t_sec%1)*100):02d}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 65), 1, cv2.LINE_AA)
        cv2.putText(bgr, "CAM-04 INTERSECTION S-12", (10, bgr.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        hero_frames[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        with imageio.get_writer(hero_webm, fps=20, codec="libvpx-vp9") as writer:
            for f in hero_frames:
                writer.append_data(f)
        print(f"Wrote {hero_webm} ({os.path.getsize(hero_webm) / 1024:.1f} KB)")
    except Exception as e:
        print(f"WebM VP9 write note: {e}. Writing MP4 fallback for hero...")
        hero_mp4 = os.path.join(SAMPLES_DIR, "hero_loop.mp4")
        with imageio.get_writer(hero_mp4, fps=20, codec="libx264", pixelformat="yuv420p") as writer:
            for f in hero_frames:
                writer.append_data(f)
        print(f"Wrote {hero_mp4} ({os.path.getsize(hero_mp4) / 1024:.1f} KB)")

    print("\nAll sample clips, poster thumbnails, and results generated successfully!")


if __name__ == "__main__":
    main()
