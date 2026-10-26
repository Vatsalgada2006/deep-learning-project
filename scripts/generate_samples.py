"""
Generate real sample clips (H.264 MP4, yuv420p, +faststart, no audio, <3MB),
poster thumbnails (JPEG), precomputed ONNX prediction JSONs, and a hero looping WebM video.
All clips are generated from 100% real footage from the CCTV dataset (no synthetic shapes).
"""

import os
import sys
import json
import cv2
import numpy as np
import imageio
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "public", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "backend", "model.onnx")
ACCIDENT_DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets", "Accident", "Accident")
NON_ACCIDENT_DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets", "NonAccident", "NonAccident")

SEQ_LENGTH = 10
FRAME_SAMPLE_INTERVAL = 5
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.5


def write_clean_mp4(output_path: str, frames: list, fps: int = 20):
    """
    Writes frames to H.264 MP4 (avc1), yuv420p, +faststart, no audio, <3MB.
    """
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart", "-an", "-preset", "fast", "-crf", "24"]
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


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
        if len(window) < SEQ_LENGTH:
            pad = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) for _ in range(SEQ_LENGTH - len(window))]
            window = pad + list(window)
        
        seq = np.expand_dims(np.stack(window, axis=0), axis=0)
        out = session.run(None, {input_name: seq})
        raw = float(out[0][0][0])
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

    # -------------------------------------------------------------
    # 1. Normal Clip: Real CCTV Traffic Sequence (frames 801 to 920)
    # -------------------------------------------------------------
    normal_mp4 = os.path.join(SAMPLES_DIR, "sample_normal_01.mp4")
    normal_poster = os.path.join(SAMPLES_DIR, "sample_normal_01.jpg")
    normal_json = os.path.join(SAMPLES_DIR, "sample_normal_01.json")

    print("\n--- Generating sample_normal_01.mp4 from real CCTV footage ---")
    normal_frames = []
    for i in range(801, 921):
        p = os.path.join(NON_ACCIDENT_DATASET_DIR, f"{i}.jpg")
        img = cv2.imread(p)
        if img is not None:
            # Resize slightly to standard 320x240 for consistency
            resized = cv2.resize(img, (320, 240))
            normal_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    print(f"Collected {len(normal_frames)} real frames for normal traffic clip")
    write_clean_mp4(normal_mp4, normal_frames, fps=20)
    print(f"Wrote {normal_mp4} ({os.path.getsize(normal_mp4) / 1024:.1f} KB)")

    # Poster from first good frame
    cv2.imwrite(normal_poster, cv2.cvtColor(normal_frames[0], cv2.COLOR_RGB2BGR))
    print(f"Wrote {normal_poster}")

    res_normal, _ = run_onnx_inference(normal_mp4, session, input_name)
    with open(normal_json, "w") as f:
        json.dump(res_normal, f, indent=2)
    print(f"Wrote {normal_json} ({len(res_normal['predictions'])} windows):")
    for p in res_normal["predictions"]:
        print(f"  Window {p['window_index']}: Conf={p['confidence']:.4f}, Alert={p['is_accident']}")

    # Also update backend/sample_video.mp4 with this real footage so backend tests don't use synthetic noise!
    backend_sample = os.path.join(BASE_DIR, "backend", "sample_video.mp4")
    write_clean_mp4(backend_sample, normal_frames, fps=20)
    print(f"Updated {backend_sample} with real footage ({os.path.getsize(backend_sample) / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # 2. Accident Clip 1: Highway Collision Sequence (frames 1240 to 1340)
    # -------------------------------------------------------------
    acc1_mp4 = os.path.join(SAMPLES_DIR, "sample_accident_01.mp4")
    acc1_poster = os.path.join(SAMPLES_DIR, "sample_accident_01.jpg")
    acc1_json = os.path.join(SAMPLES_DIR, "sample_accident_01.json")

    print("\n--- Generating sample_accident_01.mp4 ---")
    acc1_frames = []
    for i in range(1240, 1340):
        p = os.path.join(ACCIDENT_DATASET_DIR, f"{i}.jpg")
        img = cv2.imread(p)
        if img is not None:
            resized = cv2.resize(img, (320, 240))
            acc1_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    if len(acc1_frames) < 100:
        reps = int(np.ceil(100 / len(acc1_frames)))
        extended = []
        for f in acc1_frames:
            extended.extend([f] * reps)
        acc1_frames = extended[:100]

    write_clean_mp4(acc1_mp4, acc1_frames, fps=20)
    print(f"Wrote {acc1_mp4} ({os.path.getsize(acc1_mp4) / 1024:.1f} KB)")

    cv2.imwrite(acc1_poster, cv2.cvtColor(acc1_frames[0], cv2.COLOR_RGB2BGR))
    print(f"Wrote {acc1_poster}")

    res_acc1, _ = run_onnx_inference(acc1_mp4, session, input_name)
    with open(acc1_json, "w") as f:
        json.dump(res_acc1, f, indent=2)
    print(f"Wrote {acc1_json} ({len(res_acc1['predictions'])} windows):")
    for p in res_acc1["predictions"]:
        print(f"  Window {p['window_index']}: Conf={p['confidence']:.4f}, Alert={p['is_accident']}")

    # -------------------------------------------------------------
    # 3. Accident Clip 2: Intersection Collision Sequence (frames 1340 to 1440)
    # -------------------------------------------------------------
    acc2_mp4 = os.path.join(SAMPLES_DIR, "sample_accident_02.mp4")
    acc2_poster = os.path.join(SAMPLES_DIR, "sample_accident_02.jpg")
    acc2_json = os.path.join(SAMPLES_DIR, "sample_accident_02.json")

    print("\n--- Generating sample_accident_02.mp4 ---")
    acc2_frames = []
    for i in range(1340, 1440):
        p = os.path.join(ACCIDENT_DATASET_DIR, f"{i}.jpg")
        img = cv2.imread(p)
        if img is not None:
            resized = cv2.resize(img, (320, 240))
            acc2_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    if len(acc2_frames) < 100:
        reps = int(np.ceil(100 / len(acc2_frames)))
        extended = []
        for f in acc2_frames:
            extended.extend([f] * reps)
        acc2_frames = extended[:100]

    write_clean_mp4(acc2_mp4, acc2_frames, fps=20)
    print(f"Wrote {acc2_mp4} ({os.path.getsize(acc2_mp4) / 1024:.1f} KB)")

    cv2.imwrite(acc2_poster, cv2.cvtColor(acc2_frames[0], cv2.COLOR_RGB2BGR))
    print(f"Wrote {acc2_poster}")

    res_acc2, _ = run_onnx_inference(acc2_mp4, session, input_name)
    with open(acc2_json, "w") as f:
        json.dump(res_acc2, f, indent=2)
    print(f"Wrote {acc2_json} ({len(res_acc2['predictions'])} windows):")
    for p in res_acc2["predictions"]:
        print(f"  Window {p['window_index']}: Conf={p['confidence']:.4f}, Alert={p['is_accident']}")

    # -------------------------------------------------------------
    # 4. Hero looping clip (WebM VP9, no synthetic shapes)
    # -------------------------------------------------------------
    hero_webm = os.path.join(SAMPLES_DIR, "hero_loop.webm")
    print(f"\n--- Generating {hero_webm} ---")
    hero_frames = [f.copy() for f in acc1_frames[:80]]
    for i, hf in enumerate(hero_frames):
        bgr = cv2.cvtColor(hf, cv2.COLOR_RGB2BGR)
        t_sec = i / 20.0
        cv2.putText(bgr, f"REC [LIVE] 00:0{int(t_sec)}:{int((t_sec%1)*100):02d}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 65), 1, cv2.LINE_AA)
        cv2.putText(bgr, "CAM-04 TRAFFIC CAMERA", (10, bgr.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        hero_frames[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        with imageio.get_writer(hero_webm, fps=20, codec="libvpx-vp9") as writer:
            for f in hero_frames:
                writer.append_data(f)
        print(f"Wrote {hero_webm} ({os.path.getsize(hero_webm) / 1024:.1f} KB)")
    except Exception as e:
        print(f"WebM write note: {e}")

    print("\nAll real footage sample clips, posters, and precomputations generated!")


if __name__ == "__main__":
    main()
