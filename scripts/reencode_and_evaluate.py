"""
Process new sample videos:
1. Re-encode to H.264 (yuv420p, +faststart, no audio, under 3MB, max 720p)
2. Generate poster images (.jpg)
3. Generate contact sheets (grid of frames) for visual inspection
4. Run real ONNX model inference and generate precomputed JSON results
5. Verify with ffprobe
"""

import os
import sys
import json
import shutil
import subprocess
import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "public", "samples")
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model.onnx")
ARTIFACT_DIR = r"C:\Users\vatsa\.gemini\antigravity-ide\brain\06e352da-2c3b-41e9-bc34-94dc9b709240"

FFMPEG_EXE = "ffmpeg"
FFPROBE_EXE = "ffprobe"

SEQ_LENGTH = 10
FRAME_SAMPLE_INTERVAL = 5
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.5


def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}\nStderr: {p.stderr}")
    return p.stdout


def get_ffprobe_info(filepath):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate:format=duration,size -of json "{filepath}"'
    out = run_cmd(cmd)
    info = json.loads(out)
    
    # check audio
    audio_cmd = f'ffprobe -v error -select_streams a -show_entries stream=codec_name -of json "{filepath}"'
    audio_info = json.loads(run_cmd(audio_cmd))
    has_audio = len(audio_info.get("streams", [])) > 0
    return info, has_audio


def reencode_clip(input_path, output_path, target_max_mb=3.0):
    print(f"\n--- Re-encoding {os.path.basename(input_path)} -> {os.path.basename(output_path)} ---")
    crf = 24
    while crf <= 34:
        # scale down to max 720p (1280x720) while preserving aspect ratio and ensuring even dims
        vf_filter = "scale='min(1280,iw)':min'(720,ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"
        cmd = f'ffmpeg -y -i "{input_path}" -vf "{vf_filter}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart -an -preset slow -crf {crf} "{output_path}"'
        run_cmd(cmd)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"CRF {crf}: output size = {size_mb:.2f} MB")
        if size_mb < target_max_mb:
            break
        crf += 2

    final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Final size: {final_size_mb:.2f} MB (under {target_max_mb} MB: {final_size_mb < target_max_mb})")


def generate_poster(video_path, poster_path):
    # Seek to 1s or middle frame
    cmd = f'ffmpeg -y -ss 00:00:01 -i "{video_path}" -frames:v 1 -q:v 2 "{poster_path}"'
    try:
        run_cmd(cmd)
    except Exception:
        # fallback to first frame if video < 1s
        cmd = f'ffmpeg -y -i "{video_path}" -frames:v 1 -q:v 2 "{poster_path}"'
        run_cmd(cmd)
    print(f"Generated poster: {poster_path}")


def create_contact_sheet(video_path, output_path, grid=(3, 3)):
    """Extract N evenly spaced frames and tile them into a contact sheet."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    num_frames = grid[0] * grid[1]
    
    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (320, 180))
            
        # Add timestamp / frame index overlay
        t_sec = idx / fps if fps > 0 else 0
        text = f"{t_sec:.1f}s (f:{idx})"
        cv2.rectangle(frame, (5, 5), (110, 25), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 65), 1, cv2.LINE_AA)
        frames.append(frame)
    cap.release()
    
    # Tile into grid
    rows = []
    for r in range(grid[0]):
        row_frames = frames[r * grid[1] : (r + 1) * grid[1]]
        rows.append(np.hstack(row_frames))
    sheet = np.vstack(rows)
    
    cv2.imwrite(output_path, sheet)
    print(f"Created contact sheet: {output_path}")


def run_onnx_inference(video_path, session, input_name):
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
        # Model sigmoid output is P(NonAccident)
        confidence = round(1.0 - raw, 4)
        sf = start * FRAME_SAMPLE_INTERVAL
        ef = min((start + len(window)) * FRAME_SAMPLE_INTERVAL, total_frames)
        predictions.append({
            "window_index": window_idx,
            "start_frame": sf,
            "end_frame": ef,
            "start_time": round(sf / fps, 4) if fps > 0 else 0.0,
            "end_time": round(ef / fps, 4) if fps > 0 else 0.0,
            "confidence": confidence,
            "is_accident": confidence > CONFIDENCE_THRESHOLD,
        })
        window_idx += 1

    dt_ms = ((cv2.getTickCount() - t0) / cv2.getTickFrequency()) * 1000.0

    peak_confidence = max((p["confidence"] for p in predictions), default=0.0)
    has_accident = any(p["is_accident"] for p in predictions)

    return {
        "status": "success",
        "message": f"Processed {total_frames} frames ({len(sampled)} sampled).",
        "predictions": predictions,
        "inference_time_ms": round(dt_ms, 2),
        "frames_processed": len(sampled),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "peak_confidence": peak_confidence,
        "has_accident": has_accident,
    }


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1. Clean up old sample files and synthetic clips
    old_files = [
        "sample_accident_01.mp4", "sample_accident_01.jpg", "sample_accident_01.json", "sample_accident_01_gradcam.jpg",
        "sample_accident_02.mp4", "sample_accident_02.jpg", "sample_accident_02.json", "sample_accident_02_gradcam.jpg",
        "sample_normal_01.mp4", "sample_normal_01.jpg", "sample_normal_01.json", "sample_normal_01_gradcam.jpg",
        "sample_continuous_01.mp4", "sample_continuous_01.jpg", "sample_continuous_01.json", "sample_continuous_01_gradcam.jpg",
        "hero_loop.webm"
    ]
    for old in old_files:
        p = os.path.join(SAMPLES_DIR, old)
        if os.path.exists(p):
            os.remove(p)
            print(f"Deleted old sample file: {old}")

    # 2. Map of root files to destination filenames
    clips_mapping = [
        ("accident_crash_demo.mp4", "accident_crash_demo.mp4"),
        ("car-accident-dataset-example2.mp4", "car-accident-dataset-example2.mp4"),
        ("non accident.mp4", "non_accident.mp4"),
        ("Video Project.mp4", "video_project.mp4"),
    ]

    # Move root files to a temp raw holding location
    raw_dir = os.path.join(SAMPLES_DIR, "raw_incoming")
    os.makedirs(raw_dir, exist_ok=True)

    moved_clips = []
    for root_name, dest_name in clips_mapping:
        src = os.path.join(BASE_DIR, root_name)
        if os.path.exists(src):
            dest_raw = os.path.join(raw_dir, dest_name)
            shutil.move(src, dest_raw)
            print(f"Moved root file '{root_name}' -> raw holding '{dest_raw}'")
            moved_clips.append((dest_raw, dest_name))
        else:
            # check if already in raw_dir
            dest_raw = os.path.join(raw_dir, dest_name)
            if os.path.exists(dest_raw):
                moved_clips.append((dest_raw, dest_name))
            else:
                print(f"WARNING: File not found in root or raw holding: {root_name}")

    # Load ONNX model for inference
    print(f"\nLoading ONNX model from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"Model loaded: {input_name}")

    results_summary = []

    for raw_path, dest_name in moved_clips:
        base_name = os.path.splitext(dest_name)[0]
        out_mp4 = os.path.join(SAMPLES_DIR, f"{base_name}.mp4")
        out_poster = os.path.join(SAMPLES_DIR, f"{base_name}.jpg")
        out_json = os.path.join(SAMPLES_DIR, f"{base_name}.json")
        out_sheet_samples = os.path.join(SAMPLES_DIR, f"{base_name}_sheet.jpg")
        out_sheet_artifact = os.path.join(ARTIFACT_DIR, f"{base_name}_sheet.jpg")

        # Re-encode to H.264, yuv420p, +faststart, no audio, <3MB, max 720p
        reencode_clip(raw_path, out_mp4, target_max_mb=2.95)

        # Generate poster
        generate_poster(out_mp4, out_poster)

        # Create contact sheet
        create_contact_sheet(out_mp4, out_sheet_samples)
        shutil.copyfile(out_sheet_samples, out_sheet_artifact)

        # Confirm with ffprobe
        probe_info, has_audio = get_ffprobe_info(out_mp4)
        v_stream = probe_info["streams"][0]
        v_format = probe_info["format"]
        file_size_bytes = int(v_format["size"])
        file_size_mb = file_size_bytes / (1024 * 1024)

        print(f"\n=== FFPROBE VERIFICATION for {base_name}.mp4 ===")
        print(f"Codec: {v_stream['codec_name']}")
        print(f"Pixel format: {v_stream['pix_fmt']}")
        print(f"Resolution: {v_stream['width']}x{v_stream['height']}")
        print(f"Duration: {float(v_format['duration']):.2f}s")
        print(f"Size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
        print(f"Audio stream present: {has_audio} (must be False)")

        # Run ONNX inference
        inf_result = run_onnx_inference(out_mp4, session, input_name)
        with open(out_json, "w") as f:
            json.dump(inf_result, f, indent=2)
        print(f"Wrote precomputed results: {out_json} ({len(inf_result['predictions'])} windows)")

        results_summary.append({
            "filename": f"{base_name}.mp4",
            "poster": f"{base_name}.jpg",
            "json": f"{base_name}.json",
            "sheet": f"{base_name}_sheet.jpg",
            "width": v_stream['width'],
            "height": v_stream['height'],
            "duration": round(float(v_format['duration']), 2),
            "size_mb": round(file_size_mb, 2),
            "codec": v_stream['codec_name'],
            "pix_fmt": v_stream['pix_fmt'],
            "has_audio": has_audio,
            "windows": len(inf_result['predictions']),
            "peak_confidence": inf_result['peak_confidence'],
            "model_has_accident": inf_result['has_accident'],
            "predictions": inf_result['predictions']
        })

    # Clean up raw holding folder once done
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("\n\n================ SUMMARY OF MODEL RUNS ================")
    for r in results_summary:
        print(f"\nClip: {r['filename']}")
        print(f"  Dimensions: {r['width']}x{r['height']} | Duration: {r['duration']}s | Size: {r['size_mb']} MB")
        print(f"  Codec: {r['codec']} ({r['pix_fmt']}) | Audio: {r['has_audio']}")
        print(f"  Windows: {r['windows']} | Peak Model Confidence: {r['peak_confidence']:.4f}")
        print(f"  Model Accident Flag: {r['model_has_accident']}")
        for w in r['predictions']:
            print(f"    Window {w['window_index']}: [{w['start_time']}s - {w['end_time']}s] Conf: {w['confidence']:.4f} Accident: {w['is_accident']}")

    with open(os.path.join(ARTIFACT_DIR, "model_run_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()

