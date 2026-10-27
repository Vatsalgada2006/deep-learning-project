# Honest Video Data Plan for Continuous Accident Detection

## 1. Executive Summary: Why Still Images Failed
The previous model was trained on the Kaggle "Accident Detection from CCTV" image dataset by artificially taking isolated frames and repeating them 10 times (`tf.repeat(x, 10, axis=1)`). As a result:
- Consecutive frames in the training sequences had **zero temporal variation** ($\Delta t = 0$).
- When deployed on continuous video with frame jumps (e.g. sampling every 5 frames = 0.17s), the inter-frame motion of moving vehicles and camera jitter produced large temporal transitions that broke the model's static assumptions.
- Fast collisions occurring over 15–30 frames were skipped or diluted across wide window strides, causing real accidents to be classified as normal traffic.

To detect accidents reliably in continuous video, the model must be trained on **real temporal video sequences** with consecutive frames at a fixed sampling rate (10–15 FPS).

---

## 2. Candidate Public Video Datasets

| Dataset | Total Clips | Format & Resolution | Annotations | Licence & Terms | Download / Access Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DoTA (Detection of Traffic Anomaly)** | 4,677 videos (~9s each, 30 FPS) | MP4, 1280×720 H.264 | Frame-level start/end timestamps, bounding boxes, accident categories (collision, out-of-control, obstacle) | **CC BY-NC-SA 4.0** (Non-commercial, Attribution, ShareAlike) | **Manual download required** from official GitHub repository (`yujiang/DoTA_dataset`) via Google Drive or Baidu Wangpan archives. |
| **CADP (Car Accident Detection & Prediction)** | 1,416 videos (YouTube / CCTV) | Variable (360p to 720p) | Accident start/peak frames, spatial bounding boxes | **Academic / Non-commercial Research** only | **Manual download required** via GitHub (`ankitshah009/Car-Accident-Dataset-CADP`) using provided YouTube download scripts or Google Drive mirror. |
| **UCF-Crime (`RoadAccidents` subset)** | ~150 accident CCTV videos + ~950 normal videos | MP4, CCTV 30 FPS | Temporal start and end intervals of road accidents | **CRCV Academic / Non-commercial Research Use** | **Direct download available** via Kaggle (`kaggle datasets download -d odinsam/ucf-crime-dataset`) or official UCF CRCV server. |
| **DADA-2000 (Driver Attention in Driving Accidents)** | 2,000 video clips (dashcam) | 1584×660 @ 30 FPS | Temporal accident window, driver fixations | **Academic Research License** | **Manual registration required** via DADA portal (requires academic email). |

---

## 3. Recommended Acquisition Strategy for Colab / Kaggle
For fast, reproducible training on Google Colab or Kaggle:
1. **Primary Accident Class**: Download `RoadAccidents` from the UCF-Crime Kaggle dataset (`odinsam/ucf-crime-dataset`) and the `Collision` / `Out_of_control` classes from DoTA.
2. **Primary Normal Class**: Normal driving and highway surveillance sequences from UCF-Crime `Normal_Videos_event` and DoTA normal background segments (pre-accident seconds).
3. **Temporal Window Sampling**: Extract 10-frame sequences at 10–15 FPS (sampling every 2nd or 3rd frame at 30 FPS, window duration ~0.67–1.0s).
4. **Explicit Labels**:
   - `0 = Normal` ($P(\text{Accident}) \approx 0.0$)
   - `1 = Accident` ($P(\text{Accident}) \approx 1.0$)
   No label inversion required.

---

## 4. Demo Clips Isolation Protocol
All demonstration and UI test clips are strictly isolated in `demo_clips/`:
- `demo_clips/accident_crash_demo.mp4`
- `demo_clips/video_project.mp4`
- `demo_clips/car-accident-dataset-example2.mp4`
- `demo_clips/non_accident.mp4`

**Strict Rule**: These 4 clips are excluded from all training, validation, and testing splits to prevent data leakage and provide genuine unseen real-world validation.
