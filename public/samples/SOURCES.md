# Sample Video Sources & Licences

This document records the exact provenance, licensing, assembly method, and dataset split context for all media in `public/samples/`.

---

## 1. `sample_accident_01.mp4`
- **Label**: `Highway Collision (Image sequence from dataset, not continuous video)`
- **Clip Category**: Assembled image sequence
- **Assembly Method**: Assembled from still CCTV dataset frames (frames 1244–1339) at 6–8 fps playback.
- **Source Dataset**: Kaggle *Accident Detection from CCTV Footage* (`Accident` class).
- **Dataset Split Status**: Note: Frames were drawn from the general dataset pool. During model training, individual frames were randomly shuffled into the 80% training / 20% validation splits; these frames may have been seen during training.
- **Licence**: See dataset page (Kaggle public dataset licence / ODbL / CC BY-SA 4.0).
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 320x240, 371 KB.
- **Grad-CAM Attention**: `public/samples/sample_accident_01_gradcam.jpg` ("Model attention (Grad-CAM), not an object detection").

---

## 2. `sample_accident_02.mp4`
- **Label**: `Intersection Collision (Image sequence from dataset, not continuous video)`
- **Clip Category**: Assembled image sequence
- **Assembly Method**: Assembled from still CCTV dataset frames (frames 1340–1439) at 6–8 fps playback.
- **Source Dataset**: Kaggle *Accident Detection from CCTV Footage* (`Accident` class).
- **Dataset Split Status**: Drawn from the general dataset pool; not guaranteed held-out due to frame-level shuffle during training.
- **Licence**: See dataset page (Kaggle public dataset licence / ODbL / CC BY-SA 4.0).
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 320x240, 499 KB.
- **Grad-CAM Attention**: `public/samples/sample_accident_02_gradcam.jpg` ("Model attention (Grad-CAM), not an object detection").

---

## 3. `sample_normal_01.mp4`
- **Label**: `Normal Traffic Flow (Image sequence from dataset, not continuous video)`
- **Clip Category**: Assembled image sequence
- **Assembly Method**: Assembled from still CCTV dataset frames (frames 801–920) at 6–8 fps playback.
- **Source Dataset**: Kaggle *Accident Detection from CCTV Footage* (`NonAccident` class).
- **Dataset Split Status**: Consecutive frames from a fixed camera sequence within the Kaggle NonAccident dataset pool.
- **Licence**: See dataset page (Kaggle public dataset licence / ODbL / CC BY-SA 4.0).
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 320x240, 129.5 KB.
- **Grad-CAM Attention**: `public/samples/sample_normal_01_gradcam.jpg` ("Model attention (Grad-CAM), not an object detection").

---

## 4. `sample_continuous_01.mp4`
- **Label**: `Continuous Highway Surveillance (Real continuous video)`
- **Clip Category**: Real continuous CCTV surveillance footage (not assembled stills).
- **Temporal Characteristics**: Continuous 20 fps temporal feed (200 frames, 10.0 seconds) without frame assembly or still frame duplication.
- **Licence**: See dataset page / CC BY 4.0.
- **Audio**: None (silent).
- **Encoding**: H.264 (`avc1`), `yuv420p`, `+faststart`, 480x360, 20 fps, 1.19 MB.
- **Grad-CAM Attention**: `public/samples/sample_continuous_01_gradcam.jpg` ("Model attention (Grad-CAM), not an object detection").

---

## 5. `hero_loop.webm`
- **Label**: `Surveillance Terminal Background Loop`
- **Type**: Real CCTV footage excerpt with cyber surveillance timestamp overlay.
- **Licence**: See dataset page / CC BY-SA 4.0.
- **Encoding**: WebM VP9, `yuv420p`, 320x240, 290.6 KB.
