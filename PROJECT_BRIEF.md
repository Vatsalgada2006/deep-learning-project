# Project Brief: Real-Time Road Accident Detection System

Give this whole file to Claude Code as your first message. Save it as `CLAUDE.md`
in the repo root too, so Claude Code re-reads it automatically in future sessions.

## What I'm building;

A full-stack, portfolio-grade web app that detects road accidents in real time
from CCTV-style video using a CNN-LSTM deep learning model, shows live alerts on
a 3D-animated dashboard, and is deployed live for free. This is based on my
college case study report (attached/pasted below) — the code needs to actually
deliver what the report promises, not just look like it does.

## Non-negotiable success criteria

Don't just build something that runs — build to these targets and tell me
honestly if a target isn't met and why.

1. **Model accuracy**: CNN-LSTM must beat the baseline CNN on F1-score and
   ROC-AUC, not just raw accuracy (raw accuracy is misleading with class
   imbalance — 15,420 non-accident vs 6,191 accident images). Report all of:
   Accuracy, Precision, Recall, F1, ROC-AUC, and a confusion matrix, for BOTH
   models, side by side.
2. **Alert latency**: from the moment an accident frame sequence is fed to the
   backend to the moment the frontend receives and displays the alert, target
   under 2 seconds end-to-end on CPU inference. Measure and log this, don't
   just assume it.
3. **UI quality**: smooth 3D animation (React Three Fiber / Three.js) that
   runs at 60fps on a mid-range laptop, not a laggy tech demo. No placeholder
   Bootstrap-looking components.
4. **It must actually deploy and work live**, end to end, not just on
   localhost — that's the whole point of the portfolio value.

## How I want you to work

- Don't build everything in one shot. Propose a milestone plan, confirm it
  with me, then build one milestone at a time and show me it working before
  moving to the next.
- After training the model, show me the real metrics before we move to
  building the backend — if accuracy is bad, we fix the model first, we don't
  paper over it with a nicer UI.
- Ask me before making decisions that lock in architecture (e.g. which cloud
  service, which frontend framework) if you think there's a better free option
  than what I specify below.
- When something can't be free or can't hit a target, tell me directly and
  give the closest realistic alternative, don't quietly downgrade the plan.

## Tech stack (all free tier, no card required)

- **Model**: PyTorch or TensorFlow, trained on Google Colab free GPU.
  Baseline: CNN with MobileNetV2 transfer learning. Proposed: CNN-LSTM
  (TimeDistributed MobileNetV2 backbone → 2-layer LSTM → Dense+Dropout →
  sigmoid). Export to ONNX for fast CPU inference after training.
- **Dataset**: Kaggle "Road Accidents from CCTV Footages" (6,191 accident /
  15,420 non-accident images) — handle the class imbalance explicitly
  (class-weighted loss and/or oversampling), don't ignore it.
- **Backend**: FastAPI, serves the ONNX model, exposes a WebSocket for live
  push alerts (not polling), simulates a "live CCTV feed" by streaming a
  sample video file frame-by-frame through the model on a loop.
- **Alerting**: Discord webhook fires on detection (free, instant, looks
  legitimate in a demo) + alert also logged to the dashboard in real time.
- **Database**: Supabase free Postgres, logs every detection event
  (timestamp, camera ID, confidence score) for a history table in the UI.
- **Frontend**: React + Vite, React Three Fiber for 3D visuals, Framer Motion
  for UI transitions, Tailwind for styling.
- **Deployment**: Vercel for frontend (no cold start, free), Hugging Face
  Spaces or Railway for backend (pick whichever handles WebSockets better —
  your call, tell me why).

## UI/UX direction for the 3D dashboard

- A 3D scene representing camera nodes on a road/city layout (doesn't need to
  be photorealistic — stylized and clean is better than trying to be
  realistic and falling short).
- When an accident is detected: the relevant camera node pulses red, a smooth
  animated alert panel slides in with timestamp + confidence score + a
  thumbnail-style visual.
- A live metrics panel (animated 3D or clean 2D charts — your call) showing
  Accuracy/Precision/Recall/F1/ROC-AUC for both models side by side.
- A history/log table of past detected events pulled from Supabase.
- Dark, modern, "control room" aesthetic — think mission-control dashboard,
  not a school project.

## Deliverables I expect at the end

- Working repo on GitHub with clear README, architecture diagram, setup
  instructions.
- Live deployed link (frontend + backend both reachable).
- Real trained model with real metrics I can quote in my report/viva.
- A short README section explaining alert latency measurement and the actual
  number achieved.

## Attach/paste alongside this brief

Paste in your case study report content (or attach the PDF) so Claude Code has
the exact section wording (Problem Statement, Architecture, Ethical
Considerations) to stay consistent with, especially for the README and any
in-app "About this project" text.
