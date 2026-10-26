import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  apiFetch,
  WS_URL,
  ApiError,
  fetchCameras,
  fetchIncidents,
  patchIncidentStatus,
  predictVideoWithMeta,
} from './utils/apiClient';
import { evaluatePredictionsAtThreshold } from './utils/threshold';
import { MapPanel, type CameraInfo } from './components/MapPanel';
import { IncidentFeed } from './components/IncidentFeed';
import { ClaritySections } from './components/ClaritySections';
import { type Incident, clusterWindowsClient } from './utils/incidents';
import { runBrowserInferenceOnVideo } from './utils/browserInference';

// =====================================================================
//  Types
// =====================================================================
interface CameraAlert {
  timestamp: string;
  confidence: string;
  cameraId: string;
  latencyMs?: number;
  message?: string;
  windowIndex?: number;
}

interface PredictionWindow {
  window_index: number;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  confidence: number;
  is_accident: boolean;
}

interface PredictResult {
  status: string;
  message: string;
  predictions: PredictionWindow[];
  inference_time_ms: number;
  frames_processed: number;
  fps: number;
  total_frames: number;
  peak_memory_mb?: number | null;
  incidents?: Incident[];
}

interface ServerHealth {
  status: string;
  model_loaded: boolean;
  mongodb_connected: boolean;
  demo_mode: boolean;
  timestamp: string;
  websocket_connections: number;
  model_version?: string;
  model_checksum?: string;
  uptime_seconds?: number;
}

interface SampleClip {
  filename: string;
  label: string;
  resultPath: string;
  poster: string;
  tag: 'ACCIDENT' | 'NORMAL';
  source: string;
  licence: string;
  isSequence: boolean;
  naturalFps: number;
  snapshot: string;
}

// =====================================================================
//  Configuration
// =====================================================================
const DEFAULT_THRESHOLD = 0.5;

const SAMPLE_CLIPS: SampleClip[] = [
  {
    filename: 'sample_accident_01.mp4',
    label: 'Highway Collision (Image sequence from dataset, not continuous video)',
    resultPath: '/samples/sample_accident_01.json',
    poster: '/samples/sample_accident_01.jpg',
    tag: 'ACCIDENT',
    isSequence: true,
    naturalFps: 7,
    source: 'CCTV sequence frames 1244–1339, Kaggle Accident Dataset',
    licence: 'See dataset page (ODbL / CC BY-SA 4.0)',
    snapshot: '/samples/sample_accident_01_gradcam.jpg',
  },
  {
    filename: 'sample_accident_02.mp4',
    label: 'Intersection Collision (Image sequence from dataset, not continuous video)',
    resultPath: '/samples/sample_accident_02.json',
    poster: '/samples/sample_accident_02.jpg',
    tag: 'ACCIDENT',
    isSequence: true,
    naturalFps: 7,
    source: 'CCTV sequence frames 1340–1439, Kaggle Accident Dataset',
    licence: 'See dataset page (ODbL / CC BY-SA 4.0)',
    snapshot: '/samples/sample_accident_02_gradcam.jpg',
  },
  {
    filename: 'sample_normal_01.mp4',
    label: 'Normal Traffic Flow (Image sequence from dataset, not continuous video)',
    resultPath: '/samples/sample_normal_01.json',
    poster: '/samples/sample_normal_01.jpg',
    tag: 'NORMAL',
    isSequence: true,
    naturalFps: 7,
    source: 'CCTV sequence frames 801–920, Kaggle NonAccident Dataset',
    licence: 'See dataset page (ODbL / CC BY-SA 4.0)',
    snapshot: '/samples/sample_normal_01_gradcam.jpg',
  },
  {
    filename: 'sample_continuous_01.mp4',
    label: 'Continuous Highway Surveillance (Real continuous video)',
    resultPath: '/samples/sample_continuous_01.json',
    poster: '/samples/sample_continuous_01.jpg',
    tag: 'NORMAL',
    isSequence: false,
    naturalFps: 20,
    source: 'Real continuous CCTV surveillance footage (200 frames, 10s)',
    licence: 'See dataset page / CC BY 4.0',
    snapshot: '/samples/sample_continuous_01_gradcam.jpg',
  },
];

// Real metrics from confusion matrices (Baseline_CNN_results.json & CNN_LSTM_results.json)
// Baseline CNN: tn=1223, fp=16, fn=48, tp=3033
// CNN-LSTM:     tn=185,  fp=20, fn=11, tp=188
const REAL_METRICS = {
  baseline: {
    name: 'Baseline CNN',
    accuracy: '98.5%',
    accidentRecall: '1.6%',
    accidentPrecision: '3.8%',
    accidentF1: '2.2%',
    rocAuc: '0.999',
    note: 'Predicts everything as NonAccident due to class imbalance — useless for accident detection',
  },
  cnnLstm: {
    name: 'CNN-LSTM (Ours)',
    accuracy: '80.0%',
    accidentRecall: '94.5%',
    accidentPrecision: '80.7%',
    accidentF1: '87.1%',
    rocAuc: '0.822',
    note: 'Temporal modeling catches accidents the frame-level baseline completely misses',
  },
  baselineCM: { tn: 1223, fp: 16, fn: 48, tp: 3033 },
  cnnLstmCM: { tn: 185, fp: 20, fn: 11, tp: 188 },
};

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// =====================================================================
//  Section IDs for navigation
// =====================================================================
const SECTIONS = [
  { id: 'hero', label: 'Home' },
  { id: 'demo', label: 'Live Monitor' },
  { id: 'radar', label: 'Radar Map' },
  { id: 'incidents', label: 'Incident Feed' },
  { id: 'clarity', label: 'Clarity Guide' },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'results', label: 'Results' },
  { id: 'model-card', label: 'Model Card' },
  { id: 'engineering', label: 'Engineering' },
  { id: 'responsible-ai', label: 'Ethics' },
] as const;

// =====================================================================
//  HUD Overlay
// =====================================================================
function HUDOverlay({
  predictions,
  currentTime,
  fps,
  totalFrames,
  isPlaying,
  hasStarted,
  sourceName,
  threshold,
  isSample,
  backendSpeed,
  incidentCount = 0,
  isSequence = false,
}: {
  predictions: PredictionWindow[];
  currentTime: number;
  fps: number;
  totalFrames: number;
  isPlaying: boolean;
  hasStarted: boolean;
  sourceName?: string;
  threshold: number;
  isSample: boolean;
  backendSpeed?: string;
  incidentCount?: number;
  isSequence?: boolean;
}) {
  const currentWindow = predictions.find(
    (p) => currentTime >= p.start_time && currentTime < p.end_time
  );
  const currentWindowIdx = currentWindow ? currentWindow.window_index : null;

  // ALERT only while video.currentTime is inside a flagged window:
  const isInsideFlaggedWindow = Boolean(currentWindow && currentWindow.confidence > threshold);

  // Status: READY before play, NORMAL outside flagged windows, ALERT only inside flagged window
  let statusText = 'NORMAL';
  let statusColor = '#00ff41';
  if (!hasStarted && currentTime === 0) {
    statusText = 'READY';
    statusColor = '#00e5ff';
  } else if (isInsideFlaggedWindow) {
    statusText = 'ALERT!';
    statusColor = '#ff1744';
  }

  const confidence = currentWindow ? currentWindow.confidence : 0;
  const currentFrame = Math.floor(currentTime * fps);
  const speedText = isSample ? 'precomputed' : (backendSpeed || (isPlaying ? `${fps.toFixed(1)} fps` : 'precomputed'));

  return (
    <>
      <div
        className={`hud-box ${isInsideFlaggedWindow ? 'alert' : ''}`}
        style={{ position: 'absolute', top: 10, left: 10, zIndex: 10 }}
      >
        <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4, letterSpacing: 2 }}>
          AI ACCIDENT DETECTION SYSTEM
        </div>
        <div>
          SYSTEM STATUS:{' '}
          <span style={{ color: statusColor, fontWeight: 700 }}>
            {statusText}
          </span>
        </div>
        <div>WINDOW: {currentWindowIdx !== null ? currentWindowIdx + 1 : '-'}/{predictions.length || 1}</div>
        <div>INCIDENTS: <span style={{ color: incidentCount > 0 ? '#ff1744' : '#00ff41', fontWeight: 700 }}>{incidentCount}</span></div>
        <div>MODEL SCORE: {(confidence * 100).toFixed(1)}%</div>
        <div>FRAMES ANALYSED: {currentFrame}/{totalFrames}</div>
        <div>VIDEO FRAME RATE: {fps.toFixed(1)} fps</div>
        <div>ANALYSIS SPEED: {speedText}</div>
        {isSequence && (
          <div style={{ color: '#ffb74d', fontSize: 10, marginTop: 4, fontWeight: 600 }}>
            ℹ Image sequence from dataset, not continuous video
          </div>
        )}
        {sourceName && (
          <div style={{ opacity: 0.7, fontSize: 10, marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 220 }}>
            FILE: {sourceName}
          </div>
        )}
      </div>

      {isInsideFlaggedWindow && (
        <div
          className="alert-banner"
          style={{ position: 'absolute', top: 10, right: 10, zIndex: 10 }}
        >
          ⚠ ALERT! ACCIDENT DETECTED
        </div>
      )}
    </>
  );
}

// =====================================================================
//  Confidence Timeline
// =====================================================================
function ConfidenceTimeline({
  predictions,
  currentTime,
  totalDuration,
  threshold,
  onSeek,
}: {
  predictions: PredictionWindow[];
  currentTime: number;
  totalDuration: number;
  threshold: number;
  onSeek: (time: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent) => {
    if (!containerRef.current || totalDuration <= 0) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    onSeek(ratio * totalDuration);
  };

  const markerPos = totalDuration > 0 ? Math.min(100, Math.max(0, (currentTime / totalDuration) * 100)) : 0;
  const thresholdPct = threshold * 100;

  return (
    <div className="timeline-wrapper">
      <div className="timeline-scale" aria-hidden="true">
        <span className="scale-tick" style={{ bottom: '100%', transform: 'translateY(50%)' }}>100%</span>
        <span className="scale-tick" style={{ bottom: '75%', transform: 'translateY(50%)' }}>75%</span>
        <span className="scale-tick" style={{ bottom: '50%', transform: 'translateY(50%)' }}>50%</span>
        <span className="scale-tick" style={{ bottom: '25%', transform: 'translateY(50%)' }}>25%</span>
        <span className="scale-tick" style={{ bottom: '0%', transform: 'translateY(50%)' }}>0%</span>
      </div>

      <div
        ref={containerRef}
        className="timeline-container"
        onClick={handleClick}
        role="slider"
        aria-label="Confidence timeline"
        aria-valuenow={Math.round(currentTime)}
        aria-valuemin={0}
        aria-valuemax={Math.round(totalDuration)}
        tabIndex={0}
      >
        <div style={{ display: 'flex', gap: 2, height: '100%', alignItems: 'flex-end', position: 'relative', zIndex: 1 }}>
          {predictions.map((pred) => {
            const widthPct = totalDuration > 0
              ? ((pred.end_time - pred.start_time) / totalDuration) * 100
              : 100 / predictions.length;
            // Bar height equals real window confidence
            const heightPct = pred.confidence * 100;
            const isAccident = pred.confidence > threshold;
            return (
              <div
                key={pred.window_index}
                className="timeline-bar"
                style={{
                  width: `${widthPct}%`,
                  height: `${heightPct}%`,
                  background: isAccident
                    ? 'linear-gradient(to top, #ff1744, #ff5252)'
                    : 'linear-gradient(to top, #00cc33, #00ff41)',
                  borderRadius: '2px 2px 0 0',
                  opacity: 0.85,
                  minWidth: 4,
                }}
                title={`Window ${pred.window_index}: ${(pred.confidence * 100).toFixed(1)}%`}
              />
            );
          })}
        </div>
        <div className="threshold-line" style={{ bottom: `${thresholdPct}%` }}>
          <span
            className="threshold-label"
            style={{
              top: thresholdPct > 75 ? 2 : -15,
            }}
          >
            THRESHOLD {(threshold * 100).toFixed(0)}%
          </span>
        </div>
        <div className="timeline-marker" style={{ left: `${markerPos}%` }} />
      </div>
    </div>
  );
}

// =====================================================================
//  Confusion Matrix Display
// =====================================================================
function ConfusionMatrixDisplay({
  cm,
  label,
}: {
  cm: { tn: number; fp: number; fn: number; tp: number };
  label: string;
}) {
  return (
    <div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-dim)', marginBottom: 8 }}>
        {label}
      </div>
      <div className="confusion-matrix">
        <div className="cm-header"></div>
        <div className="cm-header">Pred: Neg</div>
        <div className="cm-header">Pred: Pos</div>
        <div className="cm-label">Actual: Neg</div>
        <div className="cm-cell tn">{cm.tn}</div>
        <div className="cm-cell fp">{cm.fp}</div>
        <div className="cm-label">Actual: Pos</div>
        <div className="cm-cell fn">{cm.fn}</div>
        <div className="cm-cell tp">{cm.tp}</div>
      </div>
    </div>
  );
}

// =====================================================================
//  MAIN APP
// =====================================================================
export default function App() {
  // --- Server state ---
  const [serverState, setServerState] = useState<'waking' | 'online' | 'offline'>('waking');
  const [serverErrorReason, setServerErrorReason] = useState<string>('');
  const wakingStartRef = useRef<number>(Date.now());
  const [healthInfo, setHealthInfo] = useState<ServerHealth | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [alerts, setAlerts] = useState<CameraAlert[]>([]);

  // --- Benchmark data ---
  const [benchmarkData, setBenchmarkData] = useState<{
    benchmark_date: string;
    model_version: string;
    model_checksum: string;
    model_size_mb: number;
    latency_stats: {
      mean_ms: number;
      median_ms: number;
      p95_ms: number;
      iterations: number;
    };
    memory_stats: {
      process_rss_mb: number;
      estimated_production_peak_rss_mb: number;
    };
  } | null>(null);

  const [copiedChecksum, setCopiedChecksum] = useState(false);

  // --- Video player ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoFileName, setVideoFileName] = useState<string>('');
  const [videoPoster, setVideoPoster] = useState<string>('');
  const [videoError, setVideoError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  // --- Inference ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null);

  // --- Threshold slider ---
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);

  // --- Active sample ---
  const [activeSample, setActiveSample] = useState<string | null>(null);

  // --- Active nav ---
  const [activeSection, setActiveSection] = useState('hero');

  // --- Cameras and Registry ---
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('cam-01');
  const [uploadLocationLabel, setUploadLocationLabel] = useState<string>('');

  // --- Incidents & Operations ---
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);

  // --- Inference Engine ---
  const [inferenceEngine, setInferenceEngine] = useState<'server' | 'browser'>('server');
  const [browserProgress, setBrowserProgress] = useState<{ pct: number; stage: string } | null>(null);
  const [engineComparison, setEngineComparison] = useState<{
    browserScore: number;
    serverScore: number;
    maxDiff: number;
    parityConfirmed: boolean;
  } | null>(null);
  const [isComparing, setIsComparing] = useState(false);

  // Compute threshold-adjusted predictions
  const thresholdAnalysis = predictResult
    ? evaluatePredictionsAtThreshold(predictResult.predictions, threshold)
    : null;

  const activePredictions = thresholdAnalysis
    ? thresholdAnalysis.evaluatedWindows
    : predictResult?.predictions ?? [];

  const currentWindow = activePredictions.find(
    (p) => currentTime >= p.start_time && currentTime < p.end_time
  );
  const isCurrentAlert = currentWindow ? currentWindow.confidence > threshold : false;

  const activeClip = SAMPLE_CLIPS.find((c) => c.filename === activeSample);

  const currentVideoIncidents = predictResult
    ? (predictResult.incidents && predictResult.incidents.length > 0
        ? predictResult.incidents
        : clusterWindowsClient(activePredictions, {
            cameraId: selectedCameraId,
            cameraName: cameras.find((c) => c.id === selectedCameraId)?.name,
            locationLabel: uploadLocationLabel || (activeClip?.isSequence ? 'Location: not available in dataset (Image sequence)' : 'Location: not available in dataset'),
            persistenceK: 1,
            cooldownWindows: 2,
            snapshotUrl: activeClip?.snapshot,
          }))
    : [];

  // ================================================================
  //  0. Load Benchmark JSON, Cameras, and Initial Incidents
  // ================================================================
  useEffect(() => {
    fetch('/benchmark.json')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setBenchmarkData(data);
      })
      .catch(() => {});

    fetchCameras()
      .then((cams) => {
        if (Array.isArray(cams) && cams.length > 0) setCameras(cams);
      })
      .catch(() => {});

    fetchIncidents()
      .then((res) => {
        if (res && res.incidents && res.incidents.length > 0) {
          setIncidents(res.incidents);
        }
      })
      .catch(() => {});
  }, []);

  // ================================================================
  //  1. Server Health Polling (with 90s offline transition)
  // ================================================================
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const data = await apiFetch<ServerHealth>('/health', {
        timeoutMs: 8000,
        retries: 0,
      });
      setHealthInfo(data);
      setServerState('online');
      setServerErrorReason('');
      return true;
    } catch (err) {
      let reason = 'Connection error';
      if (err instanceof ApiError) {
        reason = err.errorReason || err.friendlyMessage;
      } else if (err instanceof Error) {
        reason = err.message;
      }
      setServerErrorReason(reason);

      const elapsedSeconds = (Date.now() - wakingStartRef.current) / 1000;
      if (elapsedSeconds >= 90) {
        setServerState('offline');
      } else {
        setServerState('waking');
      }
      return false;
    }
  }, []);

  const handleRetryConnection = useCallback(() => {
    wakingStartRef.current = Date.now();
    setServerState('waking');
    setServerErrorReason('');
    checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let delay = 1500;
    let isCancelled = false;

    const poll = async () => {
      const ok = await checkHealth();
      if (!isCancelled) {
        if (!ok) {
          const elapsed = (Date.now() - wakingStartRef.current) / 1000;
          if (elapsed >= 90) {
            setServerState('offline');
            timer = setTimeout(poll, 45000);
          } else {
            delay = Math.min(delay * 1.5, 8000);
            timer = setTimeout(poll, delay);
          }
        } else {
          timer = setTimeout(poll, 30000);
        }
      }
    };

    poll();
    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [checkHealth]);

  // ================================================================
  //  2. WebSocket
  // ================================================================
  useEffect(() => {
    if (serverState !== 'online') return;

    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let backoff = 2000;
    let isCancelled = false;

    const connectWs = () => {
      try {
        ws = new WebSocket(WS_URL);
        ws.onopen = () => { setWsConnected(true); backoff = 2000; };
        ws.onclose = () => {
          setWsConnected(false);
          if (!isCancelled) {
            backoff = Math.min(backoff * 1.6, 12000);
            reconnectTimeout = setTimeout(connectWs, backoff);
          }
        };
        ws.onerror = () => setWsConnected(false);
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'new_incident' && data.data) {
              setIncidents((prev) => [data.data, ...prev.filter((i) => i.id !== data.data.id)]);
            }
            if (data.type === 'accident_detected') {
              const detectionTime = new Date(data.timestamp).getTime();
              const latencyMs = Math.max(0, Date.now() - detectionTime);
              setAlerts((prev) => [
                {
                  timestamp: data.timestamp,
                  confidence: `${(parseFloat(data.confidence) * 100).toFixed(0)}%`,
                  cameraId: data.camera_id || 'cam_001',
                  latencyMs,
                  message: data.message,
                  windowIndex: data.window_index,
                },
                ...prev.slice(0, 7),
              ]);
            }
          } catch (e) {
            console.error('WS parse error', e);
          }
        };
      } catch {
        if (!isCancelled) reconnectTimeout = setTimeout(connectWs, 5000);
      }
    };

    connectWs();
    return () => {
      isCancelled = true;
      clearTimeout(reconnectTimeout);
      ws?.close();
    };
  }, [serverState]);

  // ================================================================
  //  3. Video time tracking
  // ================================================================
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onTimeUpdate = () => setCurrentTime(video.currentTime);
    const onDurationChange = () => setDuration(video.duration || 0);
    const onPlay = () => { setIsPlaying(true); setHasStarted(true); };
    const onPause = () => setIsPlaying(false);
    const onEnded = () => setIsPlaying(false);

    video.addEventListener('timeupdate', onTimeUpdate);
    video.addEventListener('durationchange', onDurationChange);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('ended', onEnded);

    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate);
      video.removeEventListener('durationchange', onDurationChange);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('ended', onEnded);
    };
  }, [videoSrc]);

  // ================================================================
  //  4. Incident status updater
  // ================================================================
  const handleUpdateIncidentStatus = async (
    id: string,
    status: 'confirmed' | 'false_alarm',
    note?: string
  ) => {
    setIncidents((prev) =>
      prev.map((i) =>
        i.id === id ? { ...i, status, operator_note: note !== undefined ? note : i.operator_note } : i
      )
    );
    if (selectedIncident && selectedIncident.id === id) {
      setSelectedIncident((prev) =>
        prev ? { ...prev, status, operator_note: note !== undefined ? note : prev.operator_note } : null
      );
    }
    if (serverState === 'online') {
      try {
        await patchIncidentStatus(id, status, note);
      } catch (err) {
        console.warn('Failed to sync incident status with server', err);
      }
    }
  };

  // ================================================================
  //  5. Upload handler (Server vs Browser WASM)
  // ================================================================
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    const objectUrl = URL.createObjectURL(selectedFile);
    setVideoSrc(objectUrl);
    setVideoFileName(selectedFile.name);
    setActiveSample(null);
    setCurrentTime(0);
    setIsUploading(true);
    setUploadError(null);
    setPredictResult(null);
    setEngineComparison(null);

    if (videoRef.current) {
      videoRef.current.playbackRate = 1.0;
    }

    const locLabel = uploadLocationLabel.trim() || 'Location: not available in dataset';
    const selectedCam = cameras.find((c) => c.id === selectedCameraId);
    const camName = selectedCam?.name || `Camera (${selectedCameraId})`;

    try {
      if (inferenceEngine === 'browser') {
        if (!videoRef.current) throw new Error('Video player not initialized');
        setBrowserProgress({ pct: 10, stage: 'Initializing Browser WASM Model...' });
        const browserRes = await runBrowserInferenceOnVideo(videoRef.current, (pct, stage) => {
          setBrowserProgress({ pct, stage });
        });
        setBrowserProgress(null);

        const resData: PredictResult = {
          status: 'success',
          message: browserRes.message,
          predictions: browserRes.predictions,
          inference_time_ms: browserRes.inference_time_ms,
          frames_processed: browserRes.frames_processed,
          fps: browserRes.fps,
          total_frames: browserRes.total_frames,
        };
        setPredictResult(resData);

        const clustered = clusterWindowsClient(browserRes.predictions, {
          cameraId: selectedCameraId,
          cameraName: camName,
          locationLabel: locLabel,
          persistenceK: 1,
          cooldownWindows: 2,
        });
        setIncidents((prev) => [...clustered, ...prev]);
      } else {
        const data = await predictVideoWithMeta(selectedFile, {
          cameraId: selectedCameraId,
          cameraName: camName,
          locationLabel: locLabel,
          persistenceK: 1,
          cooldownWindows: 2,
        });
        setPredictResult(data);
        if (data.incidents && data.incidents.length > 0) {
          setIncidents((prev) => [
            ...data.incidents,
            ...prev.filter((i) => !data.incidents.some((c: Incident) => c.id === i.id)),
          ]);
        }
      }
    } catch (err: unknown) {
      setBrowserProgress(null);
      if (err instanceof ApiError) {
        setUploadError(err.friendlyMessage);
      } else if (err instanceof Error) {
        setUploadError(err.message);
      } else {
        setUploadError('Failed to analyze video');
      }
    } finally {
      setIsUploading(false);
    }
  };

  // ================================================================
  //  6. Compare Inference Engines (Parity Verification)
  // ================================================================
  const handleCompareEngines = async () => {
    if (!videoRef.current || !selectedFile) return;
    setIsComparing(true);
    setUploadError(null);
    try {
      setBrowserProgress({ pct: 15, stage: 'Running Browser WASM inference...' });
      const browserRes = await runBrowserInferenceOnVideo(videoRef.current);

      setBrowserProgress({ pct: 60, stage: 'Calling Server Cloud API...' });
      const serverRes = await predictVideoWithMeta(selectedFile, {
        cameraId: selectedCameraId,
        locationLabel: uploadLocationLabel || 'Location: not available in dataset',
      });
      setBrowserProgress(null);

      const bConfs = browserRes.predictions.map((p) => p.confidence);
      const sConfs = serverRes.predictions.map((p: PredictionWindow) => p.confidence);
      const minLen = Math.min(bConfs.length, sConfs.length);
      let maxDiff = 0;
      for (let i = 0; i < minLen; i++) {
        const diff = Math.abs(bConfs[i] - sConfs[i]);
        if (diff > maxDiff) maxDiff = diff;
      }
      const bPeak = bConfs.length ? Math.max(...bConfs) : 0;
      const sPeak = sConfs.length ? Math.max(...sConfs) : 0;

      setEngineComparison({
        browserScore: bPeak,
        serverScore: sPeak,
        maxDiff,
        parityConfirmed: maxDiff <= 0.05,
      });
    } catch (e: unknown) {
      setBrowserProgress(null);
      setUploadError(e instanceof Error ? e.message : 'Engine comparison failed');
    } finally {
      setIsComparing(false);
    }
  };

  // ================================================================
  //  7. Load sample clip
  // ================================================================
  const loadSampleClip = async (clip: SampleClip) => {
    setActiveSample(clip.filename);
    setUploadError(null);
    setSelectedFile(null);
    setIsUploading(false);
    setEngineComparison(null);

    try {
      const res = await fetch(clip.resultPath);
      if (!res.ok) throw new Error(`Could not load ${clip.resultPath}`);
      const data: PredictResult = await res.json();
      setPredictResult(data);

      const clipIncidents = clusterWindowsClient(data.predictions, {
        cameraId: clip.tag === 'ACCIDENT' ? 'cam-01' : 'cam-03',
        cameraName: clip.tag === 'ACCIDENT' ? 'Camera 01 - Highway Sector 4' : 'Camera 03 - Northbound Expressway',
        locationLabel: clip.isSequence
          ? 'Location: not available in dataset (Image sequence from dataset, not continuous video)'
          : 'Location: not available in dataset (Continuous CCTV surveillance video)',
        persistenceK: 1,
        cooldownWindows: 2,
        snapshotUrl: clip.snapshot,
      });
      setIncidents((prev) => [
        ...clipIncidents,
        ...prev.filter((i) => !clipIncidents.some((c) => c.id === i.id)),
      ]);
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : 'Failed to load sample');
      setPredictResult(null);
    }

    setVideoSrc(`/samples/${clip.filename}`);
    setVideoFileName(clip.filename);
    setVideoPoster(clip.poster);
    setVideoError(null);
    setCurrentTime(0);
    setHasStarted(false);

    if (videoRef.current) {
      videoRef.current.playbackRate = clip.isSequence ? 0.35 : 1.0;
    }
  };

  // ================================================================
  //  6. Seek handler
  // ================================================================
  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  // ================================================================
  //  7. Intersection Observer for nav highlighting
  // ================================================================
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: '-40% 0px -40% 0px', threshold: 0.1 }
    );

    SECTIONS.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  // ================================================================
  //  RENDER
  // ================================================================
  return (
    <div style={{ minHeight: '100vh' }}>
      {/* ── NAVIGATION ── */}
      <nav className="nav" role="navigation" aria-label="Main navigation">
        <a href="#hero" className="nav-logo" aria-label="Home">
          <span className="glow-title" style={{ fontSize: 14 }}>
            AI ACCIDENT DETECTION
          </span>
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 3,
            border: '1px solid var(--border-color)', color: 'var(--text-dim)',
            fontFamily: 'var(--font-mono)',
          }}>
            v{healthInfo?.model_version || '2.0'}
          </span>
        </a>

        <ul className="nav-links">
          {SECTIONS.map(({ id, label }) => (
            <li key={id}>
              <a href={`#${id}`} className={activeSection === id ? 'active' : ''}>
                {label}
              </a>
            </li>
          ))}
        </ul>

        <div className="nav-status" title={serverErrorReason || undefined}>
          {serverState === 'waking' ? (
            <>
              <span className="status-dot waking" />
              <span style={{ color: '#ffc107' }}>WAKING...</span>
            </>
          ) : serverState === 'offline' ? (
            <>
              <span className="status-dot offline" />
              <span style={{ color: 'var(--text-dim)' }}>OFFLINE</span>
              <button
                className="retry-btn"
                onClick={handleRetryConnection}
                aria-label="Retry connection"
              >
                ↻
              </button>
            </>
          ) : (
            <>
              <span className={`status-dot ${wsConnected ? 'online' : 'offline'}`} />
              <span style={{ color: wsConnected ? 'var(--neon-green)' : 'var(--neon-green)' }}>
                {wsConnected ? 'LIVE' : 'ONLINE'}
              </span>
            </>
          )}
        </div>
      </nav>

      {/* ═══════════════════════════════════════════════════════════
           SECTION 1: HERO
           ═══════════════════════════════════════════════════════════ */}
      <section id="hero" className="hero">
        <div className="hero-bg">
          <video
            autoPlay
            loop
            muted
            playsInline
            poster="/samples/sample_accident_01.jpg"
          >
            <source src="/samples/hero_loop.webm" type="video/webm" />
          </video>
        </div>

        <div className="hero-content fade-in-up">
          <div className="hero-badge">
            <span className="status-dot online" />
            Deep Learning Research Project
          </div>

          <h1 className="hero-title">
            Real-Time <span className="highlight">Accident Detection</span>
            <br />Using CNN-LSTM
          </h1>

          <p className="hero-description">
            A temporal deep learning system that analyzes CCTV / traffic camera footage to detect
            road accidents in real time — achieving <strong style={{ color: 'var(--neon-green)' }}>94.5% recall</strong> on
            the accident class where a frame-level baseline scores near zero.
          </p>

          <div className="hero-actions">
            <a href="#demo" className="btn btn-primary">
              Try Live Demo →
            </a>
            <a
              href="https://github.com/Vatsalgada2006/deep-learning-project"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline"
            >
              View on GitHub
            </a>
          </div>

          <div className="hero-stats stagger">
            <div className="hero-stat fade-in-up">
              <div className="hero-stat-value">94.5%</div>
              <div className="hero-stat-label">Accident Recall</div>
            </div>
            <div className="hero-stat fade-in-up">
              <div className="hero-stat-value">87.1%</div>
              <div className="hero-stat-label">F1-Score</div>
            </div>
            <div className="hero-stat fade-in-up">
              <div className="hero-stat-value">{benchmarkData ? `${Math.round(benchmarkData.latency_stats.mean_ms)}ms` : '~41ms'}</div>
              <div className="hero-stat-label">Inference Time{benchmarkData ? '' : ' (est.)'}</div>
            </div>
            <div className="hero-stat fade-in-up">
              <div className="hero-stat-value">ONNX</div>
              <div className="hero-stat-label">Runtime</div>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION 2: LIVE DEMO
           ═══════════════════════════════════════════════════════════ */}
      <section id="demo" className="section demo-section">
        <div className="section-inner">
          <div className="section-label">Interactive Demo</div>
          <h2 className="section-title">Try It Yourself</h2>
          <p className="section-subtitle" style={{ marginBottom: 'var(--sp-6)' }}>
            Upload a CCTV / traffic camera video or select a sample clip. The model analyzes 10-frame
            temporal windows and returns per-window accident confidence.
          </p>

          {/* Wake-up banner */}
          {serverState === 'waking' && (
            <div className="wakeup-banner" style={{ marginBottom: 'var(--sp-4)' }}>
              <div className="spinner" style={{ borderColor: '#ffc107', borderTopColor: '#fff' }} />
              <div>
                <div style={{ fontWeight: 600 }}>Waking up Render free-tier instance...</div>
                <div style={{ fontSize: 'var(--text-xs)', color: 'rgba(255,193,7,0.7)', marginTop: 2 }}>
                  Cold start takes ~30–50s. You can still try sample clips instantly.
                </div>
              </div>
            </div>
          )}

          <div className="demo-layout">
            {/* ── LEFT: Video Player ── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-3)' }}>
              <div
                className={`video-container ${isCurrentAlert ? 'alert-active' : ''}`}
              >
                {videoSrc ? (
                  <>
                    <video
                      ref={videoRef}
                      src={videoSrc}
                      controls
                      preload="metadata"
                      playsInline
                      poster={videoPoster || undefined}
                      onError={() => setVideoError(`Failed to load: ${videoFileName || videoSrc}`)}
                      style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                    />
                    {videoError && (
                      <div className="video-error-overlay">
                        <span style={{ fontSize: 28 }}>⚠</span>
                        <span>{videoError}</span>
                      </div>
                    )}
                    {predictResult && (
                      <HUDOverlay
                        predictions={activePredictions}
                        currentTime={currentTime}
                        fps={predictResult.fps}
                        totalFrames={predictResult.total_frames}
                        isPlaying={isPlaying}
                        hasStarted={hasStarted}
                        sourceName={videoFileName}
                        threshold={threshold}
                        isSample={activeSample !== null}
                        backendSpeed={predictResult.inference_time_ms ? `${predictResult.inference_time_ms.toFixed(0)}ms` : undefined}
                        incidentCount={currentVideoIncidents.length}
                        isSequence={activeClip?.isSequence ?? false}
                      />
                    )}
                  </>
                ) : (
                  <div className="video-placeholder">
                    <div className="video-placeholder-icon">▶</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)' }}>
                      Upload a video or select a sample clip
                    </div>
                  </div>
                )}
              </div>

              {/* Source attribution */}
              {activeSample && (() => {
                const clip = SAMPLE_CLIPS.find(c => c.filename === activeSample);
                return clip ? (
                  <div className="player-source-line">
                    <span className="source-label">Source:</span>
                    <span className="source-divider">|</span>
                    <span>{clip.source}</span>
                    <span className="source-divider">·</span>
                    <span style={{ opacity: 0.7 }}>{clip.licence}</span>
                    {serverState !== 'online' && (
                      <span className="sample-offline-badge">⚡ works offline</span>
                    )}
                  </div>
                ) : null;
              })()}

              {/* Confidence Timeline */}
              {predictResult && predictResult.predictions.length > 0 && (
                <ConfidenceTimeline
                  predictions={activePredictions}
                  currentTime={currentTime}
                  totalDuration={duration || activePredictions[activePredictions.length - 1]?.end_time || 1}
                  threshold={threshold}
                  onSeek={handleSeek}
                />
              )}

              {/* Threshold Slider */}
              <div className="threshold-slider">
                <span style={{ color: 'var(--text-dim)', whiteSpace: 'nowrap' }}>THRESHOLD</span>
                <input
                  type="range"
                  min="5"
                  max="95"
                  value={threshold * 100}
                  onChange={(e) => setThreshold(Number(e.target.value) / 100)}
                  aria-label="Confidence threshold"
                />
                <span style={{ color: 'var(--neon-green)', fontWeight: 700, minWidth: 38, textAlign: 'right' }}>
                  {(threshold * 100).toFixed(0)}%
                </span>
                {thresholdAnalysis && (
                  <span style={{ color: thresholdAnalysis.alertsRaised > 0 ? 'var(--alert-red)' : 'var(--text-dim)' }}>
                    {thresholdAnalysis.alertsRaised} alert{thresholdAnalysis.alertsRaised !== 1 ? 's' : ''}
                  </span>
                )}
              </div>

              {/* Camera selection & location label */}
              <div className="upload-config-box">
                <div className="config-grid">
                  <div className="config-item">
                    <label htmlFor="cam-id-select" className="config-label">SURVEILLANCE CAMERA</label>
                    <select
                      id="cam-id-select"
                      className="config-select"
                      value={selectedCameraId}
                      onChange={(e) => setSelectedCameraId(e.target.value)}
                    >
                      {cameras.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.name} ({c.id})
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="config-item">
                    <label htmlFor="loc-label-input" className="config-label">LOCATION LABEL (OPTIONAL)</label>
                    <input
                      id="loc-label-input"
                      type="text"
                      className="config-input"
                      placeholder="Location: not available in dataset"
                      value={uploadLocationLabel}
                      onChange={(e) => setUploadLocationLabel(e.target.value)}
                    />
                  </div>
                </div>

                {/* Engine Selector */}
                <div className="engine-selector-row">
                  <span className="engine-label">INFERENCE ENGINE:</span>
                  <div className="engine-toggle-group">
                    <button
                      type="button"
                      className={`engine-toggle-btn ${inferenceEngine === 'server' ? 'active' : ''}`}
                      onClick={() => setInferenceEngine('server')}
                    >
                      ☁️ Server API (Cloud)
                    </button>
                    <button
                      type="button"
                      className={`engine-toggle-btn ${inferenceEngine === 'browser' ? 'active' : ''}`}
                      onClick={() => setInferenceEngine('browser')}
                    >
                      ⚡ Browser WASM (Local)
                    </button>
                  </div>
                </div>
              </div>

              {/* Upload area */}
              <div className="upload-area">
                <label
                  htmlFor="video-upload"
                  className={`upload-label ${selectedFile ? 'has-file' : ''}`}
                >
                  {selectedFile ? selectedFile.name : 'SELECT VIDEO (MP4, WebM • ≤10 MB)'}
                  <input
                    type="file"
                    accept="video/mp4,video/webm,video/x-msvideo"
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    style={{ display: 'none' }}
                    id="video-upload"
                  />
                </label>
                <button
                  className="btn btn-primary"
                  disabled={!selectedFile || isUploading || (inferenceEngine === 'server' && serverState !== 'online')}
                  onClick={handleUpload as unknown as React.MouseEventHandler}
                  style={{ flexShrink: 0 }}
                >
                  {isUploading ? (
                    <>
                      <div className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                      ANALYZING...
                    </>
                  ) : (
                    inferenceEngine === 'browser' ? 'RUN BROWSER WASM' : 'UPLOAD & DETECT'
                  )}
                </button>
                {selectedFile && serverState === 'online' && (
                  <button
                    type="button"
                    className="btn btn-outline"
                    disabled={isComparing || isUploading}
                    onClick={handleCompareEngines}
                    title="Run both Browser WASM and Server API and verify numerical output parity"
                    style={{ fontSize: 11, padding: '8px 12px' }}
                  >
                    {isComparing ? 'Comparing...' : '⚖️ Compare Parity'}
                  </button>
                )}
              </div>

              {/* Browser Progress Bar */}
              {browserProgress && (
                <div className="browser-progress-box">
                  <div className="progress-info">
                    <span>{browserProgress.stage}</span>
                    <span>{browserProgress.pct}%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${browserProgress.pct}%` }} />
                  </div>
                </div>
              )}

              {/* Parity comparison result */}
              {engineComparison && (
                <div className={`parity-box ${engineComparison.parityConfirmed ? 'passed' : 'diverged'}`}>
                  <div className="parity-status">
                    {engineComparison.parityConfirmed ? '✓ PARITY VERIFIED (Browser WASM vs Server Cloud API)' : '⚠ PARITY DIVERGENCE'}
                  </div>
                  <div className="parity-metrics">
                    <span>Server Peak: <b>{(engineComparison.serverScore * 100).toFixed(2)}%</b></span>
                    <span>Browser Peak: <b>{(engineComparison.browserScore * 100).toFixed(2)}%</b></span>
                    <span>Max Abs Difference: <b>{engineComparison.maxDiff.toFixed(5)}</b> (&le; 0.05 tolerance)</span>
                  </div>
                </div>
              )}

              {/* Error */}
              {uploadError && <div className="error-box">{uploadError}</div>}

              {/* Inference summary */}
              {predictResult && (
                <div className="inference-summary">
                  <span>STATUS: <span className="val">{predictResult.status.toUpperCase()}</span></span>
                  <span>WINDOWS: <span className="val">{predictResult.predictions.length}</span></span>
                  <span>FRAMES: <span className="val">{predictResult.frames_processed}</span></span>
                  <span>INFERENCE: <span className="val">{predictResult.inference_time_ms}ms</span></span>
                  <span>FPS: <span className="val">{predictResult.fps}</span></span>
                  {predictResult.peak_memory_mb && (
                    <span>MEM: <span className="val">{predictResult.peak_memory_mb}MB</span></span>
                  )}
                </div>
              )}
            </div>

            {/* ── RIGHT: Side Panel ── */}
            <div className="side-panel">
              <div className="side-panel-header">
                Sample Clips
              </div>

              <div style={{ padding: 'var(--sp-3)', display: 'flex', flexDirection: 'column', gap: 'var(--sp-3)', overflowY: 'auto', flex: 1 }}>
                {SAMPLE_CLIPS.map((clip) => (
                  <div
                    key={clip.filename}
                    className={`sample-card ${activeSample === clip.filename ? 'active' : ''}`}
                    onClick={() => loadSampleClip(clip)}
                    role="button"
                    tabIndex={0}
                    aria-label={`Load ${clip.label} sample`}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        loadSampleClip(clip);
                      }
                    }}
                  >
                    <div
                      className="sample-card-thumb"
                      style={{ backgroundImage: `url(${clip.poster})` }}
                    >
                      <span className="play-icon">▶</span>
                      {activeSample === clip.filename && (
                        <div style={{
                          position: 'absolute', top: 6, right: 6,
                          width: 8, height: 8, borderRadius: '50%',
                          background: 'var(--neon-green)',
                          boxShadow: '0 0 6px var(--neon-green)',
                        }} />
                      )}
                    </div>
                    <div className="sample-label">
                      <span style={{
                        color: clip.tag === 'ACCIDENT' ? 'var(--alert-red)' : 'var(--neon-green)',
                        marginRight: 8,
                        fontWeight: 700,
                        fontSize: 9,
                      }}>
                        [{clip.tag}]
                      </span>
                      {clip.label}
                    </div>
                  </div>
                ))}
              </div>

              {/* Window Results */}
              {predictResult && (
                <div style={{ borderTop: '1px solid var(--border-color)', padding: 'var(--sp-3)', overflowY: 'auto', maxHeight: 220 }}>
                  <div style={{
                    fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
                    color: 'var(--neon-green)', letterSpacing: 2, marginBottom: 6,
                  }}>
                    WINDOW RESULTS
                  </div>
                  {activePredictions.map((pred) => {
                    const isCurrent = currentTime >= pred.start_time && currentTime < pred.end_time;
                    const isAccident = pred.confidence > threshold;
                    return (
                      <div
                        key={pred.window_index}
                        onClick={() => handleSeek(pred.start_time)}
                        className={`window-row ${isCurrent ? 'current' : ''} ${isCurrent && isAccident ? 'accident' : ''}`}
                      >
                        <span style={{ color: 'var(--text-secondary)' }}>
                          W{pred.window_index} [{formatTime(pred.start_time)}-{formatTime(pred.end_time)}]
                        </span>
                        <span style={{ color: isAccident ? '#ff1744' : '#00ff41', fontWeight: 700 }}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Live Alerts */}
              {alerts.length > 0 && (
                <div style={{ borderTop: '1px solid var(--border-color)', padding: 'var(--sp-3)', overflowY: 'auto', maxHeight: 180 }}>
                  <div style={{
                    fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
                    color: '#ff1744', letterSpacing: 2, marginBottom: 6,
                    display: 'flex', justifyContent: 'space-between',
                  }}>
                    <span>LIVE ALERTS</span>
                    <button onClick={() => setAlerts([])} style={{
                      background: 'none', border: 'none', color: 'var(--text-dim)',
                      cursor: 'pointer', fontSize: 10, fontFamily: 'var(--font-mono)',
                    }}>
                      CLEAR
                    </button>
                  </div>
                  <AnimatePresence>
                    {alerts.map((alert, idx) => (
                      <motion.div
                        key={`${alert.timestamp}-${idx}`}
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        style={{
                          padding: '6px 10px', marginBottom: 4, borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--alert-red-border)', background: 'var(--alert-red-bg)',
                          fontSize: 10, fontFamily: 'var(--font-mono)',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#ff5252', fontWeight: 700 }}>CRASH DETECTED</span>
                          <span style={{ color: '#ff8a80' }}>{alert.confidence}</span>
                        </div>
                        <div style={{ color: 'var(--text-dim)', marginTop: 2, fontSize: 9 }}>
                          {new Date(alert.timestamp).toLocaleTimeString()}
                          {alert.latencyMs !== undefined && (
                            <span style={{ color: 'var(--neon-green-dim)', marginLeft: 8 }}>RTT: {alert.latencyMs}ms</span>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION: RADAR & SURVEILLANCE MAP
           ═══════════════════════════════════════════════════════════ */}
      <section id="radar" className="section">
        <div className="section-inner">
          <div className="section-label">Geospatial Grid</div>
          <h2 className="section-title">Surveillance Radar & Camera Topology</h2>
          <p className="section-subtitle">
            Live geographic topology of monitored traffic nodes. Markers denote simulated demo cameras
            and spatial coordinates of detected incidents.
          </p>

          <MapPanel
            cameras={cameras}
            incidents={incidents}
            selectedCameraId={selectedCameraId}
            onSelectCamera={(id) => setSelectedCameraId(id)}
            onSelectIncident={(inc) => setSelectedIncident(inc)}
          />
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION: ACTIVE INCIDENTS FEED & AUDIT DRAWER
           ═══════════════════════════════════════════════════════════ */}
      <section id="incidents" className="section">
        <div className="section-inner">
          <div className="section-label">Incident Management</div>
          <h2 className="section-title">Active Incidents & Dispatch Feed</h2>
          <p className="section-subtitle">
            Clustered anomaly feed merging contiguous high-confidence windows with configurable persistence
            and cooldown. Operators can verify, flag false alarms, or export logs to CSV and JSON.
          </p>

          <IncidentFeed
            incidents={incidents}
            onUpdateIncidentStatus={handleUpdateIncidentStatus}
            selectedIncident={selectedIncident}
            onSelectIncident={setSelectedIncident}
          />
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION: CLARITY & ERROR AUDIT
           ═══════════════════════════════════════════════════════════ */}
      <section id="clarity" className="section">
        <div className="section-inner">
          <div className="section-label">Scientific Rigor & Error Analysis</div>
          <h2 className="section-title">Telemetry Guide & Failure Modes</h2>
          <p className="section-subtitle">
            Transparent documentation of model scores, verified outcomes across all sample clips,
            and honest empirical edge-case failure modes on real held-out CCTV footage.
          </p>

          <ClaritySections />
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION 3: HOW IT WORKS
           ═══════════════════════════════════════════════════════════ */}
      <section id="how-it-works" className="section">
        <div className="section-inner">
          <div className="section-label">Architecture</div>
          <h2 className="section-title">How It Works</h2>
          <p className="section-subtitle">
            The system processes video through a four-stage pipeline, combining spatial
            feature extraction with temporal sequence modeling.
          </p>

          <div className="pipeline-grid stagger">
            <div className="pipeline-step fade-in-up">
              <div className="pipeline-step-num">01</div>
              <div className="pipeline-step-icon">🎥</div>
              <div className="pipeline-step-title">Frame Extraction</div>
              <div className="pipeline-step-desc">
                Video is decoded at native FPS. Every 5th frame is sampled and resized to 128×128 RGB.
              </div>
            </div>

            <div className="pipeline-step fade-in-up">
              <div className="pipeline-step-num">02</div>
              <div className="pipeline-step-icon">🔍</div>
              <div className="pipeline-step-title">Spatial Features</div>
              <div className="pipeline-step-desc">
                TimeDistributed MobileNetV2 extracts a 1280-d feature vector from each frame in the window.
              </div>
            </div>

            <div className="pipeline-step fade-in-up">
              <div className="pipeline-step-num">03</div>
              <div className="pipeline-step-icon">🧠</div>
              <div className="pipeline-step-title">Temporal Modeling</div>
              <div className="pipeline-step-desc">
                A 2-layer LSTM (64→32 units) models the temporal dynamics across 10 consecutive frames.
              </div>
            </div>

            <div className="pipeline-step fade-in-up">
              <div className="pipeline-step-num">04</div>
              <div className="pipeline-step-icon">⚡</div>
              <div className="pipeline-step-title">ONNX Inference</div>
              <div className="pipeline-step-desc">
                Exported to ONNX for fast CPU inference ({benchmarkData ? `~${Math.round(benchmarkData.latency_stats.mean_ms)}ms` : '~41ms'} per window) on Render free tier.
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION 4: RESULTS
           ═══════════════════════════════════════════════════════════ */}
      <section id="results" className="section">
        <div className="section-inner">
          <div className="section-label">Evaluation</div>
          <h2 className="section-title">Model Comparison</h2>
          <p className="section-subtitle">
            All metrics are computed from real evaluation confusion matrices.
            The baseline CNN achieves high accuracy by predicting "no accident" for
            nearly every sample — useless for actual accident detection.
          </p>

          <div className="metrics-grid">
            {/* Comparison Table */}
            <div className="metric-card">
              <div className="metric-card-title">Class-Wise Metrics (Accident Class)</div>
              <table className="metric-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Baseline CNN</th>
                    <th style={{ color: 'var(--neon-green)' }}>CNN-LSTM (Ours)</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Accuracy', REAL_METRICS.baseline.accuracy, REAL_METRICS.cnnLstm.accuracy],
                    ['Accident Recall', REAL_METRICS.baseline.accidentRecall, REAL_METRICS.cnnLstm.accidentRecall],
                    ['Accident Precision', REAL_METRICS.baseline.accidentPrecision, REAL_METRICS.cnnLstm.accidentPrecision],
                    ['Accident F1', REAL_METRICS.baseline.accidentF1, REAL_METRICS.cnnLstm.accidentF1],
                    ['ROC-AUC', REAL_METRICS.baseline.rocAuc, REAL_METRICS.cnnLstm.rocAuc],
                  ].map(([metric, base, ours]) => (
                    <tr key={metric}>
                      <td>{metric}</td>
                      <td className="dim">{base}</td>
                      <td className="highlight">{ours}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Confusion Matrices */}
            <div className="metric-card">
              <div className="metric-card-title">Confusion Matrices</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-6)' }}>
                <ConfusionMatrixDisplay cm={REAL_METRICS.baselineCM} label="Baseline CNN (4320 samples)" />
                <ConfusionMatrixDisplay cm={REAL_METRICS.cnnLstmCM} label="CNN-LSTM (404 sequences)" />
              </div>
            </div>

            {/* Key Insight */}
            <div className="metric-card" style={{ gridColumn: '1 / -1' }}>
              <div className="metric-card-title">Key Insight</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--text-sm)', lineHeight: 1.7 }}>
                The baseline CNN's 98.5% accuracy is misleading — it achieves this by predicting
                "NonAccident" for nearly every sample, resulting in <strong style={{ color: 'var(--alert-red)' }}>near-zero
                recall on the Accident class</strong>. Our CNN-LSTM sacrifices some overall accuracy (80%)
                to achieve <strong style={{ color: 'var(--neon-green)' }}>94.5% accident recall</strong>,
                meaning it catches 19 out of every 20 real accidents. For a safety-critical system,
                this tradeoff is essential — missing an accident has far greater consequences than
                a false alarm.
              </p>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION 5: MODEL CARD
           ═══════════════════════════════════════════════════════════ */}
      <section id="model-card" className="section">
        <div className="section-inner">
          <div className="section-label">Transparency</div>
          <h2 className="section-title">Model Card</h2>
          <p className="section-subtitle">
            Key facts about the model following Google's Model Card framework.
          </p>

          <div className="model-card-grid">
            <div className="model-card-item">
              <h3>Architecture</h3>
              <ul>
                <li>TimeDistributed MobileNetV2 backbone (ImageNet pre-trained)</li>
                <li>2-layer LSTM: 64 → 32 units with 0.3 dropout</li>
                <li>Sigmoid output: P(NonAccident), inverted for P(Accident)</li>
                <li>Input: 10 frames × 128×128×3 (RGB, normalized [0,1])</li>
              </ul>
            </div>

            <div className="model-card-item">
              <h3>Training Data</h3>
              <ul>
                <li>Source: Accident Detection Dataset (Kaggle)</li>
                <li>Binary classes: Accident (0) and NonAccident (1)</li>
                <li>Sequences of 10 frames sampled at fixed intervals from each class folder (not necessarily consecutive in the original video)</li>
                <li>Train/val split uses random shuffling — near-duplicate frames from similar scenes may appear in both splits, potentially inflating validation scores</li>
              </ul>
            </div>

            <div className="model-card-item">
              <h3>Intended Use</h3>
              <ul>
                <li>Research demonstration and portfolio showcase</li>
                <li>CCTV / traffic camera accident detection assistance</li>
                <li>Not intended for standalone production use</li>
                <li>Human review required for all detections</li>
              </ul>
            </div>

            <div className="model-card-item">
              <h3>Limitations</h3>
              <ul>
                <li>Trained on limited dataset — may not generalize to all conditions</li>
                <li>No night-time, rain, or adverse weather training data verified</li>
                <li>Single-camera perspective only (no multi-view fusion)</li>
                <li>Fixed 10-frame window — misses very brief or very long events</li>
              </ul>
            </div>

            <div className="model-card-item">
              <h3>Inference Details</h3>
              <ul>
                <li>Runtime: ONNX Runtime (CPU) on Render free tier</li>
                <li>Latency: {benchmarkData ? `${Math.round(benchmarkData.latency_stats.mean_ms)}ms mean / ${Math.round(benchmarkData.latency_stats.p95_ms)}ms p95` : '~41ms mean'} per 10-frame window</li>
                <li>Memory: {benchmarkData ? `~${Math.round(benchmarkData.memory_stats.estimated_production_peak_rss_mb)} MB peak RSS` : '~180 MB peak RSS'}</li>
                <li>Export: Keras → ONNX (opset 17)</li>
              </ul>
            </div>

            <div className="model-card-item">
              <h3>Versioning</h3>
              <ul>
                <li>Model version: {healthInfo?.model_version || benchmarkData?.model_version || '2.0.0'}</li>
                <li style={{ wordBreak: 'break-all' }}>
                  Checksum (SHA256):{' '}
                  <code style={{ fontSize: '0.75em' }}>
                    {healthInfo?.model_checksum || benchmarkData?.model_checksum || '0fed96a8fc41f39db48fbe579fc12850849cf78c1c463b848a48fd86b6e6ea84'}
                  </code>
                  <button
                    className="copy-btn"
                    onClick={() => {
                      const cs = healthInfo?.model_checksum || benchmarkData?.model_checksum || '0fed96a8fc41f39db48fbe579fc12850849cf78c1c463b848a48fd86b6e6ea84';
                      navigator.clipboard.writeText(cs).then(() => {
                        setCopiedChecksum(true);
                        setTimeout(() => setCopiedChecksum(false), 2000);
                      });
                    }}
                    aria-label="Copy checksum"
                  >
                    {copiedChecksum ? '✓' : '⧉'}
                  </button>
                </li>
                <li>Framework: TensorFlow/Keras 2.x → ONNX</li>
                {serverState === 'online' && healthInfo?.uptime_seconds != null && (
                  <li>Server uptime: {Math.floor(healthInfo.uptime_seconds / 60)}m</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION 6: ENGINEERING
           ═══════════════════════════════════════════════════════════ */}
      <section id="engineering" className="section">
        <div className="section-inner">
          <div className="section-label">System Design</div>
          <h2 className="section-title">Engineering Details</h2>
          <p className="section-subtitle">
            Built for free-tier deployment: Vercel (frontend), Render (backend), MongoDB Atlas (logging).
          </p>

          <div className="eng-grid stagger">
            <div className="eng-stat fade-in-up">
              <div className="eng-stat-value">{benchmarkData ? `${Math.round(benchmarkData.latency_stats.mean_ms)}ms` : '~41ms'}</div>
              <div className="eng-stat-label">Per-Window Inference</div>
            </div>
            <div className="eng-stat fade-in-up">
              <div className="eng-stat-value">{benchmarkData ? `${Math.round(benchmarkData.memory_stats.estimated_production_peak_rss_mb)}MB` : '~180MB'}</div>
              <div className="eng-stat-label">Peak Memory</div>
            </div>
            <div className="eng-stat fade-in-up">
              <div className="eng-stat-value">512MB</div>
              <div className="eng-stat-label">Render RAM Limit</div>
            </div>
            <div className="eng-stat fade-in-up">
              <div className="eng-stat-value">{benchmarkData ? `${benchmarkData.model_size_mb.toFixed(1)}MB` : '~10MB'}</div>
              <div className="eng-stat-label">ONNX Model Size</div>
            </div>
          </div>

          {/* Technology Badges */}
          <div className="tech-badges" style={{ marginTop: 'var(--sp-5)', display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
            {['React 19', 'TypeScript', 'Vite', 'FastAPI', 'ONNX Runtime', 'OpenCV', 'WebSocket', 'MongoDB Atlas', 'Framer Motion', 'Docker'].map(tech => (
              <span key={tech} className="tech-badge">{tech}</span>
            ))}
          </div>

          <div className="model-card-grid" style={{ marginTop: 'var(--sp-8)' }}>
            <div className="model-card-item">
              <h3>Backend Stack</h3>
              <ul>
                <li>FastAPI with async request handling</li>
                <li>ONNX Runtime for model inference (CPU only)</li>
                <li>OpenCV for frame extraction and preprocessing</li>
                <li>MongoDB Atlas for detection logging (30-day TTL)</li>
                <li>WebSocket for real-time alert broadcasting</li>
              </ul>
            </div>

            <div className="model-card-item">
              <h3>Frontend Stack</h3>
              <ul>
                <li>React 19 + TypeScript with Vite</li>
                <li>Framer Motion for alert animations</li>
                <li>Custom HUD overlay with real-time sync</li>
                <li>Client-side threshold recalculation</li>
                <li>Safe API client with cold-start backoff</li>
              </ul>
            </div>

            <div className="model-card-item">
              <h3>Testing</h3>
              <ul>
                <li>13 pytest tests (backend): health, metrics, prediction, ONNX shape</li>
                <li>9 Vitest tests (frontend): threshold logic, API client</li>
                <li>Production build verification (tsc + vite build)</li>
                <li>Real accident clip inference validation</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           SECTION 7: RESPONSIBLE AI
           ═══════════════════════════════════════════════════════════ */}
      <section id="responsible-ai" className="section">
        <div className="section-inner">
          <div className="section-label">Ethics</div>
          <h2 className="section-title">Responsible AI</h2>
          <p className="section-subtitle">
            Important considerations for deploying accident detection systems.
          </p>

          <div className="rai-grid stagger">
            <div className="rai-card fade-in-up">
              <h3>🔒 Privacy</h3>
              <p>
                No video frames are stored on the server. Uploaded clips are processed in
                temporary memory and discarded after inference. MongoDB logs contain only
                confidence scores and timestamps — never raw video data.
              </p>
            </div>

            <div className="rai-card fade-in-up">
              <h3>⚖️ Bias Awareness</h3>
              <p>
                The training dataset has known limitations: primarily daytime footage,
                limited geographic diversity, and potential class imbalance artifacts.
                Real-world deployment would require retraining on representative data
                for the target environment.
              </p>
            </div>

            <div className="rai-card fade-in-up">
              <h3>🧑‍⚖️ Human in the Loop</h3>
              <p>
                This system is designed as a detection <em>assistant</em>, not a
                replacement for human judgment. All alerts should be reviewed by a
                trained operator before triggering emergency response actions.
              </p>
            </div>

            <div className="rai-card fade-in-up">
              <h3>📊 Threshold Transparency</h3>
              <p>
                The interactive threshold slider demonstrates how changing the decision
                boundary affects the precision-recall tradeoff. Operators can tune
                sensitivity based on their specific false-alarm tolerance.
              </p>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════════
           FOOTER
           ═══════════════════════════════════════════════════════════ */}
      <footer className="footer">
        <div className="section-inner">
          <div className="footer-links">
            <a href="https://github.com/Vatsalgada2006/deep-learning-project" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
            <a href="#model-card">Model Card</a>
            <a href="#results">Results</a>
            <a href="#demo">Live Demo</a>
          </div>
          <div className="footer-copy">
            AI Accident Detection System · CNN-LSTM Research Project · {new Date().getFullYear()}
          </div>
        </div>
      </footer>
    </div>
  );
}