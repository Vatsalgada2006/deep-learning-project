import { Component, useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';

// --- Types ---
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
  confidence: number;
  is_accident: boolean;
}

interface PredictResult {
  status: string;
  message: string;
  predictions: PredictionWindow[];
  inference_time_ms: number;
  frames_processed: number;
  peak_memory_mb?: number | null;
}

interface ServerHealth {
  status: string;
  model_loaded: boolean;
  mongodb_connected: boolean;
  demo_mode: boolean;
  timestamp: string;
  websocket_connections: number;
}

// --- Configuration ---
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const RAW_WS = import.meta.env.VITE_WS_URL || '';
const WS_URL = RAW_WS
  ? RAW_WS
  : `${API_BASE.replace(/^http/, 'ws')}/ws/alerts`;

// --- Error Boundary for 3D Scene ---
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback: ReactNode;
}
interface ErrorBoundaryState {
  hasError: boolean;
}
class SceneErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }
  componentDidCatch(error: unknown) {
    console.warn('WebGL / ThreeJS scene encountered an error, falling back to 2D mode:', error);
  }
  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// --- 3D Camera Node ---
const CameraNode = ({
  position,
  isActive,
  cameraId,
}: {
  position: [number, number, number];
  isActive: boolean;
  cameraId: string;
}) => {
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    if (isActive) {
      setPulse(true);
      const timer = setTimeout(() => setPulse(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [isActive]);

  const scale = pulse ? 1.3 : 1.0;
  const color = isActive ? '#ef4444' : '#10b981';

  return (
    <group position={position}>
      <mesh
        scale={[scale, scale, scale]}
        onPointerEnter={() => setPulse(true)}
        onPointerLeave={() => setPulse(false)}
      >
        <boxGeometry args={[0.7, 0.7, 0.7]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActive ? 0.6 : 0.15}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>
      <Text
        position={[0, 0.85, 0]}
        color={color}
        fontSize={0.32}
        anchorX="center"
        anchorY="middle"
      >
        {cameraId}
      </Text>
    </group>
  );
};

export default function App() {
  // --- States ---
  const [serverState, setServerState] = useState<'waking' | 'online' | 'offline'>('waking');
  const [healthInfo, setHealthInfo] = useState<ServerHealth | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [alerts, setAlerts] = useState<CameraAlert[]>([]);
  const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d');
  const [activeTab, setActiveTab] = useState<'live' | 'upload' | 'history' | 'metrics'>('live');

  // Upload & Inference
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null);

  // History
  const [historyItems, setHistoryItems] = useState<any[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Cameras positions
  const cameraNodes: { id: string; pos: [number, number, number] }[] = [
    { id: 'cam_001', pos: [-5, 0, -3] },
    { id: 'cam_002', pos: [0, 0, -5] },
    { id: 'cam_003', pos: [5, 0, -3] },
    { id: 'cam_004', pos: [-3, 0, 0] },
    { id: 'cam_005', pos: [3, 0, 0] },
  ];

  // --- 1. Server Wake-up & Health Polling ---
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/health`, { method: 'GET', signal: AbortSignal.timeout(6000) });
      if (res.ok) {
        const data: ServerHealth = await res.json();
        setHealthInfo(data);
        setServerState('online');
        return true;
      }
    } catch {
      // Server is likely cold-starting on Render free tier
      setServerState('waking');
    }
    return false;
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let delay = 1500;
    let isCancelled = false;

    const poll = async () => {
      const ok = await checkHealth();
      if (!isCancelled) {
        if (!ok) {
          // Exponential backoff up to 8s
          delay = Math.min(delay * 1.5, 8000);
          timer = setTimeout(poll, delay);
        } else {
          // Regular heartbeat poll every 25s
          timer = setTimeout(poll, 25000);
        }
      }
    };

    poll();

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [checkHealth]);

  // --- 2. WebSocket with Auto-Reconnect ---
  useEffect(() => {
    if (serverState !== 'online') return;

    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let backoff = 2000;
    let isCancelled = false;

    const connectWs = () => {
      try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
          setWsConnected(true);
          backoff = 2000;
        };

        ws.onclose = () => {
          setWsConnected(false);
          if (!isCancelled) {
            backoff = Math.min(backoff * 1.6, 12000);
            reconnectTimeout = setTimeout(connectWs, backoff);
          }
        };

        ws.onerror = () => {
          setWsConnected(false);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'accident_detected') {
              const detectionTime = new Date(data.timestamp).getTime();
              const receiptTime = Date.now();
              const latencyMs = Math.max(0, receiptTime - detectionTime);

              const newAlert: CameraAlert = {
                timestamp: data.timestamp,
                confidence: `${(parseFloat(data.confidence) * 100).toFixed(0)}%`,
                cameraId: data.camera_id || 'cam_001',
                latencyMs,
                message: data.message,
                windowIndex: data.window_index,
              };

              setAlerts((prev) => [newAlert, ...prev.slice(0, 7)]);

              // Echo latency report back
              if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'latency_report', latency_ms: latencyMs }));
              }
            }
          } catch (e) {
            console.error('Error parsing WS message', e);
          }
        };
      } catch (err) {
        console.warn('WS connection failed, retrying...', err);
        if (!isCancelled) {
          reconnectTimeout = setTimeout(connectWs, 5000);
        }
      }
    };

    connectWs();

    return () => {
      isCancelled = true;
      clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, [serverState]);

  // --- 3. Upload Video for Analysis ---
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadError(null);
    setPredictResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => null);
        throw new Error(errJson?.detail || `Upload failed with HTTP ${res.status}`);
      }

      const data: PredictResult = await res.json();
      setPredictResult(data);
    } catch (err: any) {
      setUploadError(err.message || 'Failed to analyze video');
    } finally {
      setIsUploading(false);
    }
  };

  // --- 4. Fetch History ---
  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE}/history?limit=15`);
      if (res.ok) {
        const data = await res.json();
        setHistoryItems(data.detections || []);
      }
    } catch (err) {
      console.warn('Could not load history', err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistory();
    }
  }, [activeTab]);

  return (
    <div className="relative h-screen w-screen bg-slate-950 text-slate-100 flex flex-col font-sans select-none overflow-hidden">
      {/* Top Navigation Bar */}
      <header className="h-14 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-4 flex items-center justify-between z-20 shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/50 flex items-center justify-center text-indigo-400 font-bold text-sm">
            AI
          </div>
          <div>
            <h1 className="font-semibold text-sm leading-tight text-white flex items-center gap-2">
              Road Accident Detection System
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">
                CNN-LSTM ONNX
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">Autonomous Edge Video Surveillance</p>
          </div>
        </div>

        {/* Server & Status Badges */}
        <div className="flex items-center space-x-2">
          {serverState === 'waking' ? (
            <div className="flex items-center space-x-2 px-2.5 py-1 bg-amber-500/10 border border-amber-500/30 text-amber-300 rounded text-xs animate-pulse">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
              <span>Waking Server (cold-start)...</span>
            </div>
          ) : (
            <div className="flex items-center space-x-1.5 px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded text-xs">
              <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-400 animate-pulse' : 'bg-yellow-400'}`} />
              <span>{wsConnected ? 'Live Feed Connected' : 'Server Online (Polling)'}</span>
            </div>
          )}

          {healthInfo?.demo_mode && (
            <span className="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
              Demo Mode
            </span>
          )}

          {/* View Toggle */}
          <div className="border border-slate-700 bg-slate-800/80 rounded flex p-0.5 text-xs">
            <button
              onClick={() => setViewMode('3d')}
              className={`px-2 py-0.5 rounded transition ${viewMode === '3d' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
            >
              3D View
            </button>
            <button
              onClick={() => setViewMode('2d')}
              className={`px-2 py-0.5 rounded transition ${viewMode === '2d' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
            >
              2D Grid
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Side: 3D Scene or 2D Camera Grid */}
        <div className="flex-1 relative bg-slate-950 flex flex-col items-center justify-center overflow-hidden">
          {serverState === 'waking' && (
            <div className="absolute top-4 z-10 px-4 py-2 bg-slate-900/90 border border-amber-500/40 rounded-lg text-amber-200 text-xs shadow-xl backdrop-blur flex items-center space-x-3">
              <div className="w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
              <div>
                <p className="font-medium">Waking up Render free-tier instance...</p>
                <p className="text-[11px] text-slate-400">Spin-up takes ~30–50s after idling. Model & DB will auto-connect.</p>
              </div>
            </div>
          )}

          {viewMode === '3d' ? (
            <SceneErrorBoundary
              fallback={
                <div className="p-6 text-center text-slate-400">
                  <p className="text-amber-400 font-semibold mb-2">WebGL not available</p>
                  <p className="text-xs">Switched to 2D view automatically.</p>
                  <button
                    onClick={() => setViewMode('2d')}
                    className="mt-3 px-3 py-1 bg-indigo-600 text-white rounded text-xs"
                  >
                    View 2D Grid
                  </button>
                </div>
              }
            >
              <Canvas camera={{ position: [0, 8, 12], fov: 45 }} className="w-full h-full">
                <ambientLight intensity={0.4} />
                <directionalLight position={[10, 15, 10]} intensity={1.2} />
                <gridHelper args={[24, 24, '#4f46e5', '#1e293b']} position={[0, -0.01, 0]} />
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
                  <planeGeometry args={[26, 26]} />
                  <meshStandardMaterial color="#0f172a" roughness={0.8} />
                </mesh>

                {cameraNodes.map((node) => {
                  const isActive = alerts.some((a) => a.cameraId === node.id);
                  return (
                    <CameraNode
                      key={node.id}
                      position={node.pos}
                      isActive={isActive}
                      cameraId={node.id}
                    />
                  );
                })}

                <OrbitControls enablePan={false} maxDistance={22} minDistance={4} />
              </Canvas>
            </SceneErrorBoundary>
          ) : (
            /* 2D Camera Matrix View */
            <div className="w-full h-full p-6 grid grid-cols-2 md:grid-cols-3 gap-4 overflow-y-auto">
              {cameraNodes.map((cam) => {
                const activeAlert = alerts.find((a) => a.cameraId === cam.id);
                return (
                  <div
                    key={cam.id}
                    className={`relative rounded-lg border p-4 flex flex-col justify-between transition ${
                      activeAlert
                        ? 'border-red-500 bg-red-950/30 ring-1 ring-red-500 animate-pulse'
                        : 'border-slate-800 bg-slate-900/60'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-xs font-semibold text-slate-300">{cam.id}</span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                          activeAlert ? 'bg-red-500/20 text-red-300' : 'bg-emerald-500/20 text-emerald-300'
                        }`}
                      >
                        {activeAlert ? 'ACCIDENT' : 'MONITORING'}
                      </span>
                    </div>

                    <div className="h-28 bg-slate-950/80 rounded border border-slate-800 flex items-center justify-center text-slate-600 text-xs">
                      {activeAlert ? (
                        <div className="text-center p-2">
                          <p className="text-red-400 font-bold text-sm">ALERT TRIGGERED</p>
                          <p className="text-xs text-red-300 font-mono mt-1">Conf: {activeAlert.confidence}</p>
                          {activeAlert.latencyMs !== undefined && (
                            <p className="text-[10px] text-slate-400">RTT: {activeAlert.latencyMs}ms</p>
                          )}
                        </div>
                      ) : (
                        <span>Simulated Optical Stream</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Quick HUD overlay in 3D view */}
          {viewMode === '3d' && (
            <div className="absolute bottom-3 left-3 bg-slate-900/80 backdrop-blur border border-slate-800 rounded px-3 py-1.5 text-[11px] text-slate-400 flex items-center space-x-3 pointer-events-none">
              <span>Drag to rotate 3D node view</span>
              <span className="text-slate-600">|</span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" /> Normal
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-red-400 inline-block" /> Incident
              </span>
            </div>
          )}
        </div>

        {/* Right Side: Tabbed Management Panel */}
        <aside className="w-96 border-l border-slate-800 bg-slate-900/95 flex flex-col shrink-0">
          {/* Tab Navigation */}
          <div className="grid grid-cols-4 border-b border-slate-800 text-xs text-slate-400">
            <button
              onClick={() => setActiveTab('live')}
              className={`py-2.5 font-medium border-b-2 ${
                activeTab === 'live'
                  ? 'border-indigo-500 text-white bg-slate-800/40'
                  : 'border-transparent hover:text-slate-200'
              }`}
            >
              Live Alerts
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-2.5 font-medium border-b-2 ${
                activeTab === 'upload'
                  ? 'border-indigo-500 text-white bg-slate-800/40'
                  : 'border-transparent hover:text-slate-200'
              }`}
            >
              Analyze Video
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`py-2.5 font-medium border-b-2 ${
                activeTab === 'history'
                  ? 'border-indigo-500 text-white bg-slate-800/40'
                  : 'border-transparent hover:text-slate-200'
              }`}
            >
              History
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`py-2.5 font-medium border-b-2 ${
                activeTab === 'metrics'
                  ? 'border-indigo-500 text-white bg-slate-800/40'
                  : 'border-transparent hover:text-slate-200'
              }`}
            >
              Metrics
            </button>
          </div>

          {/* Tab Body */}
          <div className="flex-1 overflow-y-auto p-4">
            {/* TAB 1: LIVE ALERTS */}
            {activeTab === 'live' && (
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
                  <span>Active WebSocket Notifications</span>
                  {alerts.length > 0 && (
                    <button
                      onClick={() => setAlerts([])}
                      className="text-indigo-400 hover:text-indigo-300 text-[11px]"
                    >
                      Clear
                    </button>
                  )}
                </div>

                <AnimatePresence>
                  {alerts.length === 0 ? (
                    <div className="text-center py-12 text-slate-500 text-xs">
                      <p>No active incidents detected.</p>
                      <p className="text-[11px] mt-1 text-slate-600">
                        Upload a video clip or monitor live stream to test detection.
                      </p>
                    </div>
                  ) : (
                    alerts.map((alert, idx) => (
                      <motion.div
                        key={`${alert.timestamp}-${idx}`}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="bg-red-950/40 border border-red-500/40 rounded-lg p-3 text-xs"
                      >
                        <div className="flex justify-between items-start mb-1">
                          <span className="font-bold text-red-300 flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                            CRASH DETECTED
                          </span>
                          <span className="font-mono text-red-200 text-[11px] font-semibold">
                            {alert.confidence}
                          </span>
                        </div>
                        <p className="text-slate-300 text-[11px] mb-1">{alert.message || alert.cameraId}</p>
                        <div className="flex justify-between items-center text-[10px] text-slate-500">
                          <span>{new Date(alert.timestamp).toLocaleTimeString()}</span>
                          {alert.latencyMs !== undefined && (
                            <span className="text-indigo-300">RTT: {alert.latencyMs}ms</span>
                          )}
                        </div>
                      </motion.div>
                    ))
                  )}
                </AnimatePresence>
              </div>
            )}

            {/* TAB 2: UPLOAD & INFERENCE */}
            {activeTab === 'upload' && (
              <div className="space-y-4">
                <div>
                  <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wide">
                    Run ONNX Video Inference
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Upload an MP4/WebM clip (≤10 MB, ≤15s). The CNN-LSTM model extracts sequence frames and predicts accident probabilities.
                  </p>
                </div>

                <form onSubmit={handleUpload} className="space-y-3">
                  <div className="border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-4 text-center cursor-pointer transition">
                    <input
                      type="file"
                      accept="video/mp4,video/webm,video/x-msvideo"
                      onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                      className="hidden"
                      id="video-upload-input"
                    />
                    <label htmlFor="video-upload-input" className="cursor-pointer block">
                      <div className="text-indigo-400 text-sm font-medium">
                        {selectedFile ? selectedFile.name : 'Select or drop video clip'}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1">
                        {selectedFile
                          ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
                          : 'Max 10MB • Max 15 seconds'}
                      </div>
                    </label>
                  </div>

                  <button
                    type="submit"
                    disabled={!selectedFile || isUploading}
                    className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded font-medium text-xs transition flex items-center justify-center space-x-2 shadow"
                  >
                    {isUploading ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Running Inference...</span>
                      </>
                    ) : (
                      <span>Upload & Detect</span>
                    )}
                  </button>
                </form>

                {uploadError && (
                  <div className="p-3 bg-red-950/60 border border-red-500/50 rounded text-red-300 text-xs">
                    {uploadError}
                  </div>
                )}

                {predictResult && (
                  <div className="space-y-3 pt-2 border-t border-slate-800">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-200">Analysis Summary</span>
                      <span className="text-[11px] text-indigo-400 font-mono">
                        {predictResult.inference_time_ms} ms
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <div className="p-2 bg-slate-800/60 rounded border border-slate-700/60">
                        <span className="text-slate-400 block text-[10px]">Sampled Frames</span>
                        <span className="font-semibold">{predictResult.frames_processed}</span>
                      </div>
                      <div className="p-2 bg-slate-800/60 rounded border border-slate-700/60">
                        <span className="text-slate-400 block text-[10px]">Peak Memory</span>
                        <span className="font-semibold font-mono">
                          {predictResult.peak_memory_mb ? `${predictResult.peak_memory_mb} MB` : 'N/A'}
                        </span>
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <span className="text-[10px] font-semibold text-slate-400 uppercase">Sliding Windows</span>
                      {predictResult.predictions.map((pred) => (
                        <div
                          key={pred.window_index}
                          className={`p-2 rounded border text-xs flex justify-between items-center ${
                            pred.is_accident
                              ? 'border-red-500/60 bg-red-950/30'
                              : 'border-slate-800 bg-slate-800/30'
                          }`}
                        >
                          <div>
                            <span className="font-mono text-[11px]">
                              Window #{pred.window_index} (Frames {pred.start_frame}–{pred.end_frame})
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span
                              className={`text-[10px] px-1.5 py-0.5 rounded font-bold font-mono ${
                                pred.is_accident
                                  ? 'bg-red-500 text-white'
                                  : 'bg-emerald-500/20 text-emerald-300'
                              }`}
                            >
                              {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 3: HISTORY */}
            {activeTab === 'history' && (
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
                  <span>Logged Incidents (Atlas M0)</span>
                  <button
                    onClick={fetchHistory}
                    disabled={isLoadingHistory}
                    className="text-indigo-400 hover:text-indigo-300 text-[11px]"
                  >
                    Refresh
                  </button>
                </div>

                {isLoadingHistory ? (
                  <div className="text-center py-8 text-slate-500 text-xs">
                    Loading records from database...
                  </div>
                ) : historyItems.length === 0 ? (
                  <div className="text-center py-12 text-slate-500 text-xs">
                    <p>No recorded incidents found.</p>
                    <p className="text-[11px] mt-1 text-slate-600">
                      Logged accidents will appear here and persist with a 30-day TTL index.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {historyItems.map((item, idx) => (
                      <div
                        key={idx}
                        className="bg-slate-800/40 border border-slate-700/60 rounded p-2.5 text-xs space-y-1"
                      >
                        <div className="flex justify-between font-mono text-[11px]">
                          <span className="text-slate-300">{item.camera_id || 'upload'}</span>
                          <span className="text-red-400 font-bold">
                            {(item.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-slate-400 text-[11px]">{item.message}</p>
                        <p className="text-slate-600 text-[10px]">
                          {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* TAB 4: METRICS */}
            {activeTab === 'metrics' && (
              <div className="space-y-4 text-xs">
                <div>
                  <h3 className="font-semibold text-slate-200 uppercase tracking-wide">
                    Model Architecture & Benchmarks
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Evaluating the temporal CNN-LSTM model against the static spatial CNN baseline.
                  </p>
                </div>

                <div className="overflow-x-auto border border-slate-800 rounded">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-slate-800 text-slate-300 font-semibold">
                      <tr>
                        <th className="p-2">Metric</th>
                        <th className="p-2">Baseline CNN</th>
                        <th className="p-2 text-indigo-400">CNN-LSTM</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      <tr>
                        <td className="p-2 text-slate-400">Accuracy</td>
                        <td className="p-2 font-mono">62.3%</td>
                        <td className="p-2 font-mono text-emerald-400 font-bold">80.0%</td>
                      </tr>
                      <tr>
                        <td className="p-2 text-slate-400">Precision</td>
                        <td className="p-2 font-mono text-emerald-400">87.8%</td>
                        <td className="p-2 font-mono">80.7%</td>
                      </tr>
                      <tr>
                        <td className="p-2 text-slate-400">Recall</td>
                        <td className="p-2 font-mono">54.8%</td>
                        <td className="p-2 font-mono text-emerald-400 font-bold">94.5%</td>
                      </tr>
                      <tr>
                        <td className="p-2 text-slate-400">F1-Score</td>
                        <td className="p-2 font-mono">67.5%</td>
                        <td className="p-2 font-mono text-emerald-400 font-bold">87.1%</td>
                      </tr>
                      <tr>
                        <td className="p-2 text-slate-400">ROC-AUC</td>
                        <td className="p-2 font-mono">0.8207</td>
                        <td className="p-2 font-mono text-emerald-400">0.8220</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="p-3 bg-slate-800/40 border border-slate-700/60 rounded text-[11px] text-slate-400 space-y-1">
                  <p className="font-semibold text-slate-300">Technical Takeaway</p>
                  <p>
                    Road accidents are inherently temporal sequences rather than isolated frames. The CNN-LSTM network achieves an exceptional recall of <strong>94.5%</strong>, ensuring nearly zero missed crash occurrences while preserving edge-deployable ONNX efficiency (&lt;10 MB model, ~40 ms inference).
                  </p>
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}