xyz;
import { Canvas } from '@react-three/fiber';
import { Box, OrbitControls, Plane, Text, spring } from '@react-three/drei';
import { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const CameraNode = ({ position, isActive, cameraId }: {
  position: [number, number, number];
  isActive: boolean;
  cameraId: string
}) => {
  const meshRef = useRef(null);
  const [pulse, setPulse] = useState(false);

  // Pulse effect when accident detected
  useEffect(() => {
    if (isActive) {
      setPulse(true);
      const timeout = setTimeout(() => setPulse(false), 1500);
      return () => clearTimeout(timeout);
    }
  }, [isActive]);

  const scale = pulse ? 1.3 : 1.0;
  const color = isActive ? '#ff0000' : '#00ff88';

  return (
    <group>
      <mesh
        ref={meshRef}
        position={position}
        scale={scale}
        onPointerEnter={() => setPulse(true)}
        onPointerLeave={() => setPulse(false)}
      >
        <boxGeometry args={[0.6, 0.6, 0.6]} />
        <meshStandardMaterial color={color} emissiveIntensity={isActive ? 0.5 : 0} />
      </mesh>
      <Text
        text={cameraId}
        position={[0, 0.8, 0]}
        alignment="center"
        color={color}
      />
    </group>
  );
};

const AlertPanel = ({ alert }: {
  alert: { timestamp: string; confidence: string; cameraId: string; latencyMs?: number } | null
}) => {
  if (!alert) return null;

  return (
    <motion.div
      className="fixed bottom-4 right-4 bg-red-900 bg-openess-90 text-white p-4 rounded-lg shadow-lg transform transition-all duration-500"
      initial={{ x: 100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 100, opacity: 0 }}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
        </div>
        <div>
          <h3 className="font-bold text-red-300">ACCIDENT DETECTED</h3>
          <p className="text-sm">{alert.cameraId}</p>
          <p className="text-xs text-red-200">{alert.timestamp}</p>
          <p className="font-medium">{alert.confidence}</p>
          {alert.latencyMs !== undefined && (
            <p className="text-xs text-blue-200 mt-1">
              Latency: {alert.latencyMs}ms
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

function App() {
  console.log('App component rendered');
  const [alerts, setAlerts] = useState<Array<{timestamp: string; confidence: string; cameraId: string; latencyMs?: number}>>([]);
  const [isConnected, setIsConnected] = useState(false);
  const cameraPositions = useRef([
    [-5, 0, -3], // Camera 1
    [0, 0, -5],  // Camera 2
    [5, 0, -3],  // Camera 3
    [-3, 0, 0],  // Camera 4
    [3, 0, 0],   // Camera 5
  ]);

  // WebSocket connection for real-time alerts
  useEffect(() => {
    console.log('WebSocket useEffect hook called');
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
    const wsProtocol = backendUrl.startsWith('https://') ? 'wss://' : 'ws://';
    const wsUrl = `${wsProtocol}${backendUrl.replace(/^https?:\/\//, '')}/ws/alerts`;
    const ws = new WebSocket(wsUrl);
    console.log(`WebSocket connecting to ${wsUrl}`);
    let isConnected = false;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      isConnected = true;
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      isConnected = false;
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
      isConnected = false;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Expecting alert data from backend: {type, timestamp, confidence, camera_id, message}
        if (data.type === 'accident_detected') {
          const detectionTime = new Date(data.timestamp).getTime();
          const receiptTime = new Date().getTime();
          const latencyMs = receiptTime - detectionTime;

          const newAlert = {
            timestamp: data.timestamp,
            confidence: `${(parseFloat(data.confidence) * 100).toFixed(0)}%`, // Convert to percentage string
            cameraId: data.camera_id,
            latencyMs: latencyMs // Store latency in the alert object for display
          };
          setAlerts(prev => [newAlert, ...prev.slice(0, 4)]); // Keep max 5 alerts

          // Send latency report back to backend via WebSocket
          if (ws.readyState === WebSocket.OPEN) {
            const latencyReport = {
              type: 'latency_report',
              latency_ms: latencyMs,
              timestamp: data.timestamp
            };
            ws.send(JSON.stringify(latencyReport));
          }
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    // Cleanup on unmount
    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="relative h-screen w-screen bg-gray-900 overflow-hidden">
      {/* 3D Scene */}
      <div className="absolute inset-0">
        <Canvas className="absolute inset-0" camera={{ position: [0, 8, 12], fov: 45 }} shadows>
          <ambientLight intensity={0.3} />
          <directionalLight position={[10, 15, 10]} intensity={1.2} castShadow />

          {/* Ground plane */}
          <mesh receiveShadow>
            <planeGeometry args={[20, 20]} />
            <meshStandardMaterial color="#2d3748" />
          </mesh>

          {/* Camera nodes */}
          {cameraPositions.current.map((pos, index) => {
            // Check if this camera has an active alert
            const activeAlert = alerts.some(alert => alert.cameraId === `cam_${String(index + 1).padStart(3, '0')}`);
            return (
              <CameraNode
                key={index}
                position={pos}
                isActive={activeAlert}
                cameraId={`cam_${String(index + 1).padStart(3, '0')}`}
              />
            );
          })}

          <OrbitControls
            enablePan={false}
            enableZoom={false}
            maxDistance={20}
            minDistance={5}
          />
        </Canvas>
      </div>

      {/* UI Overlay */}
      <div className="absolute inset-0 flex flex-col items-center p-4 pointer-events-none">
        {/* Status bar */}
        <div className="w-full mb-4 flex justify-between items-center">
          <div className="flex items-center space-x-3 text-green-400 text-sm">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>{isConnected ? 'System Online' : 'System Offline'}</span>
          </div>
          <div className="text-xs text-gray-500">
            {new Date().toLocaleTimeString()}
          </div>
        </div>

        {/* Alerts container */}
        <div className="w-full max-w-2xl">
          {alerts.map((alert, index) => (
            <AlertPanel key={index} alert={alert} />
          ))}
        </div>
      </div>

      {/* Dark overlay for better text visibility */}
      <div className="absolute inset-0 bg-black bg-opacity-30 pointer-events-none"></div>
    </div>
  );
}

export default App;