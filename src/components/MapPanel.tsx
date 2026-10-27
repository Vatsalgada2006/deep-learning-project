import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { type Incident } from '../utils/incidents';

export interface CameraInfo {
  id: string;
  name: string;
  label: string;
  location_label: string;
  is_simulated: boolean;
  lat: number;
  lon: number;
  fps_nominal: number;
  status: string;
}

interface MapPanelProps {
  cameras: CameraInfo[];
  incidents: Incident[];
  selectedCameraId?: string;
  onSelectCamera?: (cameraId: string) => void;
  onSelectIncident?: (incident: Incident) => void;
  onUpdateIncidentStatus?: (id: string, status: 'confirmed' | 'false_alarm') => void;
}

export const MapPanel: React.FC<MapPanelProps> = ({
  cameras,
  incidents,
  selectedCameraId,
  onSelectCamera,
  onSelectIncident,
  onUpdateIncidentStatus,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);

  const [tileError, setTileError] = useState(false);
  const [hasLoadedTile, setHasLoadedTile] = useState(false);

  // Initialize Leaflet map centred on Mumbai (19.0760, 72.8777), zoom 11
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    const mumbaiLat = 19.0760;
    const mumbaiLon = 72.8777;

    const map = L.map(mapContainerRef.current, {
      center: [mumbaiLat, mumbaiLon],
      zoom: 11,
      zoomControl: true,
      attributionControl: true,
    });

    const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    });

    tiles.on('tileerror', () => {
      setTileError(true);
    });

    tiles.on('tileload', () => {
      setHasLoadedTile(true);
      setTileError(false);
    });

    tiles.addTo(map);
    tileLayerRef.current = tiles;

    markersLayerRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  const handleRetryTiles = () => {
    setTileError(false);
    if (tileLayerRef.current) {
      tileLayerRef.current.redraw();
    }
  };

  // Update camera and incident markers
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layer = markersLayerRef.current;
    if (!map || !layer) return;

    layer.clearLayers();

    // 1. Add camera markers (Simulated demo cameras in Mumbai)
    cameras.forEach((cam) => {
      const isSelected = cam.id === selectedCameraId;
      const cameraHtml = `
        <div class="custom-cam-marker ${isSelected ? 'selected' : ''}" title="${cam.name}">
          <div class="cam-icon">📹</div>
          <div class="cam-label">${cam.id.replace('cam-mum-', 'M')}</div>
        </div>
      `;

      const icon = L.divIcon({
        html: cameraHtml,
        className: 'cam-div-icon',
        iconSize: [36, 36],
        iconAnchor: [18, 18],
      });

      const marker = L.marker([cam.lat, cam.lon], { icon }).addTo(layer);
      marker.bindPopup(`
        <div class="map-popup-card">
          <div class="popup-title">${cam.name}</div>
          <div class="popup-sub">${cam.label}</div>
          <div class="popup-tag-simulated">Simulated demo cameras, not real CCTV locations</div>
          <div class="popup-row">Status: <b style="color:var(--neon-green)">${cam.status.toUpperCase()}</b> • ${cam.fps_nominal} FPS</div>
        </div>
      `);

      marker.on('click', () => {
        onSelectCamera?.(cam.id);
      });
    });

    // 2. Add incident markers (matched to cameras)
    incidents.forEach((inc) => {
      const matchedCam = cameras.find((c) => c.id === inc.camera_id);
      if (!matchedCam) return;

      const isNew = inc.status === 'new';
      const isConfirmed = inc.status === 'confirmed';
      const markerColor = isConfirmed ? '#ff1744' : isNew ? '#ff9100' : '#718096';

      const incidentHtml = `
        <div class="custom-incident-marker ${isNew ? 'pulse-anim' : ''}" style="background-color: ${markerColor};" title="Incident on ${matchedCam.name}">
          ⚠️
        </div>
      `;

      const icon = L.divIcon({
        html: incidentHtml,
        className: 'incident-div-icon',
        iconSize: [30, 30],
        iconAnchor: [15, 15],
      });

      // Slight lat offset so camera and incident icons don't completely overlap
      const offsetLat = matchedCam.lat + 0.0016;
      const marker = L.marker([offsetLat, matchedCam.lon], { icon }).addTo(layer);

      const statusBadge =
        inc.status === 'confirmed'
          ? '<span class="status-badge badge-confirmed">CONFIRMED</span>'
          : inc.status === 'false_alarm'
          ? '<span class="status-badge badge-false">FALSE ALARM</span>'
          : '<span class="status-badge badge-new">NEW INCIDENT</span>';

      const timeStr = inc.detected_at ? inc.detected_at.replace('T', ' ').substring(0, 19) + ' UTC' : 'Live stream';

      const popupHtml = `
        <div class="map-popup-card incident-popup">
          <div class="popup-header-row">
            <span class="popup-incident-id">INCIDENT ${inc.id.substring(0, 10)}</span>
            ${statusBadge}
          </div>
          <div class="popup-camera-name">📹 ${matchedCam.name}</div>
          <div class="popup-stat-grid">
            <div class="popup-stat-item">
              <span class="stat-lbl">Time</span>
              <span class="stat-val">${timeStr}</span>
            </div>
            <div class="popup-stat-item">
              <span class="stat-lbl">Peak Model Score</span>
              <span class="stat-val score-highlight">${(inc.peak_confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
          <div class="popup-actions">
            <button class="popup-btn popup-view-btn" id="btn-view-${inc.id}">View Incident</button>
            <button class="popup-btn popup-confirm-btn" id="btn-confirm-${inc.id}">Confirm</button>
            <button class="popup-btn popup-false-alarm-btn" id="btn-false-${inc.id}">False Alarm</button>
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml, { maxWidth: 300 });

      marker.on('popupopen', () => {
        const popupEl = marker.getPopup()?.getElement();
        if (!popupEl) return;

        const viewBtn = popupEl.querySelector(`#btn-view-${inc.id}`);
        const confirmBtn = popupEl.querySelector(`#btn-confirm-${inc.id}`);
        const falseAlarmBtn = popupEl.querySelector(`#btn-false-${inc.id}`);

        if (viewBtn) {
          viewBtn.addEventListener('click', () => {
            onSelectIncident?.(inc);
            const feedEl = document.getElementById('incidents');
            feedEl?.scrollIntoView({ behavior: 'smooth' });
          });
        }

        if (confirmBtn) {
          confirmBtn.addEventListener('click', () => {
            onUpdateIncidentStatus?.(inc.id, 'confirmed');
            marker.closePopup();
          });
        }

        if (falseAlarmBtn) {
          falseAlarmBtn.addEventListener('click', () => {
            onUpdateIncidentStatus?.(inc.id, 'false_alarm');
            marker.closePopup();
          });
        }
      });

      marker.on('click', () => {
        onSelectIncident?.(inc);
      });
    });
  }, [cameras, incidents, selectedCameraId, onSelectCamera, onSelectIncident, onUpdateIncidentStatus]);

  return (
    <div className="map-panel-wrapper">
      <div className="map-header">
        <div className="map-title-row">
          <span className="map-icon">🗺️</span>
          <span className="map-title">MUMBAI SURVEILLANCE RADAR & CAMERA GRID</span>
        </div>
        <div className="map-disclaimer">
          ⚠️ Simulated demo cameras, not real CCTV locations
        </div>
      </div>

      <div className="map-viewport-wrapper" style={{ position: 'relative' }}>
        <div ref={mapContainerRef} className="leaflet-map-container" />

        {/* Friendly fallback when tiles fail to load */}
        {tileError && !hasLoadedTile && (
          <div className="map-tile-fallback-overlay">
            <div className="fallback-card">
              <div className="fallback-icon">📡</div>
              <h4>Map Tiles Offline / Unavailable</h4>
              <p>
                OpenStreetMap tiles could not be loaded (network offline, restricted proxy, or tile server unreachable).
                Camera telemetry and spatial incident markers remain fully active on coordinates.
              </p>
              <button className="btn btn-outline" onClick={handleRetryTiles} style={{ marginTop: '8px', fontSize: '11px', padding: '6px 14px' }}>
                ↻ Retry Loading Tiles
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
