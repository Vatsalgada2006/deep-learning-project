import React, { useEffect, useRef } from 'react';
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
}

export const MapPanel: React.FC<MapPanelProps> = ({
  cameras,
  incidents,
  selectedCameraId,
  onSelectCamera,
  onSelectIncident,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    // Center on default demo coordinate (San Francisco demo cluster)
    const initialLat = 37.7749;
    const initialLon = -122.4194;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLon],
      zoom: 13,
      zoomControl: true,
      attributionControl: true,
    });

    // Dark styled / standard OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    }).addTo(map);

    markersLayerRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update camera and incident markers
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layer = markersLayerRef.current;
    if (!map || !layer) return;

    layer.clearLayers();

    // 1. Add camera markers
    cameras.forEach((cam) => {
      const isSelected = cam.id === selectedCameraId;
      const cameraHtml = `
        <div class="custom-cam-marker ${isSelected ? 'selected' : ''}">
          <div class="cam-icon">📹</div>
          <div class="cam-label">${cam.id}</div>
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
        <div style="font-family: monospace; font-size: 12px; color: #111;">
          <strong>${cam.name}</strong><br/>
          <span style="color: #666;">${cam.label}</span><br/>
          <span style="color: #e53e3e; font-weight: bold;">${cam.location_label}</span><br/>
          Status: <b>${cam.status.toUpperCase()}</b>
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
      const markerColor = isConfirmed ? '#e53e3e' : isNew ? '#ed8936' : '#718096';

      const incidentHtml = `
        <div class="custom-incident-marker ${isNew ? 'pulse-anim' : ''}" style="background-color: ${markerColor}">
          ⚠️
        </div>
      `;

      const icon = L.divIcon({
        html: incidentHtml,
        className: 'incident-div-icon',
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });

      // Slight lat offset so camera and incident icons don't completely overlap
      const offsetLat = matchedCam.lat + 0.0012;
      const marker = L.marker([offsetLat, matchedCam.lon], { icon }).addTo(layer);

      marker.bindPopup(`
        <div style="font-family: monospace; font-size: 12px; color: #111;">
          <strong style="color: #e53e3e;">INCIDENT ${inc.id}</strong><br/>
          Peak Score: <b>${(inc.peak_confidence * 100).toFixed(1)}%</b><br/>
          Status: <b>${inc.status.toUpperCase()}</b><br/>
          Time: ${inc.detected_at.substring(11, 19)} UTC<br/>
          <em>${inc.location_label}</em>
        </div>
      `);

      marker.on('click', () => {
        onSelectIncident?.(inc);
      });
    });
  }, [cameras, incidents, selectedCameraId, onSelectCamera, onSelectIncident]);

  return (
    <div className="map-panel-wrapper">
      <div className="map-header">
        <div className="map-title-row">
          <span className="map-icon">🗺️</span>
          <span className="map-title">SURVEILLANCE RADAR & CAMERA GRID</span>
        </div>
        <div className="map-disclaimer">
          ⚠️ SIMULATED DEMO CAMERAS — Location not available in dataset
        </div>
      </div>
      <div ref={mapContainerRef} className="leaflet-map-container" />
    </div>
  );
};
