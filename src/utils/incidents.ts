/**
 * incidents.ts
 * Client-side incident data types, clustering, and export helpers (JSON & CSV).
 */

export interface Incident {
  id: string;
  camera_id: string;
  camera_name: string;
  detected_at: string;
  video_start_time: number;
  video_end_time: number;
  peak_confidence: number;
  mean_confidence: number;
  window_indices: number[];
  peak_frame_snapshot?: string;
  model_version: string;
  status: 'new' | 'confirmed' | 'false_alarm';
  operator_note: string;
  location_label: string;
}

export interface WindowPrediction {
  window_index: number;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  confidence: number;
  is_accident: boolean;
}

export const DEFAULT_DEMO_INCIDENTS: Incident[] = [
  {
    id: 'inc-demo-001',
    camera_id: 'cam-mum-01',
    camera_name: 'Camera 01 - Dadar TT Circle',
    detected_at: '2026-10-27T01:30:00Z',
    video_start_time: 0.0,
    video_end_time: 1.67,
    peak_confidence: 0.8724,
    mean_confidence: 0.2671,
    window_indices: [0, 1],
    peak_frame_snapshot: '/samples/accident_crash_demo.jpg',
    model_version: '2.0.0',
    status: 'confirmed',
    operator_note: 'Verified vehicle collision on lane 2. Dispatched roadside unit.',
    location_label: 'Location: Dadar, Mumbai (Simulated demo cameras, not real CCTV locations)',
  },
  {
    id: 'inc-demo-002',
    camera_id: 'cam-mum-02',
    camera_name: 'Camera 02 - Bandra WEH Flyover',
    detected_at: '2026-10-27T02:05:00Z',
    video_start_time: 0.0,
    video_end_time: 1.67,
    peak_confidence: 0.8977,
    mean_confidence: 0.7336,
    window_indices: [0, 1],
    peak_frame_snapshot: '/samples/video_project.jpg',
    model_version: '2.0.0',
    status: 'new',
    operator_note: '',
    location_label: 'Location: Bandra, Mumbai (Simulated demo cameras, not real CCTV locations)',
  },
];

export function clusterWindowsClient(
  predictions: WindowPrediction[],
  options: {
    cameraId?: string;
    cameraName?: string;
    locationLabel?: string;
    persistenceK?: number;
    cooldownWindows?: number;
    snapshotUrl?: string;
  } = {}
): Incident[] {
  const {
    cameraId = 'cam-01',
    cameraName = 'Camera 01 - Highway Sector 4',
    locationLabel = 'Location: not available in dataset',
    persistenceK = 1,
    cooldownWindows = 2,
    snapshotUrl,
  } = options;

  if (!predictions || predictions.length === 0) return [];

  const incidents: Incident[] = [];
  let currentRun: WindowPrediction[] = [];
  let lastFlaggedIdx = -1000;

  for (const w of predictions) {
    const isFlagged = Boolean(w.is_accident);
    const idx = Number(w.window_index ?? 0);

    if (isFlagged) {
      if (currentRun.length > 0) {
        if (idx - lastFlaggedIdx <= cooldownWindows + 1) {
          currentRun.push(w);
          lastFlaggedIdx = idx;
        } else {
          if (currentRun.length >= persistenceK) {
            incidents.push(buildIncident(currentRun, cameraId, cameraName, locationLabel, snapshotUrl));
          }
          currentRun = [w];
          lastFlaggedIdx = idx;
        }
      } else {
        currentRun = [w];
        lastFlaggedIdx = idx;
      }
    }
  }

  if (currentRun.length >= persistenceK) {
    incidents.push(buildIncident(currentRun, cameraId, cameraName, locationLabel, snapshotUrl));
  }

  return incidents;
}

function buildIncident(
  windows: WindowPrediction[],
  cameraId: string,
  cameraName: string,
  locationLabel: string,
  snapshotUrl?: string
): Incident {
  const confs = windows.map((w) => w.confidence);
  const peak = Math.max(...confs);
  const mean = Math.round((confs.reduce((a, b) => a + b, 0) / confs.length) * 10000) / 10000;
  const startT = Math.min(...windows.map((w) => w.start_time));
  const endT = Math.max(...windows.map((w) => w.end_time));
  const randSuffix = Math.random().toString(36).substring(2, 8);

  return {
    id: `inc-client-${randSuffix}`,
    camera_id: cameraId,
    camera_name: cameraName,
    detected_at: new Date().toISOString(),
    video_start_time: Math.round(startT * 100) / 100,
    video_end_time: Math.round(endT * 100) / 100,
    peak_confidence: Math.round(peak * 10000) / 10000,
    mean_confidence: mean,
    window_indices: windows.map((w) => w.window_index),
    peak_frame_snapshot: snapshotUrl,
    model_version: '2.0.0',
    status: 'new',
    operator_note: '',
    location_label: locationLabel,
  };
}

/**
 * Serializes incidents into a formatted JSON string.
 */
export function serializeIncidentsToJSON(incidents: Incident[]): string {
  return JSON.stringify(incidents, null, 2);
}

/**
 * Serializes incidents into RFC-4180 compliant CSV format.
 */
export function serializeIncidentsToCSV(incidents: Incident[]): string {
  const headers = [
    'ID',
    'Camera ID',
    'Camera Name',
    'Detected At (UTC)',
    'Video Start (s)',
    'Video End (s)',
    'Peak Model Score',
    'Mean Model Score',
    'Window Indices',
    'Status',
    'Operator Note',
    'Location Label',
  ];

  const rows = incidents.map((inc) => [
    `"${inc.id}"`,
    `"${inc.camera_id}"`,
    `"${inc.camera_name.replace(/"/g, '""')}"`,
    `"${inc.detected_at}"`,
    inc.video_start_time,
    inc.video_end_time,
    inc.peak_confidence,
    inc.mean_confidence,
    `"${inc.window_indices.join(';')}"`,
    `"${inc.status}"`,
    `"${(inc.operator_note || '').replace(/"/g, '""')}"`,
    `"${inc.location_label.replace(/"/g, '""')}"`,
  ]);

  return [headers.join(','), ...rows.map((r) => r.join(','))].join('\r\n');
}

/**
 * Downloads formatted JSON array of incidents.
 */
export function exportIncidentsToJSON(incidents: Incident[], filename = 'incidents_export.json'): void {
  const jsonStr = serializeIncidentsToJSON(incidents);
  if (typeof window === 'undefined' || typeof document === 'undefined') return;
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Downloads formatted CSV file of incidents.
 */
export function exportIncidentsToCSV(incidents: Incident[], filename = 'incidents_export.csv'): void {
  const csvContent = serializeIncidentsToCSV(incidents);
  if (typeof window === 'undefined' || typeof document === 'undefined') return;
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

