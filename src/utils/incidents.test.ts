import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  clusterWindowsClient,
  serializeIncidentsToJSON,
  serializeIncidentsToCSV,
  exportIncidentsToJSON,
  exportIncidentsToCSV,
  type WindowPrediction,
  type Incident,
} from './incidents';

describe('clusterWindowsClient', () => {
  it('returns empty array when predictions is empty or null', () => {
    expect(clusterWindowsClient([])).toEqual([]);
    expect(clusterWindowsClient(null as unknown as WindowPrediction[])).toEqual([]);
  });

  it('returns empty array when no windows are flagged as accidents', () => {
    const windows: WindowPrediction[] = [
      { window_index: 0, start_frame: 0, end_frame: 9, start_time: 0, end_time: 0.5, confidence: 0.1, is_accident: false },
      { window_index: 1, start_frame: 5, end_frame: 14, start_time: 0.25, end_time: 0.75, confidence: 0.2, is_accident: false },
    ];
    expect(clusterWindowsClient(windows)).toEqual([]);
  });

  it('creates an incident for an isolated flagged window when persistenceK=1', () => {
    const windows: WindowPrediction[] = [
      { window_index: 0, start_frame: 0, end_frame: 9, start_time: 0, end_time: 0.5, confidence: 0.1, is_accident: false },
      { window_index: 1, start_frame: 5, end_frame: 14, start_time: 0.25, end_time: 0.75, confidence: 0.88, is_accident: true },
      { window_index: 2, start_frame: 10, end_frame: 19, start_time: 0.5, end_time: 1.0, confidence: 0.2, is_accident: false },
    ];
    const incidents = clusterWindowsClient(windows, {
      cameraId: 'cam-test',
      cameraName: 'Test Camera',
      locationLabel: 'Simulated Lane 1',
      persistenceK: 1,
      cooldownWindows: 2,
    });

    expect(incidents).toHaveLength(1);
    const inc = incidents[0];
    expect(inc.camera_id).toBe('cam-test');
    expect(inc.camera_name).toBe('Test Camera');
    expect(inc.location_label).toBe('Simulated Lane 1');
    expect(inc.peak_confidence).toBe(0.88);
    expect(inc.mean_confidence).toBe(0.88);
    expect(inc.video_start_time).toBe(0.25);
    expect(inc.video_end_time).toBe(0.75);
    expect(inc.window_indices).toEqual([1]);
    expect(inc.status).toBe('new');
    expect(inc.model_version).toBe('2.0.0');
  });

  it('enforces persistenceK >= 2 to suppress transient isolated false spikes', () => {
    const windows: WindowPrediction[] = [
      { window_index: 0, start_frame: 0, end_frame: 9, start_time: 0, end_time: 0.5, confidence: 0.1, is_accident: false },
      { window_index: 1, start_frame: 5, end_frame: 14, start_time: 0.25, end_time: 0.75, confidence: 0.75, is_accident: true },
      { window_index: 2, start_frame: 10, end_frame: 19, start_time: 0.5, end_time: 1.0, confidence: 0.1, is_accident: false },
    ];
    // With persistenceK=2, a single window is ignored
    const resultIgnored = clusterWindowsClient(windows, { persistenceK: 2 });
    expect(resultIgnored).toHaveLength(0);

    // With 2 consecutive flagged windows, it is retained
    const consecutive: WindowPrediction[] = [
      { window_index: 0, start_frame: 0, end_frame: 9, start_time: 0, end_time: 0.5, confidence: 0.85, is_accident: true },
      { window_index: 1, start_frame: 5, end_frame: 14, start_time: 0.25, end_time: 0.75, confidence: 0.95, is_accident: true },
      { window_index: 2, start_frame: 10, end_frame: 19, start_time: 0.5, end_time: 1.0, confidence: 0.2, is_accident: false },
    ];
    const resultConfirmed = clusterWindowsClient(consecutive, { persistenceK: 2 });
    expect(resultConfirmed).toHaveLength(1);
    expect(resultConfirmed[0].peak_confidence).toBe(0.95);
    expect(resultConfirmed[0].mean_confidence).toBe(0.9);
    expect(resultConfirmed[0].window_indices).toEqual([0, 1]);
    expect(resultConfirmed[0].video_start_time).toBe(0);
    expect(resultConfirmed[0].video_end_time).toBe(0.75);
  });

  it('merges consecutive flagged windows within cooldown into ONE incident', () => {
    // Windows 1 and 3 are flagged, separated by window 2 (gap = 2 - 1 = 1 <= cooldown=2)
    const windows: WindowPrediction[] = [
      { window_index: 0, start_frame: 0, end_frame: 9, start_time: 0, end_time: 0.5, confidence: 0.1, is_accident: false },
      { window_index: 1, start_frame: 5, end_frame: 14, start_time: 0.25, end_time: 0.75, confidence: 0.8, is_accident: true },
      { window_index: 2, start_frame: 10, end_frame: 19, start_time: 0.5, end_time: 1.0, confidence: 0.3, is_accident: false },
      { window_index: 3, start_frame: 15, end_frame: 24, start_time: 0.75, end_time: 1.25, confidence: 0.9, is_accident: true },
      { window_index: 4, start_frame: 20, end_frame: 29, start_time: 1.0, end_time: 1.5, confidence: 0.1, is_accident: false },
    ];

    const incidents = clusterWindowsClient(windows, { persistenceK: 1, cooldownWindows: 2 });
    expect(incidents).toHaveLength(1);
    expect(incidents[0].window_indices).toEqual([1, 3]);
    expect(incidents[0].peak_confidence).toBe(0.9);
    expect(incidents[0].video_start_time).toBe(0.25);
    expect(incidents[0].video_end_time).toBe(1.25);
  });

  it('splits into multiple incidents when separation exceeds cooldown', () => {
    // Flagged at index 1 and index 6 (difference = 5 > cooldownWindows + 1 = 3)
    const windows: WindowPrediction[] = [
      { window_index: 1, start_frame: 5, end_frame: 14, start_time: 0.25, end_time: 0.75, confidence: 0.85, is_accident: true },
      { window_index: 2, start_frame: 10, end_frame: 19, start_time: 0.5, end_time: 1.0, confidence: 0.1, is_accident: false },
      { window_index: 3, start_frame: 15, end_frame: 24, start_time: 0.75, end_time: 1.25, confidence: 0.1, is_accident: false },
      { window_index: 4, start_frame: 20, end_frame: 29, start_time: 1.0, end_time: 1.5, confidence: 0.1, is_accident: false },
      { window_index: 5, start_frame: 25, end_frame: 34, start_time: 1.25, end_time: 1.75, confidence: 0.1, is_accident: false },
      { window_index: 6, start_frame: 30, end_frame: 39, start_time: 1.5, end_time: 2.0, confidence: 0.92, is_accident: true },
    ];

    const incidents = clusterWindowsClient(windows, { persistenceK: 1, cooldownWindows: 1 });
    expect(incidents).toHaveLength(2);
    expect(incidents[0].window_indices).toEqual([1]);
    expect(incidents[1].window_indices).toEqual([6]);
  });
});

describe('Incident Exports and Serialization', () => {
  const sampleIncidents: Incident[] = [
    {
      id: 'inc-001',
      camera_id: 'cam-01',
      camera_name: 'Camera 01 "North"',
      detected_at: '2026-10-27T00:00:00Z',
      video_start_time: 0.5,
      video_end_time: 1.75,
      peak_confidence: 0.9421,
      mean_confidence: 0.8911,
      window_indices: [1, 2, 3],
      model_version: '2.0.0',
      status: 'confirmed',
      operator_note: 'Verified collision between vehicle A and B',
      location_label: 'Highway Mile 42',
    },
  ];

  it('correctly serializes incidents to formatted JSON', () => {
    const jsonStr = serializeIncidentsToJSON(sampleIncidents);
    const parsed = JSON.parse(jsonStr);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].id).toBe('inc-001');
    expect(parsed[0].peak_confidence).toBe(0.9421);
  });

  it('correctly serializes incidents to RFC-4180 CSV with escaped quotes', () => {
    const csvStr = serializeIncidentsToCSV(sampleIncidents);
    const lines = csvStr.split('\r\n');
    expect(lines[0]).toBe(
      'ID,Camera ID,Camera Name,Detected At (UTC),Video Start (s),Video End (s),Peak Model Score,Mean Model Score,Window Indices,Status,Operator Note,Location Label'
    );
    expect(lines[1]).toContain('"inc-001"');
    expect(lines[1]).toContain('"Camera 01 ""North"""');
    expect(lines[1]).toContain('"1;2;3"');
    expect(lines[1]).toContain('"confirmed"');
  });

  describe('DOM download trigger tests', () => {
    const originalDocument = (globalThis as any).document;
    const originalWindow = (globalThis as any).window;

    beforeEach(() => {
      const mockElement = {
        href: '',
        download: '',
        click: vi.fn(),
      };
      (globalThis as any).window = {};
      (globalThis as any).document = {
        body: {
          appendChild: vi.fn(),
          removeChild: vi.fn(),
        },
        createElement: vi.fn(() => mockElement),
      };
      (globalThis as any).URL.createObjectURL = vi.fn(() => 'blob:mock-url');
      (globalThis as any).URL.revokeObjectURL = vi.fn();
    });

    afterEach(() => {
      (globalThis as any).document = originalDocument;
      (globalThis as any).window = originalWindow;
    });

    it('triggers JSON export download in browser environment', () => {
      exportIncidentsToJSON(sampleIncidents, 'test.json');
      expect((globalThis as any).document.createElement).toHaveBeenCalledWith('a');
      expect((globalThis as any).document.body.appendChild).toHaveBeenCalled();
      expect((globalThis as any).document.body.removeChild).toHaveBeenCalled();
      expect((globalThis as any).URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });

    it('triggers CSV export download in browser environment', () => {
      exportIncidentsToCSV(sampleIncidents, 'test.csv');
      expect((globalThis as any).document.createElement).toHaveBeenCalledWith('a');
      expect((globalThis as any).document.body.appendChild).toHaveBeenCalled();
      expect((globalThis as any).document.body.removeChild).toHaveBeenCalled();
      expect((globalThis as any).URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });
  });
});

