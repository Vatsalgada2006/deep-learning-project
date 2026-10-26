import React, { useState } from 'react';
import { type Incident, exportIncidentsToJSON, exportIncidentsToCSV } from '../utils/incidents';

interface IncidentFeedProps {
  incidents: Incident[];
  onUpdateIncidentStatus?: (id: string, status: 'confirmed' | 'false_alarm', note?: string) => void;
  selectedIncident?: Incident | null;
  onSelectIncident?: (incident: Incident | null) => void;
}

export const IncidentFeed: React.FC<IncidentFeedProps> = ({
  incidents,
  onUpdateIncidentStatus,
  selectedIncident,
  onSelectIncident,
}) => {
  const [filter, setFilter] = useState<'all' | 'new' | 'confirmed' | 'false_alarm'>('all');
  const [operatorNoteInput, setOperatorNoteInput] = useState('');

  const filtered = incidents.filter((inc) => {
    if (filter === 'all') return true;
    return inc.status === filter;
  });

  const handleOpenDetail = (inc: Incident) => {
    onSelectIncident?.(inc);
    setOperatorNoteInput(inc.operator_note || '');
  };

  const handleSaveNote = () => {
    if (!selectedIncident) return;
    onUpdateIncidentStatus?.(selectedIncident.id, selectedIncident.status === 'new' ? 'confirmed' : selectedIncident.status, operatorNoteInput);
  };

  return (
    <div className="incident-feed-container">
      {/* Feed Header */}
      <div className="incident-feed-header">
        <div className="feed-title-row">
          <span className="feed-icon">🚨</span>
          <span className="feed-title">ACTIVE INCIDENTS & DISPATCH FEED</span>
          <span className="feed-count-badge">{incidents.length} TOTAL</span>
        </div>

        {/* Filter and Export buttons */}
        <div className="feed-actions-row">
          <div className="feed-filters">
            {(['all', 'new', 'confirmed', 'false_alarm'] as const).map((tab) => (
              <button
                key={tab}
                className={`filter-btn ${filter === tab ? 'active' : ''}`}
                onClick={() => setFilter(tab)}
              >
                {tab === 'false_alarm' ? 'FALSE ALARMS' : tab.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="export-actions">
            <button
              className="export-btn"
              onClick={() => exportIncidentsToJSON(filtered)}
              title="Export filtered incidents as JSON"
            >
              📥 JSON
            </button>
            <button
              className="export-btn"
              onClick={() => exportIncidentsToCSV(filtered)}
              title="Export filtered incidents as CSV"
            >
              📥 CSV
            </button>
          </div>
        </div>
      </div>

      {/* Incidents List */}
      <div className="incident-cards-list">
        {filtered.length === 0 ? (
          <div className="no-incidents-msg">
            No incidents found for filter "{filter.toUpperCase()}".
          </div>
        ) : (
          filtered.map((inc) => {
            const isSelected = selectedIncident?.id === inc.id;
            return (
              <div
                key={inc.id}
                className={`incident-card ${inc.status} ${isSelected ? 'selected' : ''}`}
                onClick={() => handleOpenDetail(inc)}
              >
                <div className="card-top">
                  <div className="inc-id-row">
                    <span className="inc-id">{inc.id}</span>
                    <span className={`status-badge ${inc.status}`}>{inc.status.toUpperCase()}</span>
                  </div>
                  <div className="inc-score">
                    <span className="score-label">MODEL SCORE</span>
                    <span className="score-val">{(inc.peak_confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div className="card-mid">
                  <div className="cam-name">📹 {inc.camera_name}</div>
                  <div className="time-range">
                    ⏱️ Video: {inc.video_start_time.toFixed(1)}s – {inc.video_end_time.toFixed(1)}s (
                    {inc.window_indices.length} windows merged)
                  </div>
                  <div className="location-text">📍 {inc.location_label}</div>
                  {inc.operator_note && (
                    <div className="operator-note-preview">
                      📝 <em>{inc.operator_note}</em>
                    </div>
                  )}
                </div>

                <div className="card-actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    className="action-btn confirm"
                    disabled={inc.status === 'confirmed'}
                    onClick={() => onUpdateIncidentStatus?.(inc.id, 'confirmed')}
                  >
                    ✓ Confirm
                  </button>
                  <button
                    className="action-btn false-alarm"
                    disabled={inc.status === 'false_alarm'}
                    onClick={() => onUpdateIncidentStatus?.(inc.id, 'false_alarm')}
                  >
                    ✕ False Alarm
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Incident Detail Drawer */}
      {selectedIncident && (
        <div className="incident-drawer-overlay" onClick={() => onSelectIncident?.(null)}>
          <div className="incident-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-header">
              <div className="drawer-title">
                <h3>INCIDENT AUDIT: {selectedIncident.id}</h3>
                <span className={`status-badge ${selectedIncident.status}`}>
                  {selectedIncident.status.toUpperCase()}
                </span>
              </div>
              <button className="close-drawer-btn" onClick={() => onSelectIncident?.(null)}>
                ✕
              </button>
            </div>

            <div className="drawer-body">
              {/* Snapshot / Grad-CAM */}
              <div className="drawer-section snapshot-section">
                <h4>PEAK FRAME VISUAL AUDIT</h4>
                {selectedIncident.peak_frame_snapshot ? (
                  <div className="snapshot-container">
                    <img
                      src={selectedIncident.peak_frame_snapshot}
                      alt={`Incident ${selectedIncident.id} snapshot`}
                      className="drawer-snapshot-img"
                    />
                    <div className="snapshot-caption">
                      🔍 Model attention (Grad-CAM), not an object detection
                    </div>
                  </div>
                ) : (
                  <div className="no-snapshot-placeholder">
                    No visual snapshot attached. (Uploaded raw video processed without snapshot retention)
                  </div>
                )}
              </div>

              {/* Telemetry Grid */}
              <div className="drawer-section">
                <h4>TELEMETRY & TEMPORAL BOUNDS</h4>
                <div className="telemetry-grid">
                  <div className="telem-item">
                    <span className="telem-k">Peak Model Score</span>
                    <span className="telem-v high">
                      {(selectedIncident.peak_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="telem-item">
                    <span className="telem-k">Mean Window Score</span>
                    <span className="telem-v">
                      {(selectedIncident.mean_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="telem-item">
                    <span className="telem-k">Temporal Range</span>
                    <span className="telem-v">
                      {selectedIncident.video_start_time.toFixed(1)}s – {selectedIncident.video_end_time.toFixed(1)}s
                    </span>
                  </div>
                  <div className="telem-item">
                    <span className="telem-k">Merged Windows</span>
                    <span className="telem-v">{selectedIncident.window_indices.join(', ')}</span>
                  </div>
                  <div className="telem-item">
                    <span className="telem-k">Camera Source</span>
                    <span className="telem-v">{selectedIncident.camera_name}</span>
                  </div>
                  <div className="telem-item">
                    <span className="telem-k">Location Label</span>
                    <span className="telem-v">{selectedIncident.location_label}</span>
                  </div>
                  <div className="telem-item">
                    <span className="telem-k">Model Checksum Version</span>
                    <span className="telem-v">{selectedIncident.model_version}</span>
                  </div>
                  <div className="telem-item">
                    <span className="telem-k">Detected At (UTC)</span>
                    <span className="telem-v">{selectedIncident.detected_at}</span>
                  </div>
                </div>
              </div>

              {/* Operator Note Section */}
              <div className="drawer-section">
                <h4>OPERATOR LOG & ACTION NOTES</h4>
                <textarea
                  className="operator-note-textarea"
                  value={operatorNoteInput}
                  onChange={(e) => setOperatorNoteInput(e.target.value)}
                  placeholder="Enter operator incident review notes (e.g. Verified side collision, dispatched rescue unit, or false alarm due to glare)..."
                  rows={3}
                />
                <div className="drawer-buttons-row">
                  <button className="btn-save-note" onClick={handleSaveNote}>
                    💾 Save Note & Confirm
                  </button>
                  <button
                    className="btn-mark-false"
                    onClick={() => {
                      onUpdateIncidentStatus?.(selectedIncident.id, 'false_alarm', operatorNoteInput);
                    }}
                  >
                    ✕ Mark False Alarm
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
