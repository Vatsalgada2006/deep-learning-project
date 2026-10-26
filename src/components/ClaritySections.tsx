import React, { useState } from 'react';

export const ClaritySections: React.FC = () => {
  const [howToReadOpen, setHowToReadOpen] = useState(false);
  const [whereItFailsOpen, setWhereItFailsOpen] = useState(false);

  return (
    <div className="clarity-sections-wrapper">
      {/* 1. How to Read This Collapsible */}
      <div className="collapsible-card">
        <button
          className="collapsible-header"
          onClick={() => setHowToReadOpen((prev) => !prev)}
          aria-expanded={howToReadOpen}
        >
          <div className="collapsible-title">
            <span className="collapsible-icon">📖</span>
            <span>HOW TO READ THIS — OPERATOR TELEMETRY GUIDE</span>
          </div>
          <span className="chevron-icon">{howToReadOpen ? '▲' : '▼'}</span>
        </button>

        {howToReadOpen && (
          <div className="collapsible-content">
            <div className="guide-grid">
              <div className="guide-card">
                <div className="guide-k">
                  <span className="guide-tag">Source</span>
                  <span className="tooltip-badge" title="Origin of video material">ℹ️</span>
                </div>
                <div className="guide-v">
                  Clips are either <b>Real continuous video</b> (smooth temporal stream at 20 fps) or{' '}
                  <b>Image sequences from dataset</b> (discrete CCTV still frames played at 6–8 fps).
                  The site never represents assembled still sequences as continuous video.
                </div>
              </div>

              <div className="guide-card">
                <div className="guide-k">
                  <span className="guide-tag">Timeline</span>
                  <span className="tooltip-badge" title="Sliding window progression">ℹ️</span>
                </div>
                <div className="guide-v">
                  Videos are sampled every 5th frame and evaluated in sliding windows of 10 frames
                  (stride = 5 frames). The horizontal timeline marks each evaluated window across time.
                </div>
              </div>

              <div className="guide-card">
                <div className="guide-k">
                  <span className="guide-tag">Threshold</span>
                  <span className="tooltip-badge" title="Decision boundary">ℹ️</span>
                </div>
                <div className="guide-v">
                  Fixed decision boundary at <b>0.50</b>. Model score &gt; 0.50 flags an accident candidate.
                  Model score &le; 0.50 classifies the window as normal traffic flow.
                </div>
              </div>

              <div className="guide-card">
                <div className="guide-k">
                  <span className="guide-tag">Model Score</span>
                  <span className="tooltip-badge" title="Output metric">ℹ️</span>
                </div>
                <div className="guide-v">
                  Reported as <b>model score</b> (raw sigmoid activation between 0.0 and 1.0),
                  <b>not calibrated probability</b>. High scores indicate visual pattern alignment
                  with training accident features.
                </div>
              </div>

              <div className="guide-card">
                <div className="guide-k">
                  <span className="guide-tag">Incidents</span>
                  <span className="tooltip-badge" title="Incident clustering">ℹ️</span>
                </div>
                <div className="guide-v">
                  Raw flagged windows are clustered into <b>Incidents</b> using configurable persistence
                  (requiring <i>k</i> consecutive flagged windows) plus cooldown windows to avoid
                  alert fatigue.
                </div>
              </div>

              <div className="guide-card">
                <div className="guide-k">
                  <span className="guide-tag">Header Status</span>
                  <span className="tooltip-badge" title="Backend health">ℹ️</span>
                </div>
                <div className="guide-v">
                  The navbar health monitor polls Render backend and resolves cleanly to <b>Online</b> or{' '}
                  <b>Offline</b> within 90 seconds. If offline, a manual Retry button and diagnostic tooltip
                  are provided.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 2. Sample Outcomes Table */}
      <div className="outcomes-table-card">
        <div className="outcomes-header">
          <div className="outcomes-title">
            <span className="outcomes-icon">📊</span>
            <span>EMPIRICAL SAMPLE OUTCOMES & VERIFICATION AUDIT</span>
          </div>
          <span className="outcomes-sub">
            Real model evaluation numbers on sample clips (Ground Truth vs Model Score)
          </span>
        </div>

        <div className="table-responsive">
          <table className="outcomes-table">
            <thead>
              <tr>
                <th>Sample Clip</th>
                <th>Category</th>
                <th>Ground Truth</th>
                <th>Peak Model Score</th>
                <th>Mean Score</th>
                <th>Status</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <b>Highway Collision</b>
                  <div className="tbl-sub">sample_accident_01.mp4</div>
                </td>
                <td>
                  <span className="type-pill sequence">Image sequence</span>
                </td>
                <td><span className="truth-pill accident">Accident</span></td>
                <td><b>0.6891</b></td>
                <td>0.6809</td>
                <td><span className="status-pill flagged">Flagged</span></td>
                <td><span className="outcome-pill correct">✓ True Positive</span></td>
              </tr>

              <tr>
                <td>
                  <b>Intersection Collision</b>
                  <div className="tbl-sub">sample_accident_02.mp4</div>
                </td>
                <td>
                  <span className="type-pill sequence">Image sequence</span>
                </td>
                <td><span className="truth-pill accident">Accident</span></td>
                <td><b>0.8536</b></td>
                <td>0.5847</td>
                <td><span className="status-pill flagged">Flagged</span></td>
                <td><span className="outcome-pill correct">✓ True Positive</span></td>
              </tr>

              <tr>
                <td>
                  <b>Normal Traffic Flow</b>
                  <div className="tbl-sub">sample_normal_01.mp4</div>
                </td>
                <td>
                  <span className="type-pill sequence">Image sequence</span>
                </td>
                <td><span className="truth-pill normal">Normal</span></td>
                <td><b>0.0194</b></td>
                <td>0.0188</td>
                <td><span className="status-pill clear">Clear</span></td>
                <td><span className="outcome-pill correct">✓ True Negative</span></td>
              </tr>

              <tr>
                <td>
                  <b>Continuous Highway Surveillance</b>
                  <div className="tbl-sub">sample_continuous_01.mp4</div>
                </td>
                <td>
                  <span className="type-pill continuous">Continuous video</span>
                </td>
                <td><span className="truth-pill normal">Normal</span></td>
                <td><b>0.0163</b></td>
                <td>0.0158</td>
                <td><span className="status-pill clear">Clear (39 windows)</span></td>
                <td><span className="outcome-pill correct">✓ True Negative</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Where It Fails Section */}
      <div className="collapsible-card">
        <button
          className="collapsible-header"
          onClick={() => setWhereItFailsOpen((prev) => !prev)}
          aria-expanded={whereItFailsOpen}
        >
          <div className="collapsible-title">
            <span className="collapsible-icon">⚠️</span>
            <span>WHERE IT FAILS — EMPIRICAL ERROR & EDGE-CASE AUDIT</span>
          </div>
          <span className="chevron-icon">{whereItFailsOpen ? '▲' : '▼'}</span>
        </button>

        {whereItFailsOpen && (
          <div className="collapsible-content">
            <div className="failure-intro">
              The CNN-LSTM model is optimized for rapid anomaly scoring, but has identifiable failure
              modes measured against real CCTV dataset samples:
            </div>

            <div className="failure-modes-grid">
              <div className="failure-card fp">
                <div className="failure-type-badge fp">FALSE POSITIVE (FALSE ALARM)</div>
                <div className="failure-title">Headlight Glare & Wet Asphalt Reflections</div>
                <div className="failure-metric">Measured Example: <code>NonAccident/11979.jpg</code> (Score: <b>0.5758</b>)</div>
                <div className="failure-desc">
                  Nighttime footage where high-beam headlights reflect off wet roads or puddles creates
                  intense localized high-gradient patches that trigger false accident activations.
                </div>
              </div>

              <div className="failure-card fp">
                <div className="failure-type-badge fp">BORDERLINE ELEVATED SCORE</div>
                <div className="failure-title">Erratic Merging & Congestion Stutter</div>
                <div className="failure-metric">Measured Example: <code>NonAccident/12013.jpg</code> (Score: <b>0.4793</b>)</div>
                <div className="failure-desc">
                  Rapid bumper-to-bumper deceleration or sudden cut-ins elevates the model score close
                  to the 0.50 decision threshold without an actual collision taking place.
                </div>
              </div>

              <div className="failure-card fn">
                <div className="failure-type-badge fn">FALSE NEGATIVE (MISSED ACCIDENT)</div>
                <div className="failure-title">Low-Contrast Distant Collisions & Heavy Occlusion</div>
                <div className="failure-metric">Measured Example: <code>Accident/1424.jpg</code> (Score: <b>0.1606</b>)</div>
                <div className="failure-desc">
                  Collisions occurring in distant CCTV perspective zones or under heavy shadows blend
                  into the background, resulting in low activation scores well below 0.50.
                </div>
              </div>

              <div className="failure-card fn">
                <div className="failure-type-badge fn">FALSE NEGATIVE (MISSED ACCIDENT)</div>
                <div className="failure-title">Stationary Vehicle Post-Crash</div>
                <div className="failure-metric">Measured Example: <code>Accident/129.jpg</code> (Score: <b>0.1873</b>)</div>
                <div className="failure-desc">
                  Once damaged vehicles come to a complete standstill, temporal LSTM recurrent units
                  register zero motion variation, dropping the score below detection threshold.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
