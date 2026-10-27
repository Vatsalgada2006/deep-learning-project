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
                  <b>Highway Multi-Lane Collision</b>
                  <div className="tbl-sub">accident_crash_demo.mp4</div>
                </td>
                <td>
                  <span className="type-pill continuous">Continuous video</span>
                </td>
                <td><span className="truth-pill accident">Accident</span></td>
                <td><b>0.8724</b></td>
                <td>0.2671</td>
                <td><span className="status-pill flagged">Flagged (20 windows)</span></td>
                <td><span className="outcome-pill correct">✓ True Positive</span></td>
              </tr>

              <tr>
                <td>
                  <b>Intersection Vehicle Crash</b>
                  <div className="tbl-sub">video_project.mp4</div>
                </td>
                <td>
                  <span className="type-pill continuous">Continuous video</span>
                </td>
                <td><span className="truth-pill accident">Accident</span></td>
                <td><b>0.8977</b></td>
                <td>0.7336</td>
                <td><span className="status-pill flagged">Flagged (12 windows)</span></td>
                <td><span className="outcome-pill correct">✓ True Positive</span></td>
              </tr>

              <tr>
                <td>
                  <b>Continuous Normal Highway Flow</b>
                  <div className="tbl-sub">non_accident.mp4</div>
                </td>
                <td>
                  <span className="type-pill continuous">Continuous video</span>
                </td>
                <td><span className="truth-pill normal">Normal</span></td>
                <td><b>0.0228</b></td>
                <td>0.0173</td>
                <td><span className="status-pill clear">Clear (42 windows)</span></td>
                <td><span className="outcome-pill correct">✓ True Negative</span></td>
              </tr>

              <tr>
                <td>
                  <b>Junction Bus-Car Collision</b>
                  <div className="tbl-sub">car-accident-dataset-example2.mp4</div>
                </td>
                <td>
                  <span className="type-pill continuous">Continuous video</span>
                </td>
                <td><span className="truth-pill accident">Accident</span></td>
                <td><b>0.0174</b></td>
                <td>0.0162</td>
                <td><span className="status-pill missed">Missed (&lt;0.50)</span></td>
                <td><span className="outcome-pill fn">✗ False Negative (Missed Accident)</span></td>
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
                <div className="failure-metric">Measured Example: <code>DoTA Anomaly/Collision_104.mp4</code> (Score: <b>0.5758</b>)</div>
                <div className="failure-desc">
                  Nighttime continuous CCTV footage where high-beam headlights reflect off wet pavement or standing puddles creates
                  intense localized high-gradient patches that trigger false accident activations.
                </div>
              </div>

              <div className="failure-card fp">
                <div className="failure-type-badge fp">BORDERLINE ELEVATED SCORE</div>
                <div className="failure-title">Emergency Braking & Congestion Occlusion</div>
                <div className="failure-metric">Measured Example: <code>UCF-Crime Normal_042.mp4</code> (Score: <b>0.4793</b>)</div>
                <div className="failure-desc">
                  Rapid bumper-to-bumper deceleration, abrupt cut-ins, or sudden occlusion elevates the model score close
                  to the decision threshold without an actual physical collision taking place.
                </div>
              </div>

              <div className="failure-card fn">
                <div className="failure-type-badge fn">FALSE NEGATIVE (MISSED ACCIDENT)</div>
                <div className="failure-title">Wide-Angle Junction & Distant Collisions (&lt;8% Frame Area)</div>
                <div className="failure-metric">Measured Example: <code>car-accident-dataset-example2.mp4</code> (Yesler Way CCTV, Score: <b>0.0174</b>)</div>
                <div className="failure-desc">
                  In wide-angle junction CCTV footage (e.g. Yesler Way), the bus-car collision occupies &lt;8% of the frame.
                  Global Average Pooling averages the entire 128×128 feature map across static asphalt and zebra markings,
                  swamping the vehicle impact signal into background noise and resulting in a missed alert (Score 0.0174).
                </div>
              </div>

              <div className="failure-card fn">
                <div className="failure-type-badge fn">FALSE NEGATIVE (MISSED ACCIDENT)</div>
                <div className="failure-title">Stationary Vehicle Post-Crash (Zero Motion Delta)</div>
                <div className="failure-metric">Measured Example: <code>UCF-Crime RoadAccidents_088.mp4</code> (Score: <b>0.1873</b>)</div>
                <div className="failure-desc">
                  Once colliding vehicles come to a complete standstill, temporal inter-frame motion Δt ≈ 0,
                  causing recurrent LSTM units to revert toward baseline normal scores.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
