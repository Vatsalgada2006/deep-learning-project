import { describe, it, expect } from 'vitest';
import { evaluatePredictionsAtThreshold, type PredictionWindow } from './threshold';

describe('Threshold Slider Logic', () => {
  const samplePredictions: PredictionWindow[] = [
    {
      window_index: 0,
      start_frame: 0,
      end_frame: 50,
      start_time: 0.0,
      end_time: 2.5,
      confidence: 0.05,
      is_accident: false,
    },
    {
      window_index: 1,
      start_frame: 50,
      end_frame: 100,
      start_time: 2.5,
      end_time: 5.0,
      confidence: 0.68,
      is_accident: true,
    },
    {
      window_index: 2,
      start_frame: 100,
      end_frame: 150,
      start_time: 5.0,
      end_time: 7.5,
      confidence: 0.82,
      is_accident: true,
    },
    {
      window_index: 3,
      start_frame: 150,
      end_frame: 200,
      start_time: 7.5,
      end_time: 10.0,
      confidence: 0.35,
      is_accident: false,
    },
  ];

  it('handles empty predictions gracefully', () => {
    const result = evaluatePredictionsAtThreshold([], 0.5);
    expect(result.totalWindows).toBe(0);
    expect(result.alertsRaised).toBe(0);
    expect(result.flaggedSharePercent).toBe(0);
    expect(result.systemStatus).toBe('NORMAL');
  });

  it('correctly evaluates at standard default threshold 0.50', () => {
    const result = evaluatePredictionsAtThreshold(samplePredictions, 0.5);
    expect(result.totalWindows).toBe(4);
    expect(result.alertsRaised).toBe(2); // 0.68 and 0.82
    expect(result.flaggedSharePercent).toBe(50.0);
    expect(result.systemStatus).toBe('ALERT');
    expect(result.evaluatedWindows[0].is_accident).toBe(false);
    expect(result.evaluatedWindows[1].is_accident).toBe(true);
    expect(result.evaluatedWindows[2].is_accident).toBe(true);
    expect(result.evaluatedWindows[3].is_accident).toBe(false);
  });

  it('lowers threshold to 0.30 and flags additional windows', () => {
    const result = evaluatePredictionsAtThreshold(samplePredictions, 0.3);
    expect(result.alertsRaised).toBe(3); // 0.35, 0.68, 0.82
    expect(result.flaggedSharePercent).toBe(75.0);
    expect(result.systemStatus).toBe('ALERT');
    expect(result.evaluatedWindows[3].is_accident).toBe(true);
  });

  it('raises threshold to 0.75 and suppresses false alarms', () => {
    const result = evaluatePredictionsAtThreshold(samplePredictions, 0.75);
    expect(result.alertsRaised).toBe(1); // Only 0.82
    expect(result.flaggedSharePercent).toBe(25.0);
    expect(result.systemStatus).toBe('ALERT');
    expect(result.evaluatedWindows[1].is_accident).toBe(false);
    expect(result.evaluatedWindows[2].is_accident).toBe(true);
  });

  it('raises threshold to 0.90 resulting in zero alerts', () => {
    const result = evaluatePredictionsAtThreshold(samplePredictions, 0.9);
    expect(result.alertsRaised).toBe(0);
    expect(result.flaggedSharePercent).toBe(0.0);
    expect(result.systemStatus).toBe('NORMAL');
    expect(result.evaluatedWindows.every((w) => !w.is_accident)).toBe(true);
  });

  it('computes max and average confidence correctly', () => {
    const result = evaluatePredictionsAtThreshold(samplePredictions, 0.5);
    expect(result.maxConfidence).toBe(0.82);
    // (0.05 + 0.68 + 0.82 + 0.35) / 4 = 1.9 / 4 = 0.475
    expect(result.avgConfidence).toBe(0.475);
  });
});
