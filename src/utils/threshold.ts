export interface PredictionWindow {
  window_index: number;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  confidence: number;
  is_accident: boolean;
}

export interface ThresholdAnalysis {
  threshold: number;
  totalWindows: number;
  alertsRaised: number;
  flaggedSharePercent: number;
  maxConfidence: number;
  avgConfidence: number;
  evaluatedWindows: PredictionWindow[];
  systemStatus: 'NORMAL' | 'ALERT';
}

/**
 * Recomputes accident alerts purely on the client based on a customizable confidence threshold.
 * Allows operators to dynamically tune sensitivity without re-running backend inference.
 */
export function evaluatePredictionsAtThreshold(
  predictions: PredictionWindow[],
  threshold: number
): ThresholdAnalysis {
  if (!predictions || predictions.length === 0) {
    return {
      threshold,
      totalWindows: 0,
      alertsRaised: 0,
      flaggedSharePercent: 0,
      maxConfidence: 0,
      avgConfidence: 0,
      evaluatedWindows: [],
      systemStatus: 'NORMAL',
    };
  }

  // Clamp threshold between 0 and 1
  const clampedThreshold = Math.max(0.01, Math.min(0.99, threshold));

  let alertsCount = 0;
  let maxConf = 0;
  let sumConf = 0;

  const evaluatedWindows: PredictionWindow[] = predictions.map((p) => {
    const isAccident = p.confidence >= clampedThreshold;
    if (isAccident) alertsCount++;
    if (p.confidence > maxConf) maxConf = p.confidence;
    sumConf += p.confidence;

    return {
      ...p,
      is_accident: isAccident,
    };
  });

  const totalWindows = predictions.length;
  const flaggedSharePercent = totalWindows > 0
    ? Number(((alertsCount / totalWindows) * 100).toFixed(1))
    : 0;

  return {
    threshold: clampedThreshold,
    totalWindows,
    alertsRaised: alertsCount,
    flaggedSharePercent,
    maxConfidence: Number(maxConf.toFixed(4)),
    avgConfidence: Number((sumConf / totalWindows).toFixed(4)),
    evaluatedWindows,
    systemStatus: alertsCount > 0 ? 'ALERT' : 'NORMAL',
  };
}
