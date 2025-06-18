export type Prediction = {
  label: PredictionLabel;
  confidence: number;
};

export enum PredictionLabel {
  Mpox,
  Healthy,
  Other
}

export const predictionLabels: Record<PredictionLabel, string> = {
  [PredictionLabel.Mpox]: "Mpox",
  [PredictionLabel.Healthy]: "Healthy",
  [PredictionLabel.Other]: "Other",
};

export const getPredictionLabel = (label: PredictionLabel): string => {
  return predictionLabels[label];
};
