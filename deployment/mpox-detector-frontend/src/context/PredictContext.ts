'use client';

import { Prediction } from "@/type/Prediction";
import { createContext, useContext } from "react";

type PredictContextType = {
  isLoading: boolean;
  setIsLoading: (isLoading: boolean) => void;
  prediction: Prediction | null;
  setPrediction: (prediction: Prediction | null) => void;
  handleFileUpload: (file: File) => Promise<void>;
};

export const PredictContext = createContext<PredictContextType | undefined>(undefined);

export const usePredictContext = (): PredictContextType => {
  const context = useContext(PredictContext);
  if (!context) {
    throw new Error("usePredictContext must be used within a PredictProvider");
  }
  return context;
};
