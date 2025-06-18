'use client';

import { PredictContext } from "@/context/PredictContext";
import { useState } from "react";
import { Prediction } from "@/type/Prediction";

export const PredictProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);

  const handleFileUpload = async (file: File) => {
    if (!file) return;
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      setPrediction(await response.json() as Prediction);
    } catch (error) {
      console.error('Error uploading file:', error);
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <PredictContext.Provider value={{ isLoading, setIsLoading, prediction, setPrediction, handleFileUpload }}>
      {children}
    </PredictContext.Provider>
  );
};
export default PredictProvider;