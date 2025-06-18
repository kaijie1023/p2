'use client';

import { usePredictContext } from "@/context/PredictContext";

const PredictResult = () => {
  const { isLoading, prediction } = usePredictContext();

  return (
    <section className="w-full flex flex-col">
      <h2 className="text-2xl font-bold mb-4">Prediction Result</h2>
      <div className="my-3">
        <h3 className="text-xl font-semibold mb-2">Prediction</h3>
        <p className="text-lg font-medium">{prediction?.label || "No prediction available"}</p>
      </div>
      <div className="my-3">
        <h3 className="text-xl font-semibold mb-2">Confidence</h3>
        <p className="text-lg font-medium">{prediction?.confidence || "No prediction available"}</p>
      </div>
    </section>
  );
}
export default PredictResult;