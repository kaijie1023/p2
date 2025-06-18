'use client';

import { useState } from "react";
import DropArea from "./DropArea";
import { usePredictContext } from "@/context/PredictContext";

const ImageInput = () => {
  const [file, setFile] = useState<File | null>(null);

  const { handleFileUpload } = usePredictContext();
  
  const handleFileChange = (file: File) => {
    setFile(file);
    handleFileUpload(file);
  };

  return (
    <section className="flex flex-col items-center">
      <DropArea onFileChange={handleFileChange} />
      {file && (
        <figure>
          <img
            src={file ? URL.createObjectURL(file) : "/placeholder.png"}
            alt="Uploaded File"
            className="rounded-lg mb-4 size-120 object-contain"
          />
        </figure>
      )}
    </section>
  );
}
export default ImageInput;