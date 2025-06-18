'use client';

import { useRef, useState } from "react";

type DropAreaProps = {
  onFileChange?: (file: File) => void;
};
const DropArea = ({ onFileChange }: DropAreaProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isEntered, setIsEntered] = useState<boolean>(false);

  const handleSelectFile = () => {
    inputRef.current?.click(); // Use the ref to trigger the file input click
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileChange?.(file);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault(); // Prevent default behavior
    
    setIsEntered(false); // Reset the state when a file is dropped
    const file = event.dataTransfer.files?.[0];
    if (file) {
      onFileChange?.(file);
    }
  };
  
  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsEntered(true); // Set the state to indicate that a file is being dragged over the area
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    if (!event.currentTarget.contains(event.relatedTarget as Node)) {
      setIsEntered(false); // Reset the state when the drag leaves the area
    }
  }
  
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault(); // Prevent default to allow drop
  }

  return (
    <div className={`p-4 w-[360px] h-[200px] border-2 border-dashed ${isEntered ? 'border-blue-500' : 'border-gray-400'} rounded-lg box-border`} 
      onDragEnter={handleDragEnter} 
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <div className="flex flex-col items-center justify-center w-full h-full">
        <p className="text-gray-500">Drag and drop files here</p>
        <p className="text-gray-500">or</p>
        <button className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 cursor-pointer" onClick={handleSelectFile}>
          Select Files
        </button>
        <input ref={inputRef} className="hidden" type="file" accept="image/*" onChange={handleFileChange} />
      </div>
    </div>
  );
}
export default DropArea;