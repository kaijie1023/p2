import DropArea from "@/component/client/DropArea";
import ImageInput from "@/component/client/ImageInput";
import PredictResult from "@/component/client/PredictResult";
import PredictProvider from "@/provider/PredictProvider";
import axios from "axios";

export default function Home() {
  return (
    <div className="h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex gap-x-[10vw] row-start-2 w-full h-full">
        <PredictProvider>
          <ImageInput />
          <PredictResult />
        </PredictProvider>
      </main>
    </div>
  );
}
