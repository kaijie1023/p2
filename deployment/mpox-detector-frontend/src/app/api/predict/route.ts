import { getPredictionLabel, Prediction, PredictionLabel } from "@/type/Prediction";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return new Response(JSON.stringify({ error: 'File is required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData,
    });

    // console.log('response', await response.json());
    if (!response.ok) {
      const errorText = await response.text();
      return new Response(
        JSON.stringify({ error: `Failed to process file: ${errorText}` }),
        {
          status: response.status,
          // headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Simulate a prediction process
    // const prediction = `Predicted class for ${file.name}`;
    // console.log('response.json()', await response.json());

    const prediction: Prediction = await response.json();

    const responseData = {
      label: getPredictionLabel(prediction.label as PredictionLabel),
      confidence: prediction.confidence,
    };


    return Response.json(responseData, {
      status: 200,
      // headers: { 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error processing request:', error);
    return new Response(JSON.stringify({ error: 'Internal Server Error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
// export const dynamic = 'force-dynamic'; // Ensure the route is always dynamic