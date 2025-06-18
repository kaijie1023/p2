from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image
import torch
import io
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from densenet121_ms import MultiScaleDenseNet

app = FastAPI()

device = None
model = None
preprocess = None

@app.on_event("startup")
def load_model():
  global device, model, preprocess

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # checkpoint = torch.load("densenet121_msi.pth", map_location="cpu", weights_only=False)
  # print(type(checkpoint)) 

  # model = models.densenet121()
  # model.classifier = torch.nn.Linear(1024, 3)
  # model.load_state_dict(torch.load("densenet121_msi.pth", map_location=device))
  model = MultiScaleDenseNet(num_classes=3)
  model.load("densenet121_msi.pth", map_location=device, weights_only=True)
  model.to(device)
  model.eval()

  preprocess = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
  ])

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        return {"label": int(pred_class), "confidence": round(confidence, 4)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
