from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
import torch
import torch.nn as nn
from typing import Optional
import io
import sys
import os

class MultiScaleDenseNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiScaleDenseNet, self).__init__()
        
        # Load pretrained DenseNet
        base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # Extract the features before each dense block
        self.initial_layers = nn.Sequential(
            base_model.features.conv0,
            base_model.features.norm0,
            base_model.features.relu0,
            base_model.features.pool0
        )
        
        # Extract dense blocks
        self.block1 = base_model.features.denseblock1
        self.trans1 = base_model.features.transition1
        self.block2 = base_model.features.denseblock2
        self.trans2 = base_model.features.transition2
        self.block3 = base_model.features.denseblock3
        self.trans3 = base_model.features.transition3
        self.block4 = base_model.features.denseblock4
        
        # Final processing
        self.final_norm = base_model.features.norm5
        
        # Feature dimension for each block output (depends on model)
        self.scale1_dim = 256  # After block1
        self.scale2_dim = 512  # After block2
        self.scale3_dim = 1024 # After block3
        self.scale4_dim = 1024 # After block4
        
        # Multi-scale integration
        self.adaptation_layers = nn.ModuleDict({
            'scale1': nn.Conv2d(self.scale1_dim, 256, kernel_size=1),
            'scale2': nn.Conv2d(self.scale2_dim, 256, kernel_size=1),
            'scale3': nn.Conv2d(self.scale3_dim, 256, kernel_size=1),
            'scale4': nn.Conv2d(self.scale4_dim, 256, kernel_size=1)
        })
        
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256 * 4, num_classes)  # 256 * 4 because we concatenate 4 scales
        )
        
    def forward(self, x):
        x = self.initial_layers(x)
        
        # Scale 1 features
        scale1_features = self.block1(x)
        scale1_adapted = self.adaptation_layers['scale1'](scale1_features)
        
        # Scale 2 features
        x = self.trans1(scale1_features)
        scale2_features = self.block2(x)
        scale2_adapted = self.adaptation_layers['scale2'](scale2_features)
        
        # Scale 3 features
        x = self.trans2(scale2_features)
        scale3_features = self.block3(x)
        scale3_adapted = self.adaptation_layers['scale3'](scale3_features)
        
        # Scale 4 features (final dense block)
        x = self.trans3(scale3_features)
        scale4_features = self.block4(x)
        scale4_adapted = self.adaptation_layers['scale4'](scale4_features)
        
        # Ensure all feature maps have same spatial dimensions through upsampling
        target_size = scale1_adapted.size()[2:]
        scale2_adapted = nn.functional.interpolate(scale2_adapted, size=target_size, mode='bilinear', align_corners=True)
        scale3_adapted = nn.functional.interpolate(scale3_adapted, size=target_size, mode='bilinear', align_corners=True)
        scale4_adapted = nn.functional.interpolate(scale4_adapted, size=target_size, mode='bilinear', align_corners=True)
        
        # Concatenate features from all scales
        multi_scale_features = torch.cat([
            scale1_adapted, 
            scale2_adapted, 
            scale3_adapted, 
            scale4_adapted
        ], dim=1)
        
        # Classification
        output = self.classifier(multi_scale_features)
        
        return output
    
    def load(self, path: str, map_location: Optional[str] = None, weights_only: bool = False) -> None:
        """
        Load model weights from a file.
        
        Args:
            path (str): Path to the model weights file.
            map_location (Optional[str]): Device to map the weights to.
            weights_only (bool): If True, only load the state_dict.
        """
        if weights_only:
            self.load_state_dict(torch.load(path, map_location=map_location))
        else:
            checkpoint = torch.load(path, map_location=map_location)
            self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {path}")

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
