from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import io

# Import model class from your project
from src.model import CNN_Model

app = FastAPI(title="Rice Grain Classifier API")

# Load model once on startup
MODEL_PATH = Path("models/rice_classifier.pth")

try:
    model = CNN_Model(parameters={
        "learning_rate": 0.0001, 
        "input_channels": 1, 
        "output_dim": 5,
        "input_size": (224, 224),
        "conv_layers": [16, 32, 64, 128],
        "fc_layers": [256, 128]
    })
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

# Define the transforms you trained with
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Class names for rice types
class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # 1) Check input file
        if not file.content_type or file.content_type.split("/")[0] != "image":
            raise HTTPException(status_code=400, detail="File must be an image.")

        # 2) Read and validate image
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        # 3) Preprocess like during training
        img_tensor = transform(image).unsqueeze(0)

        # 4) Model inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = probabilities[0, pred_idx].item()
        
        # 5) Return prediction
        return {
            "class": class_names[pred_idx],
            "class_index": pred_idx,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                class_names[i]: round(probabilities[0, i].item(), 4) 
                for i in range(len(class_names))
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")