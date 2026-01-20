from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from google.cloud import storage
import os
import io

# Import model class from your project
from src.model import CNN_Model

app = FastAPI(title="Rice Grain Classifier API")

# Load model once on startup (require Lightning checkpoint)
CKPT_PATH = Path("checkpoints/best.ckpt")
GCS_BUCKET = "mlops-s204229"
GCS_BLOB = "checkpoints/custom_cnn_best.ckpt"
model = None


def _init_model() -> CNN_Model:
    # Defaults match the trained custom CNN config
    return CNN_Model(parameters={
        "learning_rate": 0.0001,
        "input_channels": 1,
        "output_dim": 5,
        "input_size": (224, 224),
        "model_name": "custom_cnn",
        "conv_layers": [16, 32, 64, 128],
        "fc_layers": [256, 128],
    })

def ensure_ckpt_available():
    if CKPT_PATH.exists():
        print(f"Using local checkpoint at {CKPT_PATH}")
        return

    print("Local checkpoint not found, downloading from GCS...")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_BLOB)
    blob.download_to_filename(CKPT_PATH)

    print(f"Downloaded checkpoint to {CKPT_PATH}")



def _load_model_weights() -> CNN_Model | None:
    if not CKPT_PATH.exists():
        print("Warning: No model weights found (expected checkpoints/best.ckpt)")
        return None

    try:
        # Explicitly disable weights_only to allow Lightning/omegaconf metadata in the checkpoint
        weights = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        state_dict = weights.get("state_dict", weights) if isinstance(weights, dict) else weights

        model_obj = _init_model()
        model_obj.load_state_dict(state_dict)
        model_obj.eval()
        print(f"Loaded model weights from {CKPT_PATH}")
        return model_obj
    except Exception as e:
        print(f"Warning: Could not load model from {CKPT_PATH}: {e}")
        return None


ensure_ckpt_available()
model = _load_model_weights()


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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure model file exists.")
    
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