import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path to import app
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app

client = TestClient(app)


def test_health():
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def create_test_image():
    """Create a simple test image."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_predict():
    """Test prediction endpoint."""
    img_bytes = create_test_image()
    response = client.post(
        "/predict/",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert "confidence" in data
    assert data["class"] in ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
    assert 0 <= data["confidence"] <= 1


def test_predict_invalid_file():
    """Test prediction with invalid file type."""
    response = client.post(
        "/predict/",
        files={"file": ("test.txt", io.BytesIO(b"text"), "text/plain")}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()

