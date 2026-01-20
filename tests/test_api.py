import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import CNN_Model

# Initialize app and client after checkpoint setup
app = None
client = None


@pytest.fixture(scope="module", autouse=True)
def setup_test_model():
    """Create a CNN checkpoint before running API tests."""
    ckpt_path = Path("checkpoints/best.ckpt")
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)

    # Always create a fresh checkpoint for tests
    model = CNN_Model(parameters={
        'learning_rate': 0.0001,
        'input_channels': 1,
        'output_dim': 5,
        'input_size': (224, 224),
        'model_name': 'custom_cnn',
        'conv_layers': [16, 32, 64, 128],
        'fc_layers': [256, 128],
    })
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    # Now import and initialize the API app with the checkpoint in place
    import src.api
    global app, client
    app = src.api.app
    client = TestClient(app)

    yield
    # Optionally clean up after tests
    # ckpt_path.unlink(missing_ok=True)


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


def test_load():
    """Simple load test: 50 concurrent requests with 10 workers."""
    num_requests = 50
    num_workers = 10
    
    results = []
    start_time = time.time()
    
    def make_request():
        img_bytes = create_test_image()
        response = client.post(
            "/predict/",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        return response.status_code == 200
    
    # Run concurrent requests
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    successful = sum(results)
    
    print(f"\n--- Load Test Results ---")
    print(f"Requests: {num_requests}")
    print(f"Concurrent workers: {num_workers}")
    print(f"Successful: {successful}/{num_requests}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests/second: {num_requests/total_time:.2f}")
    
    # Assert at least 95% success rate
    assert successful >= num_requests * 0.95
