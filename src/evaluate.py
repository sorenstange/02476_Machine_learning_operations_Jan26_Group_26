<<<<<<< HEAD
from pathlib import Path
import torch
import typer
import wandb
import hydra
from data import load_test_data
from model import CNN_Model, load_parameters
=======
import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
>>>>>>> 700c8dc33100554c98ccd65f577f8f71939ee8b0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


<<<<<<< HEAD
@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base=None,
)
def evaluate(cfg) -> None:
    """Evaluate a trained rice classifier model on test data."""
    parameters = load_parameters(cfg)
    
    model_checkpoint = Path("models/rice_classifier.pth")
    
    if not model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
    
    # Initialize wandb
    wandb.init(
        project="rice_classifier",
        job_type="evaluation",
        config={"batch_size": parameters["batch_size"]},
    )
    
    print("Evaluating rice classifier")
    print(f"Model checkpoint: {model_checkpoint}")

    # Initialize model and load weights
    model = CNN_Model(parameters).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    # Load test data
    test_dataloader = load_test_data(parameters['batch_size'], num_workers=parameters['num_workers'])
    print(f"Test samples: {len(test_dataloader.dataset)}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Log to wandb
    wandb.log({
        "test_accuracy": accuracy,
        "test_correct": correct,
        "test_total": total
    })
    
    wandb.finish()


if __name__ == "__main__":
    evaluate()
=======
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
>>>>>>> 700c8dc33100554c98ccd65f577f8f71939ee8b0
