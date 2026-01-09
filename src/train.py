from pathlib import Path
import torch
import typer
import wandb
import hydra
from pytorch_lightning import Trainer
from data import load_train_data, load_val_data
from model import CNN_Model, load_parameters

@hydra.main(
        config_path="../configs",
        config_name="config",
        version_base=None,
        )
def train(cfg) -> None:
    """Train a model on Rice Image Dataset."""

    parameters = load_parameters(cfg)
    print("Training rice classifier")
    print(f"learning rate = {parameters['learning_rate']}, batch size = {parameters['batch_size']=}, epochs = {parameters['epochs']=}")
    
    wandb.init(
        project="rice_classifier",
        config={"lr": parameters["learning_rate"], "batch_size": parameters["batch_size"], "epochs": parameters["epochs"]},
    )

    # Load datasets
    train_set = load_train_data(parameters['batch_size'])
    val_set = load_val_data(parameters['batch_size'])
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"Classes: {train_set.class_names}")

    model = CNN_Model(parameters)
    trainer = Trainer(max_epochs=parameters["epochs"], accelerator="auto")
    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=val_set)
    
    print("Training complete.")
    # Save model
    model_path = Path("models/rice_classifier.pth")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    typer.run(train)