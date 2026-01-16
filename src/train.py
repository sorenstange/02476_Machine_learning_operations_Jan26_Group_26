from pathlib import Path
import torch
<<<<<<< HEAD
import wandb
import hydra
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from data import load_train_data, load_val_data
from model import load_model, load_parameters

class WandbArtifactCallback(Callback):
    """Callback that logs the current best checkpoint from a ModelCheckpoint callback to W&B.

    It looks for a ModelCheckpoint in trainer.callbacks and logs the checkpoint file only when
    `best_model_path` changes (so we only upload the 'best' checkpoint when it updates).
    """

    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self._last_logged_best = None

    def on_validation_end(self, trainer, pl_module):
        # Only log from rank 0 in multi-GPU/TPU runs
        if getattr(trainer, "global_rank", 0) != 0:
            return

        # Find ModelCheckpoint callback
        ckpt_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_cb = cb
                break

        if ckpt_cb is None:
            # No checkpoint callback available -> nothing to do
            return

        best_path = getattr(ckpt_cb, "best_model_path", None)
        if not best_path:
            return

        # Only act when the best checkpoint changed
        if best_path == self._last_logged_best:
            return

        # Log the best checkpoint to W&B
        artifact = wandb.Artifact(
            name=f"rice_classifier_best",
            type="model",
            metadata={"best_path": best_path, **self.metadata}
        )
        artifact.add_file(str(best_path))

        if hasattr(trainer, "logger") and getattr(trainer.logger, "experiment", None):
            trainer.logger.experiment.log_artifact(artifact)
            # Update last logged to avoid duplicate uploads
            self._last_logged_best = best_path

import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg) -> None: 
    """Train a model on Rice Image Dataset."""
    log.info("Training rice classifier")
    parameters = load_parameters(cfg)
    log.info(f"learning rate = {parameters['learning_rate']}, batch size = {parameters['batch_size']}, epochs = {parameters['epochs']}")
    
    
    model = load_model(cfg)

    wandb_logger = WandbLogger(project="rice_classifier",
        config={"lr": parameters["learning_rate"], "batch_size": parameters["batch_size"], "epochs": parameters["epochs"]},
    )

    # ModelCheckpoint: keep only best model according to validation loss
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=Path(get_original_cwd()) / "checkpoints",
        filename="best",
        verbose=True,
    )

    # Artifact callback: logs the best checkpoint when ModelCheckpoint updates it
    artifact_cb = WandbArtifactCallback(
        metadata={"learning_rate": parameters["learning_rate"], "batch_size": parameters["batch_size"], "epochs": parameters["epochs"]},
    )

    # Load datasets
    train_set = load_train_data(parameters['batch_size'], num_workers=parameters['num_workers'])
    val_set = load_val_data(parameters['batch_size'], num_workers=parameters['num_workers'])
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    #print(f"Classes: {train_set.targets.unique().tolist()}")
    
    # Check for GPU availability
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        print("Using GPU for training")
    else:
        accelerator = "cpu"
        devices = 1
        print("Using CPU for training")
    
    trainer = Trainer(max_epochs=parameters["epochs"], accelerator=accelerator, devices=devices, logger=wandb_logger, callbacks=[checkpoint_cb, artifact_cb])
    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=val_set)
    
    print("Training complete.")

    # Save model into the original working directory so artifact points to correct location
    model_dir = Path(get_original_cwd()) / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "rice_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Log the best model checkpoint if available, otherwise log the saved final model
    best_ckpt = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            best_ckpt = getattr(cb, "best_model_path", None)
            break

    artifact = wandb.Artifact(
        name=parameters["model_name"] + 'rice_classifier' + "_final",
        type="model",
        metadata={"epochs": parameters["epochs"], "lr": parameters["learning_rate"], "batch_size": parameters["batch_size"]}
    )

    log_path = best_ckpt if best_ckpt else str(model_path)
    artifact.add_file(str(log_path))
    # Use the WandbLogger's experiment (the active run)
    wandb_logger.experiment.log_artifact(artifact)
    artifact.wait()  # optional: wait for upload to finish


if __name__ == "__main__":
    train()
=======
import typer
import wandb
from data import MyDataset
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    data_path: Path = Path("processed"),
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 5
) -> None:
    """Train a model on Rice Image Dataset.
    
    Args:
        data_path: Path to preprocessed data folder containing train/val/test splits
        lr: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
    """
    print("Training rice classifier")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    print(f"Using device: {DEVICE}")
    
    wandb.init(
        project="rice_classifier",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    # Load datasets
    train_set = MyDataset(data_path / "train")
    val_set = MyDataset(data_path / "val")
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"Classes: {train_set.class_names}")

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = MyAwesomeModel(num_classes=5).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (y_pred.argmax(dim=1) == target).sum().item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, iter {i}/{len(train_dataloader)}, loss: {loss.item():.4f}")
        
        train_loss /= len(train_dataloader)
        train_acc = train_correct / len(train_set)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for img, target in val_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                val_loss += loss.item()
                val_correct += (y_pred.argmax(dim=1) == target).sum().item()
        
        val_loss /= len(val_dataloader)
        val_acc = val_correct / len(val_set)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
    
    # Save model
    model_path = Path("models/rice_classifier.pth")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    typer.run(train)
>>>>>>> 700c8dc33100554c98ccd65f577f8f71939ee8b0
