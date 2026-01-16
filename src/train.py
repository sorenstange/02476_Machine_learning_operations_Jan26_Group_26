from pathlib import Path
import torch
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