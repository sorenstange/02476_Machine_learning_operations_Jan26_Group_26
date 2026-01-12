from git import List
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim

import hydra


class CNN_Model(LightningModule):
    """CNN model for rice classification (5 classes, grayscale images)."""

    def __init__(
            self, 
            parameters: dict,
        ) -> None:
        """Initialize CNN model."""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = parameters["learning_rate"]
        layers = []
        in_channels = parameters["input_channels"]
        current_size = parameters["input_size"]

        for out_channels in parameters["conv_layers"]:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))

            # Update spatial size: (N - 2) after conv, then floor(N/2) after pool
            current_size_0 = (current_size[0] - 2) // 2
            current_size_1 = (current_size[1] - 2) // 2
            current_size = (current_size_0, current_size_1)
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)

        flattened_dim = in_channels * current_size[0] * current_size[1]
        
        classifier_layers = []
        in_features = flattened_dim

        for hidden_dim in parameters["fc_layers"]:
            classifier_layers.append(nn.Linear(in_features, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(0.5))
            in_features = hidden_dim

        classifier_layers.append(nn.Linear(in_features, parameters["output_dim"]))
        self.classifier = nn.Sequential(*classifier_layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step. Needed for LightningModule."""
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for LightningModule: compute loss and accuracy."""
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        # Log metrics; accumulate across epoch
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
        """Configure optimizers. Needed for LightningModule."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    

def load_parameters(cfg):
    parameters = {
        "input_size"    : cfg.model_parameters.input_size,
        "input_channels": cfg.model_parameters.input_channels,
        "conv_layers"   : cfg.model_parameters.conv_layers,
        "fc_layers"     : cfg.model_parameters.fc_layers,
        "output_dim"    : cfg.model_parameters.output_dim,
        "learning_rate" : cfg.hyperparameters.learning_rate,
        "epochs"        : cfg.hyperparameters.epochs,
        "batch_size"    : cfg.hyperparameters.batch_size,
        "num_workers"   : cfg.hyperparameters.num_workers,
    }
    return parameters

@hydra.main(
        config_path="../configs",
        config_name="config",
        version_base=None,
        )
def main(cfg):
    parameters = load_parameters(cfg)

    model = CNN_Model(
        parameters,
    )
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with grayscale rice image dimensions
    dummy_input = torch.randn(1, parameters["input_channels"], parameters["input_size"][0], parameters["input_size"][1])
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()