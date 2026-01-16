<<<<<<< HEAD
from git import List
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim

import hydra
import timm
from torchmetrics.classification import Accuracy

class CNN_Model(LightningModule):
    """CNN model for rice classification (5 classes, grayscale images)."""
    model_name = "cnn"

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

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=parameters["output_dim"])
        self.val_acc = Accuracy(task="multiclass", num_classes=parameters["output_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.train_acc(logits, y)
        self.log(f"{self.model_name}_train_loss", loss, prog_bar=True)
        self.log(f"{self.model_name}_train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.val_acc(logits, y)
        self.log(f"{self.model_name}_val_loss", loss, prog_bar=True)
        self.log(f"{self.model_name}_val_acc", acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
        """Configure optimizers. Needed for LightningModule."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

class LitResNet18(LightningModule):
    model_name = "resnet"
    
    def __init__(self, parameters):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = parameters["learning_rate"]

        self.model = timm.create_model(
            parameters['model_name'],
            pretrained=True,
            in_chans=parameters['input_channels'],
            num_classes=parameters['output_dim']
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=parameters["output_dim"])
        self.val_acc = Accuracy(task="multiclass", num_classes=parameters["output_dim"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.train_acc(logits, y)
        self.log(f"{self.model_name}_train_loss", loss, prog_bar=True)
        self.log(f"{self.model_name}_train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.val_acc(logits, y)
        self.log(f"{self.model_name}_val_loss", loss, prog_bar=True)
        self.log(f"{self.model_name}_val_acc", acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        """Configure optimizers. Needed for LightningModule."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



def load_parameters(cfg):
    parameters = {
        "learning_rate" : cfg.training_parameters.learning_rate,
        "epochs"        : cfg.training_parameters.epochs,
        "batch_size"    : cfg.training_parameters.batch_size,
        "num_workers"   : cfg.training_parameters.num_workers,
        "input_size"    : cfg.data_parameters.augmentation.shape,
        "input_channels": cfg.data_parameters.channels,
        "output_dim"    : cfg.data_parameters.num_classes,
    }
    model_name = cfg.experiment.model_parameters.model_name
    parameters["model_name"] = model_name

    if model_name == 'custom_cnn':
        parameters["conv_layers"] = cfg.experiment.model_parameters.parameters.conv_layers
        parameters["fc_layers"] = cfg.experiment.model_parameters.parameters.fc_layers
    else:
        try:
            parameters['pretrained'] = cfg.experiment.model_parameters.pretrained
        except:
            parameters['pretrained'] = False

    return parameters

def load_model(cfg) -> LightningModule:
    parameters = load_parameters(cfg)
    model_name = parameters['model_name']

    if model_name == 'custom_cnn':
        model = CNN_Model(parameters)
    else:
        model = LitResNet18(parameters)

    return model

import logging
log = logging.getLogger(__name__)

@hydra.main(
        config_path="../configs",
        config_name="config",
        version_base=None,
        )
def main(cfg):
    model = load_model(cfg)
    parameters = load_parameters(cfg)

    log.info(f"Model architecture: {model}")
    log.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with grayscale rice image dimensions
    dummy_input = torch.randn(1, parameters["input_channels"], parameters["input_size"][0], parameters["input_size"][1])
    output = model(dummy_input)
    log.info(f"Input shape: {dummy_input.shape}")
    log.info(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
=======
import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """CNN model for rice classification (5 classes, RGB images)."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        # Input: 3 channels (RGB), output: 5 classes
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # After 4 pooling layers on 224x224: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with RGB rice image dimensions
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (1, 5) for 5 rice classes")
>>>>>>> 700c8dc33100554c98ccd65f577f8f71939ee8b0
