from pathlib import Path
import torch
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