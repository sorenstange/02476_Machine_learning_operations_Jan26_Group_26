from pathlib import Path
import random
import shutil
import json
from typing import Dict, List, Tuple

import typer
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from omegaconf import OmegaConf
config = OmegaConf.load('configs/config.yaml')

class RiceDataset(Dataset):
    """Dataset for Rice Image classification with 5 rice types.

    Supports greyscale images (single channel) by passing `grayscale=True`.
    """

    def __init__(self, data_path : Path = Path(config.data_parameters.dataset_raw_path), transform : bool = True) -> None:
        """Initialize dataset from preprocessed splits.
        
        Args:
            data_path: Path to split folder (e.g., data/train)
            transform: Optional torchvision transforms
        """
        self.data_path = data_path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Expected rice types
        self.class_names = sorted(["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        # Collect all image files and their labels
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.data_path / cls_name
            if not cls_dir.exists():
                raise ValueError(f"Class folder not found: {cls_dir}")

            # Get all image files
            image_files = [f for f in cls_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if not image_files:
                raise ValueError(f"No image files found in {cls_dir}")

            # Add (path, label) tuples
            label = self.class_to_idx[cls_name]
            self.samples.extend([(img_path, label) for img_path in image_files])

            if transform:
                self.transform = transforms.Compose([
                    transforms.Resize((config.data_parameters.augmentation.shape[0], config.data_parameters.augmentation.shape[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[config.data_parameters.augmentation.mean], std=[config.data_parameters.augmentation.std])
                ])
            else:
                self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a sample (image, label) at the given index."""
        img_path, label = self.samples[index]

        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

def preprocess():
    print("Preprocessing data...")
    dataset = RiceDataset()

    output_folder = Path(config.data_parameters.dataset_processed_path)

    # Check if output folder exists
    if output_folder.exists():
        raise FileExistsError(
            f"Output folder already exists: {output_folder}\n"
            f"Please remove it manually before running preprocessing to avoid mixing old and new data."
        )
    
    # Collect file lists per class
    files_by_class: Dict[str, List[Path]] = {}
    for cls in dataset.class_names:
        cls_dir = dataset.data_path / cls
        if not cls_dir.exists():
            raise ValueError(f"Expected class folder not found: {cls_dir}")
            
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not files:
            raise ValueError(f"No image files found in class folder: {cls_dir}")
        files_by_class[cls] = files
        
    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in dataset.class_names:
            (output_folder / split / cls).mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    split_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    split_files: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}, "test": {}}

    for cls, files in files_by_class.items():
        # Shuffle deterministically
        files = list(files)
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(config.data_parameters.train_fraction * n))
        n_val = int(round(config.data_parameters.validation_fraction * n))

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        # Copy files into split directories
        for src in train_files:
            dst = output_folder / "train" / cls / src.name
            shutil.copy2(src, dst)
        for src in val_files:
            dst = output_folder / "val" / cls / src.name
            shutil.copy2(src, dst)
        for src in test_files:
            dst = output_folder / "test" / cls / src.name
            shutil.copy2(src, dst)

        # Track counts and file names for manifest
        split_counts["train"][cls] = len(train_files)
        split_counts["val"][cls] = len(val_files)
        split_counts["test"][cls] = len(test_files)
        split_files["train"][cls] = [f.name for f in train_files]
        split_files["val"][cls] = [f.name for f in val_files]
        split_files["test"][cls] = [f.name for f in test_files]


    # Print verification summary per class
    print(f"Created stratified splits at {output_folder} (seed={config.seed}).")
    for cls in dataset.class_names:
        n_total = (
            split_counts["train"][cls]
            + split_counts["val"][cls]
            + split_counts["test"][cls]
        )
        tr = split_counts["train"][cls] / n_total
        vr = split_counts["val"][cls] / n_total
        ter = split_counts["test"][cls] / n_total
        print(
            f"  {cls}: train={split_counts['train'][cls]} ({tr:.1%}), "
            f"val={split_counts['val'][cls]} ({vr:.1%}), "
            f"test={split_counts['test'][cls]} ({ter:.1%})"
        )
    print(f"Done! Use RiceDataset(Path('{output_folder}/train')) for training.")


def load_train_data(batch_size: int, num_workers: int) -> DataLoader:
    """Load training data from data/train into a DataLoader.

    Args:
        batch_size: Batch size for the DataLoader
        num_workers: Number of workers for the DataLoader
    Returns:
        DataLoader containing training data
    """
    train_path = Path("data/train")
    train_dataset = RiceDataset(train_path)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)


def load_val_data(batch_size: int, num_workers: int) -> DataLoader:
    """Load validation data from data/val into a DataLoader.

    Args:
        batch_size: Batch size for the DataLoader
        num_workers: Number of workers for the DataLoader   
    Returns:
        DataLoader containing validation data
    """
    val_path = Path("data/val")
    val_dataset = RiceDataset(val_path)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)


def load_test_data(batch_size: int, num_workers: int) -> DataLoader:
    """Load test data from data/test into a DataLoader.

    Args:
        batch_size: Batch size for the DataLoader
        num_workers: Number of workers for the DataLoader
    Returns:
        DataLoader containing test data
    """
    test_path = Path("data/test")
    test_dataset = RiceDataset(test_path)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == "__main__":
    typer.run(preprocess)
