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


class MyDataset(Dataset):
    """Dataset for Rice Image classification with 5 rice types.

    Supports greyscale images (single channel) by passing `grayscale=True`.
    """

    def __init__(self, data_path: Path, transform=None, grayscale: bool = True) -> None:
        """Initialize dataset from preprocessed splits.
        
        Args:
            data_path: Path to split folder (e.g., data/train)
            transform: Optional torchvision transforms
            grayscale: Whether to load images as grayscale (default: True)
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        self.grayscale = grayscale

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

        # Default transform based on image type
        if transform is None:
            if grayscale:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a sample (image, label) at the given index."""
        img_path, label = self.samples[index]

        if self.grayscale:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class RicePreprocessor:
    """Helper for preprocessing raw Rice Image Dataset into stratified splits."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Expected rice types
        self.class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
        
        # Collect file lists per class
        self.files_by_class: Dict[str, List[Path]] = {}
        for cls in self.class_names:
            cls_dir = self.data_path / cls
            if not cls_dir.exists():
                raise ValueError(f"Expected class folder not found: {cls_dir}")
            
            files = [p for p in cls_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if not files:
                raise ValueError(f"No image files found in class folder: {cls_dir}")
            self.files_by_class[cls] = files

    def preprocess(self, output_folder: Path, seed: int = 42, fraction: float = 1.0) -> None:
        """Create stratified 70/15/15 splits and copy files to output folder.

        Args:
            output_folder: Path to output directory for splits
            seed: Random seed for reproducibility
            fraction: Fraction of dataset to use (0.0 to 1.0). Default 1.0 uses all data.

        The output structure will be:
            output_folder/
              train/<class>/*
              val/<class>/*
              test/<class>/*
        """
        output_folder = Path(output_folder)
        
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Fraction must be between 0.0 and 1.0, got {fraction}")
        
        # Check if output folder exists
        if output_folder.exists():
            raise FileExistsError(
                f"Output folder already exists: {output_folder}\n"
                f"Please remove it manually before running preprocessing to avoid mixing old and new data."
            )
        
        # Create output directories
        for split in ["train", "val", "test"]:
            for cls in self.class_names:
                (output_folder / split / cls).mkdir(parents=True, exist_ok=True)
        
        rng = random.Random(seed)

        split_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
        split_files: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}, "test": {}}

        for cls, files in self.files_by_class.items():
            # Shuffle deterministically
            files = list(files)
            rng.shuffle(files)
            
            # Sample only the specified fraction
            if fraction < 1.0:
                n_sample = max(1, int(len(files) * fraction))
                files = files[:n_sample]

            n = len(files)
            n_train = int(round(0.70 * n))
            n_val = int(round(0.15 * n))
            n_test = n - n_train - n_val

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

        # Write manifest with counts and file lists
        manifest = {
            "classes": self.class_names,
            "counts": split_counts,
            "files": split_files,
        }
        manifest_path = output_folder / "split_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Print verification summary per class
        if fraction < 1.0:
            print(f"Created stratified splits at {output_folder} (seed={seed}, fraction={fraction:.2%}).")
        else:
            print(f"Created stratified splits at {output_folder} (seed={seed}).")
        for cls in self.class_names:
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


def load_train_data(batch_size: int, num_workers: int) -> DataLoader:
    """Load training data from data/train into a DataLoader.

    Args:
        batch_size: Batch size for the DataLoader
        num_workers: Number of workers for the DataLoader
    Returns:
        DataLoader containing training data
    """
    train_path = Path("data/train")
    train_dataset = MyDataset(train_path, grayscale=True)
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
    val_dataset = MyDataset(val_path, grayscale=True)
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
    test_dataset = MyDataset(test_path, grayscale=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def preprocess(
    data_path: Path = typer.Argument(..., help="Path to raw rice image dataset"),
    output_folder: Path = typer.Option(
        None, 
        help="Path to output processed splits (default: data at project root)"
    ),
    fraction: float = typer.Option(
        1.0,
        help="Fraction of dataset to process (0.0 to 1.0). Use smaller values for testing."
    ),
) -> None:
    """CLI entry: preprocess raw rice images into 70/15/15 splits."""
    # Set default output folder if not provided
    if output_folder is None:
        output_folder = Path(__file__).parent.parent / "data"
    
    print("Preprocessing data...")
    preprocessor = RicePreprocessor(Path(data_path))
    preprocessor.preprocess(output_folder, fraction=fraction)
    print(f"Done! Use MyDataset(Path('{output_folder}/train')) for training.")


if __name__ == "__main__":
    typer.run(preprocess)
