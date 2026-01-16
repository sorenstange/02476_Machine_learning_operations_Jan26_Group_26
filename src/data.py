from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf

def get_config():
    return OmegaConf.load("configs/config.yaml")


class RiceDataset(Dataset):
    """Dataset for Rice Image classification with 5 rice types."""

    def __init__(
        self,
        data_path: Path,
        transform: bool = True,
        config=None,
    ) -> None:
        if config is None:
            config = get_config()

        self.data_path = data_path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        self.class_names = sorted(
            ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        if transform:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (
                            config.data_parameters.augmentation.shape[0],
                            config.data_parameters.augmentation.shape[1],
                        )
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[config.data_parameters.augmentation.mean],
                        std=[config.data_parameters.augmentation.std],
                    ),
                ]
            )
        else:
            self.transform = None

        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.data_path / cls_name
            if not cls_dir.exists():
                raise ValueError(f"Class folder not found: {cls_dir}")

            image_files = [
                f for f in cls_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
            if not image_files:
                raise ValueError(f"No image files found in {cls_dir}")

            label = self.class_to_idx[cls_name]
            self.samples.extend((img_path, label) for img_path in image_files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

def preprocess():
    config = get_config()
    print("Preprocessing data...")

    dataset = RiceDataset(
        Path(config.data_parameters.dataset_raw_path),
        config=config,
    )

    output_folder = Path(config.data_parameters.dataset_processed_path)

    if output_folder.exists():
        print(f"Output folder already exists: {output_folder}")
        print("Skipping preprocessing to avoid mixing old and new data.")
        return

    files_by_class: Dict[str, List[Path]] = {}
    for cls in dataset.class_names:
        cls_dir = dataset.data_path / cls
        files = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        if not files:
            raise ValueError(f"No image files found in class folder: {cls_dir}")
        files_by_class[cls] = files

    for split in ["train", "val", "test"]:
        for cls in dataset.class_names:
            (output_folder / split / cls).mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)

    for cls, files in files_by_class.items():
        files = list(files)
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(config.data_parameters.train_fraction * n))
        n_val = int(round(config.data_parameters.validation_fraction * n))

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        for src in train_files:
            shutil.copy2(src, output_folder / "train" / cls / src.name)
        for src in val_files:
            shutil.copy2(src, output_folder / "val" / cls / src.name)
        for src in test_files:
            shutil.copy2(src, output_folder / "test" / cls / src.name)

    print(f"Created stratified splits at {output_folder}")
    print("Done!")


def load_train_data(batch_size: int, num_workers: int) -> DataLoader:
    config = get_config()
    train_dataset = RiceDataset(Path("data/train"), config=config)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def load_val_data(batch_size: int, num_workers: int) -> DataLoader:
    config = get_config()
    val_dataset = RiceDataset(Path("data/val"), config=config)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def load_test_data(batch_size: int, num_workers: int) -> DataLoader:
    config = get_config()
    test_dataset = RiceDataset(Path("data/test"), config=config)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

if __name__ == "__main__":
    preprocess()
