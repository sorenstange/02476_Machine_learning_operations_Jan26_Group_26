from torch.utils.data import Dataset
import torch
from pathlib import Path
from data import MyDataset, load_train_data, load_val_data, load_test_data

def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(Path("Rice_Image_Dataset"))
    assert isinstance(dataset, Dataset)

def test_load_train_data():
    """Test loading training data."""
    train_loader = load_train_data(batch_size=16, num_workers=0)
    assert isinstance(train_loader, torch.utils.data.DataLoader), "Train loader is not a DataLoader"
    assert len(train_loader.dataset) == 52_500, "Train loader does not have the correct length" # Dataset is 70% of total data

def test_load_val_data():
    """Test loading validation data."""
    val_loader = load_val_data(batch_size=16, num_workers=0)
    assert isinstance(val_loader, torch.utils.data.DataLoader), "Validation loader is not a DataLoader"
    assert len(val_loader.dataset) == 11_250, "Validation loader does not have the correct length" # Dataset is 15% of total data

def test_load_test_data():
    """Test loading test data."""
    test_loader = load_test_data(batch_size=16, num_workers=0)
    assert isinstance(test_loader, torch.utils.data.DataLoader), "Test loader is not a DataLoader"
    assert len(test_loader.dataset) == 11_250, "Test loader does not have the correct length" # Dataset is 15% of total data