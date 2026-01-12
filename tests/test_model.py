import torch
from model import CNN_Model

def test_cnn_model_initialization():
    """Test CNN_Model initialization."""
    parameters = {
        "learning_rate": 0.001,
        "input_channels": 1,
        "input_size": (64, 64),
        "conv_layers": [16, 32],
        "fc_layers": [128],
        "output_dim": 5
    }
    model = CNN_Model(parameters)
    assert isinstance(model, CNN_Model), "Model is not an instance of CNN_Model"

def test_cnn_model_forward_pass():
    """Test CNN_Model forward pass."""
    parameters = {
        "learning_rate": 0.001,
        "input_channels": 1,
        "input_size": (64, 64),
        "conv_layers": [16, 32],
        "fc_layers": [128],
        "output_dim": 5
    }
    model = CNN_Model(parameters)
    dummy_input = torch.randn(4, 1, 64, 64)  # Batch size of 4
    output = model(dummy_input)
    assert output.shape == (4, 5), f"Output shape is incorrect: {output.shape}"

def test_cnn_model_training_step():
    """Test CNN_Model training step."""
    parameters = {
        "learning_rate": 0.001,
        "input_channels": 1,
        "input_size": (64, 64),
        "conv_layers": [16, 32],
        "fc_layers": [128],
        "output_dim": 5
    }
    model = CNN_Model(parameters)
    dummy_input = torch.randn(4, 1, 64, 64)  # Batch size of 4
    dummy_labels = torch.randint(0, 5, (4,))  # Random labels for 5 classes
    batch = (dummy_input, dummy_labels)
    loss = model.training_step(batch, 0)
    assert loss is not None, "Training step did not return a loss"

def test_cnn_model_validation_step():
    """Test CNN_Model validation step."""
    parameters = {
        "learning_rate": 0.001,
        "input_channels": 1,
        "input_size": (64, 64),
        "conv_layers": [16, 32],
        "fc_layers": [128],
        "output_dim": 5
    }
    model = CNN_Model(parameters)
    dummy_input = torch.randn(4, 1, 64, 64)  # Batch size of 4
    dummy_labels = torch.randint(0, 5, (4,))  # Random labels for 5 classes
    batch = (dummy_input, dummy_labels)
    loss = model.validation_step(batch, 0)
    assert loss is not None, "Validation step did not return a loss"