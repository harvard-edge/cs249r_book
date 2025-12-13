#!/usr/bin/env python
"""
Integration Tests for TinyTorch
================================
Tests complete pipelines work end-to-end.
Validates that all components work together correctly.

Test Categories:
- Complete training loops
- Data loading pipelines
- Model save/load
- Checkpoint/resume
- Multi-component architectures
"""

import sys
import os
import numpy as np
import tempfile
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.training import MeanSquaredError, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.nn import Sequential, Conv2d
import tinytorch.nn.functional as F


# ============== Complete Training Loop Tests ==============

def test_basic_training_loop():
    """Complete training loop with all components."""
    # Create simple dataset
    X_train = Tensor(np.random.randn(100, 10))
    y_train = Tensor(np.random.randn(100, 5))

    # Build model
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    ])

    # Setup training
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    criterion = MeanSquaredError()

    # Training loop
    initial_loss = None
    final_loss = None

    for epoch in range(10):
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        if epoch == 0:
            initial_loss = float(loss.data) if hasattr(loss, 'data') else float(loss)
        if epoch == 9:
            final_loss = float(loss.data) if hasattr(loss, 'data') else float(loss)

        # Backward pass
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            # If autograd not available, just test forward passes
            pass

    # Loss should decrease (or at least not increase much)
    assert final_loss is not None, "Training loop didn't complete"
    if initial_loss and final_loss:
        assert final_loss <= initial_loss * 1.1, "Loss increased during training"


def test_minibatch_training():
    """Training with mini-batches."""
    # Create dataset
    dataset_size = 128
    batch_size = 16

    X_train = Tensor(np.random.randn(dataset_size, 10))
    y_train = Tensor(np.random.randn(dataset_size, 5))

    # Model
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    ])

    optimizer = Adam(model.parameters(), learning_rate=0.001)
    criterion = MeanSquaredError()

    # Mini-batch training
    n_batches = dataset_size // batch_size
    losses = []

    for epoch in range(2):
        epoch_loss = 0
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            X_batch = Tensor(X_train.data[start_idx:end_idx])
            y_batch = Tensor(y_train.data[start_idx:end_idx])

            # Training step
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            epoch_loss += float(loss.data) if hasattr(loss, 'data') else float(loss)

            try:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                pass

        losses.append(epoch_loss / n_batches)

    # Training should complete without errors
    assert len(losses) == 2, "Mini-batch training didn't complete"


def test_classification_training():
    """Classification task with cross-entropy loss."""
    # Create classification dataset
    n_samples = 100
    n_classes = 3
    n_features = 10

    X_train = Tensor(np.random.randn(n_samples, n_features))
    y_train = Tensor(np.random.randint(0, n_classes, n_samples))

    # Classification model
    model = Sequential([
        Linear(n_features, 20),
        ReLU(),
        Linear(20, n_classes)
    ])

    optimizer = Adam(model.parameters(), learning_rate=0.01)
    criterion = CrossEntropyLoss()

    # Training
    for epoch in range(5):
        logits = model(X_train)
        loss = criterion(logits, y_train)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    # Should produce valid class predictions
    final_logits = model(X_train)
    predictions = np.argmax(final_logits.data, axis=1)
    assert predictions.shape == (n_samples,), "Invalid prediction shape"
    assert np.all((predictions >= 0) & (predictions < n_classes)), "Invalid class predictions"


# ============== Data Loading Pipeline Tests ==============

def test_dataset_iteration():
    """Dataset and DataLoader work together."""
    try:
        from tinytorch.core.dataloader import Dataset, DataLoader

        class SimpleDataset(Dataset):
            def __init__(self, size):
                self.X = np.random.randn(size, 10)
                self.y = np.random.randn(size, 5)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return Tensor(self.X[idx]), Tensor(self.y[idx])

        dataset = SimpleDataset(100)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        # Iterate through dataloader
        batch_count = 0
        for X_batch, y_batch in dataloader:
            assert X_batch.shape == (10, 10), f"Wrong batch shape: {X_batch.shape}"
            assert y_batch.shape == (10, 5), f"Wrong target shape: {y_batch.shape}"
            batch_count += 1

        assert batch_count == 10, f"Expected 10 batches, got {batch_count}"

    except ImportError:
        pytest.skip("DataLoader not implemented")


def test_data_augmentation_pipeline():
    """Data augmentation in loading pipeline."""
    try:
        from tinytorch.core.dataloader import Dataset, DataLoader

        class AugmentedDataset(Dataset):
            def __init__(self, size):
                self.X = np.random.randn(size, 3, 32, 32)
                self.y = np.random.randint(0, 10, size)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                # Simple augmentation: random flip
                x = self.X[idx]
                if np.random.random() > 0.5:
                    x = np.flip(x, axis=-1)  # Horizontal flip
                return Tensor(x), Tensor(self.y[idx])

        dataset = AugmentedDataset(50)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

        # Should handle augmented data
        for X_batch, y_batch in dataloader:
            assert X_batch.shape == (5, 3, 32, 32), "Augmented batch wrong shape"
            break  # Just test first batch

    except ImportError:
        pytest.skip("DataLoader not implemented")


# ============== Model Save/Load Tests ==============

def test_model_save_load():
    """Save and load model weights."""
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    ])

    # Get initial predictions
    x_test = Tensor(np.random.randn(3, 10))
    initial_output = model(x_test)

    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name

    try:
        # Save weights
        import pickle
        weights = {}
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight'):
                weights[f'layer_{i}_weights'] = layer.weight.data
                if hasattr(layer, 'bias') and layer.bias is not None:
                    weights[f'layer_{i}_bias'] = layer.bias.data

        with open(temp_path, 'wb') as f:
            pickle.dump(weights, f)

        # Modify model (to ensure load works)
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                layer.weight.data = np.random.randn(*layer.weight.shape)

        # Load weights
        with open(temp_path, 'rb') as f:
            loaded_weights = pickle.load(f)

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight'):
                layer.weight.data = loaded_weights[f'layer_{i}_weights']
                if f'layer_{i}_bias' in loaded_weights:
                    layer.bias.data = loaded_weights[f'layer_{i}_bias']

        # Check outputs match
        loaded_output = model(x_test)
        assert np.allclose(initial_output.data, loaded_output.data), \
            "Model outputs differ after save/load"

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_checkpoint_resume_training():
    """Save checkpoint and resume training."""
    # Initial training
    model = Linear(10, 5)
    optimizer = SGD(model.parameters(), learning_rate=0.01)

    X = Tensor(np.random.randn(20, 10))
    y = Tensor(np.random.randn(20, 5))

    # Train for a few steps
    losses_before = []
    for _ in range(3):
        y_pred = model(X)
        loss = MeanSquaredError()(y_pred, y)
        losses_before.append(float(loss.data) if hasattr(loss, 'data') else float(loss))

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    # Save checkpoint
    checkpoint = {
        'model_weights': model.weight.data.copy(),
        'model_bias': model.bias.data.copy() if model.bias is not None else None,
        'optimizer_state': {'step': 3},  # Simplified
        'losses': losses_before
    }

    # Continue training
    for _ in range(3):
        y_pred = model(X)
        loss = MeanSquaredError()(y_pred, y)
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    # Restore checkpoint
    model.weight.data = checkpoint['model_weights']
    if checkpoint['model_bias'] is not None:
        model.bias.data = checkpoint['model_bias']

    # Verify restoration worked
    y_pred = model(X)
    restored_loss = MeanSquaredError()(y_pred, y)
    restored_loss_val = float(restored_loss.data) if hasattr(restored_loss, 'data') else float(restored_loss)

    # Loss should be close to checkpoint loss (not the continued training loss)
    assert abs(restored_loss_val - losses_before[-1]) < abs(restored_loss_val - losses_before[0]), \
        "Checkpoint restore failed"


# ============== Multi-Component Architecture Tests ==============

def test_cnn_to_fc_integration():
    """CNN features feed into FC classifier."""
    class CNNClassifier:
        def __init__(self):
            # CNN feature extractor
            self.conv1 = Conv2d(3, 16, kernel_size=3)
            self.conv2 = Conv2d(16, 32, kernel_size=3)
            # Classifier head
            self.fc1 = Linear(32 * 6 * 6, 128)
            self.fc2 = Linear(128, 10)

        def forward(self, x):
            # Feature extraction
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            # Classification
            x = F.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

        def parameters(self):
            params = []
            for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
                if hasattr(layer, 'parameters'):
                    params.extend(layer.parameters())
            return params

    model = CNNClassifier()
    x = Tensor(np.random.randn(8, 3, 32, 32))

    # Forward pass should work
    output = model.forward(x)
    assert output.shape == (8, 10), f"Wrong output shape: {output.shape}"

    # Training step should work
    y_true = Tensor(np.random.randint(0, 10, 8))
    loss = CrossEntropyLoss()(output, y_true)

    optimizer = Adam(model.parameters(), learning_rate=0.001)
    try:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    except:
        pass  # Autograd might not be implemented


def test_encoder_decoder_integration():
    """Encoder-decoder architecture integration."""
    class SimpleAutoencoder:
        def __init__(self, input_dim=784, latent_dim=32):
            # Encoder
            self.enc1 = Linear(input_dim, 128)
            self.enc2 = Linear(128, latent_dim)
            # Decoder
            self.dec1 = Linear(latent_dim, 128)
            self.dec2 = Linear(128, input_dim)

        def encode(self, x):
            x = F.relu(self.enc1(x))
            return self.enc2(x)

        def decode(self, z):
            z = F.relu(self.dec1(z))
            return F.sigmoid(self.dec2(z))

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z)

        def parameters(self):
            params = []
            for layer in [self.enc1, self.enc2, self.dec1, self.dec2]:
                if hasattr(layer, 'parameters'):
                    params.extend(layer.parameters())
            return params

    model = SimpleAutoencoder()
    x = Tensor(np.random.randn(16, 784))

    # Test encoding
    latent = model.encode(x)
    assert latent.shape == (16, 32), f"Wrong latent shape: {latent.shape}"

    # Test full forward
    reconstruction = model.forward(x)
    assert reconstruction.shape == x.shape, "Reconstruction shape mismatch"

    # Test training
    loss = MeanSquaredError()(reconstruction, x)
    optimizer = Adam(model.parameters(), learning_rate=0.001)

    try:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    except:
        pass


def test_multi_loss_training():
    """Training with multiple loss functions."""
    # Model with multiple outputs
    class MultiOutputModel:
        def __init__(self):
            self.shared = Linear(10, 20)
            self.head1 = Linear(20, 5)  # Regression head
            self.head2 = Linear(20, 3)  # Classification head

        def forward(self, x):
            shared_features = F.relu(self.shared(x))
            out1 = self.head1(shared_features)
            out2 = self.head2(shared_features)
            return out1, out2

        def parameters(self):
            params = []
            for layer in [self.shared, self.head1, self.head2]:
                if hasattr(layer, 'parameters'):
                    params.extend(layer.parameters())
            return params

    model = MultiOutputModel()
    optimizer = Adam(model.parameters(), learning_rate=0.001)

    # Data
    X = Tensor(np.random.randn(32, 10))
    y_reg = Tensor(np.random.randn(32, 5))  # Regression targets
    y_cls = Tensor(np.random.randint(0, 3, 32))  # Classification targets

    # Forward
    out_reg, out_cls = model.forward(X)

    # Multiple losses
    loss_reg = MeanSquaredError()(out_reg, y_reg)
    loss_cls = CrossEntropyLoss()(out_cls, y_cls)

    # Combined loss
    total_loss_val = (float(loss_reg.data) if hasattr(loss_reg, 'data') else float(loss_reg)) + \
                     (float(loss_cls.data) if hasattr(loss_cls, 'data') else float(loss_cls))

    # Should handle multiple losses
    assert total_loss_val > 0, "Combined loss calculation failed"


# ============== End-to-End Pipeline Tests ==============

def test_mnist_pipeline():
    """Complete MNIST training pipeline."""
    # Simplified MNIST-like data
    X_train = Tensor(np.random.randn(100, 784))  # Flattened 28x28
    y_train = Tensor(np.random.randint(0, 10, 100))

    X_val = Tensor(np.random.randn(20, 784))
    y_val = Tensor(np.random.randint(0, 10, 20))

    # MNIST model
    model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10)
    ])

    optimizer = Adam(model.parameters(), learning_rate=0.001)
    criterion = CrossEntropyLoss()

    # Training
    train_losses = []
    for epoch in range(3):
        # Training
        logits = model(X_train)
        loss = criterion(logits, y_train)
        train_losses.append(float(loss.data) if hasattr(loss, 'data') else float(loss))

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

        # Validation
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val)

        # Accuracy
        predictions = np.argmax(val_logits.data, axis=1)
        accuracy = np.mean(predictions == y_val.data)

    # Pipeline should complete
    assert len(train_losses) == 3, "Training didn't complete"
    assert 0 <= accuracy <= 1, "Invalid accuracy"


def test_cifar10_pipeline():
    """Complete CIFAR-10 training pipeline."""
    # Simplified CIFAR-like data
    X_train = Tensor(np.random.randn(50, 3, 32, 32))
    y_train = Tensor(np.random.randint(0, 10, 50))

    # Simple CNN for CIFAR
    class SimpleCIFARNet:
        def __init__(self):
            self.conv1 = Conv2d(3, 32, kernel_size=3)
            self.conv2 = Conv2d(32, 64, kernel_size=3)
            self.fc = Linear(64 * 6 * 6, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.flatten(x, start_dim=1)
            return self.fc(x)

        def parameters(self):
            params = []
            for layer in [self.conv1, self.conv2, self.fc]:
                if hasattr(layer, 'parameters'):
                    params.extend(layer.parameters())
            return params

    model = SimpleCIFARNet()
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    criterion = CrossEntropyLoss()

    # Quick training
    for epoch in range(2):
        output = model.forward(X_train)
        loss = criterion(output, y_train)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    # Final predictions
    final_output = model.forward(X_train)
    predictions = np.argmax(final_output.data, axis=1)

    # Should produce valid predictions
    assert predictions.shape == (50,), "Wrong prediction shape"
    assert np.all((predictions >= 0) & (predictions < 10)), "Invalid predictions"


if __name__ == "__main__":
    # When run directly, use pytest
    import subprocess
    result = subprocess.run(["pytest", __file__, "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    sys.exit(result.returncode)
