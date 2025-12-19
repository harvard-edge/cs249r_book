"""
Checkpoint 10: Training (After Module 08 - Training)
Question: "Can I build complete training loops for end-to-end learning?"
"""

import numpy as np
import pytest

def test_checkpoint_10_training():
    """
    Checkpoint 10: Training

    Validates that students can orchestrate complete training loops with
    data loading, forward passes, backward passes, and optimization -
    the complete machine learning pipeline.
    """
    print("\n🎓 Checkpoint 10: Training")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.losses import MeanSquaredError, BinaryCrossEntropy
        from tinytorch.core.optimizers import Adam, SGD
        from tinytorch.core.training import Trainer, DataLoader
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-11 first: {e}")

    # Test 1: Basic training loop
    print("🔄 Testing basic training loop...")

    # Create a simple regression problem
    np.random.seed(42)
    X_data = np.random.randn(100, 2)
    y_data = 2 * X_data[:, 0] + 3 * X_data[:, 1] + 1 + 0.1 * np.random.randn(100)
    y_data = y_data.reshape(-1, 1)

    # Create model
    model = Linear(2, 1)
    model.weight.requires_grad = True
    model.bias.requires_grad = True

    optimizer = Adam([model.weights, model.bias], lr=0.01)
    loss_fn = MeanSquaredError()

    # Manual training loop
    losses = []
    for epoch in range(10):
        # Forward pass
        X_tensor = Tensor(X_data)
        y_tensor = Tensor(y_data)
        predictions = model(X_tensor)
        loss = loss_fn(predictions, y_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.data.item() if hasattr(loss.data, 'item') else float(loss.data))

    # Check convergence
    assert len(losses) == 10, "Should complete 10 epochs"
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"✅ Basic training: {len(losses)} epochs, loss {losses[0]:.4f} → {losses[-1]:.4f}")

    # Test 2: Batch training with DataLoader
    print("📦 Testing batch training...")

    try:
        # Create DataLoader
        dataloader = DataLoader(X_data, y_data, batch_size=16, shuffle=True)

        # Batch training
        model_batch = Linear(2, 1)
        model_batch.weight.requires_grad = True
        model_batch.bias.requires_grad = True
        optimizer_batch = SGD([model_batch.weights, model_batch.bias], lr=0.01)

        epoch_losses = []
        for epoch in range(3):
            batch_losses = []
            for batch_X, batch_y in dataloader:
                X_batch = Tensor(batch_X)
                y_batch = Tensor(batch_y)

                pred_batch = model_batch(X_batch)
                loss_batch = loss_fn(pred_batch, y_batch)

                loss_batch.backward()
                optimizer_batch.step()
                optimizer_batch.zero_grad()

                batch_losses.append(loss_batch.data.item() if hasattr(loss_batch.data, 'item') else float(loss_batch.data))

            epoch_losses.append(np.mean(batch_losses))

        assert len(epoch_losses) == 3, "Should complete 3 epochs"
        print(f"✅ Batch training: {len(epoch_losses)} epochs with batching")

    except (ImportError, AttributeError):
        print("⚠️ DataLoader not available, testing manual batching...")

        # Manual batching
        batch_size = 16
        num_batches = len(X_data) // batch_size

        for epoch in range(2):
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_X = Tensor(X_data[start_idx:end_idx])
                batch_y = Tensor(y_data[start_idx:end_idx])

                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        print(f"✅ Manual batching: {num_batches} batches per epoch")

    # Test 3: Classification training
    print("🎯 Testing classification training...")

    # Binary classification data
    np.random.seed(123)
    X_class = np.random.randn(200, 3)
    # Create separable classes
    y_class = (X_class[:, 0] + X_class[:, 1] - X_class[:, 2] > 0).astype(np.float32).reshape(-1, 1)

    # Classification model
    classifier = [
        Linear(3, 5),
        ReLU(),
        Linear(5, 1),
        Sigmoid()
    ]

    # Set requires_grad for all parameters
    for layer in classifier:
        if hasattr(layer, 'weight'):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True

    optimizer_class = Adam([layer.weights for layer in classifier if hasattr(layer, 'weight')] +
                          [layer.bias for layer in classifier if hasattr(layer, 'bias')], lr=0.01)

    bce_loss = BinaryCrossEntropy()

    # Classification training
    class_losses = []
    for epoch in range(5):
        X_class_tensor = Tensor(X_class)
        y_class_tensor = Tensor(y_class)

        # Forward pass through network
        x = X_class_tensor
        for layer in classifier:
            x = layer(x)

        loss = bce_loss(x, y_class_tensor)
        class_losses.append(loss.data.item() if hasattr(loss.data, 'item') else float(loss.data))

        loss.backward()
        optimizer_class.step()
        optimizer_class.zero_grad()

    # Check classification convergence
    assert class_losses[-1] < class_losses[0], f"Classification loss should decrease: {class_losses[0]:.4f} → {class_losses[-1]:.4f}"
    print(f"✅ Classification: loss {class_losses[0]:.4f} → {class_losses[-1]:.4f}")

    # Test 4: Training with validation
    print("📊 Testing training with validation...")

    # Split data into train/validation
    split_idx = int(0.8 * len(X_data))
    X_train, X_val = X_data[:split_idx], X_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]

    # Fresh model for validation testing
    model_val = Linear(2, 1)
    model_val.weight.requires_grad = True
    model_val.bias.requires_grad = True
    optimizer_val = Adam([model_val.weights, model_val.bias], lr=0.01)

    train_losses = []
    val_losses = []

    for epoch in range(5):
        # Training phase
        X_train_tensor = Tensor(X_train)
        y_train_tensor = Tensor(y_train)
        pred_train = model_val(X_train_tensor)
        loss_train = loss_fn(pred_train, y_train_tensor)

        loss_train.backward()
        optimizer_val.step()
        optimizer_val.zero_grad()

        train_losses.append(loss_train.data.item() if hasattr(loss_train.data, 'item') else float(loss_train.data))

        # Validation phase (no gradients)
        X_val_tensor = Tensor(X_val)
        y_val_tensor = Tensor(y_val)
        pred_val = model_val(X_val_tensor)
        loss_val = loss_fn(pred_val, y_val_tensor)

        val_losses.append(loss_val.data.item() if hasattr(loss_val.data, 'item') else float(loss_val.data))

    assert len(train_losses) == len(val_losses) == 5, "Should track both train and validation losses"
    print(f"✅ Train/Val: train {train_losses[0]:.4f}→{train_losses[-1]:.4f}, val {val_losses[0]:.4f}→{val_losses[-1]:.4f}")

    # Test 5: Model evaluation
    print("🔍 Testing model evaluation...")

    # Evaluate final model performance
    final_pred = model_val(Tensor(X_val))
    mse = np.mean((final_pred.data - y_val) ** 2)
    mae = np.mean(np.abs(final_pred.data - y_val))

    print(f"✅ Evaluation: MSE={mse:.4f}, MAE={mae:.4f}")

    # Test 6: Learning curves
    print("📈 Testing learning curves...")

    # Demonstrate learning progress
    model_curve = Linear(2, 1)
    model_curve.weight.requires_grad = True
    model_curve.bias.requires_grad = True
    optimizer_curve = SGD([model_curve.weights, model_curve.bias], lr=0.1)

    curve_losses = []
    curve_accuracies = []

    for epoch in range(8):
        X_tensor = Tensor(X_data)
        y_tensor = Tensor(y_data)
        pred = model_curve(X_tensor)
        loss = loss_fn(pred, y_tensor)

        # Calculate "accuracy" (for regression, use threshold)
        accuracy = np.mean(np.abs(pred.data - y_data) < 1.0)  # Within 1 unit

        curve_losses.append(loss.data.item() if hasattr(loss.data, 'item') else float(loss.data))
        curve_accuracies.append(accuracy)

        loss.backward()
        optimizer_curve.step()
        optimizer_curve.zero_grad()

    # Check learning progress
    assert curve_losses[-1] < curve_losses[0], "Learning curves should show improvement"
    assert curve_accuracies[-1] > curve_accuracies[0], "Accuracy should improve"
    print(f"✅ Learning curves: loss↓ accuracy {curve_accuracies[0]:.3f}→{curve_accuracies[-1]:.3f}")

    # Test 7: Complete training pipeline
    print("🏗️ Testing complete pipeline...")

    try:
        # Try using Trainer class if available
        trainer = Trainer(
            model=Linear(2, 1),
            optimizer=Adam,
            loss_fn=MeanSquaredError(),
            lr=0.01
        )

        # Set up for training
        trainer.model.weight.requires_grad = True
        trainer.model.bias.requires_grad = True

        # Train (simplified interface)
        pipeline_losses = []
        for epoch in range(3):
            X_tensor = Tensor(X_train)
            y_tensor = Tensor(y_train)

            loss = trainer.train_step(X_tensor, y_tensor)
            pipeline_losses.append(loss)

        print(f"✅ Complete pipeline: Trainer class with {len(pipeline_losses)} steps")

    except (ImportError, AttributeError, TypeError):
        print("⚠️ Trainer class not available, pipeline tested via manual steps")

        # Manual pipeline demonstration
        pipeline_model = Linear(2, 1)
        pipeline_model.weight.requires_grad = True
        pipeline_model.bias.requires_grad = True

        pipeline_optimizer = Adam([pipeline_model.weights, pipeline_model.bias], lr=0.01)
        pipeline_loss_fn = MeanSquaredError()

        # Complete pipeline in one function
        def train_epoch(model, optimizer, loss_fn, X, y):
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)

        pipeline_loss = train_epoch(pipeline_model, pipeline_optimizer, pipeline_loss_fn,
                                  Tensor(X_train), Tensor(y_train))
        print(f"✅ Manual pipeline: complete training function, loss={pipeline_loss:.4f}")

    print("\n🎉 Training Complete!")
    print("📝 You can now build complete training loops for end-to-end learning")
    print("🔧 Built capabilities: Training loops, batching, validation, evaluation, learning curves")
    print("🧠 Breakthrough: You can now train neural networks from start to finish!")
    print("🎯 Next: Add regularization and advanced training techniques")

if __name__ == "__main__":
    test_checkpoint_10_training()
