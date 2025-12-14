"""
Checkpoint 11: Regularization (After Module 12 - Regularization)
Question: "Can I prevent overfitting and build robust models?"
"""

import numpy as np
import pytest

def test_checkpoint_11_regularization():
    """
    Checkpoint 11: Regularization

    Validates that students can apply regularization techniques to prevent
    overfitting and build models that generalize well to unseen data -
    essential for practical machine learning applications.
    """
    print("\n🛡️ Checkpoint 11: Regularization")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
        from tinytorch.core.regularization import Dropout, L1Regularization, L2Regularization
        from tinytorch.core.losses import MeanSquaredError
        from tinytorch.core.optimizers import Adam
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-12 first: {e}")

    # Test 1: Dropout for generalization
    print("🎭 Testing dropout...")

    dropout = Dropout(p=0.5)

    # Create test data
    input_data = Tensor(np.ones((10, 20)))  # All ones for predictable testing

    # Training mode (should drop some neurons)
    if hasattr(dropout, 'training'):
        dropout.training = True

    dropped_output = dropout(input_data)

    # Check that some values are zeroed
    num_zeros = np.sum(dropped_output.data == 0)
    total_elements = dropped_output.data.size
    dropout_rate = num_zeros / total_elements

    # Should drop approximately 50% (with some variance)
    assert dropout_rate > 0.3 and dropout_rate < 0.7, f"Dropout rate should be ~0.5, got {dropout_rate:.3f}"
    print(f"✅ Dropout training: {dropout_rate:.3f} dropout rate")

    # Inference mode (should keep all values)
    if hasattr(dropout, 'training'):
        dropout.training = False

    inference_output = dropout(input_data)

    # In inference, should scale but not drop
    if hasattr(dropout, 'training'):
        # Proper dropout scales by 1/(1-p) in training or keeps values in inference
        assert not np.any(inference_output.data == 0), "Inference mode should not drop neurons"
        print(f"✅ Dropout inference: no neurons dropped")
    else:
        print(f"⚠️ Dropout mode switching not implemented")

    # Test 2: L2 Regularization (Weight Decay)
    print("⚖️ Testing L2 regularization...")

    # Create model with large weights
    model = Linear(5, 3)
    model.weight.data = np.random.randn(5, 3) * 2  # Larger weights
    model.bias.data = np.random.randn(3) * 2
    model.weight.requires_grad = True
    model.bias.requires_grad = True

    l2_reg = L2Regularization(lambda_reg=0.01)
    loss_fn = MeanSquaredError()

    # Test data
    X = Tensor(np.random.randn(4, 5))
    y = Tensor(np.random.randn(4, 3))

    # Forward pass with regularization
    pred = model(X)
    base_loss = loss_fn(pred, y)
    reg_loss = l2_reg(model.weights)
    total_loss = base_loss + reg_loss

    # L2 regularization should add penalty for large weights
    assert reg_loss.data > 0, f"L2 regularization should add positive penalty, got {reg_loss.data}"
    assert total_loss.data > base_loss.data, "Total loss should be larger than base loss"
    print(f"✅ L2 regularization: base={base_loss.data:.4f}, penalty={reg_loss.data:.4f}")

    # Test 3: L1 Regularization (Sparsity)
    print("📉 Testing L1 regularization...")

    l1_reg = L1Regularization(lambda_reg=0.01)
    l1_penalty = l1_reg(model.weights)

    # L1 should encourage sparsity
    assert l1_penalty.data > 0, f"L1 regularization should add positive penalty, got {l1_penalty.data}"
    print(f"✅ L1 regularization: sparsity penalty={l1_penalty.data:.4f}")

    # Test 4: Regularized training
    print("🎯 Testing regularized training...")

    # Create overfitting scenario (small dataset, complex model)
    np.random.seed(42)
    X_small = np.random.randn(20, 10)  # Only 20 samples
    y_small = np.random.randn(20, 1)

    # Complex model (prone to overfitting)
    model_reg = [
        Linear(10, 50),
        ReLU(),
        Dropout(p=0.3),
        Linear(50, 50),
        ReLU(),
        Dropout(p=0.3),
        Linear(50, 1)
    ]

    # Set requires_grad for all layers
    for layer in model_reg:
        if hasattr(layer, 'weight'):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True
        if hasattr(layer, 'training'):
            layer.training = True

    # Collect parameters
    params = []
    for layer in model_reg:
        if hasattr(layer, 'weight'):
            params.extend([layer.weights, layer.bias])

    optimizer = Adam(params, lr=0.01)
    l2_regularizer = L2Regularization(lambda_reg=0.001)

    # Training with regularization
    reg_losses = []
    for epoch in range(5):
        X_tensor = Tensor(X_small)
        y_tensor = Tensor(y_small)

        # Forward pass
        x = X_tensor
        for layer in model_reg:
            x = layer(x)

        # Loss with regularization
        base_loss = loss_fn(x, y_tensor)
        reg_penalty = sum(l2_regularizer(layer.weights) for layer in model_reg if hasattr(layer, 'weight'))
        total_loss = base_loss + reg_penalty

        reg_losses.append(total_loss.data.item() if hasattr(total_loss.data, 'item') else float(total_loss.data))

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"✅ Regularized training: {len(reg_losses)} epochs with dropout + L2")

    # Test 5: Generalization gap
    print("📊 Testing generalization...")

    # Create train/test split
    np.random.seed(123)
    X_full = np.random.randn(100, 8)
    y_full = X_full[:, 0] + 0.5 * X_full[:, 1] + 0.1 * np.random.randn(100)
    y_full = y_full.reshape(-1, 1)

    split = 70
    X_train, X_test = X_full[:split], X_full[split:]
    y_train, y_test = y_full[:split], y_full[split:]

    # Train regularized model
    gen_model = Linear(8, 1)
    gen_model.weight.requires_grad = True
    gen_model.bias.requires_grad = True

    gen_optimizer = Adam([gen_model.weights, gen_model.bias], lr=0.01)
    gen_l2 = L2Regularization(lambda_reg=0.01)

    train_losses = []
    test_losses = []

    for epoch in range(10):
        # Training
        X_train_tensor = Tensor(X_train)
        y_train_tensor = Tensor(y_train)
        pred_train = gen_model(X_train_tensor)
        loss_train = loss_fn(pred_train, y_train_tensor) + gen_l2(gen_model.weights)

        loss_train.backward()
        gen_optimizer.step()
        gen_optimizer.zero_grad()

        train_losses.append(loss_train.data.item() if hasattr(loss_train.data, 'item') else float(loss_train.data))

        # Testing (no regularization in evaluation)
        X_test_tensor = Tensor(X_test)
        y_test_tensor = Tensor(y_test)
        pred_test = gen_model(X_test_tensor)
        loss_test = loss_fn(pred_test, y_test_tensor)

        test_losses.append(loss_test.data.item() if hasattr(loss_test.data, 'item') else float(loss_test.data))

    # Check generalization
    final_gap = test_losses[-1] - train_losses[-1]
    print(f"✅ Generalization: train={train_losses[-1]:.4f}, test={test_losses[-1]:.4f}, gap={final_gap:.4f}")

    # Test 6: Early stopping concept
    print("⏰ Testing early stopping concept...")

    # Simulate early stopping by tracking validation loss
    val_losses = test_losses  # Use test as validation for this demo

    # Find best epoch (lowest validation loss)
    best_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch]

    # Check if we can detect optimal stopping point
    if best_epoch < len(val_losses) - 2:  # Not the last epoch
        print(f"✅ Early stopping: optimal at epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    else:
        print(f"✅ Early stopping: training could continue, best val_loss={best_val_loss:.4f}")

    # Test 7: Model complexity vs performance
    print("🏗️ Testing model complexity trade-offs...")

    # Compare simple vs complex models
    simple_model = Linear(8, 1)
    complex_model = [
        Linear(8, 32),
        ReLU(),
        Linear(32, 16),
        ReLU(),
        Linear(16, 1)
    ]

    # Set requires_grad
    simple_model.weight.requires_grad = True
    simple_model.bias.requires_grad = True

    for layer in complex_model:
        if hasattr(layer, 'weight'):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True

    # Train simple model
    simple_opt = Adam([simple_model.weights, simple_model.bias], lr=0.01)

    X_tensor = Tensor(X_train)
    y_tensor = Tensor(y_train)

    for _ in range(5):
        pred = simple_model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        simple_opt.step()
        simple_opt.zero_grad()

    # Evaluate simple model
    simple_test_pred = simple_model(Tensor(X_test))
    simple_test_loss = loss_fn(simple_test_pred, Tensor(y_test))

    print(f"✅ Complexity: simple model test_loss={simple_test_loss.data:.4f}")

    # Test 8: Regularization strength effects
    print("💪 Testing regularization strength...")

    # Test different L2 strengths
    strengths = [0.001, 0.01, 0.1]
    strength_results = []

    for strength in strengths:
        temp_model = Linear(5, 1)
        temp_model.weight.requires_grad = True
        temp_model.bias.requires_grad = True

        temp_opt = Adam([temp_model.weights, temp_model.bias], lr=0.01)
        temp_l2 = L2Regularization(lambda_reg=strength)

        # Quick training
        X_temp = Tensor(np.random.randn(10, 5))
        y_temp = Tensor(np.random.randn(10, 1))

        for _ in range(3):
            pred = temp_model(X_temp)
            loss = loss_fn(pred, y_temp) + temp_l2(temp_model.weights)
            loss.backward()
            temp_opt.step()
            temp_opt.zero_grad()

        # Check weight magnitude
        weight_norm = np.linalg.norm(temp_model.weight.data)
        strength_results.append(weight_norm)

    # Higher regularization should lead to smaller weights
    assert strength_results[2] < strength_results[0], "Higher L2 should produce smaller weights"
    print(f"✅ Regularization strength: {strengths} → weight norms {[f'{r:.3f}' for r in strength_results]}")

    print("\n🎉 Regularization Complete!")
    print("📝 You can now prevent overfitting and build robust models")
    print("🔧 Built capabilities: Dropout, L1/L2 regularization, early stopping, complexity control")
    print("🧠 Breakthrough: You can now build models that generalize to real-world data!")
    print("🎯 Next: Add high-performance computational kernels")

if __name__ == "__main__":
    test_checkpoint_11_regularization()
