#!/usr/bin/env python3
"""
Comprehensive Gradient Flow Tests for TinyTorch
================================================

Tests that gradients flow correctly through:
1. Simple networks (single layer)
2. Multi-layer networks (MLP)
3. Convolutional networks (CNN)
4. Attention mechanisms
5. Complete training loops

This ensures backpropagation works correctly end-to-end.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Dropout
from tinytorch.core.activations import ReLU, Sigmoid, Softmax
from tinytorch.core.losses import MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.core.spatial import Conv2d, MaxPool2d
from tinytorch.core.autograd import enable_autograd

# Enable autograd
enable_autograd()

def test_simple_linear_gradient_flow():
    """Test gradients flow through a single linear layer"""
    print("\n" + "="*70)
    print("TEST 1: Simple Linear Layer Gradient Flow")
    print("="*70)

    # Create simple network: Linear(2->1)
    layer = Linear(2, 1)

    # Input
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    target = Tensor([[3.0]])

    # Forward pass
    output = layer.forward(x)

    # Loss
    loss_fn = MSELoss()
    loss = loss_fn.forward(output, target)

    print(f"Initial loss: {float(loss.data):.4f}")
    print(f"Initial weight shape: {layer.weight.shape}")
    print(f"Initial bias shape: {layer.bias.shape}")

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert layer.weight.grad is not None, "Weight gradient is None!"
    assert layer.bias.grad is not None, "Bias gradient is None!"
    assert x.grad is not None, "Input gradient is None!"

    # Check gradients are non-zero
    weight_grad_norm = np.linalg.norm(layer.weight.grad.data)
    bias_grad_norm = np.linalg.norm(layer.bias.grad.data)
    input_grad_norm = np.linalg.norm(x.grad.data)

    print(f"\n✓ Weight gradient norm: {weight_grad_norm:.6f}")
    print(f"✓ Bias gradient norm: {bias_grad_norm:.6f}")
    print(f"✓ Input gradient norm: {input_grad_norm:.6f}")

    assert weight_grad_norm > 1e-6, f"Weight gradients too small: {weight_grad_norm}"
    assert bias_grad_norm > 1e-6, f"Bias gradients too small: {bias_grad_norm}"
    assert input_grad_norm > 1e-6, f"Input gradients too small: {input_grad_norm}"

    print("\n✅ TEST PASSED: Gradients flow correctly through linear layer")
    return True


def test_mlp_gradient_flow():
    """Test gradients flow through multi-layer perceptron"""
    print("\n" + "="*70)
    print("TEST 2: Multi-Layer Perceptron Gradient Flow")
    print("="*70)

    # Create MLP: Input(4) -> Linear(4->8) -> ReLU -> Linear(8->2)
    layer1 = Linear(4, 8)
    activation = ReLU()
    layer2 = Linear(8, 2)

    # Input and target
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    target = Tensor(np.array([[1, 0], [0, 1], [1, 0]]))

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")

    # Forward pass
    h1 = layer1.forward(x)
    h1_activated = activation.forward(h1)
    output = layer2.forward(h1_activated)

    print(f"Hidden layer shape: {h1.shape}")
    print(f"Output shape: {output.shape}")

    # Loss
    loss_fn = MSELoss()
    loss = loss_fn.forward(output, target)

    print(f"Initial loss: {float(loss.data):.4f}")

    # Backward pass
    loss.backward()

    # Check all layer gradients exist
    assert layer1.weight.grad is not None, "Layer1 weight gradient is None!"
    assert layer1.bias.grad is not None, "Layer1 bias gradient is None!"
    assert layer2.weight.grad is not None, "Layer2 weight gradient is None!"
    assert layer2.bias.grad is not None, "Layer2 bias gradient is None!"

    # Check gradient magnitudes
    l1_weight_norm = np.linalg.norm(layer1.weight.grad.data)
    l1_bias_norm = np.linalg.norm(layer1.bias.grad.data)
    l2_weight_norm = np.linalg.norm(layer2.weight.grad.data)
    l2_bias_norm = np.linalg.norm(layer2.bias.grad.data)

    print(f"\n✓ Layer1 weight gradient norm: {l1_weight_norm:.6f}")
    print(f"✓ Layer1 bias gradient norm: {l1_bias_norm:.6f}")
    print(f"✓ Layer2 weight gradient norm: {l2_weight_norm:.6f}")
    print(f"✓ Layer2 bias gradient norm: {l2_bias_norm:.6f}")

    assert l1_weight_norm > 1e-6, "Layer1 weight gradients too small"
    assert l1_bias_norm > 1e-6, "Layer1 bias gradients too small"
    assert l2_weight_norm > 1e-6, "Layer2 weight gradients too small"
    assert l2_bias_norm > 1e-6, "Layer2 bias gradients too small"

    print("\n✅ TEST PASSED: Gradients flow correctly through MLP")
    return True


def test_mlp_training_updates():
    """Test that MLP actually learns (loss decreases)"""
    print("\n" + "="*70)
    print("TEST 3: MLP Training - Loss Reduction")
    print("="*70)

    # Create simple MLP
    layer1 = Linear(2, 4)
    activation = ReLU()
    layer2 = Linear(4, 1)

    # Simple dataset (XOR-like)
    X = Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), requires_grad=False)
    y = Tensor(np.array([[0.0], [1.0], [1.0], [0.0]]))

    # Optimizer
    optimizer = SGD([layer1.weight, layer1.bias, layer2.weight, layer2.bias], lr=0.1)
    loss_fn = MSELoss()

    losses = []

    print("Training for 50 epochs...")
    for epoch in range(50):
        # Forward
        h1 = layer1.forward(X)
        h1_act = activation.forward(h1)
        output = layer2.forward(h1_act)

        # Loss
        loss = loss_fn.forward(output, y)
        losses.append(float(loss.data))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}: Loss = {float(loss.data):.6f}")

    # Check loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = initial_loss - final_loss
    reduction_pct = (reduction / initial_loss) * 100

    print(f"\n✓ Initial loss: {initial_loss:.6f}")
    print(f"✓ Final loss: {final_loss:.6f}")
    print(f"✓ Reduction: {reduction:.6f} ({reduction_pct:.1f}%)")

    assert final_loss < initial_loss, f"Loss didn't decrease! Initial: {initial_loss}, Final: {final_loss}"
    assert reduction_pct > 10, f"Loss reduction too small: {reduction_pct:.1f}%"

    print("\n✅ TEST PASSED: MLP learns successfully (loss decreases)")
    return True


def test_cnn_gradient_flow():
    """Test gradients flow through convolutional layers"""
    print("\n" + "="*70)
    print("TEST 4: CNN Gradient Flow")
    print("="*70)

    # Create simple CNN: Conv2d -> ReLU -> Linear
    conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0)
    activation = ReLU()

    # Input: batch=2, channels=1, height=8, width=8
    x = Tensor(np.random.randn(2, 1, 8, 8), requires_grad=True)

    print(f"Input shape: {x.shape}")
    print(f"Conv weight shape: {conv.weight.shape}")

    # Forward through conv
    conv_out = conv.forward(x)
    print(f"Conv output shape: {conv_out.shape}")

    activated = activation.forward(conv_out)

    # Flatten for linear layer
    batch_size = activated.shape[0]
    flattened_size = np.prod(activated.shape[1:])
    # Use reshape method to maintain gradient flow
    flattened = activated.reshape(batch_size, flattened_size)

    linear = Linear(flattened_size, 2)
    output = linear.forward(flattened)

    print(f"Flattened shape: {flattened.shape}")
    print(f"Output shape: {output.shape}")

    # Loss
    target = Tensor(np.array([[1, 0], [0, 1]]))
    loss_fn = MSELoss()
    loss = loss_fn.forward(output, target)

    print(f"Initial loss: {float(loss.data):.4f}")

    # Backward
    loss.backward()

    # Check gradients
    assert conv.weight.grad is not None, "Conv weight gradient is None!"
    assert conv.bias.grad is not None, "Conv bias gradient is None!"
    assert linear.weight.grad is not None, "Linear weight gradient is None!"

    weight_grad_norm = np.linalg.norm(conv.weight.grad.data)
    conv_bias_norm = np.linalg.norm(conv.bias.grad.data)
    linear_grad_norm = np.linalg.norm(linear.weight.grad.data)

    print(f"\n✓ Conv weight gradient norm: {weight_grad_norm:.6f}")
    print(f"✓ Conv bias gradient norm: {conv_bias_norm:.6f}")
    print(f"✓ Linear weight gradient norm: {linear_grad_norm:.6f}")

    assert weight_grad_norm > 1e-6, f"Conv weight gradients too small: {weight_grad_norm}"
    assert conv_bias_norm > 1e-6, f"Conv bias gradients too small: {conv_bias_norm}"
    assert linear_grad_norm > 1e-6, f"Linear gradients too small: {linear_grad_norm}"

    print("\n✅ TEST PASSED: Gradients flow correctly through CNN")
    return True


def test_cnn_training_updates():
    """Test that CNN actually learns on simple data"""
    print("\n" + "="*70)
    print("TEST 5: CNN Training - Loss Reduction")
    print("="*70)

    # Simple CNN
    conv = Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    activation = ReLU()

    # Simple data: 4 samples, 1 channel, 4x4 images
    X = Tensor(np.random.randn(4, 1, 4, 4), requires_grad=False)

    # After conv: (4, 2, 4, 4) -> flatten to (4, 32)
    conv_out_size = 2 * 4 * 4  # channels * height * width
    linear = Linear(conv_out_size, 2)

    y = Tensor(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]))

    # Get parameters with gradients
    params = []
    for p in [conv.weight, conv.bias, linear.weight, linear.bias]:
        if not p.requires_grad:
            p.requires_grad = True
        params.append(p)

    # Optimizer
    optimizer = SGD(params, lr=0.01)
    loss_fn = MSELoss()

    losses = []

    print("Training for 30 epochs...")
    for epoch in range(30):
        # Forward
        conv_out = conv.forward(X)
        activated = activation.forward(conv_out)

        # Flatten using reshape to maintain gradients
        batch_size = activated.shape[0]
        flattened = activated.reshape(batch_size, -1)

        output = linear.forward(flattened)

        # Loss
        loss = loss_fn.forward(output, y)
        losses.append(float(loss.data))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}: Loss = {float(loss.data):.6f}")

    # Check loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = initial_loss - final_loss
    reduction_pct = (reduction / initial_loss) * 100

    print(f"\n✓ Initial loss: {initial_loss:.6f}")
    print(f"✓ Final loss: {final_loss:.6f}")
    print(f"✓ Reduction: {reduction:.6f} ({reduction_pct:.1f}%)")

    assert final_loss < initial_loss, f"Loss didn't decrease! Initial: {initial_loss}, Final: {final_loss}"

    print("\n✅ TEST PASSED: CNN learns successfully (loss decreases)")
    return True


def test_gradient_accumulation():
    """Test that gradients accumulate correctly across batches"""
    print("\n" + "="*70)
    print("TEST 6: Gradient Accumulation")
    print("="*70)

    layer = Linear(2, 1)

    # Two batches
    x1 = Tensor([[1.0, 2.0]], requires_grad=True)
    x2 = Tensor([[3.0, 4.0]], requires_grad=True)
    target = Tensor([[1.0]])

    loss_fn = MSELoss()

    # Forward + backward on first batch (don't zero grad)
    out1 = layer.forward(x1)
    loss1 = loss_fn.forward(out1, target)
    loss1.backward()

    grad_after_first = np.array(layer.weight.grad.data)

    # Forward + backward on second batch (gradients should accumulate)
    out2 = layer.forward(x2)
    loss2 = loss_fn.forward(out2, target)
    loss2.backward()

    grad_after_second = layer.weight.grad.data

    # Gradients should have accumulated (not been replaced)
    grad_diff = np.linalg.norm(grad_after_second - grad_after_first)

    print(f"✓ Gradient after first batch norm: {np.linalg.norm(grad_after_first):.6f}")
    print(f"✓ Gradient after second batch norm: {np.linalg.norm(grad_after_second):.6f}")
    print(f"✓ Difference: {grad_diff:.6f}")

    assert grad_diff > 1e-6, "Gradients didn't accumulate properly"

    print("\n✅ TEST PASSED: Gradients accumulate correctly")
    return True


def main():
    """Run all gradient flow tests"""
    print("\n" + "="*70)
    print("  TINYTORCH GRADIENT FLOW TEST SUITE")
    print("="*70)

    tests = [
        ("Simple Linear", test_simple_linear_gradient_flow),
        ("MLP Gradient Flow", test_mlp_gradient_flow),
        ("MLP Training", test_mlp_training_updates),
        ("CNN Gradient Flow", test_cnn_gradient_flow),
        ("CNN Training", test_cnn_training_updates),
        ("Gradient Accumulation", test_gradient_accumulation),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAILED"))

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)

    for name, status in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Gradients flow correctly through TinyTorch.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
