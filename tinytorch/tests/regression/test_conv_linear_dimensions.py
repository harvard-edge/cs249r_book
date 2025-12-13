"""
BUG TRACKING:
============
Bug ID: BUG-2024-11-25-001
Date Found: 2024-11-25
Found By: PyTorch Expert Architecture Review
Severity: High

DESCRIPTION:
CNN example fails with "Inner dimensions must match: 2304 != 1600" when connecting
Conv2d outputs to Linear layer inputs in CIFAR-10 training.

REPRODUCTION:
1. Load CIFAR-10 data (32x32 images, 3 channels)
2. Pass through Conv2d(3, 32, 3) -> MaxPool2d(2) -> Conv2d(32, 64, 3) -> MaxPool2d(2)
3. Flatten and pass to Linear(1600, 128)
4. ValueError raised because actual flattened size is 2304, not 1600

ROOT CAUSE:
Incorrect manual calculation of convolution output dimensions. The example assumed
wrong dimensions after pooling operations.

FIX:
Calculate actual dimensions:
- Input: (32, 32, 3)
- Conv1: (30, 30, 32) after 3x3 kernel
- Pool1: (15, 15, 32) after 2x2 pooling
- Conv2: (13, 13, 64) after 3x3 kernel
- Pool2: (6, 6, 64) after 2x2 pooling
- Flatten: 6 * 6 * 64 = 2304 features

PREVENTION:
This regression test ensures convolution output dimensions are correctly calculated
and match Linear layer input expectations.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tinytorch.core.tensor import Tensor
from tinytorch.nn import Conv2d, Linear
import tinytorch.nn.functional as F


def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """Helper to calculate convolution output dimensions."""
    return (input_size - kernel_size + 2 * padding) // stride + 1


def test_conv_to_linear_dimension_match():
    """
    Regression test ensuring Conv2d output dimensions match Linear input.
    This exact architecture failed in examples/alexnet_2012/train_cnn.py
    """
    print("🔬 Testing Conv2d -> Linear dimension compatibility...")

    # Exact architecture from failing CNN example
    batch_size = 32
    input_channels = 3
    input_height = 32
    input_width = 32

    # Layer definitions (from CNN example)
    conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
    conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=0)

    # Create dummy CIFAR-10 batch
    x = Tensor(np.random.randn(batch_size, input_channels, input_height, input_width))

    # Forward pass with dimension tracking
    print(f"Input shape: {x.shape}")

    # Conv1 + Pool1
    x = conv1(x)
    h1 = calculate_conv_output_size(32, 3)  # 30
    assert x.shape == (batch_size, 32, h1, h1), f"Conv1 output shape mismatch: {x.shape}"
    print(f"After Conv1: {x.shape}")

    x = F.max_pool2d(x, kernel_size=2)
    h2 = h1 // 2  # 15
    assert x.shape == (batch_size, 32, h2, h2), f"Pool1 output shape mismatch: {x.shape}"
    print(f"After Pool1: {x.shape}")

    # Conv2 + Pool2
    x = conv2(x)
    h3 = calculate_conv_output_size(h2, 3)  # 13
    assert x.shape == (batch_size, 64, h3, h3), f"Conv2 output shape mismatch: {x.shape}"
    print(f"After Conv2: {x.shape}")

    x = F.max_pool2d(x, kernel_size=2)
    h4 = h3 // 2  # 6
    assert x.shape == (batch_size, 64, h4, h4), f"Pool2 output shape mismatch: {x.shape}"
    print(f"After Pool2: {x.shape}")

    # Calculate correct flattened size
    correct_flat_size = 64 * h4 * h4  # 64 * 6 * 6 = 2304
    print(f"Correct flattened size: {correct_flat_size}")

    # The bug: example used 1600 instead of 2304
    incorrect_flat_size = 1600  # What the example incorrectly used

    # Test correct dimension
    fc_correct = Linear(correct_flat_size, 128)
    x_flat = x.reshape(batch_size, -1)
    assert x_flat.shape[1] == correct_flat_size, f"Flattened size {x_flat.shape[1]} != {correct_flat_size}"

    # This should work without error
    output = fc_correct(x_flat)
    assert output.shape == (batch_size, 128), f"FC output shape mismatch: {output.shape}"
    print("✅ Correct dimensions: Conv output matches Linear input")

    # Test that incorrect dimension raises error (the original bug)
    fc_incorrect = Linear(incorrect_flat_size, 128)
    try:
        output = fc_incorrect(x_flat)
        assert False, "Should have raised ValueError for dimension mismatch"
    except ValueError as e:
        print(f"✅ Correctly caught dimension mismatch: {e}")

    print("🎯 Conv->Linear dimension test PASSED!")
    return True


def test_conv_output_size_calculation():
    """Test that convolution output size is calculated correctly."""
    print("🔬 Testing convolution output size calculations...")

    test_cases = [
        # (input_size, kernel, stride, padding, expected_output)
        (32, 3, 1, 0, 30),  # Standard conv
        (32, 3, 1, 1, 32),  # Same padding
        (32, 3, 2, 0, 15),  # Strided conv
        (32, 5, 1, 2, 32),  # 5x5 kernel with padding
    ]

    for input_size, kernel, stride, padding, expected in test_cases:
        output = calculate_conv_output_size(input_size, kernel, stride, padding)
        assert output == expected, f"Failed: {input_size}, k={kernel}, s={stride}, p={padding}"
        print(f"  Input={input_size}, Kernel={kernel}, Stride={stride}, Pad={padding} -> Output={output} ✓")

    print("✅ All convolution size calculations correct!")
    return True


def test_typical_cnn_architectures():
    """Test dimension flow through typical CNN architectures."""
    print("🔬 Testing typical CNN architecture dimensions...")

    # LeNet-style architecture
    batch_size = 16

    # LeNet on 32x32 images (CIFAR-10)
    x = Tensor(np.random.randn(batch_size, 3, 32, 32))

    # Conv block 1: 3->6 channels
    conv1 = Conv2d(3, 6, kernel_size=5)
    x = conv1(x)  # -> (16, 6, 28, 28)
    assert x.shape == (batch_size, 6, 28, 28)
    x = F.max_pool2d(x, 2)  # -> (16, 6, 14, 14)
    assert x.shape == (batch_size, 6, 14, 14)

    # Conv block 2: 6->16 channels
    conv2 = Conv2d(6, 16, kernel_size=5)
    x = conv2(x)  # -> (16, 16, 10, 10)
    assert x.shape == (batch_size, 16, 10, 10)
    x = F.max_pool2d(x, 2)  # -> (16, 16, 5, 5)
    assert x.shape == (batch_size, 16, 5, 5)

    # Flatten and FC layers
    flat_size = 16 * 5 * 5  # 400
    x_flat = x.reshape(batch_size, -1)
    assert x_flat.shape == (batch_size, flat_size)

    fc1 = Linear(flat_size, 120)
    fc2 = Linear(120, 84)
    fc3 = Linear(84, 10)

    x = fc1(x_flat)
    assert x.shape == (batch_size, 120)
    x = fc2(x)
    assert x.shape == (batch_size, 84)
    x = fc3(x)
    assert x.shape == (batch_size, 10)

    print("✅ LeNet-style architecture dimensions flow correctly!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("REGRESSION TEST: Conv2d to Linear Dimension Compatibility")
    print("="*60)

    # Run all tests
    all_pass = True
    all_pass &= test_conv_output_size_calculation()
    all_pass &= test_conv_to_linear_dimension_match()
    all_pass &= test_typical_cnn_architectures()

    if all_pass:
        print("\n🏆 ALL REGRESSION TESTS PASSED!")
        print("The Conv->Linear dimension bug is prevented.")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
