#!/usr/bin/env python
"""
Forward Pass Tests for TinyTorch
=================================
Tests that all architectures can do forward passes correctly.
This validates the "plumbing" - data flows through without errors.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
from tinytorch.nn import Sequential, Conv2d, TransformerBlock, Embedding, PositionalEncoding, LayerNorm
import tinytorch.nn.functional as F


class ForwardPassTester:
    """Test forward passes for various architectures."""

    def __init__(self):
        self.passed = []
        self.failed = []

    def test(self, name, func):
        """Run a test and track results."""
        try:
            func()
            self.passed.append(name)
            print(f"✅ {name}")
            return True
        except Exception as e:
            self.failed.append((name, str(e)))
            print(f"❌ {name}: {e}")
            return False

    def summary(self):
        """Print test summary."""
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"FORWARD PASS TESTS: {len(self.passed)}/{total} passed")
        if self.failed:
            print("\nFailed tests:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        return len(self.failed) == 0


# Test different layer types
def test_linear_forward():
    """Test Linear layer forward pass."""
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(3, 10))
    y = layer(x)
    assert y.shape == (3, 5)


def test_conv2d_forward():
    """Test Conv2d forward pass."""
    layer = Conv2d(3, 16, kernel_size=3)
    x = Tensor(np.random.randn(2, 3, 32, 32))
    y = layer(x)
    assert y.shape == (2, 16, 30, 30)


def test_conv2d_with_padding():
    """Test Conv2d with padding."""
    layer = Conv2d(3, 16, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 3, 32, 32))
    y = layer(x)
    assert y.shape == (2, 16, 32, 32)  # Same size with padding=1


def test_conv2d_with_stride():
    """Test Conv2d with stride."""
    layer = Conv2d(3, 16, kernel_size=3, stride=2)
    x = Tensor(np.random.randn(2, 3, 32, 32))
    y = layer(x)
    assert y.shape == (2, 16, 15, 15)  # (32-3)/2 + 1 = 15


# Test activation functions
def test_relu_forward():
    """Test ReLU activation."""
    x = Tensor(np.array([[-1, 0, 1], [2, -3, 4]]))
    y = F.relu(x)
    assert y.shape == x.shape


def test_sigmoid_forward():
    """Test Sigmoid activation."""
    x = Tensor(np.random.randn(2, 3))
    y = F.sigmoid(x)
    assert y.shape == x.shape
    # Check sigmoid bounds
    assert np.all(y.data >= 0) and np.all(y.data <= 1)


def test_tanh_forward():
    """Test Tanh activation."""
    x = Tensor(np.random.randn(2, 3))
    y = F.tanh(x)
    assert y.shape == x.shape
    # Check tanh bounds
    assert np.all(y.data >= -1) and np.all(y.data <= 1)


def test_softmax_forward():
    """Test Softmax activation."""
    x = Tensor(np.random.randn(2, 10))
    y = F.softmax(x, dim=-1)
    assert y.shape == x.shape
    # Check softmax sums to 1
    sums = np.sum(y.data, axis=-1)
    assert np.allclose(sums, 1.0)


# Test pooling operations
def test_maxpool2d_forward():
    """Test MaxPool2d."""
    x = Tensor(np.random.randn(2, 16, 32, 32))
    y = F.max_pool2d(x, kernel_size=2)
    assert y.shape == (2, 16, 16, 16)


def test_avgpool2d_forward():
    """Test AvgPool2d."""
    x = Tensor(np.random.randn(2, 16, 32, 32))
    y = F.avg_pool2d(x, kernel_size=2)
    assert y.shape == (2, 16, 16, 16)


# Test reshape operations
def test_flatten_forward():
    """Test flatten operation."""
    x = Tensor(np.random.randn(2, 3, 4, 5))
    y = F.flatten(x, start_dim=1)
    assert y.shape == (2, 60)  # 3*4*5 = 60


def test_reshape_forward():
    """Test reshape operation."""
    x = Tensor(np.random.randn(2, 3, 4))
    y = x.reshape(6, 4)
    assert y.shape == (6, 4)


# Test normalization layers
def test_layernorm_forward():
    """Test LayerNorm."""
    layer = LayerNorm(128)
    x = Tensor(np.random.randn(2, 10, 128))
    y = layer(x)
    assert y.shape == x.shape


def test_batchnorm_forward():
    """Test BatchNorm (if implemented)."""
    # Skip if not implemented
    try:
        from tinytorch.nn import BatchNorm1d
        layer = BatchNorm1d(128)
        x = Tensor(np.random.randn(32, 128))
        y = layer(x)
        assert y.shape == x.shape
    except ImportError:
        pass  # BatchNorm not implemented yet


# Test complex architectures
def test_sequential_forward():
    """Test Sequential container."""
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 30),
        ReLU(),
        Linear(30, 5)
    ])
    x = Tensor(np.random.randn(4, 10))
    y = model(x)
    assert y.shape == (4, 5)


def test_mlp_forward():
    """Test Multi-Layer Perceptron."""
    class MLP:
        def __init__(self):
            self.fc1 = Linear(784, 256)
            self.fc2 = Linear(256, 128)
            self.fc3 = Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    model = MLP()
    x = Tensor(np.random.randn(32, 784))  # MNIST batch
    y = model.forward(x)
    assert y.shape == (32, 10)


def test_cnn_forward():
    """Test Convolutional Neural Network."""
    class CNN:
        def __init__(self):
            self.conv1 = Conv2d(1, 32, 3)
            self.conv2 = Conv2d(32, 64, 3)
            self.fc1 = Linear(64 * 5 * 5, 128)
            self.fc2 = Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = CNN()
    x = Tensor(np.random.randn(16, 1, 28, 28))  # MNIST batch
    y = model.forward(x)
    assert y.shape == (16, 10)


def test_transformer_forward():
    """Test Transformer architecture."""
    class SimpleTransformer:
        def __init__(self):
            self.embed = Embedding(1000, 128)
            self.pos_enc = PositionalEncoding(128, 100)
            self.transformer = TransformerBlock(128, 8)
            self.ln = LayerNorm(128)
            self.output = Linear(128, 1000)

        def forward(self, x):
            x = self.embed(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            x = self.ln(x)
            # Reshape for output
            batch, seq, embed = x.shape
            x = x.reshape(batch * seq, embed)
            x = self.output(x)
            return x.reshape(batch, seq, 1000)

    model = SimpleTransformer()
    x = Tensor(np.random.randint(0, 1000, (4, 20)))  # Token batch
    y = model.forward(x)
    assert y.shape == (4, 20, 1000)


def test_residual_block_forward():
    """Test Residual Block (ResNet-style)."""
    class ResidualBlock:
        def __init__(self, channels):
            self.conv1 = Conv2d(channels, channels, 3, padding=1)
            self.conv2 = Conv2d(channels, channels, 3, padding=1)

        def forward(self, x):
            identity = x
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out = out + identity  # Residual connection
            return F.relu(out)

    block = ResidualBlock(64)
    x = Tensor(np.random.randn(2, 64, 16, 16))
    y = block.forward(x)
    assert y.shape == x.shape


def run_all_forward_tests():
    """Run comprehensive forward pass tests."""
    print("="*60)
    print("FORWARD PASS TEST SUITE")
    print("Testing data flow through all layer types")
    print("="*60)

    tester = ForwardPassTester()

    # Basic layers
    print("\n📦 Basic Layers:")
    tester.test("Linear layer", test_linear_forward)
    tester.test("Conv2d layer", test_conv2d_forward)
    tester.test("Conv2d with padding", test_conv2d_with_padding)
    tester.test("Conv2d with stride", test_conv2d_with_stride)

    # Activations
    print("\n⚡ Activation Functions:")
    tester.test("ReLU", test_relu_forward)
    tester.test("Sigmoid", test_sigmoid_forward)
    tester.test("Tanh", test_tanh_forward)
    tester.test("Softmax", test_softmax_forward)

    # Pooling
    print("\n🏊 Pooling Operations:")
    tester.test("MaxPool2d", test_maxpool2d_forward)
    tester.test("AvgPool2d", test_avgpool2d_forward)

    # Reshaping
    print("\n🔄 Reshape Operations:")
    tester.test("Flatten", test_flatten_forward)
    tester.test("Reshape", test_reshape_forward)

    # Normalization
    print("\n📊 Normalization:")
    tester.test("LayerNorm", test_layernorm_forward)
    tester.test("BatchNorm", test_batchnorm_forward)

    # Full architectures
    print("\n🏗️ Complete Architectures:")
    tester.test("Sequential container", test_sequential_forward)
    tester.test("MLP (MNIST)", test_mlp_forward)
    tester.test("CNN (Images)", test_cnn_forward)
    tester.test("Transformer (NLP)", test_transformer_forward)
    tester.test("Residual Block", test_residual_block_forward)

    return tester.summary()


if __name__ == "__main__":
    success = run_all_forward_tests()
    sys.exit(0 if success else 1)
