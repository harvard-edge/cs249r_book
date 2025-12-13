#!/usr/bin/env python
"""
Shape Validation Tests for TinyTorch
=====================================
Comprehensive shape validation ensuring all operations produce expected dimensions.
Uses pytest style - one test per specific behavior for clear reporting.

Run with: pytest tests/system/test_shapes.py -v
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
from tinytorch.nn import Conv2d, TransformerBlock, Embedding, PositionalEncoding, LayerNorm, Sequential
import tinytorch.nn.functional as F


# ============== Linear Layer Shape Tests ==============

def test_linear_basic_shape():
    """Linear layer produces correct output shape."""
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(3, 10))
    y = layer(x)
    assert y.shape == (3, 5), f"Expected (3, 5), got {y.shape}"


def test_linear_single_sample():
    """Linear handles single sample (batch=1)."""
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(1, 10))
    y = layer(x)
    assert y.shape == (1, 5), f"Expected (1, 5), got {y.shape}"


def test_linear_large_batch():
    """Linear handles large batch size."""
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(32, 10))
    y = layer(x)
    assert y.shape == (32, 5), f"Expected (32, 5), got {y.shape}"


def test_linear_chain():
    """Chain of linear layers maintains correct dimensions."""
    layer1 = Linear(784, 256)
    layer2 = Linear(256, 128)
    layer3 = Linear(128, 10)

    x = Tensor(np.random.randn(16, 784))
    x = layer1(x)
    assert x.shape == (16, 256), f"After layer1: expected (16, 256), got {x.shape}"
    x = layer2(x)
    assert x.shape == (16, 128), f"After layer2: expected (16, 128), got {x.shape}"
    x = layer3(x)
    assert x.shape == (16, 10), f"After layer3: expected (16, 10), got {x.shape}"


# ============== Conv2d Shape Tests ==============

def test_conv2d_basic():
    """Conv2d produces correct output shape with no padding."""
    layer = Conv2d(3, 16, kernel_size=3)
    x = Tensor(np.random.randn(2, 3, 32, 32))
    y = layer(x)
    # Output: (32 - 3)/1 + 1 = 30
    assert y.shape == (2, 16, 30, 30), f"Expected (2, 16, 30, 30), got {y.shape}"


def test_conv2d_with_padding():
    """Conv2d with padding=1 preserves spatial dimensions."""
    layer = Conv2d(3, 16, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 3, 32, 32))
    y = layer(x)
    assert y.shape == (2, 16, 32, 32), f"Expected (2, 16, 32, 32), got {y.shape}"


def test_conv2d_with_stride():
    """Conv2d with stride=2 halves spatial dimensions."""
    layer = Conv2d(3, 16, kernel_size=3, stride=2)
    x = Tensor(np.random.randn(2, 3, 32, 32))
    y = layer(x)
    # Output: (32 - 3)/2 + 1 = 15
    assert y.shape == (2, 16, 15, 15), f"Expected (2, 16, 15, 15), got {y.shape}"


def test_conv2d_1x1():
    """1x1 convolution preserves spatial dimensions."""
    layer = Conv2d(64, 32, kernel_size=1)
    x = Tensor(np.random.randn(4, 64, 14, 14))
    y = layer(x)
    assert y.shape == (4, 32, 14, 14), f"Expected (4, 32, 14, 14), got {y.shape}"


def test_conv2d_chain():
    """Chain of conv layers (typical CNN pattern)."""
    conv1 = Conv2d(1, 32, kernel_size=3)
    conv2 = Conv2d(32, 64, kernel_size=3)

    x = Tensor(np.random.randn(4, 1, 28, 28))  # MNIST-like
    x = conv1(x)
    assert x.shape == (4, 32, 26, 26), f"After conv1: expected (4, 32, 26, 26), got {x.shape}"
    x = conv2(x)
    assert x.shape == (4, 64, 24, 24), f"After conv2: expected (4, 64, 24, 24), got {x.shape}"


# ============== Activation Shape Tests ==============

def test_relu_preserves_2d_shape():
    """ReLU preserves 2D tensor shape."""
    x = Tensor(np.random.randn(10, 20))
    y = F.relu(x)
    assert y.shape == x.shape, f"ReLU changed shape: {x.shape} → {y.shape}"


def test_relu_preserves_4d_shape():
    """ReLU preserves 4D tensor shape (conv output)."""
    x = Tensor(np.random.randn(2, 16, 32, 32))
    y = F.relu(x)
    assert y.shape == x.shape, f"ReLU changed shape: {x.shape} → {y.shape}"


def test_sigmoid_preserves_shape():
    """Sigmoid preserves tensor shape."""
    x = Tensor(np.random.randn(5, 10))
    y = F.sigmoid(x)
    assert y.shape == x.shape, f"Sigmoid changed shape: {x.shape} → {y.shape}"


def test_tanh_preserves_shape():
    """Tanh preserves tensor shape."""
    x = Tensor(np.random.randn(5, 10))
    y = F.tanh(x)
    assert y.shape == x.shape, f"Tanh changed shape: {x.shape} → {y.shape}"


def test_softmax_preserves_shape():
    """Softmax preserves tensor shape."""
    x = Tensor(np.random.randn(5, 10))
    y = F.softmax(x, dim=-1)
    assert y.shape == x.shape, f"Softmax changed shape: {x.shape} → {y.shape}"


# ============== Pooling Shape Tests ==============

def test_maxpool2d_kernel_2():
    """MaxPool2d with kernel=2 halves spatial dimensions."""
    x = Tensor(np.random.randn(2, 16, 32, 32))
    y = F.max_pool2d(x, kernel_size=2)
    assert y.shape == (2, 16, 16, 16), f"Expected (2, 16, 16, 16), got {y.shape}"


def test_maxpool2d_kernel_4():
    """MaxPool2d with kernel=4 quarters spatial dimensions."""
    x = Tensor(np.random.randn(2, 16, 32, 32))
    y = F.max_pool2d(x, kernel_size=4)
    assert y.shape == (2, 16, 8, 8), f"Expected (2, 16, 8, 8), got {y.shape}"


def test_avgpool2d_kernel_2():
    """AvgPool2d with kernel=2 halves spatial dimensions."""
    x = Tensor(np.random.randn(2, 16, 32, 32))
    y = F.avg_pool2d(x, kernel_size=2)
    assert y.shape == (2, 16, 16, 16), f"Expected (2, 16, 16, 16), got {y.shape}"


def test_pool_after_conv():
    """Pooling after convolution (common CNN pattern)."""
    conv = Conv2d(3, 32, kernel_size=5)
    x = Tensor(np.random.randn(4, 3, 32, 32))
    x = conv(x)
    assert x.shape == (4, 32, 28, 28), f"After conv: expected (4, 32, 28, 28), got {x.shape}"
    x = F.max_pool2d(x, 2)
    assert x.shape == (4, 32, 14, 14), f"After pool: expected (4, 32, 14, 14), got {x.shape}"


# ============== Reshape Operation Tests ==============

def test_flatten_4d():
    """Flatten 4D tensor for FC after Conv."""
    x = Tensor(np.random.randn(4, 64, 5, 5))
    y = F.flatten(x, start_dim=1)
    assert y.shape == (4, 1600), f"Expected (4, 1600), got {y.shape}"


def test_flatten_cnn_to_fc():
    """Flatten for CNN→FC transition."""
    x = Tensor(np.random.randn(8, 128, 7, 7))
    y = F.flatten(x, start_dim=1)
    expected = 128 * 7 * 7
    assert y.shape == (8, expected), f"Expected (8, {expected}), got {y.shape}"


def test_reshape_3d_to_2d():
    """Reshape 3D tensor to 2D."""
    x = Tensor(np.random.randn(2, 3, 4))
    y = x.reshape(6, 4)
    assert y.shape == (6, 4), f"Expected (6, 4), got {y.shape}"


def test_reshape_to_flat():
    """Reshape to 1D (flatten completely)."""
    x = Tensor(np.random.randn(2, 3, 4))
    y = x.reshape(24)
    assert y.shape == (24,), f"Expected (24,), got {y.shape}"


def test_reshape_batch_preserve():
    """Reshape preserving batch dimension."""
    x = Tensor(np.random.randn(10, 3, 4))
    y = x.reshape(10, 12)
    assert y.shape == (10, 12), f"Expected (10, 12), got {y.shape}"


# ============== Transformer Component Tests ==============

def test_embedding_shape():
    """Embedding produces correct shape."""
    embed = Embedding(1000, 128)
    input_ids = Tensor(np.random.randint(0, 1000, (4, 10)))
    x = embed(input_ids)
    assert x.shape == (4, 10, 128), f"Expected (4, 10, 128), got {x.shape}"


def test_positional_encoding_preserves_shape():
    """Positional encoding preserves tensor shape."""
    pos_enc = PositionalEncoding(128, 50)
    x = Tensor(np.random.randn(4, 10, 128))
    y = pos_enc(x)
    assert y.shape == x.shape, f"PositionalEncoding changed shape: {x.shape} → {y.shape}"


def test_transformer_block_preserves_shape():
    """TransformerBlock preserves tensor shape."""
    block = TransformerBlock(128, num_heads=8)
    x = Tensor(np.random.randn(4, 10, 128))
    y = block(x)
    assert y.shape == x.shape, f"TransformerBlock changed shape: {x.shape} → {y.shape}"


def test_layernorm_preserves_shape():
    """LayerNorm preserves tensor shape."""
    ln = LayerNorm(128)
    x = Tensor(np.random.randn(4, 10, 128))
    y = ln(x)
    assert y.shape == x.shape, f"LayerNorm changed shape: {x.shape} → {y.shape}"


def test_transformer_output_projection():
    """Transformer output projection with reshape."""
    batch, seq, embed = 4, 10, 128
    vocab = 1000

    x = Tensor(np.random.randn(batch, seq, embed))
    x_2d = x.reshape(batch * seq, embed)
    assert x_2d.shape == (40, 128), f"Expected (40, 128), got {x_2d.shape}"

    proj = Linear(embed, vocab)
    logits_2d = proj(x_2d)
    assert logits_2d.shape == (40, 1000), f"Expected (40, 1000), got {logits_2d.shape}"

    logits = logits_2d.reshape(batch, seq, vocab)
    assert logits.shape == (4, 10, 1000), f"Expected (4, 10, 1000), got {logits.shape}"


# ============== Batch Size Flexibility Tests ==============

@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
def test_linear_batch_flexibility(batch_size):
    """Linear handles various batch sizes."""
    layer = Linear(100, 50)
    x = Tensor(np.random.randn(batch_size, 100))
    y = layer(x)
    assert y.shape == (batch_size, 50), f"Batch {batch_size}: expected ({batch_size}, 50), got {y.shape}"


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
def test_conv2d_batch_flexibility(batch_size):
    """Conv2d handles various batch sizes."""
    layer = Conv2d(3, 16, kernel_size=3)
    x = Tensor(np.random.randn(batch_size, 3, 32, 32))
    y = layer(x)
    assert y.shape == (batch_size, 16, 30, 30), f"Batch {batch_size}: got {y.shape}"


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_sequential_batch_flexibility(batch_size):
    """Sequential model handles various batch sizes."""
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    ])
    x = Tensor(np.random.randn(batch_size, 10))
    y = model(x)
    assert y.shape == (batch_size, 5), f"Batch {batch_size}: expected ({batch_size}, 5), got {y.shape}"


# ============== Edge Cases ==============

def test_conv_small_spatial():
    """Conv on very small spatial dimensions."""
    x = Tensor(np.random.randn(2, 16, 3, 3))
    conv = Conv2d(16, 32, kernel_size=3)
    y = conv(x)
    assert y.shape == (2, 32, 1, 1), f"Expected (2, 32, 1, 1), got {y.shape}"


def test_flatten_already_2d():
    """Flatten on already 2D tensor (should be no-op)."""
    x = Tensor(np.random.randn(10, 20))
    y = F.flatten(x, start_dim=1)
    assert y.shape == (10, 20), f"Expected (10, 20), got {y.shape}"


def test_single_channel_conv():
    """Conv with single input channel (grayscale images)."""
    conv = Conv2d(1, 8, kernel_size=3)
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y = conv(x)
    assert y.shape == (2, 8, 26, 26), f"Expected (2, 8, 26, 26), got {y.shape}"


# ============== Integration Pattern Tests ==============

def test_mnist_cnn_dimensions():
    """Complete MNIST CNN dimension flow."""
    x = Tensor(np.random.randn(32, 1, 28, 28))  # MNIST batch

    # Conv block 1
    conv1 = Conv2d(1, 32, kernel_size=3)
    x = conv1(x)
    assert x.shape == (32, 32, 26, 26), f"After conv1: {x.shape}"
    x = F.max_pool2d(x, 2)
    assert x.shape == (32, 32, 13, 13), f"After pool1: {x.shape}"

    # Conv block 2
    conv2 = Conv2d(32, 64, kernel_size=3)
    x = conv2(x)
    assert x.shape == (32, 64, 11, 11), f"After conv2: {x.shape}"
    x = F.max_pool2d(x, 2)
    assert x.shape == (32, 64, 5, 5), f"After pool2: {x.shape}"

    # Flatten for FC
    x = F.flatten(x, start_dim=1)
    assert x.shape == (32, 1600), f"After flatten: {x.shape}"

    # FC layers
    fc1 = Linear(1600, 128)
    x = fc1(x)
    assert x.shape == (32, 128), f"After fc1: {x.shape}"

    fc2 = Linear(128, 10)
    x = fc2(x)
    assert x.shape == (32, 10), f"Final output: {x.shape}"


def test_cifar10_cnn_dimensions():
    """Complete CIFAR-10 CNN dimension flow."""
    x = Tensor(np.random.randn(16, 3, 32, 32))  # CIFAR-10 batch

    # Conv block 1
    conv1 = Conv2d(3, 32, kernel_size=3)
    x = conv1(x)
    assert x.shape == (16, 32, 30, 30), f"After conv1: {x.shape}"
    x = F.max_pool2d(x, 2)
    assert x.shape == (16, 32, 15, 15), f"After pool1: {x.shape}"

    # Conv block 2
    conv2 = Conv2d(32, 64, kernel_size=3)
    x = conv2(x)
    assert x.shape == (16, 64, 13, 13), f"After conv2: {x.shape}"
    x = F.max_pool2d(x, 2)
    assert x.shape == (16, 64, 6, 6), f"After pool2: {x.shape}"

    # Flatten and FC
    x = F.flatten(x, start_dim=1)
    assert x.shape == (16, 2304), f"After flatten: {x.shape}"

    fc = Linear(2304, 10)
    x = fc(x)
    assert x.shape == (16, 10), f"Final output: {x.shape}"


if __name__ == "__main__":
    # When run directly, use pytest
    import subprocess
    result = subprocess.run(["pytest", __file__, "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    sys.exit(result.returncode)
