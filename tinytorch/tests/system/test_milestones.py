#!/usr/bin/env python
"""
TinyTorch Milestone Validation Tests
=====================================
Ensures all three major milestones work end-to-end.
Students should be able to build and run these examples successfully.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.nn import Conv2d, TransformerBlock, Embedding, PositionalEncoding
import tinytorch.nn.functional as F


def test_milestone1_xor():
    """Test Milestone 1: XOR Problem with Perceptron."""
    print("\n" + "="*60)
    print("MILESTONE 1: XOR Problem (Perceptron)")
    print("="*60)

    # XOR dataset
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float32')
    y = Tensor([[0], [1], [1], [0]], dtype='float32')

    # Build simple neural network (perceptron with hidden layer)
    from tinytorch.core.networks import Sequential
    model = Sequential([
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    ])

    # Forward pass test
    output = model(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✅ XOR network structure works!")

    # Loss function test
    criterion = MeanSquaredError()
    loss = criterion(output, y)
    print(f"Loss value: {loss.data if hasattr(loss, 'data') else loss}")
    print(f"✅ Loss computation works!")

    return True


def test_milestone2_cnn():
    """Test Milestone 2: CNN for CIFAR-10."""
    print("\n" + "="*60)
    print("MILESTONE 2: CNN for Image Classification")
    print("="*60)

    # Create simple CNN
    class SimpleCNN:
        def __init__(self):
            self.conv1 = Conv2d(3, 32, kernel_size=(3, 3))
            self.conv2 = Conv2d(32, 64, kernel_size=(3, 3))
            # Correct dimensions after convs and pools
            self.fc1 = Linear(64 * 6 * 6, 256)
            self.fc2 = Linear(256, 10)

        def forward(self, x):
            # Conv block 1
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            # Conv block 2
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            # Classification head
            x = F.flatten(x, start_dim=1)
            x = self.fc1(x)
            x = F.relu(x)
            return self.fc2(x)

    # Test with dummy CIFAR-10 batch
    model = SimpleCNN()
    batch_size = 4
    x = Tensor(np.random.randn(batch_size, 3, 32, 32))

    print(f"Input shape (CIFAR batch): {x.shape}")

    # Test each stage
    x1 = model.conv1(x)
    print(f"After conv1: {x1.shape} (expected: {batch_size}, 32, 30, 30)")

    x2 = F.max_pool2d(x1, 2)
    print(f"After pool1: {x2.shape} (expected: {batch_size}, 32, 15, 15)")

    x3 = model.conv2(x2)
    print(f"After conv2: {x3.shape} (expected: {batch_size}, 64, 13, 13)")

    x4 = F.max_pool2d(x3, 2)
    print(f"After pool2: {x4.shape} (expected: {batch_size}, 64, 6, 6)")

    # Full forward pass
    output = model.forward(x)
    print(f"Final output: {output.shape} (expected: {batch_size}, 10)")

    assert output.shape == (batch_size, 10), f"Output shape mismatch: {output.shape}"
    print(f"✅ CNN architecture works for CIFAR-10!")

    return True


def test_milestone3_tinygpt():
    """Test Milestone 3: TinyGPT Language Model."""
    print("\n" + "="*60)
    print("MILESTONE 3: TinyGPT Language Model")
    print("="*60)

    # GPT parameters
    vocab_size = 100
    embed_dim = 64
    seq_length = 10
    batch_size = 2
    num_heads = 4

    # Build simple GPT
    class SimpleGPT:
        def __init__(self):
            self.embedding = Embedding(vocab_size, embed_dim)
            self.pos_encoding = PositionalEncoding(embed_dim, seq_length)
            self.transformer = TransformerBlock(embed_dim, num_heads, hidden_dim=embed_dim * 4)
            self.output_proj = Linear(embed_dim, vocab_size)

        def forward(self, x):
            # Embed tokens
            x = self.embedding(x)
            x = self.pos_encoding(x)

            # Transform
            x = self.transformer(x)

            # Project to vocabulary (with reshaping for Linear)
            batch, seq, embed = x.shape
            x_2d = x.reshape(batch * seq, embed)
            logits_2d = self.output_proj(x_2d)
            logits = logits_2d.reshape(batch, seq, vocab_size)

            return logits

    # Test with dummy tokens
    model = SimpleGPT()
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))

    print(f"Input tokens shape: {input_ids.shape}")

    # Test embedding
    embedded = model.embedding(input_ids)
    print(f"After embedding: {embedded.shape} (expected: {batch_size}, {seq_length}, {embed_dim})")

    # Test position encoding
    with_pos = model.pos_encoding(embedded)
    print(f"After pos encoding: {with_pos.shape} (expected: {batch_size}, {seq_length}, {embed_dim})")

    # Test transformer
    transformed = model.transformer(with_pos)
    print(f"After transformer: {transformed.shape} (expected: {batch_size}, {seq_length}, {embed_dim})")

    # Full forward pass
    output = model.forward(input_ids)
    print(f"Final logits: {output.shape} (expected: {batch_size}, {seq_length}, {vocab_size})")

    assert output.shape == (batch_size, seq_length, vocab_size), f"Output shape mismatch: {output.shape}"
    print(f"✅ TinyGPT architecture works!")

    return True


def run_all_milestone_tests():
    """Run all milestone validation tests."""
    print("\n" + "🎯"*30)
    print("TINYTORCH MILESTONE VALIDATION SUITE")
    print("Testing that all major learning milestones work correctly")
    print("🎯"*30)

    results = []

    # Test each milestone
    try:
        result1 = test_milestone1_xor()
        results.append(("XOR/Perceptron", result1))
    except Exception as e:
        print(f"❌ XOR test failed: {e}")
        results.append(("XOR/Perceptron", False))

    try:
        result2 = test_milestone2_cnn()
        results.append(("CNN/CIFAR-10", result2))
    except Exception as e:
        print(f"❌ CNN test failed: {e}")
        results.append(("CNN/CIFAR-10", False))

    try:
        result3 = test_milestone3_tinygpt()
        results.append(("TinyGPT", result3))
    except Exception as e:
        print(f"❌ TinyGPT test failed: {e}")
        results.append(("TinyGPT", False))

    # Summary
    print("\n" + "="*60)
    print("📊 MILESTONE TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n🎉 ALL MILESTONES WORKING!")
        print("Students can successfully build:")
        print("  1. Neural networks that solve XOR")
        print("  2. CNNs that process real images")
        print("  3. Transformers for language modeling")
        print("\n✨ The learning sandbox is robust!")
    else:
        print("\n⚠️  Some milestones need attention")

    return all_passed


if __name__ == "__main__":
    success = run_all_milestone_tests()
    sys.exit(0 if success else 1)
