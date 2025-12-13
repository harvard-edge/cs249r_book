#!/usr/bin/env python3
"""
Test batched matrix multiplication gradients in autograd.

This test verifies that MatmulBackward correctly handles batched 3D+ tensors
using np.matmul and np.swapaxes instead of np.dot and .T
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

# Enable autograd
enable_autograd()


def test_batched_3d_matmul_backward():
    """Test gradient flow through batched 3D matrix multiplication."""
    print("Testing batched 3D matmul backward...")

    # Batched matmul: (batch=2, m=4, k=8) @ (batch=2, k=8, n=4)
    a = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    b = Tensor(np.random.randn(2, 8, 4), requires_grad=True)

    # Forward pass
    c = a.matmul(b)

    # Check output shape
    assert c.shape == (2, 4, 4), f"Expected (2, 4, 4), got {c.shape}"

    # Backward pass
    grad_output = np.ones((2, 4, 4))
    c.backward(grad_output)

    # Verify gradients exist and have correct shapes
    assert a.grad is not None, "a.grad should exist"
    assert b.grad is not None, "b.grad should exist"
    assert a.grad.shape == (2, 4, 8), f"a.grad shape: {a.grad.shape}"
    assert b.grad.shape == (2, 8, 4), f"b.grad shape: {b.grad.shape}"

    print(f"  ✓ Forward shape: {c.shape}")
    print(f"  ✓ a.grad shape: {a.grad.shape}")
    print(f"  ✓ b.grad shape: {b.grad.shape}")
    print("✅ Batched 3D matmul backward test passed\n")


def test_attention_pattern_matmul():
    """Test the specific pattern used in attention: Q @ K.T."""
    print("Testing attention pattern (Q @ K.T) backward...")

    # Attention scores: (batch=2, heads=4, seq=8, dim=64) @ (batch=2, heads=4, dim=64, seq=8)
    Q = Tensor(np.random.randn(2, 4, 8, 64), requires_grad=True)
    K = Tensor(np.random.randn(2, 4, 8, 64), requires_grad=True)

    # Transpose K (swap last two dims)
    K_T = K.transpose()

    # Compute attention scores
    scores = Q.matmul(K_T)

    # Check output shape
    assert scores.shape == (2, 4, 8, 8), f"Expected (2, 4, 8, 8), got {scores.shape}"

    # Backward pass
    grad_output = np.ones((2, 4, 8, 8))
    scores.backward(grad_output)

    # Verify gradients
    assert Q.grad is not None, "Q.grad should exist"
    assert K.grad is not None, "K.grad should exist"
    assert Q.grad.shape == (2, 4, 8, 64), f"Q.grad shape: {Q.grad.shape}"
    assert K.grad.shape == (2, 4, 8, 64), f"K.grad shape: {K.grad.shape}"

    print(f"  ✓ Forward shape: {scores.shape}")
    print(f"  ✓ Q.grad shape: {Q.grad.shape}")
    print(f"  ✓ K.grad shape: {K.grad.shape}")
    print("✅ Attention pattern backward test passed\n")


def test_attention_output_matmul():
    """Test the attention @ V pattern."""
    print("Testing attention @ V pattern backward...")

    # Attention output: (batch=2, heads=4, seq=8, seq=8) @ (batch=2, heads=4, seq=8, dim=64)
    attn_weights = Tensor(np.random.randn(2, 4, 8, 8), requires_grad=True)
    V = Tensor(np.random.randn(2, 4, 8, 64), requires_grad=True)

    # Compute attention output
    output = attn_weights.matmul(V)

    # Check output shape
    assert output.shape == (2, 4, 8, 64), f"Expected (2, 4, 8, 64), got {output.shape}"

    # Backward pass
    grad_output = np.ones((2, 4, 8, 64))
    output.backward(grad_output)

    # Verify gradients
    assert attn_weights.grad is not None, "attn_weights.grad should exist"
    assert V.grad is not None, "V.grad should exist"
    assert attn_weights.grad.shape == (2, 4, 8, 8), f"attn_weights.grad shape: {attn_weights.grad.shape}"
    assert V.grad.shape == (2, 4, 8, 64), f"V.grad shape: {V.grad.shape}"

    print(f"  ✓ Forward shape: {output.shape}")
    print(f"  ✓ attn_weights.grad shape: {attn_weights.grad.shape}")
    print(f"  ✓ V.grad shape: {V.grad.shape}")
    print("✅ Attention @ V pattern backward test passed\n")


def run_all_tests():
    """Run all batched matmul backward tests."""
    print("\n" + "="*70)
    print("BATCHED MATMUL BACKWARD TEST SUITE")
    print("="*70 + "\n")

    tests = [
        test_batched_3d_matmul_backward,
        test_attention_pattern_matmul,
        test_attention_output_matmul,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ All batched matmul backward tests passed!")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
