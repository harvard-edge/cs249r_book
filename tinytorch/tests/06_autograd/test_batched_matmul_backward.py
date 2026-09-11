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
rng = np.random.default_rng(7)
from tinytorch.core.tensor import Tensor
import tinytorch.core.autograd  # completes every operation with its backward half

# Enable autograd
def test_batched_3d_matmul_backward():
    """Test gradient flow through batched 3D matrix multiplication."""
    print("Testing batched 3D matmul backward...")

    # Batched matmul: (batch=2, m=4, k=8) @ (batch=2, k=8, n=4)
    a = Tensor(rng.standard_normal((2, 4, 8)), requires_grad=True)
    b = Tensor(rng.standard_normal((2, 8, 4)), requires_grad=True)

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
    Q = Tensor(rng.standard_normal((2, 4, 8, 64)), requires_grad=True)
    K = Tensor(rng.standard_normal((2, 4, 8, 64)), requires_grad=True)

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
    attn_weights = Tensor(rng.standard_normal((2, 4, 8, 8)), requires_grad=True)
    V = Tensor(rng.standard_normal((2, 4, 8, 64)), requires_grad=True)

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


import pytest


@pytest.mark.parametrize('a_shape,b_shape', [
    ((3,), (3,)), ((3,), (3, 2)), ((2, 3), (3,)),
    ((3,), (2, 3, 2)), ((2, 2, 3), (3,)),
    ((1, 2, 3), (4, 3, 2)), ((2, 1, 2, 3), (1, 3, 3, 2)),
])
def test_matmul_vector_and_broadcast_gradients_match_finite_difference(a_shape, b_shape):
    local_rng = np.random.default_rng(83)
    a_data = local_rng.normal(size=a_shape).astype(np.float32)
    b_data = local_rng.normal(size=b_shape).astype(np.float32)
    upstream = local_rng.normal(size=np.matmul(a_data, b_data).shape).astype(np.float32)
    a, b = Tensor(a_data, requires_grad=True), Tensor(b_data, requires_grad=True)
    (a @ b).backward(upstream)
    for which, value, actual in [(0, a_data, a.grad), (1, b_data, b.grad)]:
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            plus, minus = value.copy(), value.copy()
            plus[index] += 1e-3
            minus[index] -= 1e-3
            if which == 0:
                difference = np.matmul(plus, b_data) - np.matmul(minus, b_data)
            else:
                difference = np.matmul(a_data, plus) - np.matmul(a_data, minus)
            numerical[index] = np.sum(difference * upstream) / 2e-3
        np.testing.assert_allclose(actual, numerical, atol=2e-3, rtol=2e-3)
