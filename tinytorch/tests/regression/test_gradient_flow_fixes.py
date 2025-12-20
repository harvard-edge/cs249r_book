#!/usr/bin/env python3
"""
Regression Tests for Gradient Flow Fixes

This test suite verifies that specific gradient flow bugs have been fixed and don't regress.
These tests document the issues we encountered during transformer milestone implementation
and ensure the fixes remain in place.

Regression Issues Tested:
1. Module 01: np.dot → np.matmul for batched 3D tensors
2. Module 01: transpose() preserving requires_grad
3. Module 05: SubBackward and DivBackward added
4. Module 02: Softmax using Tensor operations
5. Module 03: Dropout using Tensor operations
6. Module 11: Embedding preserving requires_grad
7. Module 12: Attention using batched operations (no .data extraction)
8. Module 13: LayerNorm using Tensor operations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

# Enable autograd once for all tests
enable_autograd()


def test_regression_batched_matmul():
    """
    Regression test for Issue #1: np.dot doesn't handle batched 3D matmul.

    Bug: Using np.dot for 3D tensors produces wrong shapes.
    Fix: Changed to np.matmul in modules/01_tensor/tensor.py
    Commit: Module 01 fixes
    """
    print("Testing regression: batched 3D matmul...")

    # This pattern appears in attention: Q @ K.T
    Q = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    K = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    K_T = K.transpose()

    scores = Q.matmul(K_T)

    # Bug would produce (2, 4, 2, 4) or crash
    # Fix produces correct (2, 4, 4)
    assert scores.shape == (2, 4, 4), f"Batched matmul shape regression: {scores.shape}"
    assert scores.requires_grad, "Batched matmul should preserve requires_grad"

    print("✅ Batched 3D matmul regression test passed")


def test_regression_transpose_requires_grad():
    """
    Regression test for Issue #2: transpose() not preserving requires_grad.

    Bug: x.transpose() created Tensor without requires_grad.
    Fix: Added requires_grad parameter in modules/01_tensor/tensor.py
    Commit: Module 01 fixes
    """
    print("Testing regression: transpose requires_grad...")

    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    x_T = x.transpose()

    # Bug: x_T.requires_grad would be False
    # Fix: x_T.requires_grad is True
    assert x_T.requires_grad, "Transpose should preserve requires_grad"

    print("✅ Transpose requires_grad regression test passed")


def test_regression_subtraction_has_backward():
    """
    Regression test for Issue #3: Subtraction had no backward pass.

    Bug: Tensor.__sub__ not patched by Module 06, no gradient flow.
    Fix: Added SubBackward class and patched __sub__ in Module 06.
    Commit: Module 06 fixes
    """
    print("Testing regression: subtraction backward...")

    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([1.0, 1.0], requires_grad=True)
    c = a - b

    # Bug: c._grad_fn would be None
    # Fix: c._grad_fn is SubBackward instance
    assert hasattr(c, '_grad_fn'), "Subtraction should have _grad_fn"
    assert c._grad_fn is not None, "Subtraction _grad_fn should not be None"

    # Verify backward pass
    c.backward(np.ones(2))
    assert a.grad is not None and np.allclose(a.grad, [1.0, 1.0]), "∂(a-b)/∂a = 1"
    assert b.grad is not None and np.allclose(b.grad, [-1.0, -1.0]), "∂(a-b)/∂b = -1"

    print("✅ Subtraction backward regression test passed")


def test_regression_division_has_backward():
    """
    Regression test for Issue #4: Division had no backward pass.

    Bug: Tensor.__truediv__ not patched by Module 06, no gradient flow.
    Fix: Added DivBackward class and patched __truediv__ in Module 06.
    Commit: Module 06 fixes
    """
    print("Testing regression: division backward...")

    a = Tensor([4.0, 6.0], requires_grad=True)
    b = Tensor([2.0, 2.0], requires_grad=True)
    c = a / b

    # Bug: c._grad_fn would be None
    # Fix: c._grad_fn is DivBackward instance
    assert hasattr(c, '_grad_fn'), "Division should have _grad_fn"
    assert c._grad_fn is not None, "Division _grad_fn should not be None"

    # Verify backward pass
    c.backward(np.ones(2))
    assert a.grad is not None and np.allclose(a.grad, [0.5, 0.5]), "∂(a/b)/∂a = 1/b"

    print("✅ Division backward regression test passed")


def test_regression_layernorm_gradient_flow():
    """
    Regression test for Issue #5: LayerNorm broke gradient flow.

    Bug: LayerNorm extracted .data, creating Tensors without _grad_fn.
    Fix: Rewrote to use Tensor operations in Module 13.
    Commit: Module 13 fixes
    """
    print("Testing regression: LayerNorm gradient flow...")

    from tinytorch.core.transformer import LayerNorm

    ln = LayerNorm(4)
    ln.gamma.requires_grad = True
    ln.beta.requires_grad = True

    x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    output = ln.forward(x)

    # Bug: output.requires_grad would be False or _grad_fn None
    # Fix: output has requires_grad=True and _grad_fn set
    assert output.requires_grad, "LayerNorm output should require gradients"
    assert hasattr(output, '_grad_fn'), "LayerNorm output should have _grad_fn"

    # Verify backward
    output.backward(np.ones_like(output.data))
    assert x.grad is not None, "Gradient should flow back through LayerNorm"

    print("✅ LayerNorm gradient flow regression test passed")


def test_regression_embedding_requires_grad():
    """
    Regression test for Issue #6: Embedding didn't preserve requires_grad.

    Bug: Embedding.forward() created Tensor(embedded) without requires_grad.
    Fix: Added requires_grad=self.weight.requires_grad in Module 11.
    Commit: Module 11 fixes
    """
    print("Testing regression: Embedding requires_grad...")

    from tinytorch.text.embeddings import Embedding

    embed = Embedding(vocab_size=10, embed_dim=8)
    embed.weight.requires_grad = True

    indices = Tensor([[1, 2, 3]])
    output = embed.forward(indices)

    # Bug: output.requires_grad would be False
    # Fix: output.requires_grad is True
    assert output.requires_grad, "Embedding output should preserve requires_grad"

    print("✅ Embedding requires_grad regression test passed")


def test_regression_dropout_uses_tensor_ops():
    """
    Regression test for Issue #7: Dropout used .data extraction.

    Bug: Dropout did (x.data * mask) / keep_prob, breaking gradient flow.
    Fix: Rewrote to use Tensor operations in Module 03.
    Commit: Module 03 fixes
    """
    print("Testing regression: Dropout Tensor operations...")

    from tinytorch.core.layers import Dropout

    dropout = Dropout(0.5)
    x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    # Set seed for reproducibility
    np.random.seed(42)
    output = dropout.forward(x, training=True)

    # Bug: output wouldn't have _grad_fn
    # Fix: output has _grad_fn from Tensor multiplication
    assert output.requires_grad, "Dropout output should require gradients"

    print("✅ Dropout Tensor operations regression test passed")


def test_regression_transpose_has_backward():
    """
    Regression test for Issue #8: Transpose had no backward pass.

    Bug: Tensor.transpose() not patched by Module 06, no gradient flow.
    Fix: Added TransposeBackward class and patched transpose in Module 06.
    Commit: Module 06 fixes (TransposeBackward)
    """
    print("Testing regression: transpose backward...")

    K = Tensor(np.random.randn(2, 4, 8, 64), requires_grad=True)
    K_T = K.transpose()

    # Bug: K_T._grad_fn would be None
    # Fix: K_T._grad_fn is TransposeBackward instance
    assert hasattr(K_T, '_grad_fn'), "Transpose should have _grad_fn"
    assert K_T._grad_fn is not None, "Transpose _grad_fn should not be None"

    # Verify backward pass (attention pattern: Q @ K.T)
    Q = Tensor(np.random.randn(2, 4, 8, 64), requires_grad=True)
    scores = Q.matmul(K_T)
    scores.backward(np.ones_like(scores.data))

    assert K.grad is not None, "Gradient should flow back through transpose"
    assert K.grad.shape == K.shape, f"K.grad shape {K.grad.shape} should match K shape {K.shape}"

    print("✅ Transpose backward regression test passed")


def test_regression_matmul_backward_uses_matmul():
    """
    Regression test for Issue #9: MatmulBackward used np.dot for gradients.

    Bug: MatmulBackward used np.dot which doesn't handle batched 3D+ tensors.
    Fix: Changed to np.matmul and np.swapaxes in Module 06.
    Commit: Module 06 fixes (MatmulBackward batched)
    """
    print("Testing regression: MatmulBackward batched operations...")

    # Batched 3D matmul
    a = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    b = Tensor(np.random.randn(2, 8, 4), requires_grad=True)
    c = a.matmul(b)

    # Backward pass
    c.backward(np.ones_like(c.data))

    # Bug: Would crash with "shapes not aligned" or produce wrong shapes
    # Fix: Gradients have correct shapes
    assert a.grad is not None and a.grad.shape == (2, 4, 8), f"a.grad shape: {a.grad.shape}"
    assert b.grad is not None and b.grad.shape == (2, 8, 4), f"b.grad shape: {b.grad.shape}"

    print("✅ MatmulBackward batched operations regression test passed")


def run_all_tests():
    """Run all regression tests for gradient flow fixes."""
    print("\n" + "="*70)
    print("GRADIENT FLOW REGRESSION TEST SUITE")
    print("="*70 + "\n")

    tests = [
        test_regression_batched_matmul,
        test_regression_transpose_requires_grad,
        test_regression_subtraction_has_backward,
        test_regression_division_has_backward,
        test_regression_layernorm_gradient_flow,
        test_regression_embedding_requires_grad,
        test_regression_dropout_uses_tensor_ops,
        test_regression_transpose_has_backward,
        test_regression_matmul_backward_uses_matmul,
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
        print("")

    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ All gradient flow fixes verified - no regressions detected!")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
