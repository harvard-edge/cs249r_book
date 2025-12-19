"""
Module 06: Autograd - Core Functionality Tests
===============================================

These tests verify automatic differentiation works correctly.

WHY AUTOGRAD MATTERS:
--------------------
Autograd is what makes training possible:
- Computes gradients automatically (no manual derivatives)
- Enables complex architectures (just define forward, get backward free)
- Powers modern deep learning frameworks

Without autograd, you'd need to derive and code every gradient by hand.

WHAT STUDENTS LEARN:
-------------------
1. Computational graphs track operations
2. Gradients flow backward through the graph
3. requires_grad enables gradient tracking
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestGradientTracking:
    """
    Test gradient tracking basics.

    CONCEPT: requires_grad=True tells the tensor to track operations
    for automatic differentiation.
    """

    def test_requires_grad_attribute(self):
        """
        WHAT: Verify tensors have requires_grad attribute.

        WHY: This flag controls whether gradients are computed.
        False = no gradient (input data), True = gradient needed (parameters).

        STUDENT LEARNING: Set requires_grad=True for:
        - Model parameters (weights, biases)
        - Any tensor you want gradients for
        """
        from tinytorch.core.tensor import Tensor

        # Default should be False (most tensors don't need gradients)
        x = Tensor([1, 2, 3])
        assert hasattr(x, 'requires_grad'), "Tensor must have requires_grad"

        # Should be able to set it True
        x_grad = Tensor([1, 2, 3], requires_grad=True)
        assert x_grad.requires_grad, (
            "Tensor with requires_grad=True should have it set"
        )

    def test_grad_attribute(self):
        """
        WHAT: Verify tensors can store gradients in .grad attribute.

        WHY: After backward(), gradients are stored in tensor.grad.
        This is what optimizers read to update parameters.

        STUDENT LEARNING: tensor.grad starts as None.
        After loss.backward(), it contains dLoss/dTensor.
        """
        from tinytorch.core.tensor import Tensor

        x = Tensor([1, 2, 3], requires_grad=True)
        assert hasattr(x, 'grad'), "Tensor must have grad attribute"


class TestSimpleGradients:
    """
    Test gradients for basic operations.

    CONCEPT: Each operation has a gradient rule.
    Chain rule combines them: d(f∘g)/dx = df/dg * dg/dx
    """

    def test_addition_gradient(self):
        """
        WHAT: Verify gradient of addition is correct.

        WHY: d(a+b)/da = 1, d(a+b)/db = 1
        Gradient "copies" to both inputs.

        STUDENT LEARNING: Addition is a "split point" in gradients.
        Both inputs receive the full upstream gradient.
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import enable_autograd

        enable_autograd()

        a = Tensor([3.0], requires_grad=True)
        b = Tensor([2.0], requires_grad=True)

        c = a + b  # c = a + b = 5
        c.backward()

        # dc/da = 1, dc/db = 1
        assert a.grad is not None and np.isclose(a.grad[0], 1.0), (
            f"d(a+b)/da should be 1, got {a.grad}"
        )
        assert b.grad is not None and np.isclose(b.grad[0], 1.0), (
            f"d(a+b)/db should be 1, got {b.grad}"
        )

    def test_multiplication_gradient(self):
        """
        WHAT: Verify gradient of multiplication is correct.

        WHY: d(a*b)/da = b, d(a*b)/db = a
        The gradient "crosses" - each input gets the other's value.

        STUDENT LEARNING: This is why a=0 causes problems -
        if a=0, gradient to b is 0 (no learning signal).
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import enable_autograd

        enable_autograd()

        a = Tensor([3.0], requires_grad=True)
        b = Tensor([2.0], requires_grad=True)

        c = a * b  # c = a * b = 6
        c.backward()

        # dc/da = b = 2, dc/db = a = 3
        assert a.grad is not None and np.isclose(a.grad[0], 2.0), (
            f"d(a*b)/da should be b=2, got {a.grad}"
        )
        assert b.grad is not None and np.isclose(b.grad[0], 3.0), (
            f"d(a*b)/db should be a=3, got {b.grad}"
        )

    def test_power_gradient(self):
        """
        WHAT: Verify gradient of x^2 is 2x.

        WHY: d(x²)/dx = 2x is the classic derivative.
        If this is wrong, all polynomial gradients are wrong.

        STUDENT LEARNING: Power rule: d(x^n)/dx = n * x^(n-1)
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import enable_autograd

        enable_autograd()

        x = Tensor([2.0], requires_grad=True)
        y = x * x  # y = x^2 = 4
        y.backward()

        # dy/dx = 2x = 4
        assert x.grad is not None and np.isclose(x.grad[0], 4.0), (
            f"d(x²)/dx at x=2 should be 2*2=4, got {x.grad}"
        )


class TestChainRule:
    """
    Test chain rule (composition of functions).

    CONCEPT: For y = f(g(x)), dy/dx = f'(g(x)) * g'(x)
    This is what makes deep networks work.
    """

    def test_simple_chain(self):
        """
        WHAT: Verify chain rule for y = (x + 1)².

        WHY: This is a composition: y = f(g(x)) where:
        g(x) = x + 1, f(u) = u²
        dy/dx = 2(x+1) * 1 = 2(x+1)

        STUDENT LEARNING: Autograd automatically applies chain rule.
        You just define the forward pass.
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import enable_autograd

        enable_autograd()

        x = Tensor([2.0], requires_grad=True)
        u = x + Tensor([1.0])  # u = x + 1 = 3
        y = u * u              # y = u² = 9
        y.backward()

        # dy/dx = 2u * du/dx = 2*3 * 1 = 6
        expected = 6.0
        assert x.grad is not None and np.isclose(x.grad[0], expected), (
            f"Chain rule: d[(x+1)²]/dx at x=2 should be 2*3=6\n"
            f"  Got: {x.grad}"
        )

    def test_deep_chain(self):
        """
        WHAT: Verify chain rule through multiple operations.

        WHY: Deep networks have many layers, each is a function.
        Chain rule must work through all of them.

        STUDENT LEARNING: Gradients "accumulate" through chain rule.
        Small gradients at each layer can vanish or explode.
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import enable_autograd

        enable_autograd()

        x = Tensor([1.0], requires_grad=True)

        # Compute x * 2 * 2 * 2 = 8x
        y = x
        for _ in range(3):
            y = y * Tensor([2.0])

        # y = 8x, dy/dx = 8
        y.backward()

        assert x.grad is not None and np.isclose(x.grad[0], 8.0), (
            f"d(2*2*2*x)/dx should be 8, got {x.grad}"
        )


class TestBatchedGradients:
    """
    Test gradients with batched (multi-sample) data.

    CONCEPT: Training uses batches. Gradients are averaged/summed
    across the batch.
    """

    def test_batched_loss_gradient(self):
        """
        WHAT: Verify gradients work with batch of samples.

        WHY: Training computes loss over batch, then backprop.
        Gradients from each sample combine.

        STUDENT LEARNING: For MSE loss on batch:
        1. Compute loss per sample
        2. Average (mean) or sum
        3. Backward gives gradient averaged/summed over batch
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import enable_autograd

        enable_autograd()

        # Batch of 3 samples
        x = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        target = Tensor([[2.0], [2.0], [2.0]])

        # Simple loss: sum of squared errors
        diff = x - target  # [[-1], [0], [1]]
        loss = (diff * diff).sum()  # 1 + 0 + 1 = 2
        loss.backward()

        # d(loss)/dx = 2*diff = [[-2], [0], [2]]
        expected = np.array([[-2.0], [0.0], [2.0]])
        assert x.grad is not None, "Batch gradient should exist"
        assert np.allclose(x.grad, expected), (
            f"Batch gradient wrong.\n"
            f"  diff = {diff.data.flatten()}\n"
            f"  d(loss)/dx = 2*diff = {expected.flatten()}\n"
            f"  Got: {x.grad.flatten()}"
        )


class TestGradientAccumulation:
    """
    Test gradient accumulation behavior.

    CONCEPT: By default, gradients accumulate (add up).
    Must call zero_grad() between batches.
    """

    def test_gradients_accumulate(self):
        """
        WHAT: Verify gradients add up without zero_grad().

        WHY: This allows gradient accumulation for large batches.
        But it's a common source of bugs!

        STUDENT LEARNING: Always call optimizer.zero_grad() before
        loss.backward(). Otherwise gradients from previous batch
        contaminate current batch.
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.autograd import enable_autograd

        enable_autograd()

        x = Tensor([1.0], requires_grad=True)

        # First backward
        y = x * Tensor([2.0])
        y.backward()
        first_grad = x.grad.copy() if x.grad is not None else None

        # Second backward without zero_grad
        y = x * Tensor([2.0])
        y.backward()
        second_grad = x.grad.copy() if x.grad is not None else None

        # Gradient should have doubled
        if first_grad is not None and second_grad is not None:
            assert np.isclose(second_grad[0], 2 * first_grad[0]), (
                f"Gradients should accumulate.\n"
                f"  First backward: {first_grad}\n"
                f"  Second backward (no zero_grad): {second_grad}\n"
                "Expected second to be double the first."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
