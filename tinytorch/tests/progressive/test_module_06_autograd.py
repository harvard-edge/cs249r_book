"""
Module 06: Autograd - Progressive Testing
==========================================

🎯 LEARNING OBJECTIVES:
1. Understand automatic differentiation
2. Build computation graphs during forward pass
3. Compute gradients via backpropagation

📚 PREREQUISITE MODULES:
- Module 01: Tensor (data structure)
- Module 02: Activations (non-linear functions)
- Module 03: Layers (Linear transformation)
- Module 04: Losses (objective functions)

🔗 WHAT AUTOGRAD ENABLES:
After this module, your tensors can automatically compute gradients!
This is the foundation of neural network training.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# SECTION 1: REGRESSION TESTS
# Verify earlier modules still work after autograd patches tensors
# =============================================================================

class TestFoundationStillWorks:
    """
    🛡️ REGRESSION CHECK: Autograd must not break the foundation

    Autograd patches Tensor operations to track gradients. This test ensures
    basic tensor functionality still works correctly after enabling autograd.

    WHY THIS MATTERS:
    A common bug is breaking basic operations when adding gradient tracking.
    If tensor creation or arithmetic breaks, nothing else will work!
    """

    def test_tensor_creation_works(self):
        """
        ✅ WHAT: Basic tensor creation
        🔍 IF FAILS: Autograd broke the Tensor constructor
        """
        from tinytorch import Tensor

        # These should all still work
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[1, 2], [3, 4]])
        t3 = Tensor(np.random.randn(3, 4, 5))

        assert t1.shape == (3,), "1D tensor creation broken"
        assert t2.shape == (2, 2), "2D tensor creation broken"
        assert t3.shape == (3, 4, 5), "3D tensor creation broken"

    def test_tensor_arithmetic_works(self):
        """
        ✅ WHAT: Basic arithmetic (+, -, *, /)
        🔍 IF FAILS: Autograd broke tensor operators
        """
        from tinytorch import Tensor

        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        # All basic operations should work
        add_result = a + b
        sub_result = a - b
        mul_result = a * b
        div_result = a / b

        assert np.allclose(add_result.data, [5, 7, 9]), "Addition broken"
        assert np.allclose(sub_result.data, [-3, -3, -3]), "Subtraction broken"
        assert np.allclose(mul_result.data, [4, 10, 18]), "Multiplication broken"
        assert np.allclose(div_result.data, [0.25, 0.4, 0.5]), "Division broken"

    def test_linear_layer_still_works(self):
        """
        ✅ WHAT: Linear layer forward pass
        🔍 IF FAILS: Autograd broke layer operations
        """
        from tinytorch import Tensor, Linear

        layer = Linear(10, 5)
        x = Tensor(np.random.randn(3, 10))  # batch of 3

        output = layer(x)

        assert output.shape == (3, 5), (
            f"Linear layer output shape wrong!\n"
            f"  Input: (3, 10)\n"
            f"  Expected output: (3, 5)\n"
            f"  Got: {output.shape}\n"
            f"\n"
            f"💡 HINT: Linear(10, 5) should transform (batch, 10) → (batch, 5)"
        )


class TestActivationsStillWork:
    """
    🛡️ REGRESSION CHECK: Activations must still work with autograd-enabled tensors
    """

    def test_relu_works_with_gradients(self):
        """
        ✅ WHAT: ReLU on tensors that require gradients
        🔍 IF FAILS: ReLU doesn't handle requires_grad properly
        """
        from tinytorch import Tensor, ReLU

        relu = ReLU()
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)

        output = relu(x)

        assert np.allclose(output.data, [0, 0, 0, 1, 2]), (
            "ReLU computation wrong!\n"
            "  Input: [-2, -1, 0, 1, 2]\n"
            "  Expected: [0, 0, 0, 1, 2]\n"
            f"  Got: {output.data}\n"
            "\n"
            "💡 HINT: ReLU(x) = max(0, x)"
        )


# =============================================================================
# SECTION 2: CAPABILITY TESTS
# Verify Module 05 provides its core functionality
# =============================================================================

class TestAutogradCapabilities:
    """
    🎯 CAPABILITY CHECK: Does autograd do what it's supposed to?

    Autograd must:
    1. Track operations during forward pass (build computation graph)
    2. Compute gradients during backward pass (backpropagation)
    3. Store gradients in .grad attribute
    """

    def test_requires_grad_flag_exists(self):
        """
        ✅ WHAT: Tensors have requires_grad attribute

        📖 CONCEPT: requires_grad tells autograd whether to track this tensor
        - requires_grad=True → track operations, compute gradients
        - requires_grad=False → don't track (saves memory)
        """
        from tinytorch import Tensor

        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=False)
        t3 = Tensor([1, 2, 3])  # default

        assert hasattr(t1, 'requires_grad'), "Tensor missing requires_grad attribute"
        assert t1.requires_grad == True, "requires_grad=True not stored"
        assert t2.requires_grad == False, "requires_grad=False not stored"

    def test_grad_attribute_exists(self):
        """
        ✅ WHAT: Tensors have .grad attribute for storing gradients

        📖 CONCEPT: After backward(), gradients are stored in .grad
        """
        from tinytorch import Tensor

        t = Tensor([1, 2, 3], requires_grad=True)

        assert hasattr(t, 'grad'), (
            "Tensor missing .grad attribute!\n"
            "\n"
            "💡 HINT: Add 'self.grad = None' in Tensor.__init__()"
        )

    def test_simple_gradient_computation(self):
        """
        ✅ WHAT: Gradients computed for y = sum(x * 2)

        📖 CONCEPT: If y = sum(2x), then dy/dx = 2 for each element
        We use sum() to get a scalar for backward().

        🔍 IF FAILS: Your backward pass isn't working
        """
        from tinytorch import Tensor

        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2  # Simple operation
        loss = y.sum()  # Must be scalar for backward()

        # Backward pass
        loss.backward()

        assert x.grad is not None, (
            "Gradient not computed!\n"
            "\n"
            "For y = 2x, we expect dy/dx = 2\n"
            "\n"
            "💡 HINTS:\n"
            "1. Is backward() calling the right backward function?\n"
            "2. Are gradients being stored in .grad?"
        )

        expected_grad = np.array([2.0, 2.0, 2.0])
        assert np.allclose(x.grad, expected_grad), (
            f"Gradient value wrong!\n"
            f"  For y = 2x, dy/dx should be 2\n"
            f"  Expected: {expected_grad}\n"
            f"  Got: {x.grad}\n"
            f"\n"
            "💡 HINT: Check your multiplication backward function"
        )

    def test_chain_rule_works(self):
        """
        ✅ WHAT: Gradients flow through multiple operations (chain rule)

        📖 CONCEPT: Chain Rule
        If z = g(y) and y = f(x), then:
        dz/dx = dz/dy * dy/dx

        This is the foundation of backpropagation!

        Example: loss = sum((x * 2) + 3)
        - y = x * 2  → dy/dx = 2
        - z = y + 3  → dz/dy = 1
        - loss = sum(z) → dloss/dz = 1
        - Therefore: dloss/dx = 1 * 1 * 2 = 2
        """
        from tinytorch import Tensor

        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2      # dy/dx = 2
        z = y + 3      # dz/dy = 1
        loss = z.sum() # Must be scalar for backward()

        loss.backward()

        expected_grad = np.array([2.0, 2.0, 2.0])  # dz/dx = 2
        assert x.grad is not None, "Chain rule: gradients didn't flow back"
        assert np.allclose(x.grad, expected_grad), (
            f"Chain rule gradient wrong!\n"
            f"  z = (x * 2) + 3\n"
            f"  dz/dx = dz/dy * dy/dx = 1 * 2 = 2\n"
            f"  Expected: {expected_grad}\n"
            f"  Got: {x.grad}"
        )


class TestNeuralNetworkGradients:
    """
    🎯 CAPABILITY CHECK: Can autograd train neural networks?

    This is the real test: can we compute gradients for a neural network?
    """

    def test_linear_layer_gradients(self):
        """
        ✅ WHAT: Gradients flow through Linear layer

        📖 CONCEPT: For y = xW + b:
        - dy/dW = x^T (input transposed)
        - dy/db = 1 (gradient of bias is 1)
        - dy/dx = W^T (weight transposed)
        """
        from tinytorch import Tensor, Linear

        # Simple linear layer
        layer = Linear(3, 2)
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        # Forward
        y = layer(x)

        # Create simple loss (sum of outputs)
        loss = y.sum()

        # Backward
        loss.backward()

        # Weight should have gradients
        assert layer.weight.grad is not None, (
            "Linear layer weights didn't receive gradients!\n"
            "\n"
            "💡 HINTS:\n"
            "1. Is layer.weight.requires_grad = True?\n"
            "2. Did you implement matmul backward correctly?\n"
            "3. Are gradients propagating through the add operation?"
        )

        # Bias should have gradients
        if layer.bias is not None:
            assert layer.bias.grad is not None, (
                "Linear layer bias didn't receive gradients!"
            )

    def test_mlp_end_to_end_gradients(self):
        """
        ✅ WHAT: Multi-layer network computes gradients

        📖 CONCEPT: Backprop through multiple layers
        Each layer receives gradients from the layer above.
        """
        from tinytorch import Tensor, Linear, ReLU

        # Two-layer MLP
        layer1 = Linear(4, 8)
        relu = ReLU()
        layer2 = Linear(8, 2)

        # Forward
        x = Tensor(np.random.randn(2, 4), requires_grad=True)
        h = layer1(x)
        h = relu(h)
        y = layer2(h)

        # Loss and backward
        loss = y.sum()
        loss.backward()

        # All layers should have gradients
        assert layer1.weight.grad is not None, "Layer 1 didn't receive gradients"
        assert layer2.weight.grad is not None, "Layer 2 didn't receive gradients"

        # Gradients should be non-zero
        assert np.any(layer1.weight.grad != 0), (
            "Layer 1 has zero gradients!\n"
            "\n"
            "💡 HINT: Check if gradients are flowing through ReLU.\n"
            "ReLU gradient is 1 for positive inputs, 0 for negative."
        )


# =============================================================================
# SECTION 3: INTEGRATION TESTS
# Verify autograd works with all previous modules together
# =============================================================================

class TestAutogradLossIntegration:
    """
    🔗 INTEGRATION CHECK: Autograd + Loss functions

    Training requires computing gradients of the loss.
    """

    def test_mse_loss_gradients(self):
        """
        ✅ WHAT: MSE loss produces correct gradients

        📖 CONCEPT: MSE = mean((predictions - targets)^2)
        Gradient: d(MSE)/d(predictions) = 2 * (predictions - targets) / n
        """
        from tinytorch import Tensor, MSELoss

        predictions = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        targets = Tensor([[1.5, 2.5, 2.5]])

        loss_fn = MSELoss()
        loss = loss_fn(predictions, targets)

        loss.backward()

        assert predictions.grad is not None, (
            "MSE loss didn't produce gradients!\n"
            "\n"
            "💡 HINT: Is loss.backward() calling the right backward function?"
        )


class TestCompleteTrainingLoop:
    """
    🔗 INTEGRATION CHECK: Can we do one complete training step?

    This tests everything together:
    1. Forward pass through layers
    2. Compute loss
    3. Backward pass (autograd)
    4. Verify gradients exist for optimization
    """

    def test_training_step_computes_gradients(self):
        """
        ✅ WHAT: Complete forward-backward pass works

        This is what happens in every training step:
        1. Feed data through network
        2. Compute loss
        3. Compute gradients
        4. (Optimizer would update weights here)
        """
        from tinytorch import Tensor, Linear, ReLU, MSELoss

        # Simple network
        layer = Linear(4, 2)
        activation = ReLU()

        # Data
        x = Tensor(np.random.randn(8, 4))  # 8 samples
        target = Tensor(np.random.randn(8, 2))

        # Forward
        hidden = layer(x)
        output = activation(hidden)

        # Loss
        loss_fn = MSELoss()
        loss = loss_fn(output, target)

        # Backward
        loss.backward()

        # Verify gradients exist
        assert layer.weight.grad is not None, (
            "Training step failed: weights have no gradients!\n"
            "\n"
            "This means backpropagation didn't work.\n"
            "\n"
            "💡 DEBUG STEPS:\n"
            "1. Check loss.backward() is called\n"
            "2. Check gradients flow through activation\n"
            "3. Check gradients flow through linear layer"
        )

        # Verify gradients are not all zeros
        assert np.any(layer.weight.grad != 0), (
            "Gradients are all zeros!\n"
            "\n"
            "This usually means:\n"
            "- ReLU killed all gradients (all outputs were negative)\n"
            "- A backward function returns zeros\n"
            "\n"
            "💡 TRY: Print intermediate values to find where gradients die"
        )


# =============================================================================
# SECTION 4: COMMON MISTAKES (Educational)
# Tests that catch common student errors
# =============================================================================

class TestCommonMistakes:
    """
    ⚠️ COMMON MISTAKE DETECTION

    These tests catch mistakes students often make.
    If these fail, check the hints carefully!
    """

    def test_backward_with_scalar_loss(self):
        """
        ⚠️ COMMON MISTAKE: Calling backward() on non-scalar

        backward() should be called on the loss (a scalar).
        You can't backprop from a multi-element tensor directly.
        """
        from tinytorch import Tensor

        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2

        # Should be able to call backward on scalar
        loss = y.sum()  # scalar
        loss.backward()  # This should work

        assert x.grad is not None, "backward() on scalar loss should compute gradients"

    def test_gradient_accumulation(self):
        """
        ⚠️ COMMON MISTAKE: Forgetting that gradients accumulate

        📖 CONCEPT: Each backward() ADDS to .grad, doesn't replace it.
        This is intentional (for batch accumulation).
        But you need to zero gradients between training steps!
        """
        from tinytorch import Tensor

        x = Tensor([1.0], requires_grad=True)

        # First backward
        y1 = x * 2
        y1.backward()
        grad1 = x.grad.copy() if hasattr(x.grad, 'copy') else np.array(x.grad)

        # Second backward (gradients should accumulate)
        y2 = x * 2
        y2.backward()
        grad2 = x.grad

        # Second gradient should be double the first
        assert np.allclose(grad2, grad1 * 2), (
            "Gradients not accumulating!\n"
            "\n"
            "📖 IMPORTANT: backward() should ADD to .grad, not replace.\n"
            "This enables gradient accumulation across mini-batches.\n"
            "\n"
            "💡 In your backward functions, use:\n"
            "   if tensor.grad is None:\n"
            "       tensor.grad = gradient\n"
            "   else:\n"
            "       tensor.grad = tensor.grad + gradient"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Module 06: Autograd - Progressive Tests")
    print("=" * 70)
    print()
    print("To run these tests:")
    print("  pytest tests/progressive/test_module_06_autograd.py -v")
    print()
    print("Or via tito:")
    print("  tito module test 06")
    print()
    pytest.main([__file__, "-v"])
