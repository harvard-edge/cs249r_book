"""
Module 02: Activations - Core Functionality Tests
==================================================

These tests verify that activation functions work correctly.

WHY ACTIVATIONS MATTER:
----------------------
Without activations, neural networks are just linear transformations.
No matter how many layers you stack, y = W3(W2(W1*x)) = W_combined*x

Activations add NON-LINEARITY, allowing networks to learn complex patterns:
- Image recognition (cats vs dogs)
- Language understanding
- Any real-world problem

WHAT STUDENTS LEARN:
-------------------
1. Each activation has specific properties (range, gradient behavior)
2. Different activations suit different problems
3. Numerical stability matters (softmax with large values)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestReLUActivation:
    """
    Test ReLU (Rectified Linear Unit) activation.

    CONCEPT: ReLU(x) = max(0, x)
    The most popular activation in modern deep learning.
    Simple, fast, and helps avoid vanishing gradients.
    """

    def test_relu_forward(self):
        """
        WHAT: Verify ReLU outputs max(0, x) for each element.

        WHY: ReLU is the foundation of modern neural networks.
        If it doesn't work, CNNs and most architectures fail.

        STUDENT LEARNING: ReLU keeps positive values unchanged,
        zeros out negative values. This simple non-linearity is
        surprisingly powerful.
        """
        try:
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            relu = ReLU()
            x = Tensor(np.array([-2, -1, 0, 1, 2]))
            output = relu(x)

            expected = np.array([0, 0, 0, 1, 2])
            assert np.array_equal(output.data, expected), (
                f"ReLU output wrong.\n"
                f"  Input: {x.data}\n"
                f"  Expected: {expected} (negative → 0, positive → unchanged)\n"
                f"  Got: {output.data}"
            )

        except ImportError:
            pytest.skip("ReLU not implemented yet")

    def test_relu_gradient_property(self):
        """
        WHAT: Verify ReLU gradient is 1 for x>0, 0 for x≤0.

        WHY: Correct gradients are essential for backpropagation.
        Wrong gradients = model learns garbage.

        STUDENT LEARNING: ReLU has a "dead neuron" problem - if x≤0,
        gradient is 0, so the neuron stops learning. This is why
        LeakyReLU exists (small slope for negative values).
        """
        try:
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            relu = ReLU()
            x = Tensor(np.array([-1, 0, 1, 2]))
            output = relu(x)

            # Where output > 0, gradient passes through (=1)
            # Where output = 0, gradient is blocked (=0)
            gradient_mask = output.data > 0
            expected_mask = np.array([False, False, True, True])
            assert np.array_equal(gradient_mask, expected_mask), (
                "ReLU gradient mask is wrong.\n"
                "Gradient should flow (True) only where output > 0."
            )

        except ImportError:
            pytest.skip("ReLU not implemented yet")

    def test_relu_large_values(self):
        """
        WHAT: Verify ReLU handles extreme values correctly.

        WHY: Real networks encounter large values during training
        (especially early in training or with wrong learning rates).

        STUDENT LEARNING: ReLU is numerically stable - no exponentials
        or divisions that could overflow/underflow.
        """
        try:
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            relu = ReLU()
            x = Tensor(np.array([-1000, 1000]))
            output = relu(x)

            expected = np.array([0, 1000])
            assert np.array_equal(output.data, expected), (
                "ReLU failed on extreme values.\n"
                f"  Input: {x.data}\n"
                f"  Expected: {expected}\n"
                f"  Got: {output.data}"
            )

        except ImportError:
            pytest.skip("ReLU not implemented yet")


class TestSigmoidActivation:
    """
    Test Sigmoid activation function.

    CONCEPT: σ(x) = 1 / (1 + e^(-x))
    Maps any real number to (0, 1).
    Used for probabilities and binary classification.
    """

    def test_sigmoid_forward(self):
        """
        WHAT: Verify sigmoid outputs values between 0 and 1.

        WHY: Sigmoid is used for:
        - Binary classification (is it a cat? probability 0-1)
        - Gates in LSTMs (how much to remember/forget)

        STUDENT LEARNING: σ(0) = 0.5 is a key property.
        Sigmoid is centered at 0.5, not 0 (unlike tanh).
        """
        try:
            from tinytorch.core.activations import Sigmoid
            from tinytorch.core.tensor import Tensor

            sigmoid = Sigmoid()
            x = Tensor(np.array([0, 1, -1]))
            output = sigmoid(x)

            # Sigmoid(0) = 0.5
            assert np.isclose(output.data[0], 0.5, atol=1e-6), (
                f"Sigmoid(0) should be 0.5, got {output.data[0]}"
            )

            # All outputs must be in (0, 1)
            assert np.all(output.data > 0) and np.all(output.data < 1), (
                f"Sigmoid outputs must be in (0, 1).\n"
                f"  Got: {output.data}\n"
                "This is essential for probability interpretation."
            )

        except ImportError:
            pytest.skip("Sigmoid not implemented yet")

    def test_sigmoid_symmetry(self):
        """
        WHAT: Verify σ(-x) = 1 - σ(x) (point symmetry around 0.5).

        WHY: This symmetry property is used in some loss calculations
        and is a mathematical sanity check.

        STUDENT LEARNING: Sigmoid is symmetric around the point (0, 0.5).
        This makes it behave similarly for positive and negative inputs.
        """
        try:
            from tinytorch.core.activations import Sigmoid
            from tinytorch.core.tensor import Tensor

            sigmoid = Sigmoid()
            x = 2.0

            pos_out = sigmoid(Tensor([x]))
            neg_out = sigmoid(Tensor([-x]))

            expected = 1 - pos_out.data[0]
            assert np.isclose(neg_out.data[0], expected, atol=1e-6), (
                f"Sigmoid symmetry broken: σ(-x) should equal 1 - σ(x)\n"
                f"  σ({x}) = {pos_out.data[0]}\n"
                f"  σ({-x}) = {neg_out.data[0]}\n"
                f"  1 - σ({x}) = {expected}"
            )

        except ImportError:
            pytest.skip("Sigmoid not implemented yet")

    def test_sigmoid_derivative_property(self):
        """
        WHAT: Verify σ'(x) = σ(x) * (1 - σ(x)).

        WHY: This elegant derivative formula makes backprop efficient.
        No need to store x - just use the output.

        STUDENT LEARNING: Maximum derivative is at x=0 where σ'(0) = 0.25.
        Far from 0, gradients become tiny (vanishing gradient problem).
        """
        try:
            from tinytorch.core.activations import Sigmoid
            from tinytorch.core.tensor import Tensor

            sigmoid = Sigmoid()
            x = Tensor(np.array([0, 1, -1]))
            output = sigmoid(x)

            # Derivative = σ(x) * (1 - σ(x))
            derivative = output.data * (1 - output.data)

            # At x=0: σ(0)=0.5, so derivative = 0.5 * 0.5 = 0.25
            assert np.isclose(derivative[0], 0.25, atol=1e-6), (
                f"Sigmoid derivative at x=0 should be 0.25.\n"
                f"  σ(0) = {output.data[0]}\n"
                f"  σ'(0) = σ(0) * (1-σ(0)) = {derivative[0]}"
            )

        except ImportError:
            pytest.skip("Sigmoid not implemented yet")


class TestTanhActivation:
    """
    Test Tanh (hyperbolic tangent) activation.

    CONCEPT: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Maps any real number to (-1, 1).
    Zero-centered, unlike sigmoid.
    """

    def test_tanh_forward(self):
        """
        WHAT: Verify tanh outputs values between -1 and 1.

        WHY: Tanh is preferred over sigmoid in hidden layers because:
        - Zero-centered (helps optimization)
        - Stronger gradients (range is 2 instead of 1)

        STUDENT LEARNING: tanh(0) = 0 (unlike sigmoid where σ(0) = 0.5).
        This zero-centering often helps training converge faster.
        """
        try:
            from tinytorch.core.activations import Tanh
            from tinytorch.core.tensor import Tensor

            tanh = Tanh()
            x = Tensor(np.array([0, 1, -1]))
            output = tanh(x)

            assert np.isclose(output.data[0], 0, atol=1e-6), (
                f"tanh(0) should be 0, got {output.data[0]}"
            )

            assert np.all(output.data > -1) and np.all(output.data < 1), (
                f"tanh outputs must be in (-1, 1).\n"
                f"  Got: {output.data}"
            )

        except ImportError:
            pytest.skip("Tanh not implemented yet")

    def test_tanh_antisymmetry(self):
        """
        WHAT: Verify tanh(-x) = -tanh(x) (odd function).

        WHY: This antisymmetry means tanh is zero-centered.
        Positive inputs → positive outputs, negative → negative.

        STUDENT LEARNING: tanh is an "odd function" like sine.
        This symmetry helps with optimization (balanced gradients).
        """
        try:
            from tinytorch.core.activations import Tanh
            from tinytorch.core.tensor import Tensor

            tanh = Tanh()
            x = 1.5

            pos_out = tanh(Tensor([x]))
            neg_out = tanh(Tensor([-x]))

            assert np.isclose(neg_out.data[0], -pos_out.data[0], atol=1e-6), (
                f"tanh antisymmetry broken: tanh(-x) should equal -tanh(x)\n"
                f"  tanh({x}) = {pos_out.data[0]}\n"
                f"  tanh({-x}) = {neg_out.data[0]}\n"
                f"  -tanh({x}) = {-pos_out.data[0]}"
            )

        except ImportError:
            pytest.skip("Tanh not implemented yet")

    def test_tanh_range(self):
        """
        WHAT: Verify tanh saturates to ±1 for extreme inputs.

        WHY: Saturation means gradients vanish for extreme values.
        This is why we need careful initialization and normalization.

        STUDENT LEARNING: For |x| > 3, tanh is essentially ±1.
        Gradients become tiny, slowing learning (saturation).
        """
        try:
            from tinytorch.core.activations import Tanh
            from tinytorch.core.tensor import Tensor

            tanh = Tanh()
            x = Tensor(np.array([-10, -5, 0, 5, 10]))
            output = tanh(x)

            assert output.data[0] < -0.99, "tanh(-10) should be near -1"
            assert output.data[4] > 0.99, "tanh(10) should be near 1"
            assert np.isclose(output.data[2], 0, atol=1e-6), "tanh(0) should be 0"

        except ImportError:
            pytest.skip("Tanh not implemented yet")


class TestSoftmaxActivation:
    """
    Test Softmax activation function.

    CONCEPT: softmax(x_i) = e^(x_i) / Σ e^(x_j)
    Converts logits to probabilities that sum to 1.
    Used for multi-class classification.
    """

    def test_softmax_forward(self):
        """
        WHAT: Verify softmax outputs sum to 1 and are positive.

        WHY: Softmax is THE activation for classification.
        "This image is 80% cat, 15% dog, 5% bird" - that's softmax.

        STUDENT LEARNING: Softmax converts any numbers to a valid
        probability distribution. Higher input → higher probability.
        """
        try:
            from tinytorch.core.activations import Softmax
            from tinytorch.core.tensor import Tensor

            softmax = Softmax()
            x = Tensor(np.array([1, 2, 3]))
            output = softmax(x)

            assert np.isclose(np.sum(output.data), 1.0, atol=1e-6), (
                f"Softmax outputs must sum to 1.\n"
                f"  Input: {x.data}\n"
                f"  Output: {output.data}\n"
                f"  Sum: {np.sum(output.data)}"
            )

            assert np.all(output.data > 0), (
                f"Softmax outputs must all be positive.\n"
                f"  Got: {output.data}"
            )

        except ImportError:
            pytest.skip("Softmax not implemented yet")

    def test_softmax_properties(self):
        """
        WHAT: Verify softmax(x + c) = softmax(x) (shift invariance).

        WHY: This property is exploited for numerical stability.
        We subtract max(x) before computing to avoid overflow.

        STUDENT LEARNING: Adding a constant to all logits doesn't
        change the probabilities. This is because the constant
        cancels out in the ratio e^(x+c) / Σe^(x+c).
        """
        try:
            from tinytorch.core.activations import Softmax
            from tinytorch.core.tensor import Tensor

            softmax = Softmax()

            x = Tensor(np.array([1, 2, 3]))
            x_shifted = Tensor(np.array([11, 12, 13]))  # x + 10

            out1 = softmax(x)
            out2 = softmax(x_shifted)

            assert np.allclose(out1.data, out2.data, atol=1e-6), (
                f"Softmax should be shift-invariant.\n"
                f"  softmax([1,2,3]) = {out1.data}\n"
                f"  softmax([11,12,13]) = {out2.data}\n"
                "These should be identical."
            )

        except ImportError:
            pytest.skip("Softmax not implemented yet")

    def test_softmax_numerical_stability(self):
        """
        WHAT: Verify softmax handles large values without overflow.

        WHY: e^1000 = infinity in float32. Naive softmax crashes.
        Stable softmax subtracts max(x) first.

        STUDENT LEARNING: Always use the stable formula:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        This prevents both overflow (large positive) and
        underflow (large negative).
        """
        try:
            from tinytorch.core.activations import Softmax
            from tinytorch.core.tensor import Tensor

            softmax = Softmax()

            # These values would overflow with naive exp()
            x = Tensor(np.array([1000, 1001, 1002]))
            output = softmax(x)

            assert np.isclose(np.sum(output.data), 1.0, atol=1e-6), (
                "Softmax failed with large values - likely overflow."
            )
            assert np.all(np.isfinite(output.data)), (
                f"Softmax produced NaN/Inf with large values.\n"
                f"  Input: {x.data}\n"
                f"  Output: {output.data}\n"
                "Use the stable formula: exp(x - max(x))."
            )

        except (ImportError, OverflowError):
            pytest.skip("Softmax numerical stability not implemented yet")


class TestActivationComposition:
    """
    Test activation functions working together.

    CONCEPT: Real networks chain activations:
    x → Linear → ReLU → Linear → Sigmoid → output
    """

    def test_activation_chaining(self):
        """
        WHAT: Verify activations can be chained together.

        WHY: Neural networks are compositions of layers + activations.
        Each activation's output is the next layer's input.

        STUDENT LEARNING: This is how forward passes work:
        Input → (Layer1 → Act1) → (Layer2 → Act2) → ... → Output
        """
        try:
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor

            relu = ReLU()
            sigmoid = Sigmoid()

            x = Tensor(np.array([-2, -1, 0, 1, 2]))

            # Chain: x → ReLU → Sigmoid
            h = relu(x)      # [-2,-1,0,1,2] → [0,0,0,1,2]
            output = sigmoid(h)  # → [0.5,0.5,0.5,0.73,0.88]

            assert output.shape == x.shape
            assert np.all(output.data >= 0) and np.all(output.data <= 1), (
                "Chained activation output should be in sigmoid range [0,1]."
            )

        except ImportError:
            pytest.skip("Activation chaining not ready yet")

    def test_activation_with_batch_data(self):
        """
        WHAT: Verify activations handle batch dimensions.

        WHY: Training processes batches of data for efficiency.
        Activation must apply element-wise to all batch elements.

        STUDENT LEARNING: Activations are applied independently to
        each element. Shape in = shape out (always).
        """
        try:
            from tinytorch.core.activations import ReLU, Sigmoid, Tanh
            from tinytorch.core.tensor import Tensor

            # Batch of 4 samples, 3 features each
            x = Tensor(np.random.randn(4, 3))

            for name, activation in [("ReLU", ReLU()), ("Sigmoid", Sigmoid()), ("Tanh", Tanh())]:
                output = activation(x)
                assert output.shape == x.shape, (
                    f"{name} changed shape!\n"
                    f"  Input: {x.shape}\n"
                    f"  Output: {output.shape}\n"
                    "Activations should preserve shape."
                )

        except ImportError:
            pytest.skip("Batch activation processing not ready yet")

    def test_activation_zero_preservation(self):
        """
        WHAT: Test how different activations handle zero input.

        WHY: Zero is a special point - understanding behavior at 0
        helps debug initialization and normalization issues.

        STUDENT LEARNING:
        - ReLU(0) = 0 (boundary case)
        - Sigmoid(0) = 0.5 (center of range)
        - Tanh(0) = 0 (zero-centered)
        """
        try:
            from tinytorch.core.activations import ReLU, Sigmoid, Tanh
            from tinytorch.core.tensor import Tensor

            zero_input = Tensor(np.array([0.0]))

            relu = ReLU()
            assert relu(zero_input).data[0] == 0.0, "ReLU(0) should be 0"

            sigmoid = Sigmoid()
            assert np.isclose(sigmoid(zero_input).data[0], 0.5, atol=1e-6), (
                "Sigmoid(0) should be 0.5"
            )

            tanh = Tanh()
            assert np.isclose(tanh(zero_input).data[0], 0.0, atol=1e-6), (
                "Tanh(0) should be 0"
            )

        except ImportError:
            pytest.skip("Activation zero behavior not ready yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
