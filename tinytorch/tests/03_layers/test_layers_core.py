"""
Module 03: Layers - Core Functionality Tests
=============================================

These tests verify that Layer abstractions work correctly.

WHY LAYERS MATTER:
-----------------
Layers are the building blocks of neural networks:
- Linear (Dense): y = Wx + b
- Dropout: Regularization by random neuron deactivation
- Sequential: Container chaining layers in sequence

Every architecture (MLP, ResNet, Transformer, GPT) is built from layers.

WHAT STUDENTS LEARN:
-------------------
1. The Layer interface (forward, parameters, __call__)
2. The Linear layer mechanics (weights, bias, initialization, affine transformations)
3. Dropout regularization (training vs. inference behavior, inverted scaling)
4. Sequential container composition and parameter collection
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.layers import Layer, Linear, Dropout, Sequential
from tinytorch.core.tensor import Tensor


class TestLayerBaseClass:
    """
    Test the Layer base class.

    CONCEPT: Layer is the base class that all layers inherit from.
    It defines the common interface (forward, __call__, parameters)
    that makes layers composable.
    """

    def test_layer_creation(self):
        """
        WHAT: Verify Layer base class can be instantiated.

        WHY: Layer is the foundation - if it doesn't exist,
        no neural network layers can be built.

        STUDENT LEARNING: All layers (Linear, Conv2d, etc.) inherit
        from this base class. It defines the common interface.
        """
        layer = Layer()
        assert layer is not None

    def test_layer_interface(self):
        """
        WHAT: Verify Layer has the required interface.

        WHY: All layers must be callable (layer(x)) and have forward().
        This consistency enables layer composition.

        STUDENT LEARNING: The __call__ method typically calls forward().
        This pattern allows layers to be used like functions.
        """
        layer = Layer()
        assert hasattr(layer, 'forward'), "Layer must have forward() method"
        assert callable(layer), "Layer must be callable (implement __call__)"

    def test_layer_inheritance(self):
        """
        WHAT: Verify custom layers can inherit from Layer.

        WHY: Students need to create custom layers for specific tasks.

        STUDENT LEARNING: To create a custom layer:
        1. Inherit from Layer
        2. Override forward() method
        3. Optionally store parameters as Tensors
        """
        class IdentityLayer(Layer):
            def forward(self, x):
                return x

        layer = IdentityLayer()
        x = Tensor(np.array([1, 2, 3]))
        output = layer(x)

        assert isinstance(output, Tensor)
        assert np.array_equal(output.data, x.data), "Identity layer should return input unchanged"


class TestLinearLayer:
    """
    Test Linear (Dense) layer functionality.

    CONCEPT: Linear(in_features, out_features) performs y = x @ W + b.
    It is the most fundamental learnable transformation in deep learning.
    """

    def test_linear_creation_and_shapes(self):
        """
        WHAT: Verify Linear layer initializes weights and bias with expected shapes.

        WHY: Linear layers map an input space of dimension in_features to out_features.
        Shape mismatches break matrix multiplication immediately.

        STUDENT LEARNING: For Linear(in_features, out_features):
        - weight shape is (in_features, out_features)
        - bias shape is (out_features,) when bias=True
        - bias is None when bias=False
        """
        layer = Linear(10, 5, bias=True)
        assert layer.weight.shape == (10, 5)
        assert layer.bias is not None
        assert layer.bias.shape == (5,)
        assert isinstance(layer.weight, Tensor)
        assert isinstance(layer.bias, Tensor)

        layer_no_bias = Linear(10, 5, bias=False)
        assert layer_no_bias.weight.shape == (10, 5)
        assert layer_no_bias.bias is None

    def test_linear_weight_initialization(self):
        """
        WHAT: Verify weights are initialized with stable variance.

        WHY: Bad initialization causes vanishing or exploding activations.
        LeCun / Xavier initialization scales weights inversely with fan-in.

        STUDENT LEARNING: Weights should have zero mean and standard deviation
        close to 1/sqrt(in_features).
        """
        layer = Linear(100, 50)
        weights_data = layer.weight.data
        std = np.std(weights_data)
        # Expected std is approximately 1/sqrt(100) = 0.10
        assert 0.05 < std < 0.20, f"Weight std unexpected: {std:.4f}"

    def test_linear_forward_with_bias(self):
        """
        WHAT: Verify forward pass computes y = x @ W + b.

        WHY: The affine transformation allows neural networks to represent
        functions not centered at the origin.

        STUDENT LEARNING: When input is zero, the output equals the bias vector.
        """
        layer = Linear(4, 2, bias=True)
        zero_x = Tensor(np.zeros((1, 4)))
        output = layer(zero_x)

        assert output.shape == (1, 2)
        np.testing.assert_allclose(output.data, layer.bias.data.reshape(1, 2))

    def test_linear_forward_without_bias(self):
        """
        WHAT: Verify forward pass without bias computes strictly linear y = x @ W.

        WHY: Some architectures (e.g. projection layers before LayerNorm) omit bias.

        STUDENT LEARNING: With zero input and bias=False, the output is strictly zero.
        """
        layer = Linear(4, 2, bias=False)
        zero_x = Tensor(np.zeros((1, 4)))
        output = layer(zero_x)

        assert output.shape == (1, 2)
        np.testing.assert_allclose(output.data, np.zeros((1, 2)))

    def test_linear_exact_arithmetic(self):
        """
        WHAT: Verify exact numerical computation of matrix multiplication and bias add.

        WHY: Testing known hand-calculated arithmetic guarantees correct matrix ordering.

        STUDENT LEARNING: For batch size 1: [1, 2] @ [[1, 3], [2, 4]] + [10, 20] = [15, 31].
        """
        layer = Linear(2, 2, bias=True)
        layer.weight.data[:] = np.array([[1.0, 3.0], [2.0, 4.0]])
        layer.bias.data[:] = np.array([10.0, 20.0])

        x = Tensor(np.array([[1.0, 2.0]]))
        output = layer(x)

        expected = np.array([[15.0, 31.0]])
        np.testing.assert_allclose(output.data, expected)

    def test_linear_batch_processing(self):
        """
        WHAT: Verify Linear handles arbitrary batch sizes independently.

        WHY: Efficient training processes mini-batches in parallel.

        STUDENT LEARNING: The batch dimension is preserved throughout:
        (batch_size, in_features) @ (in_features, out_features) = (batch_size, out_features).
        """
        layer = Linear(8, 4)
        for batch_size in [1, 7, 16, 32]:
            x = Tensor(np.random.default_rng(batch_size).standard_normal((batch_size, 8)))
            output = layer(x)
            assert output.shape == (batch_size, 4)

    def test_linear_parameters_collection(self):
        """
        WHAT: Verify parameters() returns weight and bias Tensors.

        WHY: Optimizers need access to all trainable parameters of a layer.

        STUDENT LEARNING: layer.parameters() yields [weight, bias] (or just [weight] if bias=False).
        """
        layer_with_bias = Linear(4, 2, bias=True)
        params = layer_with_bias.parameters()
        assert len(params) == 2
        assert params[0] is layer_with_bias.weight
        assert params[1] is layer_with_bias.bias

        layer_without_bias = Linear(4, 2, bias=False)
        params_no_b = layer_without_bias.parameters()
        assert len(params_no_b) == 1
        assert params_no_b[0] is layer_without_bias.weight


class TestDropoutLayer:
    """
    Test Dropout regularization layer.

    CONCEPT: Dropout randomly deactivates neurons during training to prevent co-adaptation.
    During inference, it behaves as the identity pass-through.
    """

    def test_dropout_validation(self):
        """
        WHAT: Verify Dropout validates probability p is between 0.0 and 1.0.

        WHY: A probability outside [0.0, 1.0] is mathematically invalid.

        STUDENT LEARNING: p is the probability of dropping a neuron.
        """
        Dropout(0.0)
        Dropout(0.5)
        Dropout(1.0)

        with pytest.raises(ValueError):
            Dropout(-0.1)
        with pytest.raises(ValueError):
            Dropout(1.1)

    def test_dropout_eval_mode_pass_through(self):
        """
        WHAT: Verify Dropout in evaluation/inference mode returns input unchanged.

        WHY: During testing or inference, all neurons must remain active.

        STUDENT LEARNING: Pass training=False for deterministic evaluation pass-through.
        """
        dropout = Dropout(0.5)
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        output = dropout(x, training=False)

        np.testing.assert_allclose(output.data, x.data)

    def test_dropout_zero_prob_is_identity(self):
        """
        WHAT: Verify Dropout with p=0.0 does not zero any activations even in training.

        WHY: A drop rate of 0% means no regularization is applied.

        STUDENT LEARNING: Dropout(0.0) acts as an identity layer.
        """
        dropout = Dropout(0.0)
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        output = dropout(x, training=True)

        np.testing.assert_allclose(output.data, x.data)

    def test_dropout_full_prob_zeros_all(self):
        """
        WHAT: Verify Dropout with p=1.0 drops all activations during training.

        WHY: At p=1.0, every neuron is zeroed out.

        STUDENT LEARNING: Inverted dropout zeros all elements when p=1.0.
        """
        dropout = Dropout(1.0)
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        output = dropout(x, training=True)

        np.testing.assert_allclose(output.data, np.zeros_like(x.data))

    def test_dropout_inverted_scaling_expectation(self):
        """
        WHAT: Verify surviving activations are scaled by 1/(1-p) to preserve expected value.

        WHY: Inverted dropout avoids having to scale weights at inference time.

        STUDENT LEARNING: Scaling by 1/(1-p) keeps the expected sum equal to the input sum:
        E[output] = (1-p) * (x / (1-p)) = x.
        """
        p = 0.5
        dropout = Dropout(p)

        # Large tensor to test expected value empirically
        x_data = np.ones((100, 100), dtype=np.float32)
        x = Tensor(x_data)
        output = dropout(x)

        # Output elements should be either 0.0 or 1.0 / (1 - p) = 2.0
        unique_vals = np.unique(np.round(output.data, decimals=3))
        for val in unique_vals:
            assert val in [0.0, 2.0], f"Unexpected value in dropout output: {val}"

        # Mean output should be close to 1.0 (mean of input)
        mean_val = np.mean(output.data)
        assert 0.90 < mean_val < 1.10, f"Expected mean ~1.0, got {mean_val:.4f}"


class TestSequentialContainer:
    """
    Test Sequential container layer.

    CONCEPT: Sequential chains multiple layers in order: x -> Layer1 -> Layer2 -> Layer3.
    It encapsulates complex multi-layer architectures behind a single Layer interface.
    """

    def test_sequential_instantiation_list_and_args(self):
        """
        WHAT: Verify Sequential accepts either a list of layers or *args.

        WHY: Flexible API matches PyTorch conventions and student expectations.

        STUDENT LEARNING: Sequential([l1, l2]) and Sequential(l1, l2) both work.
        """
        l1 = Linear(4, 8)
        l2 = Linear(8, 2)

        s1 = Sequential([l1, l2])
        assert len(s1.layers) == 2

        s2 = Sequential(l1, l2)
        assert len(s2.layers) == 2

    def test_sequential_forward_chaining(self):
        """
        WHAT: Verify Sequential executes layers in sequential order.

        WHY: Forward pass must pipe the output of each layer as the input to the next.

        STUDENT LEARNING: output = layerN(...layer2(layer1(x))).
        """
        l1 = Linear(2, 2, bias=False)
        l2 = Linear(2, 1, bias=False)
        l1.weight.data[:] = np.array([[1.0, 0.0], [0.0, 2.0]])
        l2.weight.data[:] = np.array([[3.0], [4.0]])

        model = Sequential(l1, l2)
        x = Tensor(np.array([[2.0, 3.0]]))
        # l1(x) = [2*1 + 0, 0 + 3*2] = [2, 6]
        # l2([2, 6]) = [2*3 + 6*4] = [6 + 24] = [30]
        output = model(x)
        expected = np.array([[30.0]])
        np.testing.assert_allclose(output.data, expected)

    def test_sequential_parameters_aggregation(self):
        """
        WHAT: Verify Sequential.parameters() collects parameters from all child layers.

        WHY: An optimizer trains the entire network by iterating over model.parameters().

        STUDENT LEARNING: Sequential traverses child layers to aggregate their parameter lists.
        """
        l1 = Linear(4, 8, bias=True)
        l2 = Linear(8, 2, bias=True)
        model = Sequential(l1, l2)

        params = model.parameters()
        assert len(params) == 4
        assert params == [l1.weight, l1.bias, l2.weight, l2.bias]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
