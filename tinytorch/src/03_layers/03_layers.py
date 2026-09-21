# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
r"""
# Module 03: Layers - Building Blocks of Neural Networks

Welcome to Module 03! You're about to build the fundamental building blocks that make neural networks possible.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor class (Module 01) with all operations and activations (Module 02)
**You'll Build**: A Layer base class, Linear layers, Dropout regularization, and a Sequential container
**You'll Enable**: Multi-layer neural networks and forward passes, with parameters collected in the form the optimizers will later consume (nothing trains yet)

**Connection Pipeline**:
$$\mathbf{X} \in \text{Tensor} \xrightarrow{\text{Nonlinearity}} \sigma(\mathbf{X}) \in \text{Activations} \xrightarrow{\text{Affine Transform}} \mathbf{X}\mathbf{W} + \mathbf{b} \in \text{Layers} \xrightarrow{\text{Composition}} \text{Sequential} \in \text{Networks}$$

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement Linear layers with proper weight initialization
2. Add Dropout for regularization during training
3. Understand parameter management and counting
4. Compose layers into a network, first by hand and then with a Sequential container
5. Test individual layer components

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/03_layers/layers.ipynb`
**Building Side:** Code exports to tinytorch.core.layers

```python
# Final package structure:
from tinytorch.core.layers import Layer, Linear, Dropout, Sequential  # This module
from tinytorch.core.tensor import Tensor  # Module 01 - foundation
from tinytorch.core.activations import ReLU  # Module 02 - intelligence
```

**Why this matters:**
- **Learning:** Complete layer system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn with all layer building blocks together
- **Consistency:** All layer operations and parameter management in core.layers
- **Integration:** Works seamlessly with tensors and activations for complete neural networks
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01 (Tensor) and 02 (Activations) must be completed

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `inspect` (Python standard library; Sequential uses it to see which layers accept a training flag)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor.Tensor` (Module 01)
- `tinytorch.core.activations.ReLU` (Module 02)

This module depends on Tensor and Activations.
Ensure previous modules are completed and exported.

**Dependency Flow**:
$$\underbrace{\text{Module 01: Tensor}}_{\text{Data Container and Autograd}} \longrightarrow \underbrace{\text{Module 02: Activations}}_{\text{Nonlinear Functions}} \longrightarrow \underbrace{\text{Module 03: Layers}}_{\text{Parametric Architecture}}$$

Students completing this module will have built the neural network
layers that enable multi-layer architectures.
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.layers
#| export

import inspect
import numpy as np
# Module-level RNG is seeded so Linear weight init is deterministic by default.
# This is what the integration test suite (and any cross-run reproducibility)
# relies on. Demo scripts that want fresh weights every run rebind this name
# to an unseeded RNG locally before constructing their model — see
# milestones/01_1958_perceptron/01_rosenblatt_forward.py for the pattern.
rng = np.random.default_rng(7)

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU

# Constant for weight initialization
# Note: True Xavier/Glorot uses sqrt(2/(fan_in+fan_out)), but we use the simpler
# LeCun-style sqrt(1/fan_in) for pedagogical clarity. Both keep the output
# variance of a layer close to its input variance.
INIT_SCALE_FACTOR = 1.0  # LeCun-style initialization: sqrt(1/fan_in)

# Constants for dropout
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)

# %% [markdown]
r"""
## 💡 Introduction: What are Neural Network Layers?

Neural network layers are the fundamental building blocks that transform data as it flows through a network. Each layer encapsulates both state (trainable weights and biases) and computation (forward transformation):

- **Linear layers** apply learned affine transformations: $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$
- **Dropout layers** randomly zero elements during training for regularization
- **Sequential containers** compose multiple layers into unified callable models

<div align="center">
  <img src="layers_architecture.svg" alt="TinyTorch Layer System Architecture" width="680px">
</div>

Data flows sequentially through cascaded representations:
$$\mathbf{X} \in \mathbb{R}^{B \times D_0} \xrightarrow{\text{Layer}_1} \mathbf{H}_1 \in \mathbb{R}^{B \times D_1} \xrightarrow{\text{Layer}_2} \mathbf{H}_2 \in \mathbb{R}^{B \times D_2} \xrightarrow{\text{Layer}_3} \hat{\mathbf{Y}} \in \mathbb{R}^{B \times C}$$

Each layer plays a distinct role. Linear layers project representations into new feature spaces, activations introduce nonlinearity, and dropout forces the representation to spread across units instead of concentrating in a few. In a trained network those projections are learned. Here they are random, because nothing in this module updates a weight, and Module 07 will add the optimizers that do.
"""

# %% [markdown]
r"""
## 📐 Foundations: Mathematical Background

### Linear Layer Mathematics

A linear layer computes a batched affine transformation:
$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

$$\underbrace{\mathbf{X}}_{(B, D_{\text{in}})} \times \underbrace{\mathbf{W}}_{(D_{\text{in}}, D_{\text{out}})} + \underbrace{\mathbf{b}}_{(D_{\text{out}},)} = \underbrace{\mathbf{Y}}_{(B, D_{\text{out}})}$$

There is one layout difference worth knowing before you read PyTorch source. `torch.nn.Linear` stores its weight transposed, as $(D_{\text{out}}, D_{\text{in}})$, and computes $\mathbf{X}\mathbf{W}^{\top}$. TinyTorch stores $(D_{\text{in}}, D_{\text{out}})$ and computes $\mathbf{X}\mathbf{W}$, so the shapes read left to right in the order the data flows. The arithmetic is identical and every site in this module uses the TinyTorch layout consistently, but a weight matrix copied between the two frameworks needs a transpose.

### Weight Initialization: Preserving Signal Variance

Random initialization is crucial for breaking symmetry and preventing signals from exploding or vanishing across deep cascades:

<div align="center">
  <img src="variance_waterfall.svg" alt="Signal Variance Across Layers" width="680px">
</div>

| Initialization Scheme | Standard Deviation ($\sigma$) | Target Activation / Design Rationale |
|:---|:---|:---|
| **LeCun (used here)** | $\sigma = \sqrt{\frac{1}{D_{\text{in}}}}$ | Linear / Sigmoid inputs; maintains unit output variance for linear maps |
| **Xavier / Glorot** | $\sigma = \sqrt{\frac{2}{D_{\text{in}} + D_{\text{out}}}}$ | Tanh / symmetric activations; harmonizes forward & backward pass signal variance |
| **He / Kaiming** | $\sigma = \sqrt{\frac{2}{D_{\text{in}}}}$ | ReLU activations; compensates for the $50\%$ variance loss from negative clamping |

<div align="center">
  <img src="kaiming_scaling.svg" alt="Kaiming Scale Scaling Factor" width="280px">
</div>

We adopt LeCun initialization $\sigma = \sqrt{\frac{1}{D_{\text{in}}}}$ for clean pedagogical clarity: on zero-mean unit-variance inputs, $\text{Var}(y_j) = \sum_{i=1}^{D_{\text{in}}} \text{Var}(x_i) \text{Var}(w_{ij}) = D_{\text{in}} \cdot \frac{1}{D_{\text{in}}} = 1.0$.

### Parameter Counting

For any layer $\text{Linear}(D_{\text{in}}, D_{\text{out}})$:
$$\text{Parameters} = \underbrace{D_{\text{in}} \times D_{\text{out}}}_{\text{Weights } \mathbf{W}} + \underbrace{D_{\text{out}}}_{\text{Biases } \mathbf{b}}$$

Example:
$$\text{Linear}(784, 256): 784 \times 256 + 256 = 200{,}704 + 256 = 200{,}960 \text{ parameters}$$

In a 2-layer classifier with ReLU:
- $\text{Layer 1: } \text{Linear}(784, 256) \implies 200{,}960 \text{ params}$
- $\text{Activation: } \text{ReLU}() \implies 0 \text{ params}$
- $\text{Layer 2: } \text{Linear}(256, 10) \implies 256 \times 10 + 10 = 2{,}570 \text{ params}$
- **Total: $203{,}530$ parameters** ($203{,}530 \times 4\text{ bytes} \approx 814.1\text{ KB}$ in FP32)
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Layer Foundation

Let's build our layer system step by step. We'll implement two essential layer types on top of a shared base class, then add a container that chains them:

1. **Linear Layer** - The workhorse of neural networks
2. **Dropout Layer** - Prevents overfitting
3. **Sequential** - Chains layers so a network is one callable object

### Key Design Principles:
- Forward methods never modify the input in place; they return a Tensor computed from it
- parameters() lists exactly the tensors a layer learns. Module 07 will add optimizers that update whatever this list returns
- Gradient tracking is not the layer's job. Module 06 will add it to Tensor without changing these classes
"""

# %% [markdown]
"""
### Layer Base Class: Foundation for All Layers

All neural network layers share common functionality: forward pass, parameter management, and callable interface. The base Layer class provides this consistent interface.
"""

# %% nbgrader={"grade": false, "grade_id": "layer-base", "solution": false}
#| export
class Layer:
    """
    Base class for all neural network layers.

    All layers should inherit from this class and implement:
    - forward(x): Compute layer output
    - parameters(): Return list of trainable parameters

    The __call__ method is provided to make layers callable, and it forwards
    any extra arguments (such as Dropout's training flag) to forward().
    The default parameters() returns an empty list, which is right for any
    layer without learnable weights.
    """

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor after transformation
        """
        raise NotImplementedError(
            f"forward() not implemented in {self.__class__.__name__}\n"
            f"  ❌ The Layer base class requires subclasses to implement forward()\n"
            f"  💡 forward() defines how input data is transformed by this layer\n"
            f"  🔧 Add this method to your class:\n"
            f"     def forward(self, x):\n"
            f"         # Your transformation logic here\n"
            f"         return transformed_x"
        )

    def __call__(self, x, *args, **kwargs):
        """Allow layer to be called like a function."""
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        """
        Return list of trainable parameters.

        Returns:
            List of Tensor objects (weights and biases)
        """
        return []  # Base class has no parameters

    def __repr__(self):
        """String representation of the layer."""
        return f"{self.__class__.__name__}()"

# %% [markdown]
r"""
### Linear Layer: The Foundation of Neural Networks

Linear layers (also known as Dense or Fully Connected layers) apply an affine transformation to incoming features:

$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

Where:
- $\mathbf{X} \in \mathbb{R}^{B \times D_{\text{in}}}$: Input feature representations across batch size $B$
- $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$: Trainable weight kernel
- $\mathbf{b} \in \mathbb{R}^{D_{\text{out}}}$: Trainable bias vector, broadcast across the batch dimension
- $\mathbf{Y} \in \mathbb{R}^{B \times D_{\text{out}}}$: Projected output representations

### Why Linear Layers Matter

Linear layers learn **feature projections**. Each output column $j$ in $\mathbf{W}$ corresponds to a learned synthetic detector: $y_{bj} = \sum_i x_{bi} w_{ij} + b_j$. Stacking these projections enables networks to uncover hierarchical representations.

### Coordinate Contraction

$$\begin{matrix}
\text{Input Batch} & & \text{Weight Kernel} & & \text{Bias Offset} & & \text{Output Logits} \\
\mathbf{X} & \times & \mathbf{W} & + & \mathbf{b} & = & \mathbf{Y} \\
[B, D_{\text{in}}] & & [D_{\text{in}}, D_{\text{out}}] & & [D_{\text{out}}] & & [B, D_{\text{out}}] \\
[32, 784] & \times & [784, 10] & + & [10] & = & [32, 10]
\end{matrix}$$

### Memory Footprint Analysis: Linear(784, 256)

| Parameter Component | Matrix Shape | Data Type | Element Count | Storage Footprint (FP32) |
|:---|:---|:---|:---|:---|
| **Weight Matrix $\mathbf{W}$** | $(784, 256)$ | `float32` | $784 \times 256 = 200{,}704$ | $200{,}704 \times 4\text{ B} = 802.82\text{ KB}$ |
| **Bias Vector $\mathbf{b}$** | $(256,)$ | `float32` | $256$ | $256 \times 4\text{ B} = 1.02\text{ KB}$ |
| **Total Resident Parameters** | — | — | **$200{,}960$ parameters** | **$803.84\text{ KB}$** |

Every KB and MB in this module is decimal ($1\text{ KB} = 1000\text{ B}$, $1\text{ MB} = 10^6\text{ B}$), the convention disk and network vendors use. Divide by $1024$ instead and the same weight matrix reads $784.0\text{ KiB}$, which is the same bytes under a different name.
"""

# %% nbgrader={"grade": false, "grade_id": "linear-layer", "solution": true}
#| export
class Linear(Layer):
    """
    Linear (fully connected) layer: y = xW + b

    This is the fundamental building block of neural networks.
    Applies a linear transformation to incoming data.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize linear layer with proper weight initialization.

        TODO: Initialize weights and bias with proper scaling

        APPROACH:
        1. Create weight matrix (in_features, out_features) with LeCun scaling
        2. Create bias vector (out_features,) initialized to zeros if bias=True
        3. Store as Tensor objects for use in forward pass

        EXAMPLE:
        >>> layer = Linear(784, 10)  # MNIST classifier final layer
        >>> print(layer.weight.shape)
        (784, 10)
        >>> print(layer.bias.shape)
        (10,)

        HINTS:
        - LeCun-style init: scale = sqrt(1/in_features)
        - Use rng.standard_normal() for normal distribution
        - bias=None when bias=False
        """
        ### BEGIN SOLUTION role="scaffold"
        self.in_features = in_features
        self.out_features = out_features

        # LeCun-style initialization for stable gradients
        scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        weight_data = rng.standard_normal((in_features, out_features)) * scale
        self.weight = Tensor(weight_data, requires_grad=True)

        # Initialize bias to zeros or None
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through linear layer.

        TODO: Implement y = xW + b

        APPROACH:
        1. Matrix multiply input with weights: xW
        2. Add bias if it exists
        3. Return result as new Tensor

        EXAMPLE:
        >>> layer = Linear(3, 2)
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features
        >>> y = layer.forward(x)
        >>> print(y.shape)
        (2, 2)  # 2 samples, 2 outputs

        HINTS:
        - Use tensor.matmul() for matrix multiplication
        - Handle bias=None case
        - Broadcasting automatically handles bias addition
        """
        ### BEGIN SOLUTION
        # Linear transformation: y = xW
        output = x.matmul(self.weight)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output
        ### END SOLUTION

    def parameters(self):
        """
        Return list of trainable parameters.

        TODO: Return all tensors that need gradients

        APPROACH:
        1. Start with weight (always present)
        2. Add bias if it exists
        3. Return as list for optimizer

        EXAMPLE:
        >>> layer = Linear(10, 5)
        >>> params = layer.parameters()
        >>> len(params)
        2  # [weight, bias]
        >>> layer_no_bias = Linear(10, 5, bias=False)
        >>> len(layer_no_bias.parameters())
        1  # [weight only]

        HINTS:
        - Create list starting with self.weight
        - Check if self.bias is not None before appending
        - Return the complete list
        """
        ### BEGIN SOLUTION role="scaffold"
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
        ### END SOLUTION

    def __repr__(self):
        """String representation for debugging."""
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"

# %% [markdown]
"""
### 🧪 Unit Test: Linear Layer

This test validates our Linear layer implementation works correctly.

**What we're testing**: Weight initialization, forward pass, parameter management
**Why it matters**: Foundation for all neural network architectures
**Expected**: Proper shapes, LeCun-style scaling, parameter counting
"""

# %% nbgrader={"grade": true, "grade_id": "test-linear", "locked": true, "points": 15}
def test_unit_linear_layer():
    """🧪 Test Linear layer implementation."""
    print("🧪 Unit Test: Linear Layer...")

    # Test layer creation
    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)

    # Test LeCun-style initialization (weights should be reasonably scaled)
    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(INIT_SCALE_FACTOR / 784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, f"Weight std {weight_std} not close to expected {expected_std}"

    # Test bias initialization (should be zeros)
    assert np.allclose(layer.bias.data, 0), "Bias should be initialized to zeros"

    # Test forward pass
    x = Tensor(rng.standard_normal((32, 784)))  # Batch of 32 samples
    y = layer.forward(x)
    assert y.shape == (32, 256), f"Expected shape (32, 256), got {y.shape}"

    # Test no bias option
    layer_no_bias = Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1  # Only weight, no bias

    # Test parameters method
    params = layer.parameters()
    assert len(params) == 2  # Weight and bias
    assert params[0] is layer.weight
    assert params[1] is layer.bias

    print("✅ Linear layer works correctly!")

if __name__ == "__main__":
    test_unit_linear_layer()

# %% [markdown]
"""
### 🧪 Unit Test: Linear Edge Cases

Additional tests for edge cases and error handling.

**What we're testing**: Linear layer behavior at the boundaries, covering empty batches,
single samples, large weight magnitudes, and mismatched input widths
**Why it matters**: Edge cases are where a layer that "works" quietly stops
working, usually the first time a real dataset has a ragged final batch
**Expected**: Correct shapes at every boundary, clear errors on genuine mismatches
"""

# %% nbgrader={"grade": true, "grade_id": "test-linear-edge-cases", "locked": true, "points": 5}
def test_unit_edge_cases_linear():
    """🧪 Test Linear layer edge cases."""
    print("🧪 Unit Test: Linear Edge Cases...")

    layer = Linear(10, 5)

    # Test single sample (should handle 2D input)
    x_2d = Tensor(rng.standard_normal((1, 10)))
    y = layer.forward(x_2d)
    assert y.shape == (1, 5), "Should handle single sample"

    # Test zero batch size (edge case)
    x_empty = Tensor(rng.standard_normal((0, 10)))
    y_empty = layer.forward(x_empty)
    assert y_empty.shape == (0, 5), "Should handle empty batch"

    # Test large weight magnitudes: the output is large but exactly predictable
    layer_large = Linear(10, 5)
    layer_large.weight.data = np.ones((10, 5)) * 100  # Large but not extreme
    x = Tensor(np.ones((1, 10)))
    y = layer_large.forward(x)
    # Ten inputs of 1.0 against ten weights of 100.0, plus a zero bias, is exactly 1000.0
    assert np.allclose(y.data, 1000.0), f"Expected 10 x 100 = 1000 per output, got {y.data}"
    assert np.all(np.isfinite(y.data)), "Large weights should still produce finite output"

    # Test a genuine mismatch: the input's last axis must equal in_features
    try:
        layer.forward(Tensor(rng.standard_normal((4, 7))))
        assert False, "Should raise ValueError when input width does not match in_features"
    except ValueError as err:
        assert "shape mismatch" in str(err), f"Error should name the shape mismatch, got: {err}"

    # Test with no bias
    layer_no_bias = Linear(10, 5, bias=False)
    x = Tensor(rng.standard_normal((4, 10)))
    y = layer_no_bias.forward(x)
    assert y.shape == (4, 5), "Should work without bias"

    print("✅ Edge cases handled correctly!")

if __name__ == "__main__":
    test_unit_edge_cases_linear()

# %% [markdown]
"""
### 🧪 Unit Test: Linear Parameter Collection

Tests to ensure Linear layer parameters can be collected for optimization.

**What we're testing**: parameters() returns the weight and bias, in a form the
optimizer accepts
**Why it matters**: The optimizer trains exactly what parameters() hands it. A
parameter left out of that list is a parameter that silently never learns
**Expected**: Both tensors returned, with the shapes the layer was built with
"""

# %% nbgrader={"grade": true, "grade_id": "test-linear-params", "locked": true, "points": 5}
def test_unit_parameter_collection_linear():
    """🧪 Test Linear layer parameter collection."""
    print("🧪 Unit Test: Linear Parameter Collection...")

    layer = Linear(10, 5)

    # Verify parameter collection works
    params = layer.parameters()
    assert len(params) == 2, "Should return 2 parameters (weight and bias)"
    assert params[0].shape == (10, 5), "First param should be weight"
    assert params[1].shape == (5,), "Second param should be bias"

    # Test layer without bias
    layer_no_bias = Linear(10, 5, bias=False)
    params_no_bias = layer_no_bias.parameters()
    assert len(params_no_bias) == 1, "Should return 1 parameter (weight only)"

    print("✅ Parameter collection works correctly!")

if __name__ == "__main__":
    test_unit_parameter_collection_linear()


# %% [markdown]
r"""
### Dropout Layer: Preventing Co-Adaptation via Inverted Regularization

Dropout is an empirical regularization technique that randomly masks features during training, preventing individual neurons from co-adapting and memorizing idiosyncratic noise in the training set.

### Inverted Dropout Formulation

During training, each element is retained with probability $q = 1 - p$. In **inverted dropout**, surviving elements are scaled by $\frac{1}{1-p}$ at training time:

$$\mathbf{m} \sim \text{Bernoulli}(1 - p), \quad \hat{\mathbf{m}} = \frac{\mathbf{m}}{1 - p}$$
$$\mathbf{y}_{\text{train}} = \mathbf{x} \odot \hat{\mathbf{m}}, \quad \mathbb{E}[\mathbf{y}_{\text{train}}] = \mathbf{x} \odot \frac{\mathbb{E}[\mathbf{m}]}{1 - p} = \mathbf{x}$$

$$\mathbf{y}_{\text{eval}} = \mathbf{x}$$

Because $\mathbb{E}[\mathbf{y}_{\text{train}}] = \mathbf{y}_{\text{eval}}$, **no scaling or modification is required during inference**! Evaluation runs at full throughput as a pure identity operation.

### Training vs. Evaluation Trace ($p = 0.5 \implies \text{scale} = 2.0$)

$$\mathbf{x} = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 & 5.0 & 6.0 & 7.0 & 8.0 \end{bmatrix}$$

$$\text{Training Mask } \hat{\mathbf{m}} = \begin{bmatrix} 2.0 & 0.0 & 2.0 & 0.0 & 2.0 & 2.0 & 0.0 & 2.0 \end{bmatrix}$$

$$\mathbf{y}_{\text{train}} = \mathbf{x} \odot \hat{\mathbf{m}} = \begin{bmatrix} 2.0 & 0.0 & 6.0 & 0.0 & 10.0 & 12.0 & 0.0 & 16.0 \end{bmatrix}$$

$$\mathbf{y}_{\text{eval}} = \mathbf{x} = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 & 5.0 & 6.0 & 7.0 & 8.0 \end{bmatrix}$$

### Execution Mode & Memory Comparison

| Operational Mode | Transformation Rule | Expectation $\mathbb{E}[y]$ | Active Buffer Overhead |
|:---|:---|:---|:---|
| **Training Mode (`train()`)** | $\mathbf{x} \odot \frac{\mathbf{m}}{1-p}$ | $\mathbf{x}$ (invariant) | Input ($X\text{ MB}$) + Mask ($X\text{ MB}$) + Output ($X\text{ MB}$) $\approx 3X\text{ MB}$ peak |
| **Inference Mode (`eval()`)** | Identity ($\mathbf{x}$) | $\mathbf{x}$ (exact) | **Zero overhead**; no mask allocated or evaluated |
"""

# %% nbgrader={"grade": false, "grade_id": "dropout-layer", "solution": true}
#| export
class Dropout(Layer):
    """
    Dropout layer for regularization.

    During training: randomly zeros elements with probability p, scales survivors by 1/(1-p)
    During inference: passes input through unchanged

    This prevents overfitting by forcing the network to not rely on specific neurons.
    """

    def __init__(self, p=0.5):
        """
        Initialize dropout layer.

        TODO: Store dropout probability and validate range

        APPROACH:
        1. Validate p is between 0.0 and 1.0 (inclusive)
        2. Raise ValueError if out of range
        3. Store p as instance attribute

        Args:
            p: Probability of zeroing each element (0.0 = no dropout, 1.0 = zero everything)

        EXAMPLE:
        >>> dropout = Dropout(0.5)  # Zero 50% of elements during training
        >>> dropout.p
        0.5

        HINTS:
        - Use DROPOUT_MIN_PROB and DROPOUT_MAX_PROB constants for validation
        - Check: DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB
        - Raise descriptive ValueError if invalid
        """
        ### BEGIN SOLUTION role="scaffold"
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(
                f"Invalid dropout probability: {p}\n"
                f"  ❌ p must be between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}\n"
                f"  💡 p is the probability of DROPPING a neuron (not keeping it!)\n"
                f"     p=0.0 means keep all neurons (no dropout)\n"
                f"     p=0.5 means drop 50% of neurons randomly\n"
                f"     p=1.0 means drop all neurons (zero output)\n"
                f"  🔧 Common values: Dropout(0.1) for light, Dropout(0.3) for moderate, Dropout(0.5) for aggressive"
            )
        self.p = p
        ### END SOLUTION

    def _should_apply_dropout(self, training):
        """
        Determine whether dropout should be applied.

        Dropout is a training-time technique. During inference the full
        network is used, so dropout is skipped. It is also skipped when p=0
        (no neurons are dropped) since the result would be the identity.

        TODO: Return True only when dropout should actually modify the input

        APPROACH:
        1. Check if we are in training mode
        2. Check if dropout probability is greater than zero

        EXAMPLE:
        >>> d = Dropout(0.5)
        >>> d._should_apply_dropout(training=True)
        True
        >>> d._should_apply_dropout(training=False)
        False

        HINT: Both conditions must be true for dropout to apply
        """
        ### BEGIN SOLUTION role="scaffold"
        return training and self.p > DROPOUT_MIN_PROB
        ### END SOLUTION

    def _generate_dropout_mask(self, shape):
        """
        Generate a random dropout mask with inverted scaling.

        The mask has the same shape as the input. Each element is either
        0 (dropped) or 1/(1-p) (kept and scaled). Scaling at training time
        keeps the expected value of each element unchanged, so no adjustment
        is needed at inference. This trick is called "inverted dropout."

        ```
        Example with p=0.5 (keep_prob=0.5, scale=2.0):
        random draw:  [0.3,  0.8,  0.1,  0.6]
                        ↓     ↓     ↓     ↓
        keep?         [yes,  no,  yes,  no ]   (< 0.5?)
                        ↓     ↓     ↓     ↓
        mask:         [2.0,  0.0,  2.0,  0.0]  (kept × scale, dropped × 0)
        ```

        TODO: Build the scaled binary mask

        APPROACH:
        1. Compute keep_prob = 1 - p
        2. Draw uniform random values and threshold at keep_prob
        3. Convert the boolean mask to float and scale by 1/keep_prob

        EXAMPLE:
        >>> d = Dropout(0.5)
        >>> mask = d._generate_dropout_mask((4,))
        >>> mask.shape
        (4,)

        HINTS:
        - rng.random(shape) gives uniform [0, 1) values
        - Threshold with < keep_prob to get a boolean mask
        - Scale factor is 1.0 / keep_prob
        """
        ### BEGIN SOLUTION role="scaffold"
        keep_prob = 1.0 - self.p
        binary_mask = (rng.random(shape) < keep_prob).astype(np.float32)
        scale = 1.0 / keep_prob
        return Tensor(binary_mask * scale)
        ### END SOLUTION

    def forward(self, x, training=True):
        """
        Forward pass through dropout layer.

        Composes the two helpers: first decide whether dropout applies,
        then generate and apply the mask if it does.

        TODO: Implement dropout forward pass

        APPROACH:
        1. Use _should_apply_dropout to check if dropout is needed
        2. Handle the special case p=1 (drop everything)
        3. Use _generate_dropout_mask to create the scaled mask
        4. Element-wise multiply input by the mask

        EXAMPLE:
        >>> dropout = Dropout(0.5)
        >>> x = Tensor([1, 2, 3, 4])
        >>> y_train = dropout.forward(x, training=True)   # Some elements zeroed
        >>> y_eval = dropout.forward(x, training=False)   # All elements preserved

        HINTS:
        - _should_apply_dropout returns False for inference or p=0
        - When p=1.0 every element is dropped (return zeros)
        - Multiply x by the mask tensor for the final output
        """
        ### BEGIN SOLUTION
        if not self._should_apply_dropout(training):
            return x

        if self.p == DROPOUT_MAX_PROB:
            return x * 0.0  # Keep the operation path; Module 06 will propagate zero gradients.

        mask = self._generate_dropout_mask(x.data.shape)
        return x * mask
        ### END SOLUTION

    def __call__(self, x, training=True):
        """Allows the layer to be called like a function."""
        return self.forward(x, training)

    def parameters(self):
        """Dropout has no parameters."""
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"

# %% [markdown]
"""
### 🧪 Unit Test: Dropout Decision Logic

Before testing the full dropout forward pass, we verify the decision logic in
isolation. The `_should_apply_dropout` helper encapsulates a concept that often
trips students up: dropout is *only* active during training *and* only when
p > 0. Testing this separately makes it easy to pinpoint bugs in the
training-vs-inference distinction without interference from randomness.

**What we're testing**: Training/inference mode detection and p=0 bypass
**Why it matters**: A single wrong boolean can silently disable regularization or corrupt inference
**Expected**: True only when training=True AND p > 0
"""

# %% nbgrader={"grade": true, "grade_id": "test-should-apply-dropout", "locked": true, "points": 3}
def test_unit_should_apply_dropout():
    """🧪 Test _should_apply_dropout decision logic."""
    print("🧪 Unit Test: Dropout Decision Logic...")

    # Standard dropout (p=0.5) in training mode should apply
    d = Dropout(0.5)
    assert d._should_apply_dropout(training=True) is True, \
        "Dropout(0.5) should apply during training"

    # Same dropout in inference mode should NOT apply
    assert d._should_apply_dropout(training=False) is False, \
        "Dropout should not apply during inference"

    # Zero dropout (p=0) should never apply, even in training
    d_zero = Dropout(0.0)
    assert d_zero._should_apply_dropout(training=True) is False, \
        "Dropout(0.0) should never apply (no neurons to drop)"

    # Full dropout (p=1.0) in training mode should apply
    d_full = Dropout(1.0)
    assert d_full._should_apply_dropout(training=True) is True, \
        "Dropout(1.0) should apply during training"

    # Full dropout in inference mode should NOT apply
    assert d_full._should_apply_dropout(training=False) is False, \
        "Even Dropout(1.0) should not apply during inference"

    print("✅ Dropout decision logic works correctly!")

if __name__ == "__main__":
    test_unit_should_apply_dropout()

# %% [markdown]
r"""
### 🧪 Unit Test: Dropout Mask Generation

The mask is the heart of dropout. Each element is drawn independently:
kept with probability $1-p$, dropped otherwise. Kept elements are scaled
by $\frac{1}{1-p}$ so the expected output equals the input. This is "inverted
dropout." We test both the statistical properties (fraction of zeros)
and the scaling (surviving values equal $\frac{1}{1-p}$).

$$p = 0.5 \implies \text{keep\_prob} = 0.5, \quad \text{scale} = \frac{1}{0.5} = 2.0$$

$$\mathbf{u} = \begin{bmatrix} 0.3 & 0.8 & 0.1 & 0.6 \end{bmatrix} \xrightarrow{\mathbf{u} < 0.5} \mathbf{m} = \begin{bmatrix} 1 & 0 & 1 & 0 \end{bmatrix} \xrightarrow{\times 2.0} \hat{\mathbf{m}} = \begin{bmatrix} 2.0 & 0.0 & 2.0 & 0.0 \end{bmatrix}$$

**What we're testing**: Mask shape, scaling factor, and survival statistics
**Why it matters**: Wrong scaling silently shifts all predictions at inference time
**Expected**: Correct shape, values in $\{0, \frac{1}{1-p}\}$, $\approx 50\%$ survival for $p=0.5$
"""

# %% nbgrader={"grade": true, "grade_id": "test-generate-dropout-mask", "locked": true, "points": 3}
def test_unit_generate_dropout_mask():
    """🧪 Test _generate_dropout_mask output properties."""
    # Isolate this graded test from earlier cells and preserve the learner's RNG.
    global rng
    previous_rng = rng
    rng = np.random.default_rng(7)
    try:
        print("🧪 Unit Test: Dropout Mask Generation...")

        d = Dropout(0.5)
        mask = d._generate_dropout_mask((1000,))

        # Shape must match the requested shape
        assert mask.shape == (1000,), f"Expected shape (1000,), got {mask.shape}"

        # Every element must be either 0.0 or 2.0 (= 1/(1-0.5))
        unique_vals = set(np.unique(mask.data))
        assert unique_vals <= {0.0, 2.0}, \
            f"Mask values should be {{0.0, 2.0}}, got {unique_vals}"

        # Statistically, about 50% should survive (3-sigma tolerance)
        non_zero = np.count_nonzero(mask.data)
        std_err = np.sqrt(1000 * 0.5 * 0.5)
        assert 500 - 3 * std_err < non_zero < 500 + 3 * std_err, \
            f"Expected ~500 survivors, got {non_zero}"

        # Test with different dropout probability
        d2 = Dropout(0.3)
        mask2 = d2._generate_dropout_mask((2000,))

        # Values should be 0.0 or 1/(1-0.3) ≈ 1.4286
        expected_scale = 1.0 / 0.7
        non_zero_vals = mask2.data[mask2.data != 0.0]
        assert np.allclose(non_zero_vals, expected_scale), \
            f"Surviving values should be {expected_scale:.4f}, got {np.unique(non_zero_vals)}"

        # About 70% should survive for p=0.3
        survival_rate = np.count_nonzero(mask2.data) / 2000
        assert 0.60 < survival_rate < 0.80, \
            f"Expected ~70% survival for p=0.3, got {survival_rate:.1%}"

        print("✅ Dropout mask generation works correctly!")
    finally:
        rng = previous_rng

if __name__ == "__main__":
    test_unit_generate_dropout_mask()

# %% [markdown]
"""
## 🏗️ Sequential: Layer Container for Composition

`Sequential` chains layers together, calling forward() on each in order.

**Progressive Disclosure**: After learning to compose layers explicitly
(h = relu(linear1(x)); out = linear2(h)), you can use Sequential for convenience:

```python
model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
out = model(x)  # Chains all layers automatically
```

This is TinyTorch's equivalent of PyTorch's nn.Sequential - simpler but same idea.
"""

# %% nbgrader={"grade": false, "grade_id": "sequential", "solution": false}
#| export
class Sequential:
    """
    Container that chains layers together sequentially.

    After you understand explicit layer composition, Sequential provides
    a convenient way to bundle layers together.

    Example:
        >>> model = Sequential(
        ...     Linear(784, 128),
        ...     ReLU(),
        ...     Linear(128, 10)
        ... )
        >>> output = model(input_tensor)
        >>> params = model.parameters()  # All parameters from all layers
    """

    def __init__(self, *layers):
        """Initialize with layers to chain together."""
        # Accept both Sequential(layer1, layer2) and Sequential([layer1, layer2])
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self, x, training=True):
        """Forward pass through all layers sequentially.

        Passes training=True/False to layers that support it (e.g. Dropout),
        and falls back to a plain forward(x) call for layers that don't.
        This lets you switch between training and eval mode with one flag:

            output = model.forward(x, training=False)   # eval: Dropout disabled
            output = model.forward(x, training=True)    # train: Dropout active
        """
        for layer in self.layers:
            # Only layers whose forward takes a `training` flag (Dropout) receive it
            if 'training' in inspect.signature(layer.forward).parameters:
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    def __call__(self, x, training=True):
        """Allow model to be called like a function."""
        return self.forward(x, training=training)

    def parameters(self):
        """Collect each parameter once, even when layers share a weight."""
        params = []
        seen = set()
        for layer in self.layers:
            for param in layer.parameters():
                if id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        return params

    def __repr__(self):
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"


# %% [markdown]
"""
### 🧪 Unit Test: Dropout Layer

This test validates our Dropout layer implementation works correctly.

**What we're testing**: Training vs inference behavior, probability scaling, randomness
**Why it matters**: Essential for preventing overfitting in neural networks
**Expected**: Correct masking during training, passthrough during inference
"""

# %% nbgrader={"grade": true, "grade_id": "test-dropout", "locked": true, "points": 10}
def test_unit_dropout_layer():
    """🧪 Test Dropout layer implementation."""
    # Isolate this graded test from earlier cells and preserve the learner's RNG.
    global rng
    previous_rng = rng
    rng = np.random.default_rng(7)
    try:
        print("🧪 Unit Test: Dropout Layer...")

        # Test dropout creation
        dropout = Dropout(0.5)
        assert dropout.p == 0.5

        # Test inference mode (should pass through unchanged)
        x = Tensor([1, 2, 3, 4])
        y_inference = dropout.forward(x, training=False)
        assert np.array_equal(x.data, y_inference.data), "Inference should pass through unchanged"

        # Test training mode with zero dropout (should pass through unchanged)
        dropout_zero = Dropout(0.0)
        y_zero = dropout_zero.forward(x, training=True)
        assert np.array_equal(x.data, y_zero.data), "Zero dropout should pass through unchanged"

        # Test training mode with full dropout (should zero everything)
        dropout_full = Dropout(1.0)
        y_full = dropout_full.forward(x, training=True)
        assert np.allclose(y_full.data, 0), "Full dropout should zero everything"

        # Test training mode with partial dropout
        # Note: This is probabilistic, so we test statistical properties
        x_large = Tensor(np.ones((1000,)))  # Large tensor for statistical significance
        y_train = dropout.forward(x_large, training=True)

        # Count non-zero elements (approximately 50% should survive)
        non_zero_count = np.count_nonzero(y_train.data)
        expected = 500
        # Use 3-sigma bounds: std = sqrt(n*p*(1-p)) = sqrt(1000*0.5*0.5) ≈ 15.8
        std_error = np.sqrt(1000 * 0.5 * 0.5)
        lower_bound = expected - 3 * std_error  # ≈ 453
        upper_bound = expected + 3 * std_error  # ≈ 547
        assert lower_bound < non_zero_count < upper_bound, \
            f"Expected {expected}±{3*std_error:.0f} survivors, got {non_zero_count}"

        # Test scaling (surviving elements should be scaled by 1/(1-p) = 2.0)
        surviving_values = y_train.data[y_train.data != 0]
        expected_value = 2.0  # 1.0 / (1 - 0.5)
        assert np.allclose(surviving_values, expected_value), f"Surviving values should be {expected_value}"

        # Test no parameters
        params = dropout.parameters()
        assert len(params) == 0, "Dropout should have no parameters"

        # Test invalid probability
        try:
            Dropout(-0.1)
            assert False, "Should raise ValueError for negative probability"
        except ValueError:
            pass

        try:
            Dropout(1.1)
            assert False, "Should raise ValueError for probability > 1"
        except ValueError:
            pass

        print("✅ Dropout layer works correctly!")
    finally:
        rng = previous_rng

if __name__ == "__main__":
    test_unit_dropout_layer()

# %% [markdown]
r"""
## 🔧 Integration: Bringing It Together

Now that we've built both layer types, let's see how they work together to create a complete neural network architecture. We'll compose a realistic 3-layer MLP for MNIST digit classification.

### End-to-End Computational Pipeline

$$\mathbf{X} \in \mathbb{R}^{32 \times 784} \xrightarrow{\text{Linear}(784, 256)} \mathbf{Z}_1 \xrightarrow{\text{ReLU}} \mathbf{A}_1 \xrightarrow{\text{Dropout}(0.5)} \hat{\mathbf{A}}_1 \xrightarrow{\text{Linear}(256, 128)} \mathbf{Z}_2 \xrightarrow{\text{ReLU}} \mathbf{A}_2 \xrightarrow{\text{Dropout}(0.3)} \hat{\mathbf{A}}_2 \xrightarrow{\text{Linear}(128, 10)} \mathbf{Y} \in \mathbb{R}^{32 \times 10}$$

### Parameter Count & Memory Breakdown

| Stage | Component / Layer | Kernel Shape | Bias Shape | Trainable Parameters | Memory (FP32) |
|:---|:---|:---|:---|:---|:---|
| **Layer 1** | `Linear(784, 256)` | $(784, 256)$ | $(256,)$ | $200{,}704 + 256 = 200{,}960$ | $803.84\text{ KB}$ |
| — | `ReLU()` | — | — | $0$ | $0\text{ B}$ |
| — | `Dropout(p=0.5)` | — | — | $0$ | $0\text{ B}$ |
| **Layer 2** | `Linear(256, 128)` | $(256, 128)$ | $(128,)$ | $32{,}768 + 128 = 32{,}896$ | $131.58\text{ KB}$ |
| — | `ReLU()` | — | — | $0$ | $0\text{ B}$ |
| — | `Dropout(p=0.3)` | — | — | $0$ | $0\text{ B}$ |
| **Layer 3** | `Linear(128, 10)` | $(128, 10)$ | $(10,)$ | $1{,}280 + 10 = 1{,}290$ | $5.16\text{ KB}$ |
| **Total** | **3-Layer MLP** | — | — | **$235{,}146$ parameters** | **$940.58\text{ KB}$** |
"""


# %% [markdown]
r"""
## 📊 Systems Analysis: Memory and Performance

Understanding memory allocation lifecycles and computational FLOPs budgets allows engineers to optimize training throughput and prevent GPU out-of-memory (OOM) faults.

### Memory Hierarchy Breakdown

| Memory Category | Allocation Lifecycle | Sizing Formula | Systems Trade-off |
|:---|:---|:---|:---|
| **Parameter Memory** | Static; persistent across epochs | $\sum (D_{\text{in}} \times D_{\text{out}} + D_{\text{out}}) \times 4\text{ B}$ | Scales with network width & depth; resident in GPU VRAM |
| **Activation Memory** | Dynamic per forward pass; retained for backward | $B \times D_{\text{layer}} \times 4\text{ B}$ per layer | Dominates training footprint; linear in batch size $B$ |
| **Temporary Buffer Memory** | Ephemeral during kernel execution | $B \times D_{\text{layer}} \times 4\text{ B}$ (e.g. dropout mask) | Peak allocated at layer forward/backward boundary |

### Computational Complexity & Hardware Profile

| Operation | Computational Complexity (FLOPs) | Memory I/O Complexity | Dominant Hardware Bottleneck |
|:---|:---|:---|:---|
| **Linear Layer** | $2 \cdot B \cdot D_{\text{in}} \cdot D_{\text{out}} + B \cdot D_{\text{out}}$ | Read $\mathbf{X}, \mathbf{W}, \mathbf{b}$; Write $\mathbf{Y}$ | Compute-bound for large $B$, memory-bandwidth-bound for $B=1$ |
| **Multi-layer MLP** | $\sum_{\ell=1}^L 2 \cdot B \cdot D_{\ell-1} \cdot D_\ell$ | Intermediate tensor reads & writes | Cache line eviction across deep layer cascades |
| **Dropout Forward** | $O(B \cdot D)$ (PRNG + threshold + multiply) | Read $\mathbf{X}$; Write $\hat{\mathbf{M}}, \mathbf{Y}$ | Memory bandwidth bound (streaming kernel) |
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-layer-memory", "solution": false}
def analyze_layer_memory():
    """📊 Analyze memory usage patterns in layer operations."""
    print("📊 Analyzing Layer Memory Usage...")

    # Test different layer sizes
    layer_configs = [
        (784, 256),   # MNIST → hidden
        (256, 256),   # Hidden → hidden
        (256, 10),    # Hidden → output
        (2048, 2048), # Large hidden
    ]

    print("\nLinear Layer Memory Analysis (decimal KB, 1 KB = 1000 B, as in the tables above):")
    print("Configuration → Weight Memory → Bias Memory → Total Memory")

    for in_feat, out_feat in layer_configs:
        # Calculate memory usage
        weight_memory = in_feat * out_feat * 4  # 4 bytes per float32
        bias_memory = out_feat * 4
        total_memory = weight_memory + bias_memory

        print(f"({in_feat:4d}, {out_feat:4d}) → {weight_memory/1000:9.2f} KB → {bias_memory/1000:7.2f} KB → {total_memory/1000:9.2f} KB")

    # Analyze multi-layer memory scaling
    print("\n💡 Multi-layer Model Memory Scaling (decimal MB, 1 MB = 1,000,000 B):")
    hidden_sizes = [128, 256, 512, 1024, 2048]

    for hidden_size in hidden_sizes:
        # 3-layer MLP: 784 → hidden → hidden/2 → 10
        layer1_params = 784 * hidden_size + hidden_size
        layer2_params = hidden_size * (hidden_size // 2) + (hidden_size // 2)
        layer3_params = (hidden_size // 2) * 10 + 10

        total_params = layer1_params + layer2_params + layer3_params
        memory_mb = total_params * 4 / 1_000_000

        print(f"Hidden={hidden_size:4d}: {total_params:7,} params = {memory_mb:5.1f} MB")

if __name__ == "__main__":
    analyze_layer_memory()

# %% nbgrader={"grade": false, "grade_id": "analyze-layer-performance", "solution": false}
def analyze_layer_performance():
    """📊 Analyze computational complexity of layer operations."""
    import time

    print("📊 Analyzing Layer Computational Complexity...")

    # Test forward pass FLOPs
    batch_sizes = [1, 32, 128, 512]
    layer = Linear(784, 256)

    print("\nLinear Layer MACs Analysis:")
    print("Batch Size → Matrix Multiply MACs → Bias Adds → Estimated FLOPs")
    print("Convention: 2 FLOPs per matrix MAC, 1 FLOP per bias addition")

    for batch_size in batch_sizes:
        # Matrix multiplication: (batch, in) @ (in, out) = batch * in * out MACs
        matmul_macs = batch_size * 784 * 256
        # Bias addition has no multiplication, so count one FLOP per output.
        bias_adds = batch_size * 256
        total_flops = 2 * matmul_macs + bias_adds

        print(f"{batch_size:10d} → {matmul_macs:15,} → {bias_adds:13,} → {total_flops:11,}")

    # Add timing measurements
    print("\nLinear Layer Timing Analysis:")
    print("Batch Size → Time (ms) → Throughput (samples/sec)")

    throughputs = []
    for batch_size in batch_sizes:
        x = Tensor(rng.standard_normal((batch_size, 784)))

        # Warm up
        for _ in range(10):
            _ = layer.forward(x)

        # Time multiple iterations
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = layer.forward(x)
        elapsed = time.perf_counter() - start

        time_per_forward = (elapsed / iterations) * 1000  # Convert to ms
        throughput = (batch_size * iterations) / elapsed
        throughputs.append(throughput)

        print(f"{batch_size:10d} → {time_per_forward:8.3f} ms → {throughput:12,.0f} samples/sec")

    batch_span = batch_sizes[-1] // batch_sizes[0]
    print("\n💡 Key Insights:")
    print("🚀 Linear layer complexity: O(batch_size × in_features × out_features)")
    print("🚀 Memory grows linearly with batch size, quadratically with layer width")
    print("🚀 Dropout adds minimal computational overhead (element-wise operations)")
    print(f"🚀 Throughput is batch-invariant here: a {batch_span}x change in batch size moved it")
    print(f"   only between {min(throughputs):,.0f} and {max(throughputs):,.0f} samples/sec, "
          f"a {max(throughputs)/min(throughputs):.2f}x spread that is")
    print("   mostly timing noise. Module 01's 2D matmul is an explicit Python loop over")
    print("   output elements, so total cost is exactly linear in batch size and there is no")
    print("   per-sample interpreter overhead left to amortize. Batching only pays off once")
    print("   that inner loop is vectorized, which is exactly what vectorization buys you")

if __name__ == "__main__":
    analyze_layer_performance()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""


# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_linear_layer()
    test_unit_edge_cases_linear()
    test_unit_parameter_collection_linear()
    test_unit_should_apply_dropout()
    test_unit_generate_dropout_mask()
    test_unit_dropout_layer()

    print("\nRunning integration scenarios...")

    # Test realistic neural network construction with manual composition
    print("🧪 Integration Test: Multi-layer Network...")

    # Build individual layers for manual composition
    layer1 = Linear(784, 128)
    activation1 = ReLU()
    dropout1 = Dropout(0.5)
    layer2 = Linear(128, 64)
    activation2 = ReLU()
    dropout2 = Dropout(0.3)
    layer3 = Linear(64, 10)

    # Test end-to-end forward pass with manual composition
    batch_size = 16
    x = Tensor(rng.standard_normal((batch_size, 784)))

    # Manual forward pass
    x = layer1.forward(x)
    x = activation1.forward(x)
    x = dropout1.forward(x)
    x = layer2.forward(x)
    x = activation2.forward(x)
    x = dropout2.forward(x)
    output = layer3.forward(x)

    assert output.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {output.shape}"

    # Test parameter counting from individual layers
    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    expected_params = 6  # 3 weights + 3 biases from 3 Linear layers
    assert len(all_params) == expected_params, f"Expected {expected_params} parameters, got {len(all_params)}"

    # Test individual layer functionality
    test_x = Tensor(rng.standard_normal((4, 784)))
    # Test dropout in training vs inference
    dropout_test = Dropout(0.5)
    train_output = dropout_test.forward(test_x, training=True)
    infer_output = dropout_test.forward(test_x, training=False)
    assert np.array_equal(test_x.data, infer_output.data), "Inference mode should pass through unchanged"

    # Reusing the same layer must collect its weights only once.
    shared = Linear(2, 2)
    shared_model = Sequential(shared, ReLU(), Sequential(shared))
    assert shared_model.parameters() == [shared.weight, shared.bias]

    print("✅ Multi-layer network integration works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 03")

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of layer operations and their systems implications:

### Question 1: Parameter Scaling and Memory
**Question**: Consider three different network architectures for MNIST (28x28 = 784 input features, 10 output classes):
- Architecture A: 784 -> 128 -> 10
- Architecture B: 784 -> 256 -> 10
- Architecture C: 784 -> 512 -> 10

**Consider**:
- Without calculating exactly, which architecture has approximately 2x the parameters of Architecture A?
- What does this tell you about how hidden layer size affects model capacity?
- If a Linear(784, 256) layer uses ~800KB of memory, how does this scale?

**Real-world context**: Parameter memory is the floor, not the total. For the 784->256->128->10 network in this module the three layer outputs kept for backward come to 50.4 KB at batch 32, against 940.6 KB of parameters, roughly 0.05x, and even counting the 100.4 KB input batch the activations stay under a fifth of the weights. It would take a batch near 6,000 before the layer outputs alone matched the parameters. The ratio inverts in the architectures you will meet later. A transformer retains per-layer attention scores that grow with the square of the sequence length, and there activation memory really does dominate.

---

### Question 2: Dropout Training vs Inference
**Question**: You have a Dropout layer with p=0.5 in your network. During training, we scale surviving values by 1/(1-p) = 2.0.

**Consider**:
- Why do we scale by 2.0 during training?
- During inference, dropout returns the input unchanged. Why don't we scale by 0.5?
- What mathematical property are we preserving with this scaling?

**Think about**:
- If training drops 50% of neurons and inference keeps all, outputs would differ by 2x
- The scaling ensures expected values match between training and inference
- This is called "inverted dropout" - scaling at train time instead of inference

---

### Question 3: Weight Initialization Trade-offs
**Question**: We initialize weights with scale = sqrt(1/in_features) (LeCun-style). For Linear(1000, 10), how does this compare to Linear(10, 1000)?

**Calculate**:
- Linear(1000, 10): scale = sqrt(1/1000) = ___________
- Linear(10, 1000): scale = sqrt(1/10) = ___________

**Trade-offs to consider**:
- Why do we want smaller initial weights for layers with more inputs?
- What would happen if we initialized all weights to 0? To 1?
- How does initialization affect signal propagation in deep networks?

---

### Question 4: Layer Ordering Effects
**Question**: In a typical layer block, we compose: Linear -> Activation -> Dropout. What happens if you change the order to: Linear -> Dropout -> Activation?

**Consider**:
- With ReLU the two orderings give *identical* outputs for the same mask. Why? (ReLU is positively homogeneous, so relu(c*x) = c*relu(x) for any c >= 0, and an inverted-dropout mask holds only 0 and 1/(1-p), both non-negative)
- With Sigmoid they differ, and not slightly. Where does the argument above break? (sigmoid(0) = 0.5, so an input that dropout zeroed still leaves 0.5 on the other side)
- Given that, which ordering would you pick for a Sigmoid block, and what does your answer say about where dropout belongs in general?

**Real-world implications**:
- Whether the two orderings commute is a property of the specific activation, not a general law about layer order
- An activation that passes through the origin with a non-negative slope commutes with the mask; anything with a non-zero output at zero does not, and then the ordering changes both the forward values and the training dynamics

---

### Question 5: Production Deployment Memory
**Question**: You're deploying a 3-layer network (784->256->128->10) to a mobile device with 10MB free memory.

**Calculate**:
- Parameters memory: 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = ___________
- With batch_size=1, activation memory per layer = ___________
- Total memory needed = ___________

**Real-world implications**:
- Can you increase batch size to 32? To 64?
- What's the trade-off between batch size and latency on mobile?
- Why do mobile inference engines optimize for batch_size=1?

---

### Bonus Question: Manual Composition Analysis

**Question**: This module builds networks two ways: layer by layer in the integration test, and chained inside a `Sequential`. Both express the same computation. What does the hand-written version show you at each step that the one-line `Sequential` call hides?

**Consider**:
1. Data shape transformations at each step
2. Which operations create new tensors vs modify in-place
3. How parameters flow through the network
4. What `Sequential.forward` has to decide for every layer it calls (look at the signature check it runs)

**Key insight**: Sequential is the convenience you reach for once the pipeline is understood. Composing by hand first is what lets you debug shape mismatches, memory issues, and gradient flow problems when the container is in the way.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Layers Transform Shapes

**What you built:** Linear layers that transform data from one dimension to another.

**Why it matters:** A Linear layer is the workhorse of neural networks. The transformation
from 784 features (a flattened 28×28 image) to 10 classes (digits 0-9) is exactly what
happens in digit recognition. You just built the core component!

In the next module, you'll add loss functions that measure how wrong predictions are.
Combined with your layers, this creates the foundation for learning.
"""

# %%
def demo_layers():
    """🎯 See how layers transform shapes."""
    print("🎯 AHA MOMENT: Layers Transform Shapes")
    print("=" * 45)

    # Create a layer that transforms 784 → 10 (like MNIST)
    layer = Linear(784, 10)

    # Simulate a batch of 32 flattened images
    batch = Tensor(rng.standard_normal((32, 784)))

    # Forward pass
    output = layer(batch)

    print(f"Input shape:  {batch.shape}  ← 32 images, 784 pixels each")
    print(f"Output shape: {output.shape}  ← 32 images, 10 classes each")
    print(f"Parameters:   {784 * 10 + 10:,} (weights + biases)")

    print("\n✨ Your layer transforms images to class predictions!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_layers()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Layers

Congratulations! You've built the fundamental building blocks that make neural networks possible!

### Key Accomplishments
- Built Linear layers with proper weight initialization and parameter management
- Created Dropout layers for regularization with training/inference mode handling
- Demonstrated manual layer composition, then bundled the same chain into a Sequential
  container that forwards the training flag and collects each parameter exactly once
- Analyzed memory scaling and computational complexity of layer operations
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Discovered
- **Parameter memory is the floor**: a Linear layer stores in_features x out_features
  weights once, while the activations kept for backward scale with batch size. For this
  module's 784->256->128->10 network that is 50.4 KB of layer outputs against 940.6 KB of
  parameters at batch 32, so the weights still dominate. Flipping the ratio takes either a
  batch near 6,000 or a much wider architecture
- **Initialization is not cosmetic**: LeCun scaling (sqrt(1/fan_in)) holds variance at 1.0
  through a linear map, which is exactly what it was derived for. ReLU then zeros half the
  signal, so a Linear+ReLU pair loses roughly 2x. Stacking this module's own Linear and
  ReLU eight times at 256->256 on unit-variance input, the activation variance falls
  1.00 -> 0.34 -> 0.17 -> 0.089 -> 0.048 -> 0.028 -> 0.014 -> 0.0076 -> 0.0035, a ~285x
  collapse. Recovering that missing factor of 2 is the whole reason He/Kaiming
  (sqrt(2/fan_in)) exists
- **Dropout costs memory, not just compute**: the mask is a full float32 tensor
  the same shape as the activations it gates
- **Composition is the whole idea**: layers are interchangeable because they almost all
  agree on one contract, forward(x) -> Tensor. Dropout is the seam, since it needs
  forward(x, training=True), and that single extra flag is why Sequential.forward has to
  inspect each layer's signature before calling it

### Ready for Next Steps
Your layer implementation enables building complete neural networks! The Linear layer provides learnable transformations, manual composition chains them together, and Dropout prevents overfitting.

Export with: `tito module complete 03`

**Next**: Module 04 will add loss functions (CrossEntropyLoss, MSELoss) that measure how wrong your model is - the foundation for learning!
"""
