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
# Module 02: Activations - Intelligence Through Nonlinearity

Welcome to Module 02! Today you'll add the one ingredient a stack of linear layers cannot supply on its own: **nonlinearity**.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor with data manipulation and basic operations
**You'll Build**: Activation functions that add nonlinearity to transformations
**You'll Enable**: Neural networks with the ability to learn complex patterns

**Connection Pipeline**:
$$\mathbf{X} \in \text{Tensor (data)} \xrightarrow{\text{Nonlinearity}} \sigma(\mathbf{X}) \in \text{Activations} \xrightarrow{\text{Parameterization}} \mathbf{W}\mathbf{X} + \mathbf{b} \in \text{Layers}$$

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement 5 core activation functions (Sigmoid, ReLU, Tanh, GELU, Softmax)
2. Understand how nonlinearity enables neural network intelligence
3. Test activation behaviors and output ranges
4. Connect activations to real neural network components

Let's add intelligence to your tensors!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/02_activations/activations.ipynb`
**Building Side:** Code exports to tinytorch.core.activations

```python
# Final package structure:
from tinytorch.core.activations import Sigmoid, ReLU, Tanh, GELU, Softmax  # This module
from tinytorch.core.tensor import Tensor  # Foundation (Module 01)
```

**Why this matters:**
- **Learning:** Complete activation system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn.functional with all activation operations together
- **Consistency:** All activation functions and behaviors in core.activations
- **Integration:** Works seamlessly with Tensor for complete nonlinear transformations
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

**Prerequisites**: Module 01 (Tensor) must be completed

**External Dependencies**:
- `numpy` (for numerical operations)

**TinyTorch Dependencies**:
- **Module 01 (Tensor)**: Foundation for all activation computations and data flow
  - Used for: Input/output data structures, shape operations, element-wise operations
  - Required: Yes - activations operate on Tensor objects

**Dependency Flow**:
$$\underbrace{\text{Module 01: Tensor}}_{\text{Foundation and Autograd Core}} \longrightarrow \underbrace{\text{Module 02: Activations}}_{\text{Nonlinear Functions}} \longrightarrow \underbrace{\text{Module 03: Layers}}_{\text{Network Architecture}}$$
"""

# %% nbgrader={"grade": false, "grade_id": "setup", "solution": false}
#| default_exp core.activations
#| export

import numpy as np

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor, Function

# Constants for numerical comparisons
TOLERANCE = 1e-10  # Small tolerance for floating-point comparisons in tests

# %% [markdown]
r"""
## 💡 Introduction: What Makes Neural Networks Intelligent?

Consider two architectural scenarios:

**1. Without Activations (Linear Transformations Only):**
Any cascade of purely linear layers collapses mathematically into a single matrix multiplication:
$$\mathbf{h}_1 = \mathbf{W}_1 \mathbf{x}, \quad \mathbf{h}_2 = \mathbf{W}_2 \mathbf{h}_1, \quad \hat{\mathbf{y}} = \mathbf{W}_3 \mathbf{h}_2$$
$$\hat{\mathbf{y}} = \mathbf{W}_3 (\mathbf{W}_2 (\mathbf{W}_1 \mathbf{x})) = (\mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1) \mathbf{x} = \mathbf{W}_{\text{eff}} \mathbf{x}$$
Regardless of whether you stack 3 layers or 300 layers, the resulting function can only construct linear hyperplanes.

**2. With Activations (Nonlinear Transformations):**
Interleaving nonlinear activation functions $\sigma(\cdot)$ prevents linear collapse:
$$\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x}), \quad \mathbf{h}_2 = \sigma(\mathbf{W}_2 \mathbf{h}_1), \quad \hat{\mathbf{y}} = \mathbf{W}_3 \mathbf{h}_2$$
Each activation function selectively warps and bends the coordinate space, allowing deep networks to approximate arbitrary continuous functions (Universal Approximation Theorem).

<div align="center">
  <img src="activations_overview.svg" alt="TinyTorch Activation Functions Overview" width="680px">
</div>

## 📐 Foundations: Five Activation Functions

Each activation function provides a distinct mathematical behavior tuned for specific systems and architectural roles:

| Activation | Mathematical Formulation | Output Range | Systems & Architectural Role |
|:---|:---|:---|:---|
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ | Binary classification outputs, gating mechanisms (LSTMs, GRUs) |
| **ReLU** | $f(x) = \max(0, x)$ | $[0, \infty)$ | Standard hidden layer default; extreme hardware speed and activation sparsity |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Zero-centered representations preventing gradient bias drift |
| **GELU** | $x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$ | $[\approx -0.17, \infty)$ | Modern Transformer standard (GPT, BERT); smooth gradient flow |
| **Softmax** | $\frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$ | $[0, 1], \sum = 1$ | Multi-class probability distributions, attention score normalization |

Let's implement each one with clear mathematical formulations and unit testing!
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Activation Functions

### Implementation Pattern

Each activation is two classes. The `Function` does the math on NumPy arrays;
the wrapper is what a network holds and calls:
```python
class ActivationNameFunction(Function):
    def forward(self, x):          # x is a NumPy array
        # Apply mathematical transformation
        # Return the result array (Module 06 adds backward)

class ActivationName:
    def parameters(self):
        return []                  # nothing to train

    def forward(self, x: Tensor) -> Tensor:
        return ActivationNameFunction.apply(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)     # so relu(x) works like relu.forward(x)
```

Every module-like object in TinyTorch answers `parameters()`. Module 03 will use
it to collect the weights a layer trains, and Module 07 will hand that list to
an optimizer. An activation owns no weights, so it returns an empty list; the
method exists so a stack of layers and activations can be walked with one
uniform call. You write only the `forward()` of each `Function`; the wrappers
are given.
"""

# %% [markdown]
r"""
### Sigmoid: The Probability Gatekeeper

Sigmoid maps any real number to the range $(0, 1)$, making it the standard function for probability estimation and binary decisions.

#### Mathematical Formulation & Properties

$$\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}$$

| Property | Specification |
|:---|:---|
| **Input Domain** | $x \in (-\infty, \infty)$ |
| **Output Range** | $\sigma(x) \in (0, 1)$ |
| **Derivative** | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ with peak $\sigma'(0) = 0.25$ |
| **Symmetry** | Centered at $\sigma(0) = 0.5$ |

#### Numerical Vector Trace

$$\begin{bmatrix} -3.0 & -1.0 & 0.0 & 1.0 & 3.0 \end{bmatrix} \xrightarrow{\sigma(x)} \begin{bmatrix} 0.0474 & 0.2689 & 0.5000 & 0.7311 & 0.9526 \end{bmatrix}$$

<div align="center">
  <img src="sigmoid_curve.svg" alt="Sigmoid Squashing and Saturation" width="320px">
</div>

**Why Sigmoid matters**: In binary classification, we need outputs between $0$ and $1$ to represent probabilities. Notice the red saturation zones where $|x| \ge 3$: the gradient vanishes toward zero ($\sigma'(x) \to 0$), which will motivate ReLU and GELU in deep hidden layers.
"""

# %% nbgrader={"grade": false, "grade_id": "sigmoid-impl", "solution": true}
#| export
class SigmoidFunction(Function):
    """
    The Sigmoid operation. forward() works on NumPy arrays; Module 06 adds backward().
    """
    def forward(self, x):
        """
        Apply sigmoid activation element-wise.

        TODO: Implement sigmoid function

        APPROACH:
        1. Compute z = exp(-abs(x)); its exponent is always <= 0, so it cannot overflow
        2. For x >= 0 use 1 / (1 + z)
        3. For x < 0 use z / (1 + z), the equivalent formula on the negative side
        4. Pick the branch per element with np.where(x >= 0, branch_a, branch_b)

        EXAMPLE:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor([-2, 0, 2])
        >>> result = sigmoid(x)
        >>> print(result.data)
        [0.119, 0.5, 0.881]  # All values between 0 and 1

        HINT: The one-line formula 1 / (1 + np.exp(-x)) overflows at x = -1000
        (exp(1000) is inf in float32). The unit test runs your code with NumPy
        set to raise on overflow, so the naive form fails it
        """
        ### BEGIN SOLUTION role="core"
        # Both branches use the same bounded exponential, so neither overflows.
        z = np.exp(-np.abs(x))
        result = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
        return result
        ### END SOLUTION


class Sigmoid:
    """
    Sigmoid activation: σ(x) = 1/(1 + e^(-x))

    Maps any real number to (0, 1) range.
    Perfect for probabilities and binary classification.
    """

    def parameters(self):
        """No weights to train, so Module 03's layers and Module 07's optimizers will have nothing to collect."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigmoid through its operation, so Module 06 will be able to record it for gradients."""
        return SigmoidFunction.apply(x)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Sigmoid

This test validates sigmoid activation behavior.

**What we're testing**: Sigmoid maps inputs to (0, 1) range
**Why it matters**: Ensures proper probability-like outputs
**Expected**: All outputs between 0 and 1, sigmoid(0) = 0.5
"""

# %% nbgrader={"grade": true, "grade_id": "test-sigmoid", "locked": true, "points": 10}
def test_unit_sigmoid():
    """🧪 Test Sigmoid implementation."""
    print("🧪 Unit Test: Sigmoid...")

    sigmoid = Sigmoid()

    # Test basic cases
    x = Tensor([0.0])
    result = sigmoid.forward(x)
    assert np.allclose(result.data, [0.5]), f"sigmoid(0) should be 0.5, got {result.data}"

    # Test range property - all outputs should be in (0, 1)
    x = Tensor([-10, -1, 0, 1, 10])
    result = sigmoid.forward(x)
    assert np.all(result.data > 0) and np.all(result.data < 1), "All sigmoid outputs should be in (0, 1)"

    # Test extreme values. NumPy is told to raise on overflow and invalid results
    # here, so the naive 1 / (1 + exp(-x)) (which computes exp(1000) = inf) fails
    # loudly instead of limping through with a warning and a 0.0.
    x = Tensor([-1000, 1000])  # Extreme values
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = sigmoid.forward(x)
    except FloatingPointError as err:
        raise AssertionError(
            f"sigmoid overflowed on extreme inputs ({err}). Keep every exponent <= 0 "
            "and silence the discarded np.where branch with np.errstate"
        ) from err
    assert np.allclose(result.data[0], 0, atol=TOLERANCE), "sigmoid(-∞) should approach 0"
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "sigmoid(+∞) should approach 1"

    print("✅ Sigmoid works correctly!")

if __name__ == "__main__":
    test_unit_sigmoid()

# %% [markdown]
r"""
### ReLU: The Sparsity Creator

ReLU (Rectified Linear Unit) is the standard hidden layer activation across deep learning. It clamps negative values to zero while passing positive values unchanged, introducing nonlinearity at near-zero hardware computational overhead.

#### Mathematical Formulation & Properties

$$f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$

| Property | Specification |
|:---|:---|
| **Input Domain** | $x \in (-\infty, \infty)$ |
| **Output Range** | $f(x) \in [0, \infty)$ |
| **Derivative** | $f'(x) = \mathbb{I}(x > 0)$ (constant $1$ for $x > 0$, $0$ for $x < 0$) |
| **Hardware Profile** | Branchless SIMD comparison; no transcendentals ($\exp$) required |

#### Numerical Vector Trace

$$\begin{bmatrix} -2.0 & -1.0 & 0.0 & 1.0 & 2.0 \end{bmatrix} \xrightarrow{\text{ReLU}} \begin{bmatrix} 0.0 & 0.0 & 0.0 & 1.0 & 2.0 \end{bmatrix}$$

<div align="center">
  <img src="relu_curve.svg" alt="ReLU Piecewise Linear Hinge" width="320px">
</div>

**Why ReLU matters**: By zeroing negative values, ReLU creates representation sparsity (often $\approx 50\%$ dead/zeroed units in trained networks). Because a comparison instruction (`max`) executes in a single clock cycle compared to multi-cycle transcendentals (`exp`), ReLU dramatically accelerates deep architectures.
"""

# %% nbgrader={"grade": false, "grade_id": "relu-impl", "solution": true}
#| export
class ReLUFunction(Function):
    """
    The ReLU operation. forward() works on NumPy arrays; Module 06 adds backward().
    """
    def forward(self, x):
        """
        Apply ReLU activation element-wise.

        TODO: Implement ReLU function

        APPROACH:
        1. Use np.maximum(0, x) for element-wise max with zero
        2. Return result as a NumPy array

        EXAMPLE:
        >>> relu = ReLU()
        >>> x = Tensor([-2, -1, 0, 1, 2])
        >>> result = relu(x)
        >>> print(result.data)
        [0, 0, 0, 1, 2]  # Negative values become 0, positive unchanged

        HINT: np.maximum handles element-wise maximum automatically
        """
        ### BEGIN SOLUTION
        # Apply ReLU: max(0, x)
        result = np.maximum(0, x)
        return result
        ### END SOLUTION


class ReLU:
    """
    ReLU activation: f(x) = max(0, x)

    Sets negative values to zero, keeps positive values unchanged.
    Most popular activation for hidden layers.
    """

    def parameters(self):
        """No weights to train, so Module 03's layers and Module 07's optimizers will have nothing to collect."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU through its operation, so Module 06 will be able to record it for gradients."""
        return ReLUFunction.apply(x)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: ReLU

This test validates ReLU activation behavior.

**What we're testing**: ReLU zeros negative values, preserves positive
**Why it matters**: ReLU's sparsity helps neural networks train efficiently
**Expected**: Negative becomes 0, positive unchanged, zero becomes 0
"""

# %% nbgrader={"grade": true, "grade_id": "test-relu", "locked": true, "points": 10}
def test_unit_relu():
    """🧪 Test ReLU implementation."""
    print("🧪 Unit Test: ReLU...")

    relu = ReLU()

    # Test mixed positive/negative values
    x = Tensor([-2, -1, 0, 1, 2])
    result = relu.forward(x)
    expected = [0, 0, 0, 1, 2]
    assert np.allclose(result.data, expected), f"ReLU failed, expected {expected}, got {result.data}"

    # Test all negative
    x = Tensor([-5, -3, -1])
    result = relu.forward(x)
    assert np.allclose(result.data, [0, 0, 0]), "ReLU should zero all negative values"

    # Test all positive
    x = Tensor([1, 3, 5])
    result = relu.forward(x)
    assert np.allclose(result.data, [1, 3, 5]), "ReLU should preserve all positive values"

    # Test sparsity property
    x = Tensor([-1, -2, -3, 1])
    result = relu.forward(x)
    zeros = np.sum(result.data == 0)
    assert zeros == 3, f"ReLU should create sparsity, got {zeros} zeros out of 4"

    print("✅ ReLU works correctly!")

if __name__ == "__main__":
    test_unit_relu()

# %% [markdown]
r"""
### Tanh: The Zero-Centered Alternative

Tanh (hyperbolic tangent) rescales the sigmoid curve to range $(-1, 1)$, centered symmetrically around the origin $(0, 0)$.

#### Mathematical Formulation & Properties

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

| Property | Specification |
|:---|:---|
| **Input Domain** | $x \in (-\infty, \infty)$ |
| **Output Range** | $\tanh(x) \in (-1, 1)$ |
| **Derivative** | $\tanh'(x) = 1 - \tanh^2(x)$ with peak $\tanh'(0) = 1.0$ |
| **Zero-Centered** | Odd function: $\tanh(-x) = -\tanh(x)$ with $\tanh(0) = 0$ |

#### Numerical Vector Trace

$$\begin{bmatrix} -2.0 & 0.0 & 2.0 \end{bmatrix} \xrightarrow{\tanh(x)} \begin{bmatrix} -0.9640 & 0.0000 & 0.9640 \end{bmatrix}$$

<div align="center">
  <img src="tanh_curve.svg" alt="Tanh Zero-Centered S-Curve" width="320px">
</div>

**Why Tanh matters**: Because Sigmoid outputs are strictly positive ($> 0$), downstream gradients all inherit the same sign, causing systematic zig-zagging in weight space. Tanh outputs have zero mean on symmetric inputs, preserving gradient balance across intermediate layers.
"""

# %% nbgrader={"grade": false, "grade_id": "tanh-impl", "solution": true}
#| export
class TanhFunction(Function):
    """
    The Tanh operation. forward() works on NumPy arrays; Module 06 adds backward().
    """
    def forward(self, x):
        """
        Apply tanh activation element-wise.

        TODO: Implement tanh function

        APPROACH:
        1. Use np.tanh(x) for hyperbolic tangent
        2. Return result as a NumPy array

        EXAMPLE:
        >>> tanh = Tanh()
        >>> x = Tensor([-2, 0, 2])
        >>> result = tanh(x)
        >>> print(result.data)
        [-0.964, 0.0, 0.964]  # Range (-1, 1), symmetric around 0

        HINT: NumPy provides np.tanh function
        """
        ### BEGIN SOLUTION role="scaffold"
        # Apply tanh using NumPy
        result = np.tanh(x)
        return result
        ### END SOLUTION


class Tanh:
    """
    Tanh activation: f(x) = (e^x - e^(-x))/(e^x + e^(-x))

    Maps any real number to (-1, 1) range.
    Zero-centered alternative to sigmoid.
    """

    def parameters(self):
        """No weights to train, so Module 03's layers and Module 07's optimizers will have nothing to collect."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply Tanh through its operation, so Module 06 will be able to record it for gradients."""
        return TanhFunction.apply(x)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Tanh

This test validates tanh activation behavior.

**What we're testing**: Tanh maps inputs to (-1, 1) range, zero-centered
**Why it matters**: Zero-centered outputs keep the next stage's inputs balanced around zero
**Expected**: All outputs in [-1, 1] (saturating to exactly ±1 in float32 for |x| > 9), tanh(0) = 0, symmetric behavior
"""

# %% nbgrader={"grade": true, "grade_id": "test-tanh", "locked": true, "points": 10}
def test_unit_tanh():
    """🧪 Test Tanh implementation."""
    print("🧪 Unit Test: Tanh...")

    tanh = Tanh()

    # Test zero
    x = Tensor([0.0])
    result = tanh.forward(x)
    assert np.allclose(result.data, [0.0]), f"tanh(0) should be 0, got {result.data}"

    # Test range property - all outputs in [-1, 1]; float32 tanh(10) rounds to exactly 1.0
    x = Tensor([-10, -1, 0, 1, 10])
    result = tanh.forward(x)
    assert np.all(result.data >= -1) and np.all(result.data <= 1), "All tanh outputs should be in [-1, 1]"

    # Test symmetry: tanh(-x) = -tanh(x)
    x = Tensor([2.0])
    pos_result = tanh.forward(x)
    x_neg = Tensor([-2.0])
    neg_result = tanh.forward(x_neg)
    assert np.allclose(pos_result.data, -neg_result.data), "tanh should be symmetric: tanh(-x) = -tanh(x)"

    # Test extreme values
    x = Tensor([-1000, 1000])
    result = tanh.forward(x)
    assert np.allclose(result.data[0], -1, atol=TOLERANCE), "tanh(-∞) should approach -1"
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "tanh(+∞) should approach 1"

    print("✅ Tanh works correctly!")

if __name__ == "__main__":
    test_unit_tanh()

# %% [markdown]
r"""
### GELU: The Smooth Modern Choice

GELU (Gaussian Error Linear Unit) is a smooth, probabilistically motivated alternative to ReLU adopted across modern Transformer architectures (GPT, BERT, RoBERTa).

#### Mathematical Formulation & Approximations

GELU weights inputs by their standard normal cumulative distribution $\Phi(x) = P(X \le x), X \sim \mathcal{N}(0, 1)$:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

In practice, two high-performance approximations are used (Hendrycks & Gimpel, 2016):
$$\text{Sigmoid-GELU (Fast, used in TinyTorch): } \text{GELU}(x) \approx x \cdot \sigma(1.702 x)$$
$$\text{Tanh-GELU (PyTorch default): } \text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

| Property | Specification |
|:---|:---|
| **Input Domain** | $x \in (-\infty, \infty)$ |
| **Output Range** | $[\approx -0.170, \infty)$ |
| **Smoothness** | Infinitely differentiable everywhere ($C^\infty$) |
| **Curvature** | Curvature well dipping to $\approx -0.17$ near $x \approx -0.75$, smoothly passing through $(0, 0)$ |

#### Numerical Vector Trace

$$\begin{bmatrix} -1.0 & 0.0 & 1.0 \end{bmatrix} \xrightarrow{\text{GELU}} \begin{bmatrix} -0.1543 & 0.0000 & 0.8457 \end{bmatrix}$$

<div align="center">
  <img src="gelu_curve.svg" alt="GELU vs ReLU Curvature" width="320px">
</div>

**Why GELU matters**: Unlike ReLU's sharp non-differentiable hinge at $x = 0$, GELU provides smooth non-zero gradients everywhere, eliminating the "dying neuron" failure mode in deep Transformers.
"""

# %% nbgrader={"grade": false, "grade_id": "gelu-impl", "solution": true}
#| export
class GELUFunction(Function):
    """
    The GELU operation. forward() works on NumPy arrays; Module 06 adds backward().
    """
    def forward(self, x):
        """
        Apply GELU activation element-wise.

        TODO: Implement GELU approximation

        APPROACH:
        1. Use approximation: x * sigmoid(1.702 * x)
        2. Reuse the stable SigmoidFunction; scaled extreme inputs may saturate to infinity
        3. Multiply by x element-wise; define the negative-infinity limit as zero
        4. Return result as a NumPy array

        EXAMPLE:
        >>> gelu = GELU()
        >>> x = Tensor([-1, 0, 1])
        >>> result = gelu(x)
        >>> print(result.data)
        [-0.15, 0.0, 0.85]  # Smooth, like ReLU but differentiable everywhere

        HINT: The 1.702 constant is empirically fitted so that sigmoid(1.702x) ≈ Φ(x)
        """
        ### BEGIN SOLUTION role="scaffold"
        # Overflow in the scaled gate means saturation, which sigmoid handles.
        with np.errstate(over="ignore"):
            sig = SigmoidFunction().forward(1.702 * x)
        with np.errstate(invalid="ignore"):
            out = x * sig
        if np.any(np.isneginf(x)):
            out = np.where(np.isneginf(x), 0.0, out)
        return out
        ### END SOLUTION


class GELU:
    """
    GELU activation: f(x) = x * Φ(x) ≈ x * Sigmoid(1.702 * x)

    Smooth approximation to ReLU, used in modern architectures.
    Where Φ(x) is the cumulative distribution function of standard normal.
    """

    def parameters(self):
        """No weights to train, so Module 03's layers and Module 07's optimizers will have nothing to collect."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU through its operation, so Module 06 will be able to record it for gradients."""
        return GELUFunction.apply(x)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: GELU

This test validates GELU activation behavior.

**What we're testing**: GELU provides smooth ReLU-like behavior
**Why it matters**: GELU is used in modern transformers like GPT and BERT
**Expected**: Smooth curve, GELU(0) approximately 0, positive values preserved roughly
"""

# %% nbgrader={"grade": true, "grade_id": "test-gelu", "locked": true, "points": 10}
def test_unit_gelu():
    """🧪 Test GELU implementation."""
    print("🧪 Unit Test: GELU...")

    gelu = GELU()

    # Test zero (should be approximately 0)
    x = Tensor([0.0])
    result = gelu.forward(x)
    assert np.allclose(result.data, [0.0], atol=TOLERANCE), f"GELU(0) should be ≈0, got {result.data}"

    # Test positive values (should be roughly preserved)
    x = Tensor([1.0])
    result = gelu.forward(x)
    assert result.data[0] > 0.8, f"GELU(1) should be ≈0.84, got {result.data[0]}"

    # Test negative values (should be small but not zero)
    x = Tensor([-1.0])
    result = gelu.forward(x)
    assert result.data[0] < 0 and result.data[0] > -0.2, f"GELU(-1) should be ≈-0.16, got {result.data[0]}"

    # Test smoothness property (no sharp corners like ReLU)
    x = Tensor([-0.001, 0.0, 0.001])
    result = gelu.forward(x)
    # Values should be close to each other (smooth)
    diff1 = abs(result.data[1] - result.data[0])
    diff2 = abs(result.data[2] - result.data[1])
    assert diff1 < 0.01 and diff2 < 0.01, "GELU should be smooth around zero"

    print("✅ GELU works correctly!")

if __name__ == "__main__":
    test_unit_gelu()

# %% [markdown]
r"""
### Softmax: The Probability Distributor

Softmax generalizes the sigmoid function to multidimensional vectors, converting unconstrained logit scores $\mathbf{z}$ into a normalized categorical probability distribution that sums to $1.0$.

#### Mathematical Formulation & Numerical Stabilization

Directly computing exponentials $\exp(z_i)$ can overflow 32-bit floating point registers (e.g. $\exp(89) > 10^{38}$). Subtracting the maximum value $\max(\mathbf{z})$ provides exact mathematical invariance while guaranteeing the exponent is $\le 0$:

$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^C e^{z_j - \max(\mathbf{z})}}$$

| Property | Specification |
|:---|:---|
| **Input Domain** | $\mathbf{z} \in \mathbb{R}^C$ |
| **Output Range** | $p_i \in (0, 1)$ such that $\sum_{i=1}^C p_i = 1.0$ |
| **Invariance** | $\text{Softmax}(\mathbf{z} - c) = \text{Softmax}(\mathbf{z})$ for any scalar $c$ |
| **Masking Convention** | Unbounded negative infinity ($-\infty$) exponentiates safely to $0.0$ |

#### Step-by-Step Numerical Vector Trace

$$\mathbf{z} = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \\ 4.0 \end{bmatrix} \xrightarrow{\text{Shift } (z_i - 4.0)} \begin{bmatrix} -3.0 \\ -2.0 \\ -1.0 \\ 0.0 \end{bmatrix} \xrightarrow{\exp(\cdot)} \begin{bmatrix} 0.0498 \\ 0.1353 \\ 0.3679 \\ 1.0000 \end{bmatrix} \xrightarrow{\div \sum=1.5530} \begin{bmatrix} 0.0321 \\ 0.0871 \\ 0.2369 \\ 0.6439 \end{bmatrix}$$

$$\sum_{i=1}^4 p_i = 0.0321 + 0.0871 + 0.2369 + 0.6439 = 1.0000$$

**Why Softmax matters**: In multi-class classification and attention mechanisms (Module 12), Softmax produces valid probability distributions where the highest logit dominates while lower logits receive proportional mass.
"""

# %% nbgrader={"grade": false, "grade_id": "softmax-impl", "solution": true}
#| export
class SoftmaxFunction(Function):
    """
    The Softmax operation. forward() works on NumPy arrays; Module 06 adds backward().
    """
    dim = -1

    def forward(self, x):
        """
        Apply softmax activation along specified dimension.

        TODO: Implement numerically stable softmax

        APPROACH:
        1. Find the maximum along the chosen dimension, keeping dimensions
        2. Replace an all-negative-infinity maximum with zero before subtracting
        3. Compute exponentials and their sum along the same dimension
        4. Replace a zero sum with one, so fully masked slices remain zero
        5. Divide by the safe sum and return the NumPy array

        EXAMPLE:
        >>> softmax = Softmax()
        >>> x = Tensor([1, 2, 3])
        >>> result = softmax(x)
        >>> print(result.data)
        [0.090, 0.245, 0.665]  # Sums to 1.0, larger inputs get higher probability

        HINTS:
        - Use np.max(x, axis=self.dim, keepdims=True) for max
        - Use np.sum(exp_values, axis=self.dim, keepdims=True) for sum
        - The max subtraction prevents overflow in exponentials
        - A fully masked slice contains only -inf and must return zeros
        - An extreme finite difference may become -inf; exp(-inf) correctly gives zero
        """
        ### BEGIN SOLUTION
        # Numerical stability: subtract max to prevent overflow
        x_max = np.max(x, axis=self.dim, keepdims=True)
        safe_max = np.where(np.isneginf(x_max), 0.0, x_max)
        # An extreme negative difference may overflow to -inf, whose exp is zero.
        with np.errstate(over="ignore"):
            x_shifted = x - safe_max

        # Compute exponentials
        exp_values = np.exp(x_shifted)

        # Sum along dimension
        exp_sum = np.sum(exp_values, axis=self.dim, keepdims=True)
        safe_sum = np.where(exp_sum == 0, 1.0, exp_sum)

        # Normalize to get probabilities
        result = exp_values / safe_sum
        return result
        ### END SOLUTION


class Softmax:
    """
    Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))

    Finite scores normalize to a probability distribution along the chosen axis.
    Negative infinity masks entries out. Fully masked slices return all zeros.
    Positive infinity and NaN scores are outside this contract.
    """

    def parameters(self):
        """No weights to train, so Module 03's layers and Module 07's optimizers will have nothing to collect."""
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply Softmax through its operation, so Module 06 will be able to record it for gradients."""
        return SoftmaxFunction.apply(x, dim=dim)

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x, dim)

# %% [markdown]
"""
### 🧪 Unit Test: Softmax

This test validates softmax activation behavior.

**What we're testing**: Softmax normalizes scores and handles masked entries
**Why it matters**: Essential for multi-class classification outputs
**Expected**: Nonnegative outputs sum to one for unmasked slices; fully masked slices are zero; the largest finite input gets the highest probability
"""

# %% nbgrader={"grade": true, "grade_id": "test-softmax", "locked": true, "points": 10}
def test_unit_softmax():
    """🧪 Test Softmax implementation."""
    print("🧪 Unit Test: Softmax...")

    softmax = Softmax()

    # Test basic probability properties
    x = Tensor([1, 2, 3])
    result = softmax.forward(x)

    # Should sum to 1
    assert np.allclose(np.sum(result.data), 1.0), f"Softmax should sum to 1, got {np.sum(result.data)}"

    # All values should be positive
    assert np.all(result.data > 0), "All softmax values should be positive"

    # All values should be less than 1
    assert np.all(result.data < 1), "All softmax values should be less than 1"

    # Largest input should get largest output
    max_input_idx = np.argmax(x.data)
    max_output_idx = np.argmax(result.data)
    assert max_input_idx == max_output_idx, "Largest input should get largest softmax output"

    # Test numerical stability with large numbers
    x = Tensor([1000, 1001, 1002])  # Would overflow without max subtraction
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data), 1.0), "Softmax should handle large numbers"
    assert not np.any(np.isnan(result.data)), "Softmax should not produce NaN"
    assert not np.any(np.isinf(result.data)), "Softmax should not produce infinity"

    # Test with 2D tensor (batch dimension)
    x = Tensor([[1, 2], [3, 4]])
    result = softmax.forward(x, dim=-1)  # Softmax along last dimension
    assert result.shape == (2, 2), "Softmax should preserve input shape"
    # Each row should sum to 1
    row_sums = np.sum(result.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), "Each row should sum to 1"

    # Fully masked rows have no available choice; partial masks still normalize.
    x = Tensor([[-np.inf, -np.inf], [0.0, -np.inf]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = softmax(x)
    np.testing.assert_array_equal(result.data, [[0.0, 0.0], [1.0, 0.0]])

    print("✅ Softmax works correctly!")

if __name__ == "__main__":
    test_unit_softmax()

# %% [markdown]
"""
## 🔧 Integration: Bringing It Together

Now let's test how all our activation functions work together and understand their different behaviors.
"""


# %% [markdown]
"""
### Understanding the Output Patterns

From the demonstration above, notice how each activation serves a different purpose:

**Sigmoid**: Squashes everything to (0, 1) - good for probabilities
**ReLU**: Zeros negatives, keeps positives - creates sparsity
**Tanh**: Like sigmoid but centered at zero (-1, 1) - better mathematical properties for composing transformations
**GELU**: Smooth ReLU-like behavior - modern choice for advanced architectures
**Softmax**: Converts to probability distribution - sum equals 1

These different behaviors make each activation suitable for different computational tasks.
"""

# %% [markdown]
"""
## 📊 Systems Analysis: Activation Computation Costs

Let's understand ONE key systems concept: **computational cost differences between activations**.

This analysis reveals why ReLU dominates hidden layers while more expensive activations are reserved for specific use cases.
"""

# %%
def analyze_activation_performance():
    """Demonstrate computational cost differences between activation functions."""
    print("Analyzing Activation Computation Costs...")
    print("=" * 60)

    import time

    # Seeded here rather than at module scope: this timing cell is the only user,
    # and the package should not ship a fixed global seed.
    rng = np.random.default_rng(7)

    # Create test data (realistic hidden layer size)
    size = 1000000  # 1 million elements (like a large hidden layer)
    test_data = Tensor(rng.standard_normal(size).astype(np.float32))

    print(f"\nTesting with {size:,} elements (simulating large hidden layer)")
    print("-" * 60)

    # Initialize activations
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    gelu = GELU()

    # Warm up
    _ = relu(test_data)
    _ = sigmoid(test_data)

    # Time each activation (multiple runs for accuracy)
    n_runs = 10

    # ReLU timing
    start = time.time()
    for _ in range(n_runs):
        _ = relu(test_data)
    relu_time = (time.time() - start) / n_runs * 1000

    # Sigmoid timing
    start = time.time()
    for _ in range(n_runs):
        _ = sigmoid(test_data)
    sigmoid_time = (time.time() - start) / n_runs * 1000

    # Tanh timing
    start = time.time()
    for _ in range(n_runs):
        _ = tanh(test_data)
    tanh_time = (time.time() - start) / n_runs * 1000

    # GELU timing
    start = time.time()
    for _ in range(n_runs):
        _ = gelu(test_data)
    gelu_time = (time.time() - start) / n_runs * 1000

    print("\n🧪 Activation Performance Results:")
    print(f"   ReLU:    {relu_time:.2f}ms (baseline)")
    print(f"   Sigmoid: {sigmoid_time:.2f}ms ({sigmoid_time/relu_time:.1f}x slower)")
    print(f"   Tanh:    {tanh_time:.2f}ms ({tanh_time/relu_time:.1f}x slower)")
    print(f"   GELU:    {gelu_time:.2f}ms ({gelu_time/relu_time:.1f}x slower)")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("   1. ReLU is fastest: Just max(0, x) - no exponentials")
    print("   2. Sigmoid/Tanh require exp() - expensive operation")
    print("   3. GELU uses sigmoid internally - inherits its cost")
    print("   4. For hidden layers: ReLU's speed advantage adds up!")

    print("\nREAL-WORLD IMPLICATIONS:")
    print("   - ResNet uses ReLU: billions of activations per forward pass")
    print("   - GPT uses GELU: worth the cost for better gradients")
    print("   - Sigmoid/Tanh: reserved for output layers or gates")
    print("=" * 60)

if __name__ == "__main__":
    analyze_activation_performance()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-test", "locked": true, "points": 20}

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
    test_unit_sigmoid()
    test_unit_relu()
    test_unit_tanh()
    test_unit_gelu()
    test_unit_softmax()

    print("\nRunning integration scenarios...")

    # Test 1: All activations preserve tensor properties
    print("🧪 Integration Test: Tensor property preservation...")
    test_data = Tensor([[1, -1], [2, -2]])  # 2D tensor

    activations = [Sigmoid(), ReLU(), Tanh(), GELU()]
    for activation in activations:
        result = activation.forward(test_data)
        assert result.shape == test_data.shape, f"Shape not preserved by {activation.__class__.__name__}"
        assert isinstance(result, Tensor), f"Output not Tensor from {activation.__class__.__name__}"

    print("✅ All activations preserve tensor properties!")

    # Test 2: Softmax works with different dimensions
    print("🧪 Integration Test: Softmax dimension handling...")
    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # (2, 2, 3)
    softmax = Softmax()

    # Test different dimensions
    result_last = softmax(data_3d, dim=-1)
    assert result_last.shape == (2, 2, 3), "Softmax should preserve shape"

    # Check that last dimension sums to 1
    last_dim_sums = np.sum(result_last.data, axis=-1)
    assert np.allclose(last_dim_sums, 1.0), "Last dimension should sum to 1"

    print("✅ Softmax handles different dimensions correctly!")

    # Test 3: Activation chaining (simulating neural network)
    print("🧪 Integration Test: Activation chaining...")

    # Chain activations the way a network does between its layers: ReLU → Softmax
    x = Tensor([[-1, 0, 1, 2]])  # Batch of 1, 4 features

    # Apply ReLU (hidden layer activation)
    relu = ReLU()
    hidden = relu.forward(x)

    # Apply Softmax (output layer activation)
    softmax = Softmax()
    output = softmax.forward(hidden)

    # Verify the chain
    assert hidden.data[0, 0] == 0, "ReLU should zero negative input"
    assert np.allclose(np.sum(output.data), 1.0), "Final output should be probability distribution"

    print("✅ Activation chaining works correctly!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 02")


# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of activation functions and their systems implications:

### Question 1: Computational Cost Comparison
**Question**: ReLU is the most popular activation function in hidden layers. Given what you implemented, why is ReLU computationally cheaper than Sigmoid or GELU?

**Consider**:
- What mathematical operations does ReLU require? (hint: just max(0, x))
- What operations does Sigmoid require? (hint: exponentials)
- If you have a hidden layer with 1 million neurons, how many exp() calls does each activation require?

**Real-world context**: In production models with billions of parameters, even small per-element costs add up. ReLU's simplicity makes it several times cheaper than Sigmoid per element.

---

### Question 2: Numerical Stability
**Question**: Look at your Softmax implementation. Why did we subtract the maximum value before computing exponentials?

**Consider**:
- What happens when you compute exp(1000)?
- What about exp(1000) / (exp(1000) + exp(1001))?
- Does subtracting a constant from all inputs change the final softmax output?

**Mathematical insight**: exp(x - max) / sum(exp(x - max)) = exp(x) / sum(exp(x)) because the constant cancels.

---

### Question 3: Sparsity and Efficiency
**Question**: ReLU creates "sparsity" by zeroing negative values. Why might having many zero activations be beneficial for computation?

**Consider**:
- Memory: Do zeros need to be stored differently than non-zeros?
- Computation: What happens when you multiply by zero?
- Learning: If 50% of neurons are "off" for a given input, what does that mean for the representation?

**Think about**:
- Sparse matrix representations and their memory benefits
- How GPUs handle sparse operations
- Whether sparsity helps or hurts different types of computations

---

### Question 4: Activation Selection for Different Stages
**Question**: Why do we typically use different activations for hidden stages vs. output stages of a computation?

**Consider the requirements**:
- Hidden stages: Need to preserve information flowing through the computation, be efficient, add nonlinearity
- Binary classification output: Need values in (0, 1) representing probability
- Multi-class classification output: Need probability distribution (sum = 1)

**Match the activation to the use case**:
- ReLU for hidden stages (why?)
- Sigmoid for binary output (why?)
- Softmax for multi-class output (why?)

---

### Question 5: The "Dying ReLU" Problem
**Question**: If ReLU outputs 0 for some inputs, those inputs get no signal passed through -- they effectively "die." What situations might cause this, and why is it a problem?

**Consider**:
- If inputs to a ReLU are always negative, its output is permanently zero
- Once a ReLU "dies" (always outputs 0), no information flows through it -- it contributes nothing to the computation
- A dead ReLU is wasted capacity: it takes up memory and compute but produces no useful signal

**Solutions used in practice**:
- LeakyReLU: f(x) = max(0.01*x, x) - allows a small signal even for negative inputs
- PReLU: Adjustable slope for negative values (a learnable parameter controls the slope)
- GELU: Smooth approximation that never fully zeroes out

---

### Bonus Question: Memory Analysis

**Scenario**: You're running inference on a model with a hidden layer of size (batch=32, features=4096) using different activations.

**Calculate for each activation**:
1. How many bytes of output memory are needed? (assume float32)
2. How many temporary buffers does Softmax need vs ReLU?
3. If you switch from float32 to float16, what's the memory savings?

**Key insight**: Activation functions are memory-light (output same size as input), but the choice affects computational speed and numerical precision significantly.
"""


# %% [markdown]
"""
## ⭐ Aha Moment: Activations Add Intelligence

**What you built:** Five activation functions that introduce nonlinearity to neural networks.

**Why it matters:** Without activations, stacking layers would just be matrix multiplication -
a linear operation. ReLU's simple "zero out negatives" rule is what allows networks to learn
complex patterns like recognizing faces or understanding language.

Module 03 will combine your activations with Linear layers!
"""

# %%
def demo_activations():
    """🎯 See how activations transform data."""
    print("🎯 AHA MOMENT: Activations Add Intelligence")
    print("=" * 45)

    # Test input with positive and negative values
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    print(f"Input:   {x.data}")

    # ReLU - zeros out negatives
    relu = ReLU()
    relu_out = relu(x)
    print(f"ReLU:    {relu_out.data}")
    print("         Negatives become 0, positives unchanged!")

    # Sigmoid - squashes to (0, 1)
    sigmoid = Sigmoid()
    sigmoid_out = sigmoid(x)
    print(f"\nSigmoid: {np.round(sigmoid_out.data, 2)}")
    print("         All values squashed to (0, 1) range!")

    # Softmax - probability distribution
    softmax = Softmax()
    softmax_out = softmax(x)
    print(f"\nSoftmax: {np.round(softmax_out.data, 3)}")
    print(f"         Sum = {softmax_out.data.sum():.1f} (valid probability distribution!)")

    print("\n✨ Activations add nonlinearity—the key to deep learning!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_activations()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Activations

Congratulations! You've built the intelligence engine of neural networks!

### Key Accomplishments
- **Built 5 core activation functions** with distinct behaviors and use cases
- **Implemented forward passes** for Sigmoid, ReLU, Tanh, GELU, and Softmax
- **Discovered computational cost differences** between activations (ReLU fastest)
- **Handled numerical stability** with max subtraction techniques
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **ReLU efficiency**: A max is far cheaper than an exponential, so ReLU costs a fraction of Sigmoid or Tanh per element
- **Numerical stability**: Softmax's max subtraction prevents overflow without changing results
- **Sparsity benefits**: ReLU's zero outputs create sparse representations
- **Activation selection**: Different layers need different activations (ReLU for hidden, Softmax for output)

### Ready for Next Steps
Your activation functions give every layer the nonlinearity it needs. Without
them a stack of Linear layers collapses into a single Linear layer, no matter
how deep you make it.

Export with: `tito module complete 02`

**Next**: Module 03 will combine your Tensors and Activations to build complete neural network Layers!
"""
