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
"""
# Activations - Intelligence Through Nonlinearity

Welcome to Activations! Today you'll add the secret ingredient that makes neural networks intelligent: **nonlinearity**.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor with data manipulation and basic operations
**You'll Build**: Activation functions that add nonlinearity to transformations
**You'll Enable**: Neural networks with the ability to learn complex patterns

**Connection Map**:
```
Tensor → Activations → Layers
(data)   (intelligence) (architecture)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement 5 core activation functions (Sigmoid, ReLU, Tanh, GELU, Softmax)
2. Understand how nonlinearity enables neural network intelligence
3. Test activation behaviors and output ranges
4. Connect activations to real neural network components

Let's add intelligence to your tensors!
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/02_activations/activations_dev.py
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
"""
## 📋 Module Dependencies

**Prerequisites**: Module 01 (Tensor) must be completed

**External Dependencies**:
- `numpy` (for numerical operations)

**TinyTorch Dependencies**:
- **Module 01 (Tensor)**: Foundation for all activation computations and data flow
  - Used for: Input/output data structures, shape operations, element-wise operations
  - Required: Yes - activations operate on Tensor objects

**Dependency Flow**:
```
Module 01 (Tensor) → Module 02 (Activations) → Module 03 (Layers)
     ↓                      ↓                         ↓
  Foundation          Nonlinearity              Architecture
```

**Import Strategy**:
This module imports directly from the TinyTorch package (`from tinytorch.core.*`).
**Assumption**: Module 01 (Tensor) has been completed and exported to the package.
If you see import errors, ensure you've run `tito export` after completing Module 01.
"""

# %% nbgrader={"grade": false, "grade_id": "setup", "solution": true}
#| default_exp core.activations
#| export

import numpy as np
from typing import Optional

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor

# Constants for numerical comparisons
TOLERANCE = 1e-10  # Small tolerance for floating-point comparisons in tests

# Export only activation classes
__all__ = ['Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax']

# %% [markdown]
"""
## 💡 Introduction - What Makes Neural Networks Intelligent?

Consider two scenarios:

**Without Activations (Linear Only):**
```
Input → Linear Transform → Output
[1, 2] → [3, 4] → [11]  # Just weighted sum
```

**With Activations (Nonlinear):**
```
Input → Linear → Activation → Linear → Activation → Output
[1, 2] → [3, 4] → [3, 4] → [7] → [7] → Complex Pattern!
```

The magic happens in those activation functions. They introduce **nonlinearity** - the ability to curve, bend, and create complex decision boundaries instead of just straight lines.

### Why Nonlinearity Matters

Without activation functions, stacking multiple linear layers is pointless:
```
Linear(Linear(x)) = Linear(x)  # Same as single layer!
```

With activation functions, each layer can learn increasingly complex patterns:
```
Layer 1: Simple edges and lines
Layer 2: Curves and shapes
Layer 3: Complex objects and concepts
```

This is how deep networks build intelligence from simple mathematical operations.
"""

# %% [markdown]
"""
## 📐 Mathematical Foundations

Each activation function serves a different purpose in neural networks:

### The Five Essential Activations

1. **Sigmoid**: Maps to (0, 1) - perfect for probabilities
2. **ReLU**: Removes negatives - creates sparsity and efficiency
3. **Tanh**: Maps to (-1, 1) - zero-centered for better training
4. **GELU**: Smooth ReLU - modern choice for transformers
5. **Softmax**: Creates probability distributions - essential for classification

Let's implement each one with clear explanations and immediate testing!
"""

# %% [markdown]
"""
## 🏗️ Implementation - Building Activation Functions

### 🏗️ Implementation Pattern

Each activation follows this structure:
```python
class ActivationName:
    def forward(self, x: Tensor) -> Tensor:
        # Apply mathematical transformation
        # Return new Tensor with result

    def backward(self, grad: Tensor) -> Tensor:
        # Stub for Module 06 - gradient computation
        pass
```
"""

# %% [markdown]
"""
## 🏗️ Sigmoid - The Probability Gatekeeper

Sigmoid maps any real number to the range (0, 1), making it perfect for probabilities and binary decisions.

### Mathematical Definition
```
σ(x) = 1/(1 + e^(-x))
```

### Visual Behavior
```
Input:  [-3, -1,  0,  1,  3]
         ↓   ↓   ↓   ↓   ↓  Sigmoid Function
Output: [0.05, 0.27, 0.5, 0.73, 0.95]
```

### ASCII Visualization
```
Sigmoid Curve:
    1.0 ┤     ╭─────
        │    ╱
    0.5 ┤   ╱
        │  ╱
    0.0 ┤─╱─────────
       -3  0  3
```

**Why Sigmoid matters**: In binary classification, we need outputs between 0 and 1 to represent probabilities. Sigmoid gives us exactly that!
"""

# %% nbgrader={"grade": false, "grade_id": "sigmoid-impl", "solution": true}
#| export

class Sigmoid:
    """
    Sigmoid activation: σ(x) = 1/(1 + e^(-x))

    Maps any real number to (0, 1) range.
    Perfect for probabilities and binary classification.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply sigmoid activation element-wise.

        TODO: Implement sigmoid function

        APPROACH:
        1. Apply sigmoid formula: 1 / (1 + exp(-x))
        2. Use np.exp for exponential
        3. Return result wrapped in new Tensor

        EXAMPLE:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor([-2, 0, 2])
        >>> result = sigmoid(x)
        >>> print(result.data)
        [0.119, 0.5, 0.881]  # All values between 0 and 1

        HINT: Use np.exp(-x.data) for numerical stability
        """
        ### BEGIN SOLUTION
        # Apply sigmoid: 1 / (1 + exp(-x))
        # Clip extreme values to prevent overflow (sigmoid(-500) ≈ 0, sigmoid(500) ≈ 1)
        # Clipping at ±500 ensures exp() stays within float64 range
        z = np.clip(x.data, -500, 500)

        # Use numerically stable sigmoid
        # For positive values: 1 / (1 + exp(-x))
        # For negative values: exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x)) after clipping
        result_data = np.zeros_like(z)

        # Positive values (including zero)
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        # Negative values
        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        """Compute gradient (implemented in Module 06)."""
        pass  # Will implement backward pass in Module 06

# %% [markdown]
"""
### 🔬 Unit Test: Sigmoid
This test validates sigmoid activation behavior.
**What we're testing**: Sigmoid maps inputs to (0, 1) range
**Why it matters**: Ensures proper probability-like outputs
**Expected**: All outputs between 0 and 1, sigmoid(0) = 0.5
"""

# %% nbgrader={"grade": true, "grade_id": "test-sigmoid", "locked": true, "points": 10}
def test_unit_sigmoid():
    """🔬 Test Sigmoid implementation."""
    print("🔬 Unit Test: Sigmoid...")

    sigmoid = Sigmoid()

    # Test basic cases
    x = Tensor([0.0])
    result = sigmoid.forward(x)
    assert np.allclose(result.data, [0.5]), f"sigmoid(0) should be 0.5, got {result.data}"

    # Test range property - all outputs should be in (0, 1)
    x = Tensor([-10, -1, 0, 1, 10])
    result = sigmoid.forward(x)
    assert np.all(result.data > 0) and np.all(result.data < 1), "All sigmoid outputs should be in (0, 1)"

    # Test specific values
    x = Tensor([-1000, 1000])  # Extreme values
    result = sigmoid.forward(x)
    assert np.allclose(result.data[0], 0, atol=TOLERANCE), "sigmoid(-∞) should approach 0"
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "sigmoid(+∞) should approach 1"

    print("✅ Sigmoid works correctly!")

if __name__ == "__main__":
    test_unit_sigmoid()

# %% [markdown]
"""
## 🏗️ ReLU - The Sparsity Creator

ReLU (Rectified Linear Unit) is the most popular activation function. It simply removes negative values, creating sparsity that makes neural networks more efficient.

### Mathematical Definition
```
f(x) = max(0, x)
```

### Visual Behavior
```
Input:  [-2, -1,  0,  1,  2]
         ↓   ↓   ↓   ↓   ↓  ReLU Function
Output: [ 0,  0,  0,  1,  2]
```

### ASCII Visualization
```
ReLU Function:
        ╱
    2  ╱
      ╱
    1╱
    ╱
   ╱
  ╱
─┴─────
-2  0  2
```

**Why ReLU matters**: By zeroing negative values, ReLU creates sparsity (many zeros) which makes computation faster and helps prevent overfitting.
"""

# %% nbgrader={"grade": false, "grade_id": "relu-impl", "solution": true}
#| export
class ReLU:
    """
    ReLU activation: f(x) = max(0, x)

    Sets negative values to zero, keeps positive values unchanged.
    Most popular activation for hidden layers.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU activation element-wise.

        TODO: Implement ReLU function

        APPROACH:
        1. Use np.maximum(0, x.data) for element-wise max with zero
        2. Return result wrapped in new Tensor

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
        result = np.maximum(0, x.data)
        return Tensor(result)
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        """Compute gradient (implemented in Module 06)."""
        pass  # Will implement backward pass in Module 06

# %% [markdown]
"""
### 🔬 Unit Test: ReLU
This test validates ReLU activation behavior.
**What we're testing**: ReLU zeros negative values, preserves positive
**Why it matters**: ReLU's sparsity helps neural networks train efficiently
**Expected**: Negative → 0, positive unchanged, zero → 0
"""

# %% nbgrader={"grade": true, "grade_id": "test-relu", "locked": true, "points": 10}
def test_unit_relu():
    """🔬 Test ReLU implementation."""
    print("🔬 Unit Test: ReLU...")

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
"""
## 🏗️ Tanh - The Zero-Centered Alternative

Tanh (hyperbolic tangent) is like sigmoid but centered around zero, mapping inputs to (-1, 1). This zero-centering helps with gradient flow during training.

### Mathematical Definition
```
f(x) = (e^x - e^(-x))/(e^x + e^(-x))
```

### Visual Behavior
```
Input:  [-2,  0,  2]
         ↓   ↓   ↓  Tanh Function
Output: [-0.96, 0, 0.96]
```

### ASCII Visualization
```
Tanh Curve:
    1 ┤     ╭─────
      │    ╱
    0 ┤───╱─────
      │  ╱
   -1 ┤─╱───────
     -3  0  3
```

**Why Tanh matters**: Unlike sigmoid, tanh outputs are centered around zero, which can help gradients flow better through deep networks.
"""

# %% nbgrader={"grade": false, "grade_id": "tanh-impl", "solution": true}
#| export
class Tanh:
    """
    Tanh activation: f(x) = (e^x - e^(-x))/(e^x + e^(-x))

    Maps any real number to (-1, 1) range.
    Zero-centered alternative to sigmoid.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply tanh activation element-wise.

        TODO: Implement tanh function

        APPROACH:
        1. Use np.tanh(x.data) for hyperbolic tangent
        2. Return result wrapped in new Tensor

        EXAMPLE:
        >>> tanh = Tanh()
        >>> x = Tensor([-2, 0, 2])
        >>> result = tanh(x)
        >>> print(result.data)
        [-0.964, 0.0, 0.964]  # Range (-1, 1), symmetric around 0

        HINT: NumPy provides np.tanh function
        """
        ### BEGIN SOLUTION
        # Apply tanh using NumPy
        result = np.tanh(x.data)
        return Tensor(result)
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        """Compute gradient (implemented in Module 06)."""
        pass  # Will implement backward pass in Module 06

# %% [markdown]
"""
### 🔬 Unit Test: Tanh
This test validates tanh activation behavior.
**What we're testing**: Tanh maps inputs to (-1, 1) range, zero-centered
**Why it matters**: Zero-centered activations can help with gradient flow
**Expected**: All outputs in (-1, 1), tanh(0) = 0, symmetric behavior
"""

# %% nbgrader={"grade": true, "grade_id": "test-tanh", "locked": true, "points": 10}
def test_unit_tanh():
    """🔬 Test Tanh implementation."""
    print("🔬 Unit Test: Tanh...")

    tanh = Tanh()

    # Test zero
    x = Tensor([0.0])
    result = tanh.forward(x)
    assert np.allclose(result.data, [0.0]), f"tanh(0) should be 0, got {result.data}"

    # Test range property - all outputs should be in (-1, 1)
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
"""
## 🏗️ GELU - The Smooth Modern Choice

GELU (Gaussian Error Linear Unit) is a smooth approximation to ReLU that's become popular in modern architectures like transformers. Unlike ReLU's sharp corner, GELU is smooth everywhere.

### Mathematical Definition
```
f(x) = x * Φ(x) ≈ x * Sigmoid(1.702 * x)
```
Where Φ(x) is the cumulative distribution function of standard normal distribution.

### Visual Behavior
```
Input:  [-1,  0,  1]
         ↓   ↓   ↓  GELU Function
Output: [-0.16, 0, 0.84]
```

### ASCII Visualization
```
GELU Function:
        ╱
    1  ╱
      ╱
     ╱
    ╱
   ╱ ↙ (smooth curve, no sharp corner)
  ╱
─┴─────
-2  0  2
```

**Why GELU matters**: Used in GPT, BERT, and other transformers. The smoothness helps with optimization compared to ReLU's sharp corner.
"""

# %% nbgrader={"grade": false, "grade_id": "gelu-impl", "solution": true}
#| export
class GELU:
    """
    GELU activation: f(x) = x * Φ(x) ≈ x * Sigmoid(1.702 * x)

    Smooth approximation to ReLU, used in modern transformers.
    Where Φ(x) is the cumulative distribution function of standard normal.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply GELU activation element-wise.

        TODO: Implement GELU approximation

        APPROACH:
        1. Use approximation: x * sigmoid(1.702 * x)
        2. Compute sigmoid part: 1 / (1 + exp(-1.702 * x))
        3. Multiply by x element-wise
        4. Return result wrapped in new Tensor

        EXAMPLE:
        >>> gelu = GELU()
        >>> x = Tensor([-1, 0, 1])
        >>> result = gelu(x)
        >>> print(result.data)
        [-0.159, 0.0, 0.841]  # Smooth, like ReLU but differentiable everywhere

        HINT: The 1.702 constant comes from √(2/π) approximation
        """
        ### BEGIN SOLUTION
        # GELU approximation: x * sigmoid(1.702 * x)
        # First compute sigmoid part
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        # Then multiply by x
        result = x.data * sigmoid_part
        return Tensor(result)
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        """Compute gradient (implemented in Module 06)."""
        pass  # Will implement backward pass in Module 06

# %% [markdown]
"""
### 🔬 Unit Test: GELU
This test validates GELU activation behavior.
**What we're testing**: GELU provides smooth ReLU-like behavior
**Why it matters**: GELU is used in modern transformers like GPT and BERT
**Expected**: Smooth curve, GELU(0) ≈ 0, positive values preserved roughly
"""

# %% nbgrader={"grade": true, "grade_id": "test-gelu", "locked": true, "points": 10}
def test_unit_gelu():
    """🔬 Test GELU implementation."""
    print("🔬 Unit Test: GELU...")

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
"""
## 🏗️ Softmax - The Probability Distributor

Softmax converts any vector into a valid probability distribution. All outputs are positive and sum to exactly 1.0, making it essential for multi-class classification.

### Mathematical Definition
```
f(x_i) = e^(x_i) / Σ(e^(x_j))
```

### Visual Behavior
```
Input:  [1, 2, 3]
         ↓  ↓  ↓  Softmax Function
Output: [0.09, 0.24, 0.67]  # Sum = 1.0
```

### ASCII Visualization
```
Softmax Transform:
Raw scores: [1, 2, 3, 4]
           ↓ Exponential ↓
          [2.7, 7.4, 20.1, 54.6]
           ↓ Normalize ↓
          [0.03, 0.09, 0.24, 0.64]  ← Sum = 1.0
```

**Why Softmax matters**: In multi-class classification, we need outputs that represent probabilities for each class. Softmax guarantees valid probabilities.
"""

# %% nbgrader={"grade": false, "grade_id": "softmax-impl", "solution": true}
#| export
class Softmax:
    """
    Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))

    Converts any vector to a probability distribution.
    Sum of all outputs equals 1.0.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Apply softmax activation along specified dimension.

        TODO: Implement numerically stable softmax

        APPROACH:
        1. Subtract max for numerical stability: x - max(x)
        2. Compute exponentials: exp(x - max(x))
        3. Sum along dimension: sum(exp_values)
        4. Divide: exp_values / sum
        5. Return result wrapped in new Tensor

        EXAMPLE:
        >>> softmax = Softmax()
        >>> x = Tensor([1, 2, 3])
        >>> result = softmax(x)
        >>> print(result.data)
        [0.090, 0.245, 0.665]  # Sums to 1.0, larger inputs get higher probability

        HINTS:
        - Use np.max(x.data, axis=dim, keepdims=True) for max
        - Use np.sum(exp_values, axis=dim, keepdims=True) for sum
        - The max subtraction prevents overflow in exponentials
        """
        ### BEGIN SOLUTION
        # Numerical stability: subtract max to prevent overflow
        x_max_data = np.max(x.data, axis=dim, keepdims=True)
        x_max = Tensor(x_max_data)
        x_shifted = x - x_max  # Tensor subtraction

        # Compute exponentials
        exp_values = Tensor(np.exp(x_shifted.data))

        # Sum along dimension
        exp_sum_data = np.sum(exp_values.data, axis=dim, keepdims=True)
        exp_sum = Tensor(exp_sum_data)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return result
        ### END SOLUTION

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x, dim)

# %% [markdown]
"""
### 🔬 Unit Test: Softmax
This test validates softmax activation behavior.
**What we're testing**: Softmax creates valid probability distributions
**Why it matters**: Essential for multi-class classification outputs
**Expected**: Outputs sum to 1.0, all values in (0, 1), largest input gets highest probability
"""

# %% nbgrader={"grade": true, "grade_id": "test-softmax", "locked": true, "points": 10}
def test_unit_softmax():
    """🔬 Test Softmax implementation."""
    print("🔬 Unit Test: Softmax...")

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

    print("✅ Softmax works correctly!")

if __name__ == "__main__":
    test_unit_softmax()

# %% [markdown]
"""
## 🔧 Integration - Bringing It Together

Now let's test how all our activation functions work together and understand their different behaviors.
"""


# %% [markdown]
"""
### Understanding the Output Patterns

From the demonstration above, notice how each activation serves a different purpose:

**Sigmoid**: Squashes everything to (0, 1) - good for probabilities
**ReLU**: Zeros negatives, keeps positives - creates sparsity
**Tanh**: Like sigmoid but centered at zero (-1, 1) - better gradient flow
**GELU**: Smooth ReLU-like behavior - modern choice for transformers
**Softmax**: Converts to probability distribution - sum equals 1

These different behaviors make each activation suitable for different parts of neural networks.
"""

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
    print("🔬 Integration Test: Tensor property preservation...")
    test_data = Tensor([[1, -1], [2, -2]])  # 2D tensor

    activations = [Sigmoid(), ReLU(), Tanh(), GELU()]
    for activation in activations:
        result = activation.forward(test_data)
        assert result.shape == test_data.shape, f"Shape not preserved by {activation.__class__.__name__}"
        assert isinstance(result, Tensor), f"Output not Tensor from {activation.__class__.__name__}"

    print("✅ All activations preserve tensor properties!")

    # Test 2: Softmax works with different dimensions
    print("🔬 Integration Test: Softmax dimension handling...")
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
    print("🔬 Integration Test: Activation chaining...")

    # Simulate: Input → Linear → ReLU → Linear → Softmax (like a simple network)
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

# Run comprehensive module test
if __name__ == "__main__":
    test_module()


# %% [markdown]
"""
## 🤔 ML Systems Thinking

Now that you've built activation functions, let's think about their systems-level characteristics.
Understanding computational cost, numerical stability, and gradient behavior helps you make
informed choices when building neural networks.

### Computational Cost Analysis

Different activations have different computational profiles:

**ReLU: O(n) comparisons**
- Simple element-wise comparison: max(0, x)
- Fastest activation function (baseline)
- No exponentials, no divisions
- Ideal for large hidden layers

**Sigmoid/Tanh: O(n) exponentials**
- Each element requires exp() computation
- 3-4× slower than ReLU
- Exponentials are expensive operations
- Use sparingly in hidden layers

**GELU: O(n) exponentials + multiplications**
- Approximation involves sigmoid (exponential)
- 4-5× slower than ReLU
- Worth the cost in transformers (better gradients)
- Trade-off: performance vs. optimization quality

**Softmax: O(n) exponentials + O(n) sum + O(n) divisions**
- Most expensive: exp, sum, divide for entire vector
- Use only for output layers (not hidden layers)
- Requires synchronization across dimension
- Numerical stability tricks add overhead

### Numerical Stability Considerations

Activations can fail catastrophically without proper handling:

**Sigmoid/Tanh overflow:**
```
Problem: exp(1000) = inf, exp(-1000) = 0
Solution: Clip inputs to reasonable range (±500)
Your implementation: Uses stable computation for Sigmoid
```

**Softmax catastrophic overflow:**
```
Problem: exp(1000) = inf, causing NaN
Solution: Subtract max before exp (doesn't change result)
Your implementation: Uses max subtraction in Softmax.forward()
```

**ReLU dying neurons:**
```
Problem: Large negative gradient → weights become negative → ReLU always outputs 0
Solution: Monitor dead neuron percentage, use LeakyReLU variants
Your implementation: Basic ReLU (watch for this in Module 08 training)
```

### Gradient Behavior Preview

While you'll implement gradients in Module 06, understanding gradient characteristics helps:

**ReLU gradient: Sharp discontinuity**
- Gradient = 1 if x > 0, else 0
- Sharp corner at zero
- Dead neurons never recover (gradient = 0 forever)

**Sigmoid/Tanh gradient: Vanishing problem**
- Gradient ≈ 0 for large |x|
- Deep networks struggle (gradients die in early layers)
- Why ReLU replaced sigmoid in hidden layers

**GELU gradient: Smooth everywhere**
- No sharp corners (unlike ReLU)
- No vanishing at extremes (like sigmoid)
- Best of both worlds (modern architectures use this)

**Softmax gradient: Coupled across dimension**
- Changing one input affects all outputs
- Jacobian matrix (not element-wise)
- More complex backward pass than others

### Memory Considerations

**Forward pass memory:**
- All activations: Same size as input (element-wise operations)
- Softmax temporary buffers: exp array + sum array (small overhead)

**Backward pass memory (Module 06):**
- Must cache inputs for gradient computation
- 2× memory per activation layer (input + gradient)
- For 1000-layer network: Memory adds up!

### Key Insights for Module 02

**For early modules, focus on correctness:**
- Your activations work correctly (test_module validates this)
- Numerical stability is handled (Sigmoid clipping, Softmax max-subtraction)
- Integration ready (Module 03 will use these in layers)

**Systems awareness for later:**
- ReLU is fastest, use for hidden layers by default
- Sigmoid/Tanh: Output layers only (or special cases like gates)
- GELU: Worth the cost in transformers (Module 13)
- Softmax: Output layer for classification only

You've built activations that are both correct AND production-ready!
"""


# %% [markdown]
"""
## 📊 Real-World Production Context

Now that you've implemented these activations, let's understand how they're used in real ML systems.

### Activation Selection Guide

**When to Use Each Activation:**

**Sigmoid**
- **Use case**: Binary classification output layers, gates in LSTMs/GRUs
- **Production example**: Spam detection (output: probability of spam)
- **Why**: Outputs valid probabilities in (0, 1)
- **Avoid**: Hidden layers in deep networks (vanishing gradients)

**ReLU**
- **Use case**: Hidden layers in CNNs, feedforward networks
- **Production example**: Image classification networks (ResNet, VGG)
- **Why**: Fast computation, prevents vanishing gradients, creates sparsity
- **Avoid**: Output layers (can't output negative values or probabilities)

**Tanh**
- **Use case**: RNN hidden states, when zero-centered outputs matter
- **Production example**: Sentiment analysis RNNs, time series prediction
- **Why**: Zero-centered helps with gradient flow in recurrent networks
- **Avoid**: Very deep networks (still suffers from vanishing gradients)

**GELU**
- **Use case**: Transformer models, modern architectures
- **Production example**: GPT, BERT, modern language models
- **Why**: Smooth approximation of ReLU, better gradient flow, state-of-the-art results
- **Avoid**: When computational efficiency is critical (slightly slower than ReLU)

**Softmax**
- **Use case**: Multi-class classification output layers
- **Production example**: ImageNet classification (1000 classes), NLP token prediction
- **Why**: Converts logits to valid probability distribution (sums to 1)
- **Avoid**: Hidden layers (loses information through normalization)

### Common Production Patterns

**Pattern 1: CNN Image Classification**
```
Input → Conv+ReLU → Conv+ReLU → ... → Linear → Softmax → Class Probabilities
```

**Pattern 2: Binary Classifier**
```
Input → Linear+ReLU → Linear+ReLU → Linear → Sigmoid → Binary Probability
```

**Pattern 3: Modern Transformer**
```
Input → Attention → Linear+GELU → Linear+GELU → Output
```

### Common Pitfalls and Debugging

**Sigmoid/Tanh Pitfalls:**
- **Vanishing gradients**: Gradients near 0 for extreme inputs
- **Saturation**: Outputs plateau, learning slows
- **Debug tip**: Check activation distribution - avoid all values near 0 or 1

**ReLU Pitfalls:**
- **Dying ReLU**: Neurons output 0 forever after large negative gradient
- **No negative outputs**: Can't represent negative relationships
- **Debug tip**: Monitor % of dead neurons (always output 0)

**Softmax Pitfalls:**
- **Numerical overflow**: exp(x) explodes for large x (solved by max subtraction)
- **Dimension confusion**: Must apply along correct axis for batched data
- **Debug tip**: Verify outputs sum to 1.0 along correct dimension

**GELU Pitfalls:**
- **Approximation error**: Using wrong approximation constant
- **Speed**: Slightly slower than ReLU
- **Debug tip**: Compare outputs to reference implementation

### Performance Characteristics

**Computational Cost (relative to ReLU = 1.0):**
- ReLU: 1.0× (fastest - just comparison and max)
- Sigmoid: ~3×-4× (exponential computation)
- Tanh: ~3×-4× (two exponentials)
- GELU: ~4×-5× (exponential in approximation)
- Softmax: ~5×+ (exponentials + division across all elements)

**Memory Impact:**
- All activations: Minimal memory overhead (output same size as input)
- Softmax: Slightly higher (temporary buffers for exp and sum)
- For autograd (Module 06): Must cache inputs for backward pass

### Integration with TinyTorch

Your activation functions integrate seamlessly with other modules:

**Module 03 (Layers)**: Will use these activations
```python
# Coming in Module 03
class Linear:
    def __init__(self, in_features, out_features, activation=None):
        self.activation = activation  # Your ReLU, Sigmoid, etc.

    def forward(self, x):
        out = self.compute_linear(x)
        if self.activation:
            out = self.activation(out)  # Uses your forward()
        return out
```

**Module 06 (Autograd)**: Will add gradient computation
```python
# Coming in Module 06
class Sigmoid:
    def backward(self, grad):
        # ∂sigmoid/∂x = sigmoid(x) * (1 - sigmoid(x))
        return grad * self.output * (1 - self.output)
```
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Activations Transform Data

**What you built:** Five activation functions that introduce nonlinearity to neural networks.

**Why it matters:** Without activations, stacking layers would just be matrix multiplication—
a linear operation. ReLU's simple "zero out negatives" rule is what allows networks to learn
complex patterns like recognizing faces or understanding language.

In the next module, you'll combine these activations with Linear layers to build complete
neural network architectures. The nonlinearity you just implemented is the secret sauce!
"""

# %%
def demo_activations():
    """🎯 See how activations transform data."""
    print("🎯 AHA MOMENT: Activations Transform Data")
    print("=" * 45)

    # Test input with positive and negative values
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    print(f"Input: {x.data}")

    # ReLU - zeros out negatives
    relu = ReLU()
    relu_out = relu(x)
    print(f"ReLU:  {relu_out.data}  ← Negatives become 0!")

    # Sigmoid - squashes to (0, 1)
    sigmoid = Sigmoid()
    sigmoid_out = sigmoid(x)
    print(f"Sigmoid: {np.round(sigmoid_out.data, 2)}  ← Squashed to (0,1)")

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
- Built 5 core activation functions with distinct behaviors and use cases
- Implemented forward passes for Sigmoid, ReLU, Tanh, GELU, and Softmax
- Discovered how nonlinearity enables complex pattern learning
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your activation implementations enable neural network layers to learn complex, nonlinear patterns instead of just linear transformations.

Export with: `tito module complete 02`

**Next**: Module 03 will combine your Tensors and Activations to build complete neural network Layers!
"""
