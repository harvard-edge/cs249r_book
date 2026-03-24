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
# Module 06: Autograd - The Gradient Engine

Welcome to Module 06! Today you'll awaken the gradient engine and unlock automatic differentiation.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor operations, activations, layers, losses, and DataLoader
**You'll Build**: The autograd system that computes gradients automatically
**You'll Enable**: Learning! Training! The ability to optimize neural networks!

**Connection Map**:
```
Modules 01-05 → Autograd → Optimizers → Training
(forward pass)  (Module 06)  (Module 07)  (Module 08)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. **Enhance Tensor** with automatic differentiation capabilities
2. **Build computation graphs** that track operations for gradient flow
3. **Implement backward()** method for reverse-mode differentiation
4. **Create Function classes** for operation-specific gradient rules
5. **Test gradient correctness** with mathematical validation

**CRITICAL**: This module enhances the existing Tensor class - no new wrapper classes needed!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/06_autograd/autograd_dev.py`
**Building Side:** Code exports to `tinytorch.core.autograd`

```python
# How to use this module:
from tinytorch.core.autograd import Function, enable_autograd
```

**Why this matters:**
- **Learning:** Complete autograd system enabling automatic differentiation
- **Production:** PyTorch-style computational graph and backward pass
- **Consistency:** All gradient operations in core.autograd
- **Integration:** Enhances existing Tensor without breaking anything

Let's get started!
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.autograd
#| export

import numpy as np
from typing import Optional, List, Tuple
import sys
import os

from tinytorch.core.tensor import Tensor

# Constants for numerical differentiation
EPSILON = 1e-7  # Small perturbation for numerical gradient computation

# %% [markdown]
"""
## 💡 Introduction: What is Automatic Differentiation?

Automatic differentiation (autograd) is the magic that makes neural networks learn. Instead of manually computing gradients for every parameter, autograd tracks operations and automatically computes gradients via the chain rule.

### The Challenge
In previous modules, you implemented layers and loss functions. To train a model, you need:
```
Loss = f(W₃, f(W₂, f(W₁, x)))
∂Loss/∂W₁ = ?  ∂Loss/∂W₂ = ?  ∂Loss/∂W₃ = ?
```

Manual gradient computation becomes impossible for complex models with millions of parameters.

### The Solution: Computational Graphs
```
Forward Pass:  x → Linear₁ → ReLU → Linear₂ → Loss
Backward Pass: ∇x ← ∇Linear₁ ← ∇ReLU ← ∇Linear₂ ← ∇Loss
```

**Complete Autograd Process Visualization:**
```
┌─ FORWARD PASS ─────────────────────────────────────────────────┐
│                                                                │
│ x ──┬── W₁ ──┐                                                 │
│     │        ├──[Linear₁]──→ z₁ ──[ReLU]──→ a₁ ──┬── W₂ ──┐    │
│     └── b₁ ──┘                               │        ├─→ Loss │
│                                              └── b₂ ──┘        │
│                                                                │
└─ COMPUTATION GRAPH BUILT ──────────────────────────────────────┘
                             │
                             ▼
┌─ BACKWARD PASS ─────────────────────────────────────────────┐
│                                                             │
│∇x ←┬← ∇W₁ ←┐                                                │
│    │       ├←[Linear₁]←─ ∇z₁ ←[ReLU]← ∇a₁ ←┬← ∇W₂ ←┐        │
│    └← ∇b₁ ←┘                             │       ├← ∇Loss   │
│                                          └← ∇b₂ ←┘          │
│                                                             │
└─ GRADIENTS COMPUTED ────────────────────────────────────────┘

Key Insight: Each [operation] stores how to compute its backward pass.
The chain rule automatically flows gradients through the entire graph.
```

Each operation records how to compute its backward pass. The chain rule connects them all.
"""

# %% [markdown]
"""
## 📐 Foundations: The Chain Rule in Action

### Mathematical Foundation
For composite functions: f(g(x)), the derivative is:
```
df/dx = (df/dg) × (dg/dx)
```

### Computational Graph Example
```
Simple computation: L = (x * y + 5)²

Forward Pass:
  x=2 ──┐
        ├──[×]──→ z=6 ──[+5]──→ w=11 ──[²]──→ L=121
  y=3 ──┘

Backward Pass (Chain Rule in Action):
  ∂L/∂x = ∂L/∂w × ∂w/∂z × ∂z/∂x
        = 2w  ×  1  ×  y
        = 2(11) × 1 × 3 = 66

  ∂L/∂y = ∂L/∂w × ∂w/∂z × ∂z/∂y
        = 2w  ×  1  ×  x
        = 2(11) × 1 × 2 = 44

Gradient Flow Visualization:
  ∇x=66 ←──┐
           ├──[×]←── ∇z=22 ←──[+]←── ∇w=22 ←──[²]←── ∇L=1
  ∇y=44 ←──┘
```

### Memory Layout During Backpropagation
```
Computation Graph Memory Structure:
┌─────────────────────────────────────────────────────────┐
│ Forward Pass (stored for backward)                      │
├─────────────────────────────────────────────────────────┤
│ Node 1: x=2 (leaf, requires_grad=True) │ grad: None→66  │
│ Node 2: y=3 (leaf, requires_grad=True) │ grad: None→44  │
│ Node 3: z=x*y (MulFunction)            │ grad: None→22  │
│         saved: (x=2, y=3)              │ inputs: [x,y]  │
│ Node 4: w=z+5 (AddFunction)            │ grad: None→22  │
│         saved: (z=6, 5)                │ inputs: [z]    │
│ Node 5: L=w² (PowFunction)             │ grad: 1        │
│         saved: (w=11)                  │ inputs: [w]    │
└─────────────────────────────────────────────────────────┘

Memory Cost: 2× parameters (data + gradients) + graph overhead
```
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building the Autograd Engine

Let's implement the autograd system step by step. We'll enhance the existing Tensor class and create supporting infrastructure.

### The Function Architecture

Every differentiable operation needs two things:
1. **Forward pass**: Compute the result
2. **Backward pass**: Compute gradients for inputs

```
Function Class Design:
┌─────────────────────────────────────┐
│ Function (Base Class)               │
├─────────────────────────────────────┤
│ • saved_tensors    ← Store data     │
│ • apply()          ← Compute grads  │
└─────────────────────────────────────┘
          ↑
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
│  Add   │ │  Mul   │ │ Matmul │ │  Sum   │
│Backward│ │Backward│ │Backward│ │Backward│
└────────┘ └────────┘ └────────┘ └────────┘
```

Each operation inherits from Function and implements specific gradient rules.
"""

# %% [markdown]
"""
### Function Base Class - The Foundation of Autograd

The Function class is the foundation that makes autograd possible. Every differentiable operation (addition, multiplication, etc.) inherits from this class.

**Why Functions Matter:**
- They remember inputs needed for backward pass
- They implement gradient computation via apply()
- They connect to form computation graphs
- They enable the chain rule to flow gradients

**The Pattern:**
```
Forward:  inputs → Function.forward() → output
Backward: grad_output → Function.apply() → grad_inputs
```

This pattern enables the chain rule to flow gradients through complex computations.
"""

# %% nbgrader={"grade": false, "grade_id": "function-base", "solution": true}
#| export
class Function:
    """
    Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class and implement the apply() method.

    **Key Concepts:**
    - **saved_tensors**: Store inputs needed for backward pass
    - **apply()**: Compute gradients using chain rule

    **Example Usage:**
    ```python
    class AddBackward(Function):
        def apply(self, grad_output):
            # Addition distributes gradients equally
            return grad_output, grad_output
    ```
    """

    def __init__(self, *tensors):
        """
        Initialize function with input tensors.

        Args:
            *tensors: Input tensors that will be saved for backward pass
        """
        self.saved_tensors = tensors

    def apply(self, grad_output):
        """
        Compute gradients for inputs.

        Args:
            grad_output: Gradient flowing backward from the output

        Returns:
            Tuple of gradients for each input tensor

        **Must be implemented by subclasses**
        """
        raise NotImplementedError("Each Function must implement apply() method")

# %% [markdown]
"""
### Operation Functions - Implementing Gradient Rules

Now we'll implement specific operations that compute gradients correctly. Each operation has mathematical rules for how gradients flow backward.

**Gradient Flow Visualization:**
```
Addition (z = a + b):
    ∂z/∂a = 1    ∂z/∂b = 1

    a ──┐           grad_a ←──┐
        ├─[+]─→ z          ├─[+]←── grad_z
    b ──┘           grad_b ←──┘

Multiplication (z = a * b):
    ∂z/∂a = b    ∂z/∂b = a

    a ──┐           grad_a = grad_z * b
        ├─[×]─→ z
    b ──┘           grad_b = grad_z * a

Matrix Multiplication (Z = A @ B):
    ∂Z/∂A = grad_Z @ B.T
    ∂Z/∂B = A.T @ grad_Z

    A ──┐           grad_A = grad_Z @ B.T
        ├─[@]─→ Z
    B ──┘           grad_B = A.T @ grad_Z
```

Each operation stores the inputs it needs for computing gradients.
"""

# %% [markdown]
"""
### 🔍 Understanding Broadcasting in Gradients

Before implementing gradient operations, we need to understand a critical challenge:
**Broadcasting in Forward Pass vs. Gradient Reduction in Backward Pass**

#### The Broadcasting Problem

NumPy automatically broadcasts tensors of different shapes during forward operations:

```
Forward Pass (Broadcasting):
┌─────────────────────────────────────────────────────────────┐
│ Example: Adding bias to batched data                        │
│                                                              │
│ x:    (32, 128)  ← Batch of 32 samples, 128 features        │
│ bias: (128,)     ← Just 128 features (no batch dimension)   │
│                                                              │
│ Forward: y = x + bias                                        │
│          NumPy broadcasts bias from (128,) to (32, 128)     │
│          Result shape: (32, 128)                             │
└─────────────────────────────────────────────────────────────┘

Backward Pass (Gradient Reduction):
┌─────────────────────────────────────────────────────────────┐
│ grad_output: (32, 128)  ← Gradient from upstream            │
│                                                              │
│ grad_x:    (32, 128)    ← Same shape as x ✓                 │
│ grad_bias: (128,)       ← Must match bias shape!            │
│                                                              │
│ Problem: grad_output is (32, 128) but bias is (128,)        │
│ Solution: Sum gradients over batch dimension                │
│           grad_bias = grad_output.sum(axis=0)               │
│           Result: (128,) ✓                                   │
└─────────────────────────────────────────────────────────────┘
```

#### Why Gradient Reduction is Necessary

**Mathematical Intuition:**
When bias broadcasts to multiple samples, it contributes to each sample's loss.
The total gradient w.r.t. bias is the SUM of gradients from all samples.

**Concrete Example:**
```
x = [[1, 2],      bias = [0.1, 0.2]
     [3, 4]]

Forward:
  y[0] = [1, 2] + [0.1, 0.2] = [1.1, 2.2]  ← bias[0]=0.1 affects sample 0
  y[1] = [3, 4] + [0.1, 0.2] = [3.1, 4.2]  ← bias[0]=0.1 affects sample 1

Backward:
  grad_output = [[1, 1],   ← ∂Loss/∂y[0]
                 [1, 1]]   ← ∂Loss/∂y[1]
  
  grad_bias[0] = ∂Loss/∂bias[0] 
               = ∂Loss/∂y[0,0] + ∂Loss/∂y[1,0]  ← Chain rule: sum contributions
               = 1 + 1 = 2
  
  grad_bias = [2, 2]  ← Sum over batch dimension
```

#### Broadcasting Scenarios to Handle

```
Scenario 1: Batch Dimension Broadcasting
  x:    (32, 128)  +  bias: (128,)    →  y: (32, 128)
  grad: (32, 128)  →  grad_bias = sum(grad, axis=0) → (128,)

Scenario 2: Multiple Dimension Broadcasting  
  x:    (32, 10, 5)  +  y: (10, 1)    →  z: (32, 10, 5)
  grad: (32, 10, 5)  →  
    1. Sum over extra dim: sum(grad, axis=0) → (10, 5)
    2. Sum over singleton: sum(grad, axis=2, keepdims=True) → (10, 1)

Scenario 3: Scalar Broadcasting
  x:    (32, 128)  +  scalar: ()      →  y: (32, 128)
  grad: (32, 128)  →  grad_scalar = sum(grad) → scalar
```

#### The General Algorithm

To reduce gradient back to original input shape:
1. **Remove extra dimensions**: Sum over leading dimensions that weren't in input
2. **Collapse singleton dimensions**: Sum over dimensions where input had size 1
3. **Preserve shape**: Use keepdims=True when collapsing to maintain dimensionality

This ensures gradients flow correctly regardless of broadcasting patterns!
"""

# %% nbgrader={"grade": false, "grade_id": "broadcast-grad-helper", "solution": true}
#| export
def _reduce_broadcast_grad(grad, original_shape):
    """
    Reduce gradient to match original tensor shape after broadcasting.

    TODO: Implement gradient shape reduction for broadcast operations.

    APPROACH:
    1. Remove leading dimensions: while grad has more dims than original, sum axis=0
    2. Collapse singleton dimensions: where original had size 1, sum with keepdims=True

    EXAMPLE:
    >>> # Bias case: (32, 128) → (128,)
    >>> grad = np.ones((32, 128))
    >>> reduced = _reduce_broadcast_grad(grad, (128,))
    >>> reduced.shape  # (128,)

    >>> # Singleton case: (32, 10, 5) → (10, 1)
    >>> grad = np.ones((32, 10, 5))
    >>> reduced = _reduce_broadcast_grad(grad, (10, 1))
    >>> reduced.shape  # (10, 1)

    HINT: Two separate loops — one for leading dims, one for singleton dims.
    """
    ### BEGIN SOLUTION
    # Step 1: Remove leading dimensions that weren't in original tensor
    # Example: grad (32, 128) with original (128,) → sum over axis 0
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    
    # Step 2: Collapse dimensions where original had size 1
    # Example: grad (10, 5) with original (10, 1) → sum over axis 1 with keepdims
    for i in range(len(original_shape)):
        if original_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad
    ### END SOLUTION

# %% [markdown]
"""
### 🔬 Unit Test: Broadcast Gradient Reduction

This test validates our gradient reduction helper handles all broadcasting scenarios.

**What we're testing**: Correct shape reduction for gradients after broadcasting
**Why it matters**: Wrong gradient shapes cause crashes or silent bugs in training
**Expected**: Reduced gradients match original tensor shapes for all broadcasting patterns
"""

# %% nbgrader={"grade": true, "grade_id": "test-reduce-broadcast-grad", "locked": true, "points": 5}
def test_unit_reduce_broadcast_grad():
    """🔬 Test _reduce_broadcast_grad helper function."""
    print("🔬 Unit Test: _reduce_broadcast_grad...")
    
    # Test 1: Remove leading dimension
    grad = np.ones((32, 128))
    original_shape = (128,)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (128,), f"Expected (128,), got {reduced.shape}"
    assert np.allclose(reduced, np.ones(128) * 32), "Should sum over batch dimension"
    print("  ✓ Leading dimension reduction works")
    
    # Test 2: Collapse singleton dimension
    grad = np.ones((10, 5))
    original_shape = (10, 1)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 1), f"Expected (10, 1), got {reduced.shape}"
    assert np.allclose(reduced, np.ones((10, 1)) * 5), "Should sum over singleton axis"
    print("  ✓ Singleton dimension reduction works")
    
    # Test 3: Both operations
    grad = np.ones((32, 10, 5))
    original_shape = (10, 1)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 1), f"Expected (10, 1), got {reduced.shape}"
    expected = np.ones((10, 1)) * 32 * 5  # Sum over batch and last dim
    assert np.allclose(reduced, expected), "Should handle multiple reductions"
    print("  ✓ Multiple dimension reduction works")
    
    # Test 4: No broadcasting (should return unchanged)
    grad = np.ones((10, 5))
    original_shape = (10, 5)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 5), f"Expected (10, 5), got {reduced.shape}"
    assert np.allclose(reduced, grad), "Should return unchanged when shapes match"
    print("  ✓ No-op case works")
    
    # Test 5: Scalar case (reduce to scalar)
    grad = np.ones((32, 128))
    original_shape = ()
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (), f"Expected scalar, got {reduced.shape}"
    assert np.allclose(reduced, 32 * 128), "Should sum to scalar"
    print("  ✓ Scalar reduction works")
    
    print("✅ _reduce_broadcast_grad works correctly!")

if __name__ == "__main__":
    test_unit_reduce_broadcast_grad()

# %% [markdown]
"""
### AddBackward - Gradient Rules for Addition

Addition is the simplest gradient operation: gradients flow unchanged to both inputs.

**Mathematical Principle:**
```
If z = a + b, then:
∂z/∂a = 1  (gradient of z w.r.t. a)
∂z/∂b = 1  (gradient of z w.r.t. b)

By chain rule:
∂Loss/∂a = ∂Loss/∂z × ∂z/∂a = grad_output × 1 = grad_output
∂Loss/∂b = ∂Loss/∂z × ∂z/∂b = grad_output × 1 = grad_output
```

**Broadcasting Challenge:**
When tensors have different shapes, NumPy broadcasts automatically in forward pass,
but we must "unbroadcast" gradients in backward pass to match original shapes.
"""

# %% nbgrader={"grade": false, "grade_id": "add-backward", "solution": true}
#| export
class AddBackward(Function):
    """
    Gradient computation for tensor addition.

    **Mathematical Rule:** If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1

    **Key Insight:** Addition distributes gradients equally to both inputs.
    The gradient flowing backward is passed unchanged to each input.

    **Broadcasting Handling:** When input shapes differ due to broadcasting,
    we sum gradients appropriately to match original tensor shapes.
    """

    def apply(self, grad_output):
        """
        Compute gradients for addition.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) for the two inputs

        **Mathematical Foundation:**
        - ∂(a+b)/∂a = 1 → grad_a = grad_output
        - ∂(a+b)/∂b = 1 → grad_b = grad_output

        TODO: Implement gradient computation for addition operation.

        APPROACH:
        1. Extract input tensors from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if it requires gradients:
           - Set grad_a = grad_output
           - Use _reduce_broadcast_grad() to handle shape mismatch if needed
        4. For second input (b): if it requires gradients:
           - Set grad_b = grad_output
           - Use _reduce_broadcast_grad() to handle shape mismatch if needed
        5. Return tuple (grad_a, grad_b)

        EXAMPLE (Same Shape):
        >>> a = Tensor([1, 2, 3], requires_grad=True)
        >>> b = Tensor([4, 5, 6], requires_grad=True)
        >>> z = a + b  # z = [5, 7, 9]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # Result: grad_a = [1, 1, 1], grad_b = [1, 1, 1]

        EXAMPLE (Broadcasting):
        >>> x = Tensor(np.ones((32, 128)), requires_grad=True)
        >>> bias = Tensor(np.zeros(128), requires_grad=True)
        >>> y = x + bias  # bias broadcasts to (32, 128)
        >>> # During backward: grad_output shape (32, 128)
        >>> # grad_bias must be reduced to (128,) by summing over batch
        >>> # Result: grad_x shape (32, 128), grad_bias shape (128,)

        HINTS:
        - Addition distributes gradients equally (derivative of a+b w.r.t. both is 1)
        - Use _reduce_broadcast_grad(grad, tensor.data.shape) to handle broadcasting
        - Check isinstance(tensor, Tensor) and tensor.requires_grad before computing
        - Return None for inputs that don't require gradients
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        # Gradient for second input
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b
        ### END SOLUTION

# %% [markdown]
"""
### MulBackward - Gradient Rules for Element-wise Multiplication

Element-wise multiplication follows the product rule of calculus.

**Mathematical Principle:**
```
If z = a * b (element-wise), then:
∂z/∂a = b  (gradient w.r.t. a equals the other input)
∂z/∂b = a  (gradient w.r.t. b equals the other input)

By chain rule:
∂Loss/∂a = grad_output * b
∂Loss/∂b = grad_output * a
```

**Visual Example:**
```
Forward:  a=[2,3] * b=[4,5] = z=[8,15]
Backward: grad_z=[1,1]
          grad_a = grad_z * b = [1,1] * [4,5] = [4,5]
          grad_b = grad_z * a = [1,1] * [2,3] = [2,3]
```
"""

# %% nbgrader={"grade": false, "grade_id": "mul-backward", "solution": true}
#| export
class MulBackward(Function):
    """
    Gradient computation for tensor multiplication.

    **Mathematical Rule:** If z = a * b, then ∂z/∂a = b and ∂z/∂b = a

    **Key Insight:** Each input's gradient equals the gradient output
    multiplied by the OTHER input's value (product rule).

    **Applications:** Used in weight scaling, binary masking,
    and anywhere element-wise multiplication occurs.
    """

    def apply(self, grad_output):
        """
        Compute gradients for multiplication.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) for the two inputs

        **Mathematical Foundation:**
        - ∂(a*b)/∂a = b → grad_a = grad_output * b
        - ∂(a*b)/∂b = a → grad_b = grad_output * a

        TODO: Implement gradient computation for element-wise multiplication.

        APPROACH:
        1. Extract input tensors a, b from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if requires_grad:
           - Compute grad_a = grad_output * b
           - Use _reduce_broadcast_grad() to handle shape mismatch
        4. For second input (b): if requires_grad:
           - Compute grad_b = grad_output * a
           - Use _reduce_broadcast_grad() to handle shape mismatch
        5. Handle both Tensor and scalar cases for b
        6. Return tuple (grad_a, grad_b)

        EXAMPLE:
        >>> a = Tensor([2, 3], requires_grad=True)
        >>> b = Tensor([4, 5], requires_grad=True)
        >>> z = a * b  # z = [8, 15]
        >>> # During backward: grad_output = [1, 1]
        >>> # grad_a = [1, 1] * [4, 5] = [4, 5]
        >>> # grad_b = [1, 1] * [2, 3] = [2, 3]

        HINTS:
        - Product rule: each input's gradient equals grad_output times the OTHER input
        - Use _reduce_broadcast_grad() to handle broadcasting correctly
        - Check if b is a Tensor or scalar before accessing .data
        - Use b.data if Tensor, or b directly if scalar
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output * b
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        # Gradient for second input: grad_output * a
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b
        ### END SOLUTION

# %% [markdown]
"""
### SubBackward - Gradient Rules for Subtraction

Subtraction is mathematically simple but important for operations like normalization.

**Mathematical Principle:**
```
If z = a - b, then:
∂z/∂a = 1
∂z/∂b = -1
```

**Key Insight:** Gradient flows forward to the first operand, but **negated** to the second.
This is crucial for operations like centering data (`x - mean`).
"""

# %% nbgrader={"grade": false, "grade_id": "sub-backward", "solution": true}
#| export
class SubBackward(Function):
    """
    Gradient computation for tensor subtraction.

    **Mathematical Rule:** If z = a - b, then ∂z/∂a = 1 and ∂z/∂b = -1
    """

    def apply(self, grad_output):
        """
        Compute gradients for subtraction.

        Returns:
            Tuple of (grad_a, grad_b) where grad_b is negated

        TODO: Implement gradient computation for subtraction operation.

        APPROACH:
        1. Extract input tensors from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if requires_grad:
           - Set grad_a = grad_output
           - Use _reduce_broadcast_grad() to handle shape mismatch
        4. For second input (b): if requires_grad:
           - Set grad_b = -grad_output (note the negative!)
           - Use _reduce_broadcast_grad() to handle shape mismatch
        5. Return tuple (grad_a, grad_b)

        EXAMPLE:
        >>> a = Tensor([5, 7], requires_grad=True)
        >>> b = Tensor([2, 3], requires_grad=True)
        >>> z = a - b  # z = [3, 4]
        >>> # During backward: grad_output = [1, 1]
        >>> # grad_a = [1, 1], grad_b = -[1, 1] = [-1, -1]

        HINTS:
        - ∂(a-b)/∂a = 1 (gradient flows unchanged to first operand)
        - ∂(a-b)/∂b = -1 (gradient is negated for second operand)
        - Use _reduce_broadcast_grad() to handle broadcasting correctly
        - The negative sign is crucial for correct gradient flow
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output  # ∂(a-b)/∂a = 1
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output  # ∂(a-b)/∂b = -1 (note the negative!)
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b
        ### END SOLUTION

# %% [markdown]
"""
### DivBackward - Gradient Rules for Division

Division requires the quotient rule from calculus.

**Mathematical Principle:**
```
If z = a / b, then:
∂z/∂a = 1/b
∂z/∂b = -a/b²
```

**Quotient Rule:** For z = f/g, dz = (g·df - f·dg)/g²
"""

# %% nbgrader={"grade": false, "grade_id": "div-backward", "solution": true}
#| export
class DivBackward(Function):
    """
    Gradient computation for tensor division.

    **Mathematical Rule:** If z = a / b, then:
    - ∂z/∂a = 1/b
    - ∂z/∂b = -a/b²
    """

    def apply(self, grad_output):
        """
        Compute gradients for division using quotient rule.

        Returns:
            Tuple of (grad_a, grad_b)

        TODO: Implement gradient computation for division operation.

        APPROACH:
        1. Extract input tensors from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if requires_grad:
           - Compute grad_a = grad_output / b
           - Use _reduce_broadcast_grad() to handle shape mismatch
        4. For second input (b): if requires_grad:
           - Compute grad_b = -grad_output * a / (b²)
           - Use _reduce_broadcast_grad() to handle shape mismatch
        5. Handle both Tensor and scalar cases for b
        6. Return tuple (grad_a, grad_b)

        EXAMPLE:
        >>> a = Tensor([8.0, 12.0], requires_grad=True)
        >>> b = Tensor([2.0, 3.0], requires_grad=True)
        >>> z = a / b  # z = [4.0, 4.0]
        >>> # During backward: grad_output = [1, 1]
        >>> # grad_a = [1, 1] / [2, 3] = [0.5, 0.333...]
        >>> # grad_b = -[1, 1] * [8, 12] / ([2, 3]²) = [-2, -1.333...]

        HINTS:
        - Quotient rule: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
        - Use _reduce_broadcast_grad() to handle broadcasting correctly
        - Use b.data if Tensor, or b directly if scalar
        - b² means b.data ** 2 for tensors
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            # ∂(a/b)/∂a = 1/b
            if isinstance(b, Tensor):
                grad_a = grad_output / b.data
            else:
                grad_a = grad_output / b
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        if isinstance(b, Tensor) and b.requires_grad:
            # ∂(a/b)/∂b = -a/b²
            grad_b = -grad_output * a.data / (b.data ** 2)
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b
        ### END SOLUTION

# %% [markdown]
"""
### MatmulBackward - Gradient Rules for Matrix Multiplication

Matrix multiplication has more complex gradient rules based on matrix calculus.

**Mathematical Principle:**
```
If Z = A @ B (matrix multiplication), then:
∂Z/∂A = grad_Z @ B.T
∂Z/∂B = A.T @ grad_Z
```

**Why These Rules Work:**
```
For element Z[i,j] = Σ_k A[i,k] * B[k,j]
∂Z[i,j]/∂A[i,k] = B[k,j]  ← This gives us grad_Z @ B.T
∂Z[i,j]/∂B[k,j] = A[i,k]  ← This gives us A.T @ grad_Z
```

**Dimension Analysis:**
```
Forward:  A(m×k) @ B(k×n) = Z(m×n)
Backward: grad_Z(m×n) @ B.T(n×k) = grad_A(m×k) ✓
          A.T(k×m) @ grad_Z(m×n) = grad_B(k×n) ✓
```
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-backward", "solution": true}
#| export
class MatmulBackward(Function):
    """
    Gradient computation for matrix multiplication.

    **Mathematical Rule:** If Z = A @ B, then:
    - ∂Z/∂A = grad_Z @ B.T
    - ∂Z/∂B = A.T @ grad_Z

    **Key Insight:** Matrix multiplication gradients involve transposing
    one input and multiplying with the gradient output.

    **Applications:** Core operation in neural networks for computing outputs
    in linear layers and combining feature representations.
    """

    def apply(self, grad_output):
        """
        Compute gradients for matrix multiplication.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) for the two matrix inputs

        **Mathematical Foundation:**
        - ∂(A@B)/∂A = grad_output @ B.T
        - ∂(A@B)/∂B = A.T @ grad_output

        **Batched Operation:** For 3D+ tensors, we transpose only the last two
        dimensions using np.swapaxes, preserving batch dimensions.

        TODO: Implement gradient computation for matrix multiplication.

        APPROACH:
        1. Extract input tensors a, b from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a):
           - Transpose b: use np.swapaxes(b.data, -2, -1) for batched tensors
           - Compute grad_a = grad_output @ b_T using np.matmul
        4. For second input (b):
           - Transpose a: use np.swapaxes(a.data, -2, -1) for batched tensors
           - Compute grad_b = a_T @ grad_output using np.matmul
        5. Return tuple (grad_a, grad_b)

        EXAMPLE:
        >>> A = Tensor([[1, 2]], requires_grad=True)  # (1, 2)
        >>> B = Tensor([[3], [4]], requires_grad=True)  # (2, 1)
        >>> C = A @ B  # (1, 1), result = [[11]]
        >>> # During backward: grad_output = [[1]]
        >>> # grad_A = [[1]] @ [[3, 4]] = [[3, 4]]
        >>> # grad_B = [[1, 2]].T @ [[1]] = [[1], [2]]

        HINTS:
        - Matrix multiplication gradients involve transposing one input
        - Use np.swapaxes(array, -2, -1) to transpose last two dimensions
        - This preserves batch dimensions for 3D+ tensors
        - Use np.matmul for the actual matrix multiplication
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output @ b.T
        if isinstance(a, Tensor) and a.requires_grad:
            # For batched tensors, transpose only last two dims
            if b.data.ndim >= 2:
                b_T = np.swapaxes(b.data, -2, -1)
            else:
                b_T = b.data.T
            grad_a = np.matmul(grad_output, b_T)

        # Gradient for second input: a.T @ grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            # For batched tensors, transpose only last two dims
            if a.data.ndim >= 2:
                a_T = np.swapaxes(a.data, -2, -1)
            else:
                a_T = a.data.T
            grad_b = np.matmul(a_T, grad_output)

        return grad_a, grad_b
        ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "transpose-backward", "solution": true}
#| export
class TransposeBackward(Function):
    """
    Gradient computation for transpose operation.

    **Mathematical Rule:** If Y = X.T, then:
    - ∂Y/∂X = grad_Y.T

    **Key Insight:** The gradient of transpose is just transpose the gradient!
    This is because transpose is a linear operation that just rearranges elements.

    **Applications:** Used in weight gradient computation (W.T), data reshaping,
    and any operation that needs to swap matrix dimensions.
    """

    def __init__(self, tensor, dim0, dim1):
        """
        Args:
            tensor: Input tensor
            dim0: First dimension to swap (None for default)
            dim1: Second dimension to swap (None for default)
        """
        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        """
        Compute gradient for transpose.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - ∂(X.T)/∂X = grad_output.T
        - Just transpose the gradient back!

        TODO: Implement gradient computation for transpose operation.

        APPROACH:
        1. Extract input tensor x from self.saved_tensors
        2. Initialize grad_x to None
        3. If x requires gradients:
           - Check if default transpose (last two dims) or specific dims
           - For default: swap last two dimensions of grad_output
           - For specific dims: swap the specified dimensions back
        4. Return tuple (grad_x,)

        EXAMPLE:
        >>> X = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> Y = X.transpose()  # [[1, 3], [2, 4]]
        >>> # During backward: grad_output = [[a, b], [c, d]]
        >>> # grad_X = grad_output.T = [[a, c], [b, d]]

        HINTS:
        - Transpose gradient is simply transposing the gradient back
        - Use np.transpose(grad_output, axes) to specify axis order
        - For default transpose, swap axes[-2] and axes[-1]
        - Return as single-element tuple: (grad_x,)
        """
        ### BEGIN SOLUTION
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Transpose gradient using the same dims
            if self.dim0 is None and self.dim1 is None:
                # Default: transpose last two dimensions
                if grad_output.ndim < 2:
                    grad_x = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    grad_x = np.transpose(grad_output, axes)
            else:
                # Specific dimensions: swap them back
                axes = list(range(grad_output.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_x = np.transpose(grad_output, axes)

        return (grad_x,)
        ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "permute-backward", "solution": true}
#| export
class PermuteBackward(Function):
    """
    Gradient computation for arbitrary axis permutation (general transpose).

    **Mathematical Rule:** If Y = X.permute(axes), then:
    - ∂Y/∂X = grad_Y.permute(inverse_axes)

    **Example:** If axes = (0, 2, 1, 3), the inverse is (0, 2, 1, 3) (self-inverse).
    More generally, if axes = (2, 0, 1), the inverse is (1, 2, 0).

    **Key Insight:** To reverse a permutation, we need to know where each axis went.
    If axis i went to position axes[i], then in the inverse, position axes[i] should go to i.

    **Applications:** Rearranging tensor dimensions for different computation patterns (e.g., swapping batch and feature dimensions).
    """

    def __init__(self, tensor, axes):
        """
        Args:
            tensor: Input tensor
            axes: Tuple of axis indices defining the permutation
        """
        super().__init__(tensor)
        self.axes = axes
        # Compute inverse permutation: if axes[i] = j, then inverse_axes[j] = i
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_output):
        """
        Compute gradient for permutation.

        The gradient is permuted back using the inverse permutation.

        **Mathematical Foundation:**
        - ∂(X.permute(axes))/∂X = grad_output.permute(inverse_axes)

        TODO: Implement gradient computation for permutation operation.

        APPROACH:
        1. Extract input tensor x from self.saved_tensors
        2. Initialize grad_x to None
        3. If x requires gradients:
           - Permute grad_output using self.inverse_axes
           - Use np.transpose(grad_output, self.inverse_axes)
        4. Return tuple (grad_x,)

        EXAMPLE:
        >>> X = Tensor([[[1, 2], [3, 4]]], requires_grad=True)  # (1, 2, 2)
        >>> Y = X.permute((0, 2, 1))  # Swap last two dims → (1, 2, 2)
        >>> # During backward: inverse_axes computed in __init__
        >>> # grad_X = np.transpose(grad_output, inverse_axes)

        HINTS:
        - Inverse permutation is precomputed in __init__ using np.argsort
        - Simply apply np.transpose with inverse_axes
        - Return as single-element tuple: (grad_x,)
        """
        ### BEGIN SOLUTION
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Permute gradient back to original axis order
            grad_x = np.transpose(grad_output, self.inverse_axes)

        return (grad_x,)
        ### END SOLUTION

#| export

class SliceBackward(Function):
    """
    Gradient computation for tensor slicing/indexing operations.

    **Mathematical Rule:** If Y = X[key], then:
    - ∂Loss/∂X[key] = grad_output
    - ∂Loss/∂X[other positions] = 0

    **Key Insight:** Slicing is a masking operation. The backward
    places gradients back into the original tensor positions, with
    zeros everywhere else.

    **Applications:** Sequence slicing, batch selection, and selecting subsets
    of tensor data for computation.

    **Examples:**
    >>> x = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    >>> y = x[:3]  # Slice first 3 elements
    >>> loss = y.sum()
    >>> loss.backward()
    >>> # x.grad = [1, 1, 1, 0, 0] - gradients only for sliced positions
    """

    def __init__(self, tensor, key):
        """
        Args:
            tensor: Original tensor being sliced
            key: Slicing key (index, slice, tuple of slices, etc.)
        """
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_output):
        """
        Compute gradient for slicing operation.

        Args:
            grad_output: Gradient flowing backward from sliced output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - Slicing extracts a subset of elements
        - Backward scatters gradients back to original positions
        - Unsliced positions receive zero gradient

        **Example:**
        If X = [a, b, c, d, e] and Y = X[1:4] = [b, c, d]
        Then dL/dX = [0, dL/db, dL/dc, dL/dd, 0]

        TODO: Implement gradient computation for slicing/indexing operation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. Initialize grad_input to None
        3. If tensor requires gradients:
           - Create zeros array: grad_input = np.zeros(self.original_shape)
           - Place gradients back: grad_input[self.key] = grad_output
        4. Return tuple (grad_input,)

        EXAMPLE:
        >>> X = Tensor([1, 2, 3, 4, 5], requires_grad=True)
        >>> Y = X[:3]  # Slice first 3 elements → [1, 2, 3]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # grad_X = [1, 1, 1, 0, 0] (gradients only for sliced positions)

        HINTS:
        - Create zero gradient array with original tensor shape
        - Use fancy indexing: grad_input[self.key] = grad_output
        - This automatically handles all slice types (single index, ranges, tuples)
        - Return as single-element tuple: (grad_input,)
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors
        grad_input = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Create gradient array with same shape as original tensor
            grad_input = np.zeros(self.original_shape, dtype=np.float32)

            # Place gradients back into the sliced positions
            # This is the inverse of the forward slicing operation
            grad_input[self.key] = grad_output

        return (grad_input,)
        ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "reshape-backward", "solution": true}
#| export
class ReshapeBackward(Function):
    """
    Gradient computation for reshape operation.

    **Mathematical Rule:** If Y = X.reshape(new_shape), then:
    - ∂Y/∂X = grad_Y.reshape(X.shape)

    **Key Insight:** Reshape just rearranges the same elements.
    The gradient is simply reshaped back to the original shape!

    **Applications:** Flattening tensors for linear layers, reshaping
    between convolutional and dense layers.
    """

    def __init__(self, tensor, original_shape):
        """
        Args:
            tensor: Input tensor
            original_shape: Shape before reshape
        """
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        """
        Compute gradient for reshape.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - ∂(X.reshape(...))/∂X = grad_output.reshape(X.shape)
        - Just reshape the gradient back!

        TODO: Implement gradient computation for reshape operation.

        APPROACH:
        1. Extract input tensor x from self.saved_tensors
        2. Initialize grad_x to None
        3. If x requires gradients:
           - Reshape grad_output back to original shape
           - Use grad_output.reshape(self.original_shape)
        4. Return tuple (grad_x,)

        EXAMPLE:
        >>> X = Tensor([[1, 2], [3, 4]], requires_grad=True)  # (2, 2)
        >>> Y = X.reshape(4)  # [1, 2, 3, 4]
        >>> # During backward: grad_output = [1, 1, 1, 1]
        >>> # grad_X = grad_output.reshape((2, 2)) = [[1, 1], [1, 1]]

        HINTS:
        - Reshape just rearranges elements, doesn't change values
        - Simply reshape gradient back to original shape
        - Use .reshape() method on grad_output numpy array
        - Return as single-element tuple: (grad_x,)
        """
        ### BEGIN SOLUTION
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Reshape gradient back to original shape
            grad_x = grad_output.reshape(self.original_shape)

        return (grad_x,)
        ### END SOLUTION

# %% [markdown]
"""
### SumBackward - Gradient Rules for Reduction Operations

Sum operations reduce tensor dimensions, so gradients must be broadcast back.

**Mathematical Principle:**
```
If z = sum(a), then ∂z/∂a[i] = 1 for all i
Gradient is broadcasted from scalar result back to input shape.
```

**Gradient Broadcasting Examples:**
```
Case 1: Full sum
  Forward:  a=[1,2,3] → sum() → z=6 (scalar)
  Backward: grad_z=1 → broadcast → grad_a=[1,1,1]

Case 2: Axis sum
  Forward:  a=[[1,2],[3,4]] → sum(axis=0) → z=[4,6]
  Backward: grad_z=[1,1] → broadcast → grad_a=[[1,1],[1,1]]
```
"""

# %% nbgrader={"grade": false, "grade_id": "sum-backward", "solution": true}
#| export
class SumBackward(Function):
    """
    Gradient computation for tensor sum.

    **Mathematical Rule:** If z = sum(a), then ∂z/∂a[i] = 1 for all i

    **Key Insight:** Sum distributes the gradient equally to all input elements.
    The gradient is broadcast from the reduced output back to input shape.

    **Applications:** Used in loss functions, mean operations, and
    anywhere tensor reduction occurs.
    """

    def __init__(self, tensor, axis=None, keepdims=False):
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def apply(self, grad_output):
        """
        Compute gradients for sum operation.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple containing gradient for the input tensor

        **Mathematical Foundation:**
        - ∂sum(a)/∂a[i] = 1 → grad_a = ones_like(a) * grad_output

        TODO: Implement gradient computation for sum reduction operation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Create ones array: np.ones_like(tensor.data)
           - Multiply by grad_output: ones * grad_output
           - Return as tuple: (grad_tensor,)
        3. Else return (None,)

        EXAMPLE:
        >>> X = Tensor([1, 2, 3], requires_grad=True)
        >>> Y = X.sum()  # Y = 6 (scalar)
        >>> # During backward: grad_output = 1 (scalar)
        >>> # grad_X = [1, 1, 1] * 1 = [1, 1, 1]

        HINTS:
        - Sum distributes gradient equally to all elements
        - Use np.ones_like(tensor.data) to create gradient template
        - Multiply ones by grad_output (broadcasting handles scalar/tensor)
        - Return as single-element tuple: (grad_result,)
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # For axis-reduced sums, expand grad_output back along the summed
            # axis before broadcasting, so each row/column gets its own gradient.
            if self.axis is not None and not self.keepdims:
                grad_output = np.expand_dims(grad_output, axis=self.axis)
            return np.ones_like(tensor.data) * grad_output,
        return None,
        ### END SOLUTION

# %% [markdown]
"""
### 🔬 Unit Test: Function Classes

This test validates our Function classes compute gradients correctly.

**What we're testing**: Forward and backward passes for each operation
**Why it matters**: These are the building blocks of autograd
**Expected**: Correct gradients that satisfy mathematical definitions
"""

# %% nbgrader={"grade": true, "grade_id": "test-function-classes", "locked": true, "points": 15}
def test_unit_function_classes():
    """🔬 Test Function classes."""
    print("🔬 Unit Test: Function Classes...")

    # Note: We manually set requires_grad here because enable_autograd() hasn't been
    # called yet. This tests the Backward classes in isolation before the full
    # autograd system is enabled.

    # Test AddBackward
    a = Tensor([1, 2, 3])
    a.requires_grad = True  # Set manually (enable_autograd not called yet)
    b = Tensor([4, 5, 6])
    b.requires_grad = True
    add_func = AddBackward(a, b)
    grad_output = np.array([1, 1, 1])
    grad_a, grad_b = add_func.apply(grad_output)
    assert np.allclose(grad_a, grad_output), f"AddBackward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, grad_output), f"AddBackward grad_b failed: {grad_b}"

    # Test MulBackward
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(grad_output)
    assert np.allclose(grad_a, b.data), f"MulBackward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, a.data), f"MulBackward grad_b failed: {grad_b}"

    # Test MatmulBackward
    a_mat = Tensor([[1, 2], [3, 4]])
    a_mat.requires_grad = True
    b_mat = Tensor([[5, 6], [7, 8]])
    b_mat.requires_grad = True
    matmul_func = MatmulBackward(a_mat, b_mat)
    grad_output = np.ones((2, 2))
    grad_a, grad_b = matmul_func.apply(grad_output)
    assert grad_a.shape == a_mat.shape, f"MatmulBackward grad_a shape: {grad_a.shape}"
    assert grad_b.shape == b_mat.shape, f"MatmulBackward grad_b shape: {grad_b.shape}"

    print("✅ Function classes work correctly!")

# NOTE: test_unit_function_classes() is called from test_module() which runs AFTER
# enable_autograd(). Do NOT call it here - Tensor class doesn't have requires_grad yet!
# See GitHub issue #1128 for details.

# %% [markdown]
"""
### 🔬 Unit Test: Broadcasting in Gradients
This test validates that gradient reduction works correctly when operations
involve broadcasting. This is crucial for real-world training where bias terms,
normalization parameters, and other operations broadcast over batches.

**What we're testing**: Gradient shape correctness with broadcasting
**Why it matters**: Without proper reduction, bias gradients would have wrong shapes
**Expected**: Gradients match original tensor shapes after reduction
"""

# %% nbgrader={"grade": true, "grade_id": "test-broadcast-gradients", "locked": true, "points": 10}
def test_unit_broadcast_gradients():
    """🔬 Test gradient broadcasting reduction."""
    print("🔬 Unit Test: Broadcasting in Gradients...")
    
    # Scenario 1: Bias-like broadcasting (most common case)
    # Shape: (batch, features) + (features,) → (batch, features)
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    bias = Tensor(np.ones(3), requires_grad=True)
    
    add_func = AddBackward(x, bias)
    grad_output = np.ones((4, 3))
    grad_x, grad_bias = add_func.apply(grad_output)
    
    # Check shapes
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    assert grad_bias.shape == bias.data.shape, \
        f"Expected grad_bias shape {bias.data.shape}, got {grad_bias.shape}"
    
    # Check values: bias gradient should sum over batch dimension
    expected_bias_grad = np.ones(3) * 4  # Sum of 4 ones = 4
    assert np.allclose(grad_bias, expected_bias_grad), \
        f"Expected bias grad {expected_bias_grad}, got {grad_bias}"
    
    print("  ✓ Bias-like broadcasting works")
    
    # Scenario 2: Scalar broadcasting
    # Shape: (3, 4) + scalar → (3, 4)
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    scalar_val = 5.0
    
    add_func = AddBackward(x, scalar_val)
    grad_output = np.ones((3, 4))
    grad_x, grad_scalar = add_func.apply(grad_output)
    
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    # Scalar gradients don't get reduced (not a Tensor)
    
    print("  ✓ Scalar broadcasting works")
    
    # Scenario 3: Multiple dimension broadcasting
    # Shape: (32, 10, 5) + (10, 1) → (32, 10, 5)
    x = Tensor(np.random.randn(32, 10, 5), requires_grad=True)
    y = Tensor(np.random.randn(10, 1), requires_grad=True)
    
    mul_func = MulBackward(x, y)
    grad_output = np.ones((32, 10, 5))
    grad_x, grad_y = mul_func.apply(grad_output)
    
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    assert grad_y.shape == y.data.shape, \
        f"Expected grad_y shape {y.data.shape}, got {grad_y.shape}"
    
    print("  ✓ Multi-dimension broadcasting works")
    
    # Scenario 4: Test all operations (Add, Mul, Sub, Div)
    a = Tensor(np.random.randn(8, 16), requires_grad=True)
    b = Tensor(np.random.randn(16), requires_grad=True)
    
    # Test Addition
    add_func = AddBackward(a, b)
    grad_a, grad_b = add_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"AddBackward: Expected (16,), got {grad_b.shape}"
    
    # Test Multiplication
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"MulBackward: Expected (16,), got {grad_b.shape}"
    
    # Test Subtraction
    sub_func = SubBackward(a, b)
    grad_a, grad_b = sub_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"SubBackward: Expected (16,), got {grad_b.shape}"
    
    # Test Division
    div_func = DivBackward(a, b)
    grad_a, grad_b = div_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"DivBackward: Expected (16,), got {grad_b.shape}"
    
    print("  ✓ All operations handle broadcasting correctly")
    
    # Scenario 5: Real-world case - Linear layer gradient
    # Simulates: output = input @ weight + bias
    # where bias is (out_features,) and output is (batch, out_features)
    batch_size, out_features = 32, 128
    output_grad = np.random.randn(batch_size, out_features)
    bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    # In real Linear layer, bias gradient comes from output gradient
    add_func = AddBackward(
        Tensor(np.zeros((batch_size, out_features)), requires_grad=True),
        bias
    )
    _, grad_bias = add_func.apply(output_grad)
    
    assert grad_bias.shape == (out_features,), \
        f"Linear layer bias: Expected ({out_features},), got {grad_bias.shape}"
    
    # Verify gradient is sum over batch
    expected = output_grad.sum(axis=0)
    assert np.allclose(grad_bias, expected), \
        "Bias gradient should equal sum over batch dimension"
    
    print("  ✓ Real-world Linear layer scenario works")
    
    print("✅ Broadcasting gradient tests pass!")

if __name__ == "__main__":
    test_unit_broadcast_gradients()

# %% [markdown]
"""
## 🏗️ Enhancing Tensor with Autograd Capabilities

Now we'll enhance the existing Tensor class to use these gradient functions and build computation graphs automatically.

**Computation Graph Formation:**
```
Before Autograd:             After Autograd:
  x → operation → y           x → [Function] → y
                                     ↓
                               Stores operation
                               for backward pass
```

**The Enhancement Strategy:**
1. **Add backward() method** - Triggers gradient computation
2. **Enhance operations** - Replace simple ops with gradient-tracking versions
3. **Track computation graphs** - Each tensor remembers how it was created
4. **Maintain compatibility** - All existing code continues to work

**Critical Design Decision:**
We enhance the EXISTING Tensor class rather than creating a new one.
This means:
- ✅ All previous modules continue working unchanged
- ✅ No import changes needed
- ✅ Gradients are "opt-in" via requires_grad=True
- ✅ No confusion between Tensor types
"""

# %% [markdown]
"""
### The enable_autograd() Function

This function is the magic that brings gradients to life! It enhances the existing Tensor class with autograd capabilities by:

1. **Monkey-patching operations** - Replaces `__add__`, `__mul__`, etc. with gradient-aware versions
2. **Adding backward() method** - Implements reverse-mode automatic differentiation
3. **Maintaining compatibility** - All existing code continues to work unchanged

**The Pattern:**
```
Original: x + y → simple addition
Enhanced: x + y → addition + gradient tracking (if requires_grad=True)
```

This approach follows PyTorch 2.0 style - clean, modern, and educational.
"""

# %% nbgrader={"grade": false, "grade_id": "relu-backward", "solution": true}
#| export
class ReLUBackward(Function):
    """
    Gradient computation for ReLU activation.

    ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    """

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """
        Compute gradient for ReLU.

        TODO: Implement gradient computation for ReLU activation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Compute ReLU mask: (tensor.data > 0).astype(np.float32)
           - Multiply grad_output by mask: grad_output * relu_grad
           - Return as tuple: (result,)
        3. Else return (None,)

        EXAMPLE:
        >>> X = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> Y = relu(X)  # [0, 0, 0, 1, 2]
        >>> # During backward: grad_output = [1, 1, 1, 1, 1]
        >>> # relu_mask = [0, 0, 0, 1, 1] (1 where x > 0)
        >>> # grad_X = [0, 0, 0, 1, 1]

        HINTS:
        - ReLU derivative: 1 if x > 0, else 0
        - Use boolean mask: tensor.data > 0
        - Convert to float32 for gradient computation
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # ReLU gradient: 1 if x > 0, else 0
            relu_grad = (tensor.data > 0).astype(np.float32)
            return grad_output * relu_grad,
        return None,
        ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "sigmoid-backward", "solution": true}
#| export
class SigmoidBackward(Function):
    """
    Gradient computation for sigmoid activation.

    Sigmoid: σ(x) = 1/(1 + exp(-x))
    Derivative: σ'(x) = σ(x) * (1 - σ(x))
    """

    def __init__(self, input_tensor, output_tensor):
        """
        Initialize with both input and output.

        Args:
            input_tensor: Original input to sigmoid
            output_tensor: Output of sigmoid (saves recomputation)
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        """
        Compute gradient for sigmoid.

        TODO: Implement gradient computation for sigmoid activation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Use saved output: σ(x) = self.output_data
           - Compute sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
           - Multiply by grad_output: grad_output * sigmoid_grad
           - Return as tuple: (result,)
        3. Else return (None,)

        EXAMPLE:
        >>> X = Tensor([0.0], requires_grad=True)
        >>> Y = sigmoid(X)  # Y = 0.5
        >>> # During backward: grad_output = 1
        >>> # σ'(0) = 0.5 * (1 - 0.5) = 0.25
        >>> # grad_X = 1 * 0.25 = 0.25

        HINTS:
        - Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
        - Output is already computed and saved in self.output_data
        - This avoids recomputing sigmoid during backward pass
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # σ'(x) = σ(x) * (1 - σ(x))
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return grad_output * sigmoid_grad,
        return None,
        ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "softmax-backward", "solution": true}
#| export
class SoftmaxBackward(Function):
    """
    Gradient computation for softmax activation.

    Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x))
    Derivative: ∂softmax/∂x[i] = softmax[i] * (δ[i,j] - softmax[j])

    For gradient computation:
    grad_x[i] = softmax[i] * (grad_y[i] - sum(grad_y * softmax))

    **Key Insight:** The gradient depends on all elements of softmax due to
    the normalization, not just the element being differentiated.
    """

    def __init__(self, input_tensor, output_tensor, dim=-1):
        """
        Initialize with input, output, and dimension.

        Args:
            input_tensor: Original input to softmax
            output_tensor: Output of softmax (needed for gradient)
            dim: Dimension along which softmax was applied
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self, grad_output):
        """
        Compute gradient for softmax.

        Mathematical formula:
        ∂L/∂x[i] = softmax[i] * (∂L/∂y[i] - sum_j(∂L/∂y[j] * softmax[j]))

        This can be vectorized as:
        grad_x = softmax * (grad_y - sum(grad_y * softmax, keepdims=True))

        TODO: Implement gradient computation for softmax activation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Compute sum term: np.sum(grad_output * self.output_data, axis=self.dim, keepdims=True)
           - Compute gradient: self.output_data * (grad_output - sum_term)
           - Return as tuple: (grad_x,)
        3. Else return (None,)

        EXAMPLE:
        >>> X = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        >>> Y = softmax(X)  # [[0.09, 0.24, 0.67]] approximately
        >>> # During backward: grad_output = [[1, 0, 0]]
        >>> # sum_term = sum([1*0.09, 0*0.24, 0*0.67]) = 0.09
        >>> # grad_X[i] = softmax[i] * (grad_output[i] - sum_term)

        HINTS:
        - Softmax gradient depends on all elements due to normalization
        - Use keepdims=True in np.sum to maintain dimensions for broadcasting
        - Vectorized formula: softmax * (grad_output - sum(grad_output * softmax))
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Compute sum(grad_output * softmax) along the softmax dimension
            sum_term = np.sum(grad_output * self.output_data, axis=self.dim, keepdims=True)

            # Softmax gradient: softmax * (grad_output - sum_term)
            grad_x = self.output_data * (grad_output - sum_term)

            return (grad_x,)
        return (None,)
        ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "gelu-backward", "solution": true}
#| export
class GELUBackward(Function):
    """
    Gradient computation for GELU activation.

    GELU: f(x) = x * Φ(x) where Φ is the CDF of standard normal
    Approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    **Key Insight:** GELU is smoother than ReLU, providing non-zero gradients
    for negative values, which helps training deep networks.
    """

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """
        Compute gradient for GELU.

        Mathematical formula (using approximation):
        ∂gelu/∂x ≈ 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * (...)

        Simplified: We compute the derivative numerically or use the formula.

        TODO: Implement gradient computation for GELU activation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Compute tanh approximation components
           - Compute sech² (derivative of tanh)
           - Apply GELU derivative formula
           - Multiply by grad_output
        3. Else return (None,)

        HINTS:
        - GELU is smoother than ReLU, providing gradients for negative values
        - Use tanh approximation for numerical stability
        - Formula: 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d(tanh_arg)/dx
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data
            # GELU derivative approximation
            # Using the tanh approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            x_cubed = x ** 3
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            tanh_out = np.tanh(tanh_arg)
            sech_squared = 1 - tanh_out ** 2

            # Derivative: 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d(tanh_arg)/dx
            d_tanh_arg = sqrt_2_over_pi * (1 + 0.134145 * x ** 2)
            gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x * sech_squared * d_tanh_arg

            return (grad_output * gelu_grad,)
        return (None,)
        ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "mse-backward", "solution": true}
#| export
class MSEBackward(Function):
    """
    Gradient computation for Mean Squared Error Loss.

    MSE: L = mean((predictions - targets)²)
    Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """
        Compute gradient for MSE loss.

        TODO: Implement gradient computation for Mean Squared Error loss.

        APPROACH:
        1. Extract predictions tensor from self.saved_tensors
        2. If predictions requires gradients:
           - Compute difference: predictions.data - self.targets_data
           - Apply MSE derivative: 2 * difference / N
           - Multiply by grad_output: grad * grad_output
           - Return as tuple: (result,)
        3. Else return (None,)

        EXAMPLE:
        >>> predictions = Tensor([2.0, 3.0], requires_grad=True)
        >>> targets = Tensor([1.0, 2.0])
        >>> loss = MSE(predictions, targets)  # (1² + 1²)/2 = 1.0
        >>> # During backward: grad_output = 1
        >>> # grad = 2 * ([2, 3] - [1, 2]) / 2 = [1, 1]

        HINTS:
        - MSE derivative: ∂MSE/∂pred = 2 * (pred - target) / N
        - N = self.num_samples (total number of elements)
        - Multiply by grad_output for chain rule
        """
        ### BEGIN SOLUTION
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            # Gradient: 2 * (predictions - targets) / N
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples

            return grad * grad_output,
        return None,
        ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "bce-backward", "solution": true}
#| export
class BCEBackward(Function):
    """
    Gradient computation for Binary Cross-Entropy Loss.

    BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """
        Compute gradient for BCE loss.

        TODO: Implement gradient computation for Binary Cross-Entropy loss.

        APPROACH:
        1. Extract predictions tensor from self.saved_tensors
        2. If predictions requires gradients:
           - Clip predictions: p = np.clip(predictions.data, eps, 1-eps)
           - Get targets: y = self.targets_data
           - Apply BCE derivative: (p - y) / (p * (1-p) * N)
           - Multiply by grad_output
           - Return as tuple: (result,)
        3. Else return (None,)

        EXAMPLE:
        >>> predictions = Tensor([0.7, 0.3], requires_grad=True)
        >>> targets = Tensor([1.0, 0.0])
        >>> loss = BCE(predictions, targets)
        >>> # During backward: grad = (p - y) / (p * (1-p) * N)

        HINTS:
        - BCE derivative: ∂BCE/∂p = (p - y) / (p * (1-p)) per sample
        - Clip predictions to avoid log(0) instability
        - Divide by N for mean loss
        """
        ### BEGIN SOLUTION
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = EPSILON
            p = np.clip(predictions.data, eps, 1 - eps)
            y = self.targets_data

            # Gradient: (p - y) / (p * (1-p) * N)
            grad = (p - y) / (p * (1 - p) * self.num_samples)

            return grad * grad_output,
        return None,
        ### END SOLUTION

# %% [markdown]
"""
### Helper: Numerically Stable Softmax

Computing softmax naively as `exp(x) / sum(exp(x))` overflows for large values.
The fix is to subtract the maximum value first, which is mathematically equivalent
but numerically stable.

```
Naive (overflows):     softmax(x) = exp(x) / sum(exp(x))
Stable (safe):         softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Why it works:
  exp(x - max(x)) / sum(exp(x - max(x)))
= exp(x) * exp(-max(x)) / (sum(exp(x)) * exp(-max(x)))
= exp(x) / sum(exp(x))
```

This helper is used by CrossEntropyBackward to convert logits to probabilities.
"""

# %% nbgrader={"grade": false, "grade_id": "stable-softmax-helper", "solution": true}
#| export
def _stable_softmax(logits_data):
    """
    Compute softmax probabilities with numerical stability.

    Subtracts the max value per row before exponentiating to prevent overflow.

    Args:
        logits_data: numpy array of shape (batch_size, num_classes)

    Returns:
        numpy array of softmax probabilities, same shape as input

    TODO: Implement numerically stable softmax computation.

    APPROACH:
    1. Find the maximum value per row: np.max(logits_data, axis=1, keepdims=True)
    2. Subtract max from logits: logits_data - max_logits
    3. Exponentiate: np.exp(shifted_logits)
    4. Normalize by row sum: exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    EXAMPLE:
    >>> logits = np.array([[2.0, 1.0, 0.1]])
    >>> probs = _stable_softmax(logits)
    >>> # probs ≈ [[0.659, 0.242, 0.099]]
    >>> # Each row sums to 1.0

    HINT: keepdims=True is essential for correct broadcasting.
    """
    ### BEGIN SOLUTION
    max_logits = np.max(logits_data, axis=1, keepdims=True)
    exp_logits = np.exp(logits_data - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ### END SOLUTION

# %% [markdown]
"""
### Unit Test: Stable Softmax Helper

**What we're testing**: Numerically stable softmax computation
**Why it matters**: Unstable softmax causes NaN/Inf in cross-entropy gradients
**Expected**: Probabilities sum to 1.0, correct values, no overflow on large inputs
"""

# %% nbgrader={"grade": true, "grade_id": "test-stable-softmax-helper", "locked": true, "points": 3}
def test_unit_stable_softmax():
    """Test stable softmax helper."""
    print("Testing stable softmax helper...")

    # Basic correctness
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = _stable_softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0), f"Softmax should sum to 1, got {probs.sum(axis=1)}"

    # Verify correct values
    expected = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    assert np.allclose(probs, expected), f"Softmax values wrong: {probs}"

    # Numerical stability: large values that would overflow naive exp()
    large_logits = np.array([[1000.0, 1001.0, 1002.0]])
    probs_large = _stable_softmax(large_logits)
    assert not np.any(np.isnan(probs_large)), "Stable softmax should handle large values"
    assert not np.any(np.isinf(probs_large)), "Stable softmax should not overflow"
    assert np.allclose(probs_large.sum(axis=1), 1.0), "Large softmax should sum to 1"

    # Batch dimension
    batch_logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch_probs = _stable_softmax(batch_logits)
    assert batch_probs.shape == (3, 2), f"Expected (3, 2), got {batch_probs.shape}"
    assert np.allclose(batch_probs.sum(axis=1), np.ones(3)), "Each row should sum to 1"

    print("  Stable softmax helper works correctly!")

if __name__ == "__main__":
    test_unit_stable_softmax()

# %% [markdown]
"""
### Helper: One-Hot Encoding

Converts class indices to one-hot vectors. This is needed by the cross-entropy
gradient formula: `grad = softmax - one_hot`.

```
Indices: [0, 2, 1]  with 3 classes

One-hot:
  [[1, 0, 0],    ← class 0
   [0, 0, 1],    ← class 2
   [0, 1, 0]]    ← class 1
```
"""

# %% nbgrader={"grade": false, "grade_id": "one-hot-helper", "solution": true}
#| export
def _one_hot_encode(targets, batch_size, num_classes):
    """
    Convert class indices to one-hot vectors.

    Args:
        targets: numpy array of integer class indices, shape (batch_size,)
        batch_size: number of samples
        num_classes: number of classes

    Returns:
        numpy array of one-hot vectors, shape (batch_size, num_classes)

    TODO: Implement one-hot encoding of target class indices.

    APPROACH:
    1. Create zeros array: np.zeros((batch_size, num_classes), dtype=np.float32)
    2. Set target positions to 1.0: result[np.arange(batch_size), targets] = 1.0
    3. Return the one-hot array

    EXAMPLE:
    >>> targets = np.array([0, 2, 1])
    >>> one_hot = _one_hot_encode(targets, batch_size=3, num_classes=3)
    >>> # one_hot = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

    HINT: Use advanced indexing with np.arange for the row indices.
    """
    ### BEGIN SOLUTION
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), targets] = 1.0
    return one_hot
    ### END SOLUTION

# %% [markdown]
"""
### Unit Test: One-Hot Encoding Helper

**What we're testing**: Conversion from class indices to one-hot vectors
**Why it matters**: Incorrect one-hot encoding produces wrong cross-entropy gradients
**Expected**: Each row has exactly one 1.0, at the correct class position
"""

# %% nbgrader={"grade": true, "grade_id": "test-one-hot-helper", "locked": true, "points": 3}
def test_unit_one_hot_encode():
    """Test one-hot encoding helper."""
    print("Testing one-hot encoding helper...")

    # Basic test
    targets = np.array([0, 2, 1])
    result = _one_hot_encode(targets, batch_size=3, num_classes=3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    assert np.allclose(result, expected), f"One-hot encoding wrong: {result}"

    # Single sample
    result_single = _one_hot_encode(np.array([1]), batch_size=1, num_classes=4)
    assert result_single.shape == (1, 4), f"Expected (1, 4), got {result_single.shape}"
    assert result_single[0, 1] == 1.0, "Target class should be 1.0"
    assert result_single.sum() == 1.0, "Should have exactly one 1.0"

    # Each row sums to 1
    targets_batch = np.array([0, 1, 2, 3, 4])
    result_batch = _one_hot_encode(targets_batch, batch_size=5, num_classes=5)
    assert np.allclose(result_batch.sum(axis=1), np.ones(5)), "Each row should sum to 1"

    print("  One-hot encoding helper works correctly!")

if __name__ == "__main__":
    test_unit_one_hot_encode()

# %% [markdown]
"""
### CrossEntropyBackward - Gradient Rules for Cross-Entropy Loss

The cross-entropy gradient combines three sub-computations:
1. **Stable softmax**: Convert raw logits to probabilities
2. **One-hot encoding**: Convert target indices to indicator vectors
3. **Gradient formula**: `(softmax - one_hot) / batch_size`

```
Logits: [2.0, 1.0, 0.1]     Target: class 0

Step 1 - Softmax:   [0.659, 0.242, 0.099]
Step 2 - One-hot:   [1.000, 0.000, 0.000]
Step 3 - Gradient:  [0.659-1, 0.242-0, 0.099-0] / 1 = [-0.341, 0.242, 0.099]
```

The gradient is simply "how far off each class probability is from the target."
This is one of the most elegant results in machine learning.
"""

# %% nbgrader={"grade": false, "grade_id": "ce-backward", "solution": true}
#| export
class CrossEntropyBackward(Function):
    """
    Gradient computation for Cross-Entropy Loss.

    CrossEntropy: L = -mean(log_softmax(logits)[targets])

    The gradient with respect to logits is remarkably elegant:
    ∂L/∂logits = (softmax(logits) - one_hot(targets)) / N

    This is one of the most beautiful results in machine learning:
    - The gradient is simply the difference between predictions and targets
    - It naturally scales with how wrong we are
    - It's numerically stable when computed via softmax
    """

    def __init__(self, logits, targets):
        """Initialize with logits and target class indices."""
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_output):
        """
        Compute gradient for cross-entropy loss.

        Uses helper functions for each sub-computation:
        - _stable_softmax(): Converts logits to probabilities (numerically stable)
        - _one_hot_encode(): Converts target indices to one-hot vectors

        TODO: Implement gradient computation for Cross-Entropy loss.

        APPROACH:
        1. Extract logits tensor from self.saved_tensors
        2. If logits requires gradients:
           - Compute softmax using _stable_softmax(logits.data)
           - Encode targets using _one_hot_encode(targets, batch_size, num_classes)
           - Apply CE derivative: (softmax - one_hot) / batch_size
           - Multiply by grad_output
           - Return as tuple: (result,)
        3. Else return (None,)

        EXAMPLE:
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> targets = Tensor([0])  # Correct class is 0
        >>> loss = CrossEntropy(logits, targets)
        >>> # softmax ≈ [0.66, 0.24, 0.10]
        >>> # one_hot = [1, 0, 0]
        >>> # grad = ([0.66, 0.24, 0.10] - [1, 0, 0]) / 1 = [-0.34, 0.24, 0.10]

        HINTS:
        - CE gradient: (softmax(logits) - one_hot(targets)) / batch_size
        - Use _stable_softmax() for numerically stable softmax
        - Use _one_hot_encode() for target encoding
        """
        ### BEGIN SOLUTION
        logits, = self.saved_tensors

        if isinstance(logits, Tensor) and logits.requires_grad:
            softmax = _stable_softmax(logits.data)
            one_hot = _one_hot_encode(self.targets_data, self.batch_size, self.num_classes)

            # Gradient: (softmax - one_hot) / batch_size
            grad = (softmax - one_hot) / self.batch_size

            return grad * grad_output,
        return None,
        ### END SOLUTION


# %% nbgrader={"grade": false, "grade_id": "no-grad-context", "solution": true}
#| export
# ===== Global Gradient Tracking Flag =====
# Why this exists: During inference or parameter updates, we don't need to build
# computation graphs. Skipping graph construction saves memory and time.
# This matches PyTorch's torch.no_grad() behavior.
_GRAD_TRACKING_ENABLED = True


def is_grad_enabled():
    """Check if gradient tracking is currently enabled.

    Returns True when operations should build computation graphs,
    False when inside a no_grad() context.
    """
    return _GRAD_TRACKING_ENABLED


class no_grad:
    """Context manager that disables gradient tracking.

    When entering this context, all operations will skip computation graph
    construction — tensors produced inside will have requires_grad=False
    regardless of their inputs. This is essential for:

    1. **Inference**: No need to track gradients when making predictions
    2. **Parameter updates**: Optimizers modify weights without recording history
    3. **Memory savings**: Skipping graph construction reduces memory usage

    Matches PyTorch's torch.no_grad() API.

    **Example:**
    ```python
    x = Tensor([2.0], requires_grad=True)

    with no_grad():
        y = x * 2  # No graph built, y.requires_grad = False

    z = x * 3  # Graph IS built (outside no_grad)
    z.backward()  # Works normally
    ```

    **Nesting is safe:**
    ```python
    with no_grad():
        with no_grad():  # Inner context
            y = x * 2    # Still no graph
        z = x * 3        # Still no graph (outer context active)
    w = x * 4  # Graph IS built (all contexts exited)
    ```
    """

    def __enter__(self):
        """Save previous state and disable gradient tracking."""
        global _GRAD_TRACKING_ENABLED
        # Save previous state so nested contexts restore correctly
        self._prev_state = _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous gradient tracking state."""
        global _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = self._prev_state
        return False  # Don't suppress exceptions


# %% nbgrader={"grade": false, "grade_id": "enable-autograd", "solution": true}
#| export
def enable_autograd(quiet=False):
    """
    Enable gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.
    Call this once to activate gradients globally.

    **Args:**
        quiet (bool): If True, suppress status messages. Default: False.

    **What it does:**
    - Replaces Tensor operations with gradient-tracking versions
    - Adds backward() method for reverse-mode differentiation
    - Enables computation graph building
    - Maintains full backward compatibility

    **After calling this:**
    - Tensor operations will track computation graphs
    - backward() method becomes available
    - Gradients will flow through operations
    - requires_grad=True enables tracking per tensor

    **Example:**
    ```python
    enable_autograd()  # Call once
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    y.backward()
    print(x.grad)  # [3.0]
    ```
    """

    # Educational Note: hasattr() is LEGITIMATE here because:
    # 1. This is a runtime monkey-patch system (meta-programming)
    # 2. We're checking if a class has been dynamically modified
    # 3. _autograd_enabled is a marker attribute we add at runtime
    # This is the CORRECT use of hasattr() for dynamic class modification
    if hasattr(Tensor, '_autograd_enabled'):
        # Silently return if already enabled - no need to warn
        return

    # ===== STEP 1: Add gradient infrastructure to Tensor =====
    # Store original __init__ to extend it
    _original_init = Tensor.__init__

    def gradient_aware_init(self, data, requires_grad=False):
        """Extended Tensor init that supports gradient tracking."""
        _original_init(self, data)
        self.requires_grad = requires_grad
        self.grad = None

    # Replace __init__ with gradient-aware version
    Tensor.__init__ = gradient_aware_init

    # Store original operations
    # These are guaranteed to exist from Module 01 (Tensor class)
    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_div = Tensor.__truediv__
    _original_getitem = Tensor.__getitem__

    # These methods are also guaranteed from Module 01 - trust Single Tensor Class
    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    # Helper to safely check requires_grad (handles tensors created before enable_autograd)
    def _get_requires_grad(tensor):
        """Safely get requires_grad, defaulting to False for pre-autograd tensors.

        Also returns False when inside a no_grad() context, since we should
        not build computation graphs there.
        """
        if not _GRAD_TRACKING_ENABLED:
            return False
        return getattr(tensor, 'requires_grad', False) if isinstance(tensor, Tensor) else False

    def _ensure_grad_attrs(tensor):
        """Ensure tensor has gradient attributes (for tensors created before enable_autograd)."""
        if isinstance(tensor, Tensor):
            if not hasattr(tensor, 'requires_grad'):
                tensor.requires_grad = False
            if not hasattr(tensor, 'grad'):
                tensor.grad = None

    # Enhanced operations that track gradients
    def tracked_add(self, other):
        """
        Addition with gradient tracking.

        Enhances the original __add__ method to build computation graphs
        when requires_grad=True for any input.
        """
        # Ensure self has gradient attributes
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_add(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)

        return result

    def tracked_mul(self, other):
        """
        Multiplication with gradient tracking.

        Enhances the original __mul__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed for consistency
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        # Call original operation
        result = _original_mul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other)

        return result

    def tracked_matmul(self, other):
        """
        Matrix multiplication with gradient tracking.

        Enhances the original matmul method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)

        # Call original matmul from Module 01
        result = _original_matmul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = MatmulBackward(self, other)

        return result

    def tracked_transpose(self, dim0=None, dim1=None):
        """
        Transpose with gradient tracking.

        Enhances the original transpose method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original transpose from Module 01
        result = _original_transpose(self, dim0, dim1)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = TransposeBackward(self, dim0, dim1)

        return result

    def tracked_reshape(self, *shape):
        """
        Reshape with gradient tracking.

        Enhances the original reshape method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)
        original_shape = self.shape

        # Call original reshape from Module 01
        result = _original_reshape(self, *shape)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = ReshapeBackward(self, original_shape)

        return result

    def tracked_sub(self, other):
        """
        Subtraction with gradient tracking.

        Enhances the original __sub__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_sub(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = SubBackward(self, other)

        return result

    def tracked_div(self, other):
        """
        Division with gradient tracking.

        Enhances the original __truediv__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_div(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = DivBackward(self, other)

        return result

    def tracked_getitem(self, key):
        """
        Indexing/slicing with gradient tracking.

        Enhances the original __getitem__ method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original __getitem__ from Module 01
        result = _original_getitem(self, key)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SliceBackward(self, key)

        return result

    def sum_op(self, axis=None, keepdims=False):
        """
        Sum operation with gradient tracking.

        Creates a new sum method that builds computation graphs
        when requires_grad=True.
        """
        _ensure_grad_attrs(self)

        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SumBackward(self, axis=axis, keepdims=keepdims)

        return result

    def backward(self, gradient=None, retain_graph=False):
        """
        Compute gradients via backpropagation.

        This is the key method that makes training possible!
        It implements reverse-mode automatic differentiation.

        **Algorithm:**
        1. Initialize gradient if not provided (for scalar outputs)
        2. Accumulate gradient in self.grad
        3. If this tensor has a _grad_fn, call it to propagate gradients
        4. Recursively call backward() on parent tensors
        5. Release computation graph (unless retain_graph=True)

        **Args:**
            gradient: External gradient to seed backpropagation. If None, assumes
                scalar output and uses ones_like as the seed.
            retain_graph: If False (default), releases the computation graph after
                backward to free memory. Set True if you need to call backward()
                multiple times on the same graph (e.g., for higher-order gradients).
                Matches PyTorch's retain_graph parameter.

        **Example:**
        ```python
        x = Tensor([2.0], requires_grad=True)
        y = x * 3
        y.backward()  # Computes gradients for x, then releases graph
        print(x.grad)  # [3.0]
        # y.backward()  # Would fail — graph already released!

        # To backward twice, use retain_graph=True:
        y2 = x * 3
        y2.backward(retain_graph=True)  # Graph kept alive
        y2.backward()  # Works! (releases graph this time)
        ```
        """
        # Ensure gradient attributes exist
        _ensure_grad_attrs(self)

        # Only compute gradients if required
        if not _get_requires_grad(self):
            return

        # Initialize gradient if not provided (for scalar outputs)
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(
                    f"backward() called on non-scalar tensor without gradient argument.\n"
                    f"  Tensor shape: {self.shape}\n"
                    f"  Issue: For non-scalar outputs, you must provide the gradient from the next layer.\n"
                    f"  Fix: Call backward(gradient) with the gradient tensor from the loss function."
                )

        # Initialize or accumulate gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        # Handle broadcasting: sum gradient to match self.data shape
        # This happens when operations broadcast tensors (e.g., adding bias to batch)
        if gradient.shape != self.grad.shape:
            # Step 1: Remove extra leading dimensions added during forward pass
            # Example: gradient (batch_size, features) → self.grad (features,)
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)

            # Step 2: Sum over dimensions that were size-1 in original tensor
            # Example: bias with shape (1,) broadcast to (batch_size,) during forward
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)

        self.grad += gradient

        # Propagate gradients through computation graph
        # _grad_fn is set by autograd enhancement when tensor is created from an operation
        grad_fn = getattr(self, '_grad_fn', None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)

            # Recursively call backward on parent tensors
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad, retain_graph=retain_graph)

            # Release computation graph to free memory (matches PyTorch's default)
            # Why: The graph stores references to all intermediate tensors. Without
            # cleanup, these references prevent garbage collection, causing memory
            # to grow linearly with the number of training steps.
            if not retain_graph:
                self._grad_fn = None

    def zero_grad(self):
        """
        Reset gradients to zero.

        Call this before each backward pass to prevent gradient accumulation
        from previous iterations.
        """
        self.grad = None

    # Install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__sub__ = tracked_sub
    Tensor.__mul__ = tracked_mul
    Tensor.__truediv__ = tracked_div
    Tensor.__getitem__ = tracked_getitem
    Tensor.matmul = tracked_matmul
    Tensor.transpose = tracked_transpose
    Tensor.reshape = tracked_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    # Patch activations and losses to track gradients
    try:
        from tinytorch.core.activations import Sigmoid, ReLU, Softmax, GELU
        from tinytorch.core.losses import BinaryCrossEntropyLoss, MSELoss, CrossEntropyLoss

        # Store original methods
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward

        def tracked_sigmoid_forward(self, x):
            """Sigmoid with gradient tracking."""
            result_data = 1.0 / (1.0 + np.exp(-x.data))
            result = Tensor(result_data)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SigmoidBackward(x, result)

            return result

        def tracked_relu_forward(self, x):
            """ReLU with gradient tracking."""
            result_data = np.maximum(0, x.data)
            result = Tensor(result_data)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)

            return result

        def tracked_softmax_forward(self, x, dim=-1):
            """Softmax with gradient tracking."""
            # Call original forward to get result using Tensor operations
            result = _original_softmax_forward(self, x, dim=dim)

            # Attach the correct gradient function
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SoftmaxBackward(x, result, dim)

            return result

        def tracked_gelu_forward(self, x):
            """GELU with gradient tracking."""
            # Call original forward to get result
            result = _original_gelu_forward(self, x)

            # Attach the correct gradient function
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = GELUBackward(x)

            return result

        def tracked_bce_forward(self, predictions, targets):
            """Binary cross-entropy with gradient tracking."""
            # Compute BCE loss
            eps = EPSILON
            clamped_preds = np.clip(predictions.data, eps, 1 - eps)
            log_preds = np.log(clamped_preds)
            log_one_minus_preds = np.log(1 - clamped_preds)
            bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)
            bce_loss = np.mean(bce_per_sample)

            result = Tensor(bce_loss)

            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = BCEBackward(predictions, targets)

            return result

        def tracked_mse_forward(self, predictions, targets):
            """MSE loss with gradient tracking."""
            # Compute MSE loss
            diff = predictions.data - targets.data
            squared_diff = diff ** 2
            mse = np.mean(squared_diff)

            result = Tensor(mse)

            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = MSEBackward(predictions, targets)

            return result

        def tracked_ce_forward(self, logits, targets):
            """Cross-entropy loss with gradient tracking."""
            from tinytorch.core.losses import log_softmax

            # Compute log-softmax for numerical stability
            log_probs = log_softmax(logits, dim=-1)

            # Select log-probabilities for correct classes
            batch_size = logits.shape[0]
            target_indices = targets.data.astype(int)
            selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

            # Return negative mean
            ce_loss = -np.mean(selected_log_probs)

            result = Tensor(ce_loss)

            if _GRAD_TRACKING_ENABLED and logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits, targets)

            return result

        # Install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward
        CrossEntropyLoss.forward = tracked_ce_forward

    except ImportError:
        # Activations/losses not yet available (happens during module development)
        pass

    # Mark as enabled
    Tensor._autograd_enabled = True

    if not quiet:
        print("✅ Autograd enabled! Tensors now track gradients.")
        print("   - Operations build computation graphs")
        print("   - backward() computes gradients")
        print("   - requires_grad=True enables tracking")

# Auto-enable when module is imported
# Always quiet to avoid cluttering user imports
import os
enable_autograd(quiet=True)

# %% [markdown]
"""
## ⚠️ DANGER: In-Place Operations Break Autograd

**THIS IS THE MOST COMMON SILENT FAILURE IN TINYTORCH!**

### Critical Rule: Never Modify Tensors In-Place When requires_grad=True

**WRONG ❌ - This Corrupts the Gradient Graph:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x * 2
x.data[0] = 999  # ❌ CORRUPTS GRADIENT GRAPH WITHOUT ERROR!
y.backward()     # ❌ Wrong gradients or crash
```

**RIGHT ✅ - Create New Tensors Instead:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x * 2
x = Tensor([999, 2, 3], requires_grad=True)  # ✅ New tensor, safe
y.backward()  # ✅ Correct gradients
```

### Why This Breaks Everything

Autograd records operations on the **original tensor values**. When you modify `.data` directly:

1. **Forward pass** records: "y = x * 2" where x = [1, 2, 3]
2. **You corrupt**: x.data[0] = 999, so x = [999, 2, 3]
3. **Backward pass** uses: corrupted x values, causing wrong gradients or crashes

**The computation graph becomes inconsistent** - forward used [1, 2, 3], backward uses [999, 2, 3].

### Common In-Place Operations to AVOID

```python
# ❌ FORBIDDEN - Direct index assignment
x.data[0] = value
x.data[:, 0] = values
x.data[mask] = values

# ❌ FORBIDDEN - In-place arithmetic
x.data += other
x.data *= scalar
x.data -= value

# ❌ FORBIDDEN - NumPy in-place operations
np.fill(x.data, value)
np.add(x.data, other, out=x.data)
x.data.fill(value)

# ✅ CORRECT - Create new tensors
x = x + other              # Creates new tensor
x = Tensor(x.data + other) # Explicit new tensor
x = Tensor([new_values])   # Complete replacement
```

### Real-World Example: Parameter Update Gone Wrong

```python
# ❌ WRONG - This is a common mistake in custom optimizers
W = Tensor([[0.5, 0.3]], requires_grad=True)
y = x.matmul(W.T)
loss = compute_loss(y, target)
loss.backward()

# Student writes custom optimizer:
W.data -= 0.01 * W.grad  # ❌ CORRUPTS GRAPH! Next forward pass is broken!

# ✅ CORRECT - Create new parameter tensor
W = Tensor(W.data - 0.01 * W.grad, requires_grad=True)  # ✅ Safe
```

### How to Debug In-Place Corruption

If your gradients look wrong or you get mysterious errors:

1. **Search your code** for `.data[` assignments
2. **Search for** in-place operators: `+=`, `-=`, `*=`, `/=` on `.data`
3. **Check custom functions** that modify tensors
4. **Verify** all parameter updates create new tensors

### Why PyTorch Has torch.no_grad()

PyTorch explicitly disables gradient tracking during parameter updates to allow safe in-place operations.
TinyTorch now supports this via the `no_grad` context manager:

```python
# PyTorch pattern
with torch.no_grad():
    W -= 0.01 * W.grad  # Safe inside no_grad context

# TinyTorch equivalent
from tinytorch.core.autograd import no_grad
with no_grad():
    W -= 0.01 * W.grad  # No graph built, safe for parameter updates
```

**How it works**: `no_grad()` sets a global flag that all tracked operations check.
When the flag is off, operations skip graph construction entirely -- the result tensor
will have `requires_grad=False` regardless of its inputs.

### Memory Impact

**Question**: "Doesn't creating new tensors waste memory?"

**Answer**: Gradient tracking already stores intermediate tensors for backprop. Creating new tensors is negligible compared to the computation graph memory overhead. Correctness > premature perf.

**Bottom Line**: If a tensor has `requires_grad=True`, treat it as **immutable**. Always create new tensors instead of modifying in-place.

---
"""

# %% [markdown]
"""
### 🔬 Unit Test: Tensor Autograd Enhancement

This test validates our enhanced Tensor class computes gradients correctly.

**What we're testing**: Gradient computation and chain rule implementation
**Why it matters**: This is the core of automatic differentiation
**Expected**: Correct gradients for various operations and computation graphs
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-autograd", "locked": true, "points": 20}
def test_unit_tensor_autograd():
    """🔬 Test Tensor autograd enhancement."""
    print("🔬 Unit Test: Tensor Autograd Enhancement...")

    # Test simple gradient computation
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1  # z = 3x + 1, so dz/dx = 3

    z.backward()
    assert np.allclose(x.grad, [3.0]), f"Expected [3.0], got {x.grad}"

    # Test matrix multiplication gradients
    a = Tensor([[1.0, 2.0]], requires_grad=True)  # 1x2
    b = Tensor([[3.0], [4.0]], requires_grad=True)  # 2x1
    c = a.matmul(b)  # 1x1, result = [[11.0]]

    c.backward()
    assert np.allclose(a.grad, [[3.0, 4.0]]), f"Expected [[3.0, 4.0]], got {a.grad}"
    assert np.allclose(b.grad, [[1.0], [2.0]]), f"Expected [[1.0], [2.0]], got {b.grad}"

    # Test computation graph with multiple operations
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2      # y = [2, 4]
    z = y.sum()    # z = 6

    z.backward()
    assert np.allclose(x.grad, [2.0, 2.0]), f"Expected [2.0, 2.0], got {x.grad}"

    print("✅ Tensor autograd enhancement works correctly!")

if __name__ == "__main__":
    test_unit_tensor_autograd()

# %% [markdown]
"""
## 📊 Systems Analysis: Computation Graph Memory

Let's understand ONE key systems concept: **computation graph memory overhead**.

This single analysis reveals why gradient tracking is expensive and why frameworks make gradient tracking opt-in.
"""

# %%
def analyze_computation_graph_memory():
    """📊 Demonstrate memory overhead of computation graphs."""
    print("📊 Analyzing Computation Graph Memory...")
    print("=" * 60)

    import sys

    # Create tensors with different sizes
    sizes = [(100, 100), (500, 500), (1000, 1000)]

    print("\nMemory comparison: With vs Without Gradient Tracking")
    print("-" * 60)

    for shape in sizes:
        # Without gradient tracking
        x_no_grad = Tensor(np.random.randn(*shape))
        base_memory = x_no_grad.data.nbytes

        # With gradient tracking
        x_with_grad = Tensor(np.random.randn(*shape), requires_grad=True)
        y = x_with_grad * 2  # Simple operation that builds graph
        z = y + 1

        # Estimate graph overhead: saved tensors in grad_fn
        graph_overhead = 0
        if hasattr(z, '_grad_fn') and z._grad_fn is not None:
            for tensor in z._grad_fn.saved_tensors:
                if isinstance(tensor, Tensor):
                    graph_overhead += tensor.data.nbytes

        print(f"\nShape {shape}:")
        print(f"   Base tensor: {base_memory / 1024:.1f} KB")
        print(f"   Graph overhead: {graph_overhead / 1024:.1f} KB")
        print(f"   Overhead ratio: {(graph_overhead / base_memory):.1f}x")

    print("\n" + "=" * 60)
    print("📊 KEY INSIGHTS:")
    print("   1. Each operation saves inputs for backward pass")
    print("   2. Memory scales with number of operations, not just parameters")
    print("   3. Deep networks have more graph overhead than shallow ones")
    print("   4. This is why requires_grad is opt-in, not default!")

    print("\n🚀 REAL-WORLD IMPLICATIONS:")
    print("   - Training uses ~2-3x memory of inference")
    print("   - torch.no_grad() context saves memory during evaluation")

    print("\n" + "=" * 60)

# Run the systems analysis
if __name__ == "__main__":
    analyze_computation_graph_memory()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 25}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Autograd works for complex computation graphs
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_stable_softmax()
    test_unit_one_hot_encode()
    test_unit_function_classes()
    test_unit_tensor_autograd()

    print("\nRunning integration scenarios...")

    # Test 1: Multi-layer computation graph
    print("🔬 Integration Test: Multi-layer Neural Network...")

    # Create a 3-layer computation: x -> Linear -> Linear -> Linear -> loss
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    W1 = Tensor([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]], requires_grad=True)
    b1 = Tensor([[0.1, 0.2, 0.3]], requires_grad=True)

    # First layer
    h1 = x.matmul(W1) + b1
    assert h1.shape == (1, 3)
    assert h1.requires_grad == True

    # Second layer
    W2 = Tensor([[0.1], [0.2], [0.3]], requires_grad=True)
    h2 = h1.matmul(W2)
    assert h2.shape == (1, 1)

    # Compute simple loss (just square the output for testing)
    loss = h2 * h2

    # Backward pass
    loss.backward()

    # Verify all parameters have gradients
    assert x.grad is not None
    assert W1.grad is not None
    assert b1.grad is not None
    assert W2.grad is not None
    assert x.grad.shape == x.shape
    assert W1.grad.shape == W1.shape

    print("✅ Multi-layer neural network gradients work!")

    # Test 2: Gradient accumulation
    print("🔬 Integration Test: Gradient Accumulation...")

    x = Tensor([2.0], requires_grad=True)

    # First computation
    y1 = x * 3
    y1.backward()
    first_grad = x.grad.copy()

    # Second computation (should accumulate)
    y2 = x * 5
    y2.backward()

    assert np.allclose(x.grad, first_grad + 5.0), "Gradients should accumulate"
    print("✅ Gradient accumulation works!")

    # Test 3: Complex mathematical operations
    print("🔬 Integration Test: Complex Operations...")

    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)

    # Complex computation: ((a @ b) + a) * b
    temp1 = a.matmul(b)  # Matrix multiplication
    temp2 = temp1 + a    # Addition
    result = temp2 * b   # Element-wise multiplication
    final = result.sum() # Sum reduction

    final.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    print("✅ Complex mathematical operations work!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 06")

# Test function defined above, will be called in main block

# %%
# Run comprehensive module test
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Before we wrap up, reflect on these systems-level questions. Use only knowledge from Modules 01-05 (no forward references to concepts you haven't learned yet).

### Question 1: Computational Graph Memory
**Scenario**: A 10-layer neural network processes a single sample. Each layer performs matrix multiplication (matmul) and addition (bias).

**Question**: How much memory does the computation graph use compared to just storing the weights?

**Consider**:
- What tensors must be saved during forward pass for backward pass?
- If weights take 10MB total, estimate graph memory overhead
- When is the graph freed?

---

### Question 2: Gradient Accumulation
**Scenario**: A weight matrix is shared between two computation paths in a network (like a tied-weights architecture).

**Question**: Why does gradient accumulation (`grad = grad + new_grad`) save memory during training? What's the trade-off?

**Consider**:
- What happens if you process a large batch all at once vs. multiple smaller batches?
- Memory usage: storing intermediate activations vs. recomputing forward passes
- Training behavior: does gradient accumulation change what the model learns?

---

### Question 3: Backward Pass Cost
**Scenario**: A forward pass through a 3-layer MLP takes 10ms.

**Question**: Is the backward pass faster, slower, or the same speed as the forward pass? Why?

**Consider**:
- Operations in forward pass: matmul, activation, addition
- Operations in backward pass: matmul (for gradients), element-wise multiplication (chain rule)
- Number of matmul operations: forward vs. backward
- Memory access patterns: reading vs. writing gradients

**Hint**: Think about matrix multiplication gradients:
```
Forward:  y = x @ W       (one matmul)
Backward: grad_x = grad_y @ W.T     (one matmul)
          grad_W = x.T @ grad_y     (another matmul)
```

---

### Question 4: Graph Retention
**Scenario**: You're training a language model that processes sequences of varying lengths.

**Question**: When should you call `.zero_grad()`? What happens if you forget?

**Consider**:
- Gradient accumulation behavior (Question 2)
- Memory growth over multiple iterations
- Training correctness: what values do parameters see?

**Example**:
```python
for batch in dataloader:
    # Should zero_grad() go here?
    loss = model(batch)
    loss.backward()
    optimizer.step()
    # Or should zero_grad() go here?
```

---

### Question 5: Production Pattern
**Scenario**: PyTorch and TensorFlow use `requires_grad` flags instead of always tracking gradients for every tensor.

**Question**: Why? What's the performance benefit of making gradient tracking opt-in?

**Consider**:
- Memory: What gets stored when requires_grad=True vs. False?
- Compute: What operations are skipped when requires_grad=False?
- Typical model: What percentage of tensors need gradients?
  - Inputs (data): requires_grad = ?
  - Weights: requires_grad = ?
  - Intermediate activations: requires_grad = ?
  - Targets (labels): requires_grad = ?

**Hint**: In a typical training loop, think about:
- How many tensors are created per forward pass?
- How many of those tensors are actually parameters that need updates?
- What's the memory multiplier for gradient tracking?

---

### Reflection Prompts

After answering these questions, consider:
1. **Which surprised you most?** What behavior was counterintuitive?
2. **What trade-offs exist?** Memory vs. compute? Simplicity vs. efficiency?
3. **How does this connect to Module 01?** Why did we include requires_grad, grad, and backward() from the start?
4. **What production patterns emerged?** What choices would you make differently for a research prototype vs. production system?

These questions prepare you for Module 07 (Optimizers), where you'll use these gradients to actually update parameters and train models!
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Gradients Flow Automatically

**What you built:** An autograd engine that computes gradients through computation graphs.

**Why it matters:** Before autograd, you had to derive and code gradients by hand for every
operation—error-prone and tedious. Your engine does this automatically! When you call
`backward()`, gradients flow from the loss back through every operation to every parameter.

This is the magic behind deep learning. PyTorch, TensorFlow, and JAX all have autograd
engines at their core. You just built one yourself!
"""

# %%
def demo_autograd():
    """🎯 See gradients computed automatically."""
    print("🎯 AHA MOMENT: Gradients Flow Automatically")
    print("=" * 45)

    # Simple example: y = x^2, so dy/dx = 2x
    x = Tensor(np.array([3.0]), requires_grad=True)
    y = x * x  # y = x^2

    print(f"x = {x.data[0]}")
    print(f"y = x^2 = {y.data[0]}")

    # Backward pass computes gradient
    y.backward()

    # Show computed vs expected gradient
    expected_grad = 2 * x.data[0]
    print(f"\nExpected: dy/dx = 2x = 2 * {x.data[0]} = {expected_grad}")
    print(f"Computed: {x.grad[0]}")
    print(f"Match: {np.allclose(x.grad[0], expected_grad)}")

    print("\n✨ Gradients computed automatically—no manual derivatives!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_autograd()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Autograd Engine

Congratulations! You've built the gradient engine that makes neural networks learn!

### Key Accomplishments
- **Enhanced Tensor class** with backward() method (no new wrapper classes!)
- **Built computation graph tracking** for automatic differentiation
- **Implemented Function classes** (Add, Mul, Matmul, Sum) with correct gradients
- **Created enable_autograd()** function that activates gradients globally
- **Tested complex multi-layer** computation graphs with gradient propagation
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory overhead**: Computation graphs store tensors for backward pass (2x memory)
- **Gradient accumulation**: Allows processing large batches in smaller chunks
- **Backward pass cost**: Approximately same as forward pass (similar number of matmuls)
- **Graph retention**: Must call zero_grad() to prevent gradient accumulation across iterations

### Ready for Next Steps
Your autograd implementation enables optimization!
Export with: `tito module complete 06`

**Next**: Module 07 will add optimizers (SGD, Adam) that use these gradients to actually train neural networks!
"""
