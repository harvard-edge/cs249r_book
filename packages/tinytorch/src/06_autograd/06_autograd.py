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
4. **Complete the Function classes** from Module 01 with their backward() rules
5. **Test gradient correctness** with mathematical validation

**CRITICAL**: This module enhances the existing Tensor class - no new wrapper classes needed!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/06_autograd/autograd.ipynb`
**Building Side:** Code exports to `tinytorch.core.autograd`

```python
# How to use this module:
from tinytorch.core.autograd import method_of, no_grad
```

**Why this matters:**
- **Learning:** Complete autograd system enabling automatic differentiation
- **Production:** PyTorch-style computational graph and backward pass
- **Consistency:** All gradient operations in core.autograd
- **Integration:** Enhances existing Tensor without breaking anything

Let's get started!
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.autograd
#| export

import numpy as np
rng = np.random.default_rng(7)
from typing import Optional, List, Tuple
import sys
import os

from tinytorch.core.tensor import (
    Tensor, Function,
    Add, Sub, Mul, Div, MatMul,
    Reshape, Permute, Copy, Slice, MaskedFill,
    Sum, Mean, Max,
)
from tinytorch.core.activations import SigmoidFunction, ReLUFunction, TanhFunction, GELUFunction, SoftmaxFunction
from tinytorch.core.losses import LogSoftmax, MSEFunction, BinaryCrossEntropyFunction, CrossEntropyFunction

# Constants for numerical differentiation
EPSILON = 1e-7  # Small perturbation for numerical gradient computation

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01, 02, and 04 must be complete
- Module 01: Tensor and the operation classes (their forward halves)
- Module 02: the activation operations
- Module 04: the loss operations

This module imports those operation classes and completes each one with its
backward half. Nothing else is imported: the gradient engine stays independent
of any particular layer.

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `typing` (for type hints)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor.Tensor` - Core tensor operations

**Dependency Flow**:
```
Module 01 (Tensor) → Module 06 (Autograd) → Module 07 (Optimizers)
        ↓                     ↓                      ↓
   data structure      gradient engine        parameter updates
```

This module completes the operation classes built in Modules 01, 02, and 04
with their `backward()` halves and gives the Tensor class its `backward()` method. Everything downstream that trains -- optimizers,
training loops, transformers -- depends on the graph this module builds.
"""

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
│ Node 3: z=x*y (Mul)                    │ grad: None→22  │
│         saved: (x=2, y=3)              │ inputs: [x,y]  │
│ Node 4: w=z+5 (Add)                    │ grad: None→22  │
│         saved: (z=6, 5)                │ inputs: [z]    │
│ Node 5: L=w*w (Mul)                    │ grad: 1        │
│         saved: (w=11, w=11)            │ inputs: [w]    │
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
│ • inputs           ← Store data     │
│ • forward()        ← Module 01      │
│ • backward()       ← Module 06      │
└─────────────────────────────────────┘
          ↑
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
│  Add   │ │  Mul   │ │ MatMul │ │  Sum   │
│forward │ │forward │ │forward │ │forward │
└────────┘ └────────┘ └────────┘ └────────┘
```

Each operation inherits from Function and implements specific gradient rules.
"""

# %% [markdown]
"""
### Function Base Class: The Foundation of Autograd

The Function class is the foundation that makes autograd possible. Every differentiable operation (addition, multiplication, etc.) inherits from this class.

**Why Functions Matter:**
- They remember inputs needed for backward pass
- They implement gradient computation via backward()
- They connect to form computation graphs
- They enable the chain rule to flow gradients

**The Pattern:**
```
Forward:  inputs → Function.forward() → output
Backward: grad_output → Function.backward() → grad_inputs
```

This pattern enables the chain rule to flow gradients through complex computations.

The `Function` class itself was built in Module 01 (`tinytorch.core.tensor`), where every operation received its `forward()`. This module fills in the `backward()` of those same classes. The `method_of` helper below attaches each one to its class, so the exported file reads as a list of completions with the class named on the line above each.
"""

# %% nbgrader={"grade": false, "grade_id": "function-base", "solution": false}
#| export
def method_of(cls):
    """
    Attach the decorated function to `cls` under the function's own name.

    Module 01 defined every operation with a forward() and left backward()
    raising NotImplementedError. This module writes those backward() halves:

    ```python
    @method_of(Add)
    def backward(self, grad_output):
        ...
    ```

    is the same as writing backward() inside `class Add` in Module 01, without
    editing that module. The attachment happens when this module is imported,
    which is a teaching convenience: PyTorch defines both halves in one place.
    The same helper gives Function.apply() its recording lines and Tensor its
    backward() method further down.
    """
    def attach(fn):
        raw = fn.__func__ if isinstance(fn, (classmethod, staticmethod)) else fn
        raw.__qualname__ = f"{cls.__name__}.{raw.__name__}"   # so tracebacks read Add.backward
        setattr(cls, raw.__name__, fn)
        return fn
    return attach

# %% [markdown]
"""
### Operation Functions: Implementing Gradient Rules

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
### Understanding Broadcasting in Gradients

Before implementing gradient operations, we need to understand a critical challenge:
**Broadcasting in Forward Pass vs. Gradient Reduction in Backward Pass**

#### The Broadcasting Problem

NumPy automatically broadcasts tensors of different shapes during forward operations:

```
Forward Pass (Broadcasting):
┌─────────────────────────────────────────────────────────────┐
│ Example: Adding bias to batched data                        │
│                                                             │
│ x:    (32, 128)  ← Batch of 32 samples, 128 features        │
│ bias: (128,)     ← Just 128 features (no batch dimension)   │
│                                                             │
│ Forward: y = x + bias                                       │
│          NumPy broadcasts bias from (128,) to (32, 128)     │
│          Result shape: (32, 128)                            │
└─────────────────────────────────────────────────────────────┘

Backward Pass (Gradient Reduction):
┌─────────────────────────────────────────────────────────────┐
│ grad_output: (32, 128)  ← Gradient from upstream            │
│                                                             │
│ grad_x:    (32, 128)    ← Same shape as x ✓                 │
│ grad_bias: (128,)       ← Must match bias shape!            │
│                                                             │
│ Problem: grad_output is (32, 128) but bias is (128,)        │
│ Solution: Sum gradients over batch dimension                │
│           grad_bias = grad_output.sum(axis=0)               │
│           Result: (128,) ✓                                  │
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
### 🧪 Unit Test: Broadcast Gradient Reduction

This test validates our gradient reduction helper handles all broadcasting scenarios.

**What we're testing**: Correct shape reduction for gradients after broadcasting
**Why it matters**: Wrong gradient shapes cause crashes or silent bugs in training
**Expected**: Reduced gradients match original tensor shapes for all broadcasting patterns
"""

# %% nbgrader={"grade": true, "grade_id": "test-reduce-broadcast-grad", "locked": true, "points": 5}
def test_unit_reduce_broadcast_grad():
    """🧪 Test _reduce_broadcast_grad helper function."""
    print("🧪 Unit Test: _reduce_broadcast_grad...")
    
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
### Add.backward: Gradient Rules for Addition

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
#| exporti
@method_of(Add)
def backward(self, grad_output):
    """
    Gradient computation for tensor addition.

    **Mathematical Rule:** If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1

    **Key Insight:** Addition distributes gradients equally to both inputs.
    The gradient flowing backward is passed unchanged to each input.

    **Broadcasting Handling:** When input shapes differ due to broadcasting,
    we sum gradients appropriately to match original tensor shapes.

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
    1. Extract input tensors from self.inputs
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
    a, b = self.inputs
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
### Mul.backward: Gradient Rules for Element-wise Multiplication

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
#| exporti
@method_of(Mul)
def backward(self, grad_output):
    """
    Gradient computation for tensor multiplication.

    **Mathematical Rule:** If z = a * b, then ∂z/∂a = b and ∂z/∂b = a

    **Key Insight:** Each input's gradient equals the gradient output
    multiplied by the OTHER input's value (product rule).

    **Applications:** Used in weight scaling, binary masking,
    and anywhere element-wise multiplication occurs.

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
    1. Extract input tensors a, b from self.inputs
    2. Initialize grad_a and grad_b to None
    3. For first input (a): if requires_grad:
       - Compute grad_a = grad_output * b
       - Use _reduce_broadcast_grad() to handle shape mismatch
    4. For second input (b): if requires_grad:
       - Compute grad_b = grad_output * a
       - Use _reduce_broadcast_grad() to handle shape mismatch
    5. Return tuple (grad_a, grad_b)

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
    - b is always a Tensor here (Tensor.__mul__ wrapped any scalar), so b.data is safe
    """
    ### BEGIN SOLUTION
    a, b = self.inputs
    grad_a = grad_b = None

    # Gradient for first input: grad_output * b
    if isinstance(a, Tensor) and a.requires_grad:
        grad_a = grad_output * b.data
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
### Sub.backward: Gradient Rules for Subtraction

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
#| exporti
@method_of(Sub)
def backward(self, grad_output):
    """
    Gradient computation for tensor subtraction.

    **Mathematical Rule:** If z = a - b, then ∂z/∂a = 1 and ∂z/∂b = -1

    Compute gradients for subtraction.

    Returns:
        Tuple of (grad_a, grad_b) where grad_b is negated

    TODO: Implement gradient computation for subtraction operation.

    APPROACH:
    1. Extract input tensors from self.inputs
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
    a, b = self.inputs
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
### Div.backward: Gradient Rules for Division

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
#| exporti
@method_of(Div)
def backward(self, grad_output):
    """
    Gradient computation for tensor division.

    **Mathematical Rule:** If z = a / b, then:
    - ∂z/∂a = 1/b
    - ∂z/∂b = -a/b²

    Compute gradients for division using quotient rule.

    Returns:
        Tuple of (grad_a, grad_b)

    TODO: Implement gradient computation for division operation.

    APPROACH:
    1. Extract input tensors from self.inputs
    2. Initialize grad_a and grad_b to None
    3. For first input (a): if requires_grad:
       - Compute grad_a = grad_output / b
       - Use _reduce_broadcast_grad() to handle shape mismatch
    4. For second input (b): if requires_grad:
       - Compute grad_b = -grad_output * a / (b²)
       - Use _reduce_broadcast_grad() to handle shape mismatch
    5. Return tuple (grad_a, grad_b)

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
    - b is always a Tensor here (Tensor.__truediv__ wrapped any scalar), so b.data is safe
    - b² means b.data ** 2
    """
    ### BEGIN SOLUTION
    a, b = self.inputs
    grad_a = grad_b = None

    if isinstance(a, Tensor) and a.requires_grad:
        # ∂(a/b)/∂a = 1/b
        grad_a = grad_output / b.data
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
### MatMul.backward: Gradient Rules for Matrix Multiplication

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
#| exporti
@method_of(MatMul)
def backward(self, grad_output):
    """
    Gradient computation for matrix multiplication.

    **Mathematical Rule:** If Z = A @ B, then:
    - ∂Z/∂A = grad_Z @ B.T
    - ∂Z/∂B = A.T @ grad_Z

    **Key Insight:** Matrix multiplication gradients involve transposing
    one input and multiplying with the gradient output.

    **Applications:** Core operation in neural networks for computing outputs
    in linear layers and combining feature representations.

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
    1. Extract input tensors a, b from self.inputs
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
    a, b = self.inputs
    grad_a = grad_b = None

    # Gradient for first input: grad_output @ b.T
    if isinstance(a, Tensor) and a.requires_grad:
        if b.data.ndim >= 2:
            # Batched: transpose only the last two dims
            b_T = np.swapaxes(b.data, -2, -1)
            grad_a = np.matmul(grad_output, b_T)
        else:
            # 1D b: A(m,k) @ b(k,) -> out(m,)
            # grad_A = outer(grad_output, b): (m,) x (k,) -> (m, k)
            grad_a = np.outer(grad_output, b.data)

    # Gradient for second input: a.T @ grad_output
    if isinstance(b, Tensor) and b.requires_grad:
        if a.data.ndim >= 2:
            # Batched: transpose only the last two dims
            a_T = np.swapaxes(a.data, -2, -1)
            grad_b = np.matmul(a_T, grad_output)
        else:
            # 1D a: a(k,) @ B(k,n) -> out(n,)
            # grad_B = outer(a, grad_output): (k,) x (n,) -> (k, n)
            grad_b = np.outer(a.data, grad_output)

    return grad_a, grad_b
    ### END SOLUTION

# %% [markdown]
"""
### Permute: Gradient Rules for Transpose and General Reordering

Permute is transpose for more than two axes. The gradient is the same idea, with
one extra step: you have to invert the permutation rather than just reapply it.

**Mathematical Principle:**
```
If Z = A.permute(axes), then grad_A = grad_Z.permute(argsort(axes))
```

**Why argsort and not axes:**
```
axes = (2, 0, 1)  means  Z's axis 0 came from A's axis 2
                          Z's axis 1 came from A's axis 0
                          Z's axis 2 came from A's axis 1
To send gradients home you need the inverse map: (1, 2, 0) = argsort(axes)
```

Reapplying `axes` works for a 2D transpose only because swapping twice is the
identity. It is wrong in general, which is what makes this one worth writing out.
"""

# %% nbgrader={"grade": false, "grade_id": "permute-backward", "solution": true}
#| exporti
@method_of(Permute)
def backward(self, grad_output):
    """
    Gradient computation for arbitrary axis permutation (general transpose).

    **Mathematical Rule:** If Y = X.permute(axes), then:
    - ∂Y/∂X = grad_Y.permute(inverse_axes)

    **Example:** If axes = (0, 2, 1, 3), the inverse is (0, 2, 1, 3) (self-inverse).
    More generally, if axes = (2, 0, 1), the inverse is (1, 2, 0).

    **Key Insight:** To reverse a permutation, we need to know where each axis went.
    If axis i went to position axes[i], then in the inverse, position axes[i] should go to i.

    **Applications:** Rearranging tensor dimensions for different computation patterns (e.g., swapping batch and feature dimensions).

    Compute gradient for permutation.

    The gradient is permuted back using the inverse permutation.

    **Mathematical Foundation:**
    - ∂(X.permute(axes))/∂X = grad_output.permute(inverse_axes)

    TODO: Implement gradient computation for permutation operation.

    APPROACH:
    1. Extract input tensor x from self.inputs
    2. Initialize grad_x to None
    3. If x requires gradients:
       - Permute grad_output using np.argsort(self.axes)
       - Use np.transpose(grad_output, np.argsort(self.axes))
    4. Return tuple (grad_x,)

    EXAMPLE:
    >>> X = Tensor([[[1, 2], [3, 4]]], requires_grad=True)  # (1, 2, 2)
    >>> Y = X.permute((0, 2, 1))  # Swap last two dims → (1, 2, 2)
    >>> # During backward: inverse axes computed with np.argsort(self.axes)
    >>> # grad_X = np.transpose(grad_output, inverse_axes)

    HINTS:
    - inverse_axes = tuple(np.argsort(self.axes)); if axes[i] = j then inverse_axes[j] = i
    - Apply np.transpose(grad_output, inverse_axes)
    - Return as single-element tuple: (grad_x,)
    """
    ### BEGIN SOLUTION
    x, = self.inputs
    grad_x = None

    if isinstance(x, Tensor) and x.requires_grad:
        # Permute gradient back to original axis order.
        # If axes[i] = j, then inverse_axes[j] = i.
        inverse_axes = tuple(np.argsort(self.axes))
        grad_x = np.transpose(grad_output, inverse_axes)

    return (grad_x,)
    ### END SOLUTION

# %% [markdown]
"""
### Slice.backward: Gradient Rules for Indexing

Slicing keeps some values and drops the rest. Gradients follow the same split:
the kept positions receive their gradient, and everything else receives zero.

**Mathematical Principle:**
```
If Z = A[key], then grad_A = zeros_like(A); grad_A[key] = grad_Z
```

**Why a zeros scaffold:**
```
Forward:  A=[1,2,3,4,5] → A[1:3] → Z=[2,3]
Backward: grad_Z=[g0,g1] → grad_A=[0,g0,g1,0,0]
                            ↑ positions that were never read contributed
                              nothing to the output, so their gradient is 0
```

This is the first backward that has to build a tensor the shape of its INPUT
rather than reshaping its output. Embedding lookup in Module 11 is the same
pattern at scale.
"""

# %% nbgrader={"grade": false, "grade_id": "slice-backward", "solution": true}
#| exporti
@method_of(Slice)
def backward(self, grad_output):
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
    1. Extract input tensor from self.inputs
    2. Initialize grad_input to None
    3. If tensor requires gradients:
       - Create zeros array: grad_input = np.zeros(the input's shape (self.inputs[0].shape))
       - Scatter gradients back: np.add.at(grad_input, self.key, grad_output)
    4. Return tuple (grad_input,)

    EXAMPLE:
    >>> X = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    >>> Y = X[:3]  # Slice first 3 elements → [1, 2, 3]
    >>> # During backward: grad_output = [1, 1, 1]
    >>> # grad_X = [1, 1, 1, 0, 0] (gradients only for sliced positions)

    HINTS:
    - Create zero gradient array with original tensor shape
    - np.add.at(grad_input, self.key, grad_output) scatters through any index
      (single index, ranges, tuples) and ACCUMULATES when an index repeats,
      where plain assignment would silently keep only the last value
    - Return as single-element tuple: (grad_input,)
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs
    grad_input = None

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        # Create gradient array with same shape as original tensor
        grad_input = np.zeros(tensor.shape, dtype=np.float32)

        # Scatter gradients back into the sliced positions (the inverse of the
        # forward slice). np.add.at accumulates when an index was used twice.
        np.add.at(grad_input, self.key, grad_output)

    return (grad_input,)
    ### END SOLUTION

# %% [markdown]
"""
### Reshape.backward: Gradient Rules for Changing Shape

Reshape reinterprets the same values under a new shape. Nothing is added,
dropped, or combined, so the gradient just needs its original shape back.

**Mathematical Principle:**
```
If Z = A.reshape(new_shape), then grad_A = grad_Z.reshape(A.shape)
```

**Why the input shape must be saved:**
```
Forward:  A(2,6) → reshape(3,4) → Z(3,4)
Backward: grad_Z(3,4) → reshape(?, ?) → grad_A must be (2,6)
                                ↑ the only way to know is to have stored A.shape
```

Every backward in this group saves something from the forward pass. Reshape
saves a shape, slice saves a key, permute saves an axis order. That is the
general rule: the backward pass needs whatever the forward pass consumed.
"""

# %% nbgrader={"grade": false, "grade_id": "reshape-backward", "solution": true}
#| exporti
@method_of(Reshape)
def backward(self, grad_output):
    """
    Gradient computation for reshape operation.

    **Mathematical Rule:** If Y = X.reshape(new_shape), then:
    - ∂Y/∂X = grad_Y.reshape(X.shape)

    **Key Insight:** Reshape just rearranges the same elements.
    The gradient is simply reshaped back to the original shape!

    **Applications:** Flattening tensors for linear layers, reshaping
    between convolutional and dense layers.

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
    1. Extract input tensor x from self.inputs
    2. Initialize grad_x to None
    3. If x requires gradients:
       - Reshape grad_output back to original shape
       - Use grad_output.reshape(the input's shape (self.inputs[0].shape))
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
    x, = self.inputs
    grad_x = None

    if isinstance(x, Tensor) and x.requires_grad:
        # Reshape gradient back to original shape
        grad_x = grad_output.reshape(x.shape)

    return (grad_x,)
    ### END SOLUTION

# %% [markdown]
"""
### Sum.backward: Gradient Rules for Reduction Operations

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
#| exporti
@method_of(Sum)
def backward(self, grad_output):
    """
    Gradient computation for tensor sum.

    **Mathematical Rule:** If z = sum(a), then ∂z/∂a[i] = 1 for all i

    **Key Insight:** Sum distributes the gradient equally to all input elements.
    The gradient is broadcast from the reduced output back to input shape.

    **Applications:** Used in loss functions, mean operations, and
    anywhere tensor reduction occurs.

    Compute gradients for sum operation.

    Args:
        grad_output: Gradient flowing backward from output

    Returns:
        Tuple containing gradient for the input tensor

    **Mathematical Foundation:**
    - ∂sum(a)/∂a[i] = 1 → grad_a = ones_like(a) * grad_output

    TODO: Implement gradient computation for sum reduction operation.

    APPROACH:
    1. Extract input tensor from self.inputs
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
    tensor, = self.inputs

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
### Mean, Max, Copy, MaskedFill.backward: The Remaining Module 01 Operations

Every operation Module 01 gave a forward needs a backward, or the first `loss.mean()` you write will stop here. These four are short:

- **Mean** is a Sum divided by the number of elements averaged, so each element receives the output gradient over that count.
- **Max** routes the gradient to the element that won and nothing to the rest. `apply()` saves the operation's output on the node as `self.output`, so the winners are wherever the input equals it. If several elements tie, the gradient is split evenly among them (PyTorch's `max()` picks one of them instead).
- **Copy** changed nothing, so the gradient passes straight through.
- **MaskedFill** overwrote the masked positions with a constant, so they receive no gradient; everything else passes through.
"""

# %% nbgrader={"grade": false, "grade_id": "mean-max-copy-maskedfill-backward", "solution": true}
#| exporti
def _expand_reduced(grad_output, shape, axis, keepdims):
    """Broadcast a reduced gradient back to the shape it was reduced from."""
    if axis is not None and not keepdims:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        for ax in sorted(ax % len(shape) for ax in axes):
            grad_output = np.expand_dims(grad_output, axis=ax)
    return np.broadcast_to(grad_output, shape)


@method_of(Mean)
def backward(self, grad_output):
    """
    Gradient computation for tensor mean.

    **Mathematical Rule:** If z = mean(a) over N elements, then ∂z/∂a[i] = 1/N

    TODO: Spread the gradient evenly, divided by the number of elements averaged.

    APPROACH:
    1. tensor, = self.inputs
    2. count = tensor.data.size // self.output.data.size (elements averaged per output)
    3. Expand grad_output back to tensor.data.shape with _expand_reduced, divide by count
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        count = tensor.data.size // self.output.data.size
        return _expand_reduced(grad_output, tensor.data.shape, self.axis, self.keepdims) / count,
    return None,
    ### END SOLUTION


@method_of(Max)
def backward(self, grad_output):
    """
    Gradient computation for tensor max.

    **Mathematical Rule:** ∂max(a)/∂a[i] = 1 where a[i] is the maximum, else 0

    TODO: Send the gradient to the winning positions only.

    APPROACH:
    1. Expand self.output.data back to the input shape with _expand_reduced
    2. winners = (tensor.data == expanded output)
    3. Return the expanded grad_output times winners, divided by the number of ties per output
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if not (isinstance(tensor, Tensor) and tensor.requires_grad):
        return None,

    shape = tensor.data.shape
    # 1. Which positions won? Compare the input against the max broadcast back.
    winners = tensor.data == _expand_reduced(self.output.data, shape, self.axis, self.keepdims)
    # 2. Share the gradient equally among tied winners.
    if self.axis is None:
        ties = np.sum(winners)
    else:
        ties = _expand_reduced(np.sum(winners, axis=self.axis, keepdims=True), shape, self.axis, True)
    # 3. Route the upstream gradient to the winners only.
    grad = _expand_reduced(grad_output, shape, self.axis, self.keepdims)
    return grad * winners / ties,
    ### END SOLUTION


@method_of(Copy)
def backward(self, grad_output):
    """
    Gradient computation for a copy: the values did not change, so neither does the gradient.

    TODO: Return grad_output unchanged (as a one-element tuple).
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        return grad_output,
    return None,
    ### END SOLUTION


@method_of(MaskedFill)
def backward(self, grad_output):
    """
    Gradient computation for masked_fill.

    **Mathematical Rule:** Masked positions received a constant, so their gradient is 0.
    Every other position passed through unchanged.

    TODO: Zero the gradient where self.mask is True.

    HINT: grad_output * ~self.mask
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        return grad_output * ~self.mask,
    return None,
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Function Classes

This test validates our Function classes compute gradients correctly.

**What we're testing**: Forward and backward passes for each operation
**Why it matters**: These are the building blocks of autograd
**Expected**: Correct gradients that satisfy mathematical definitions
"""

# %% nbgrader={"grade": true, "grade_id": "test-function-classes", "locked": true, "points": 15}
def test_unit_function_classes():
    """🧪 Test Function classes."""
    print("🧪 Unit Test: Function Classes...")

    # Note: We call backward() on each operation directly, in isolation, before
    # Tensor.backward() exists. requires_grad is a constructor argument from Module 01.

    # Test Add.backward
    a = Tensor([1, 2, 3])
    a.requires_grad = True
    b = Tensor([4, 5, 6])
    b.requires_grad = True
    add_func = Add(a, b)
    grad_output = np.array([1, 1, 1])
    grad_a, grad_b = add_func.backward(grad_output)
    assert np.allclose(grad_a, grad_output), f"Add.backward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, grad_output), f"Add.backward grad_b failed: {grad_b}"

    # Test Mul.backward
    mul_func = Mul(a, b)
    grad_a, grad_b = mul_func.backward(grad_output)
    assert np.allclose(grad_a, b.data), f"Mul.backward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, a.data), f"Mul.backward grad_b failed: {grad_b}"

    # Test MatMul.backward
    a_mat = Tensor([[1, 2], [3, 4]])
    a_mat.requires_grad = True
    b_mat = Tensor([[5, 6], [7, 8]])
    b_mat.requires_grad = True
    matmul_func = MatMul(a_mat, b_mat)
    grad_output = np.ones((2, 2))
    grad_a, grad_b = matmul_func.backward(grad_output)
    assert grad_a.shape == a_mat.shape, f"MatMul.backward grad_a shape: {grad_a.shape}"
    assert grad_b.shape == b_mat.shape, f"MatMul.backward grad_b shape: {grad_b.shape}"

    print("✅ Function classes work correctly!")

if __name__ == "__main__":
    test_unit_function_classes()

# %% [markdown]
"""
### 🧪 Unit Test: Broadcasting in Gradients
This test validates that gradient reduction works correctly when operations
involve broadcasting. This is crucial for real-world training where bias terms,
normalization parameters, and other operations broadcast over batches.

**What we're testing**: Gradient shape correctness with broadcasting
**Why it matters**: Without proper reduction, bias gradients would have wrong shapes
**Expected**: Gradients match original tensor shapes after reduction
"""

# %% nbgrader={"grade": true, "grade_id": "test-broadcast-gradients", "locked": true, "points": 10}
def test_unit_broadcast_gradients():
    """🧪 Test gradient broadcasting reduction."""
    print("🧪 Unit Test: Broadcasting in Gradients...")

    # Note: We call backward() on each operation directly, in isolation, before
    # Tensor.backward() exists.

    # Scenario 1: Bias-like broadcasting (most common case)
    # Shape: (batch, features) + (features,) → (batch, features)
    x = Tensor(rng.standard_normal((4, 3)))
    x.requires_grad = True
    bias = Tensor(np.ones(3))
    bias.requires_grad = True
    
    add_func = Add(x, bias)
    grad_output = np.ones((4, 3))
    grad_x, grad_bias = add_func.backward(grad_output)
    
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
    x = Tensor(rng.standard_normal((3, 4)))
    x.requires_grad = True
    scalar_val = 5.0
    
    add_func = Add(x, Tensor(scalar_val))
    grad_output = np.ones((3, 4))
    grad_x, grad_scalar = add_func.backward(grad_output)
    
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    print("  ✓ Scalar broadcasting works")
    
    # Scenario 3: Multiple dimension broadcasting
    # Shape: (32, 10, 5) + (10, 1) → (32, 10, 5)
    x = Tensor(rng.standard_normal((32, 10, 5)))
    x.requires_grad = True
    y = Tensor(rng.standard_normal((10, 1)))
    y.requires_grad = True
    
    mul_func = Mul(x, y)
    grad_output = np.ones((32, 10, 5))
    grad_x, grad_y = mul_func.backward(grad_output)
    
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    assert grad_y.shape == y.data.shape, \
        f"Expected grad_y shape {y.data.shape}, got {grad_y.shape}"
    
    print("  ✓ Multi-dimension broadcasting works")
    
    # Scenario 4: Test all operations (Add, Mul, Sub, Div)
    a = Tensor(rng.standard_normal((8, 16)))
    a.requires_grad = True
    b = Tensor(rng.standard_normal(16))
    b.requires_grad = True
    
    # Test Addition
    add_func = Add(a, b)
    grad_a, grad_b = add_func.backward(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"Add.backward: Expected (16,), got {grad_b.shape}"
    
    # Test Multiplication
    mul_func = Mul(a, b)
    grad_a, grad_b = mul_func.backward(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"Mul.backward: Expected (16,), got {grad_b.shape}"
    
    # Test Subtraction
    sub_func = Sub(a, b)
    grad_a, grad_b = sub_func.backward(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"Sub.backward: Expected (16,), got {grad_b.shape}"
    
    # Test Division
    div_func = Div(a, b)
    grad_a, grad_b = div_func.backward(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"Div.backward: Expected (16,), got {grad_b.shape}"
    
    print("  ✓ All operations handle broadcasting correctly")
    
    # Scenario 5: Real-world case - Linear layer gradient
    # Simulates: output = input @ weight + bias
    # where bias is (out_features,) and output is (batch, out_features)
    batch_size, out_features = 32, 128
    output_grad = rng.standard_normal((batch_size, out_features))
    bias = Tensor(np.zeros(out_features))
    bias.requires_grad = True

    # In real Linear layer, bias gradient comes from output gradient
    activations = Tensor(np.zeros((batch_size, out_features)))
    activations.requires_grad = True
    add_func = Add(activations, bias)
    _, grad_bias = add_func.backward(output_grad)
    
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
## 🔧 Integration: Enhancing Tensor with Autograd Capabilities

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
2. **Complete operations** - Give every operation from Modules 01-04 its backward() rule
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
### Gradient Rules for Activations and Losses

The Function classes above covered arithmetic and shape. This next group covers
the two remaining kinds of node in a network's graph: the activations that sit
between layers, and the losses that sit at the very end.

They share a property the earlier group did not. Every one of them can compute
its gradient from values it already has -- usually its own output -- so none of
them re-runs the forward computation. Watch for that as you write them; it is
the difference between an autograd engine that is usable and one that doubles
the cost of every backward pass.

```
ReLU     -> gate: gradient passes or it does not
Sigmoid  -> z·(1-z)          from the output alone
Tanh     -> 1-z²             from the output alone
Softmax  -> outputs coupled: raising one lowers the others
GELU     -> smooth gate: both what passes AND how open the gate is
MSE, BCE -> the losses, where every backward pass begins
```
"""

# %% nbgrader={"grade": false, "grade_id": "relu-backward", "solution": true}
#| exporti
@method_of(ReLUFunction)
def backward(self, grad_output):
    """
    Gradient computation for ReLU activation.

    ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0

    Compute gradient for ReLU.

    TODO: Implement gradient computation for ReLU activation.

    APPROACH:
    1. Extract input tensor from self.inputs
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
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        # ReLU gradient: 1 if x > 0, else 0
        relu_grad = (tensor.data > 0).astype(np.float32)
        return grad_output * relu_grad,
    return None,
    ### END SOLUTION

# %% [markdown]
"""
### SigmoidFunction.backward: Gradient Rules for Sigmoid

Sigmoid's derivative has a property worth exploiting: it can be written entirely
in terms of the OUTPUT, so the backward pass never needs the input.

**Mathematical Principle:**
```
If z = σ(a) = 1 / (1 + e^-a), then ∂z/∂a = z · (1 - z)
```

**Why saving the output beats saving the input:**
```
Saving a:  recompute σ(a), then σ(a)·(1-σ(a))   -> one exp per element, again
Saving z:  z·(1-z) directly                     -> two multiplies, no exp
```

**The systems consequence**: the gradient peaks at 0.25 (when z = 0.5) and decays
toward zero at both tails. Stack ten sigmoid layers and the gradient reaching the
first one is scaled by at most 0.25^10 -- about one in a million. That is the
vanishing gradient problem, and it is why ReLU replaced sigmoid in hidden layers.
"""

# %% nbgrader={"grade": false, "grade_id": "sigmoid-backward", "solution": true}
#| exporti
@method_of(SigmoidFunction)
def backward(self, grad_output):
    """
    Gradient computation for sigmoid activation.

    Sigmoid: σ(x) = 1/(1 + exp(-x))
    Derivative: σ'(x) = σ(x) * (1 - σ(x))

    Compute gradient for sigmoid.

    TODO: Implement gradient computation for sigmoid activation.

    APPROACH:
    1. Extract input tensor from self.inputs
    2. If tensor requires gradients:
       - Use saved output: σ(x) = self.output.data
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
    - Output is saved as self.output by apply(), which this module completes further down
    - This avoids recomputing sigmoid during backward pass
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        # σ'(x) = σ(x) * (1 - σ(x)), using the output apply() saved on this node
        sigmoid_grad = self.output.data * (1 - self.output.data)
        return grad_output * sigmoid_grad,
    return None,
    ### END SOLUTION

# %% [markdown]
"""
### TanhFunction.backward: Gradient Rules for Tanh

Tanh is sigmoid's zero-centered cousin, and its gradient is also expressible from
the output alone.

**Mathematical Principle:**
```
If z = tanh(a), then ∂z/∂a = 1 - z²
```

**Compared with sigmoid:**
```
tanh max gradient: 1.0   (at z = 0)
sigmoid max gradient: 0.25
```

Tanh saturates too, but its gradient peaks four times higher, so it vanishes more
slowly through depth. That is the whole reason tanh outlasted sigmoid inside RNNs
long after ReLU had taken over feed-forward networks.
"""

# %% nbgrader={"grade": false, "grade_id": "tanh-backward", "solution": true}
#| exporti
@method_of(TanhFunction)
def backward(self, grad_output):
    """
    Gradient computation for tanh activation.

    Tanh: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
    Derivative: tanh'(x) = 1 - tanh(x)²

    Compute gradient for tanh.

    TODO: Implement gradient computation for tanh activation.

    APPROACH:
    1. Extract input tensor from self.inputs
    2. If tensor requires gradients:
       - Use saved output: tanh(x) = self.output.data
       - Compute tanh derivative: tanh'(x) = 1 - tanh(x)²
       - Multiply by grad_output: grad_output * tanh_grad
       - Return as tuple: (result,)
    3. Else return (None,)

    EXAMPLE:
    >>> X = Tensor([0.0], requires_grad=True)
    >>> Y = tanh(X)  # Y = 0.0
    >>> # During backward: grad_output = 1
    >>> # tanh'(0) = 1 - 0² = 1
    >>> # grad_X = 1 * 1 = 1

    HINTS:
    - Tanh derivative: tanh'(x) = 1 - tanh(x)²
    - Output is saved as self.output by apply(), which this module completes further down
    - This avoids recomputing tanh during backward pass
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        # tanh'(x) = 1 - tanh(x)², using the output apply() saved on this node
        tanh_grad = 1 - self.output.data ** 2
        return grad_output * tanh_grad,
    return None,
    ### END SOLUTION

# %% [markdown]
"""
### SoftmaxFunction.backward: Gradient Rules for a Coupled Output

Softmax is the first backward here where outputs are not independent. Raising one
logit lowers every other probability, because they must still sum to one. The
gradient has to account for that coupling.

**Mathematical Principle:**
```
If z = softmax(a), the Jacobian is
    ∂z_i/∂a_j = z_i · (δ_ij - z_j)
which contracts with an incoming gradient g to
    grad_a = z · (g - Σ_k g_k · z_k)
```

**Why the subtraction term exists:**
```
Σ_k g_k · z_k  is the probability-weighted average of the incoming gradient.
Subtracting it removes the component that would push all logits the same way --
which softmax ignores anyway, since adding a constant to every logit changes
nothing.
```

Computing the full n×n Jacobian would cost O(n²) per sample. The contracted form
above is O(n). For a 50,000-token vocabulary that is the difference between
feasible and not.
"""

# %% nbgrader={"grade": false, "grade_id": "softmax-backward", "solution": true}
#| exporti
@method_of(SoftmaxFunction)
def backward(self, grad_output):
    """
    Gradient computation for softmax activation.

    Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x))
    Derivative: ∂softmax/∂x[i] = softmax[i] * (δ[i,j] - softmax[j])

    For gradient computation:
    grad_x[i] = softmax[i] * (grad_y[i] - sum(grad_y * softmax))

    **Key Insight:** The gradient depends on all elements of softmax due to
    the normalization, not just the element being differentiated.

    Compute gradient for softmax.

    Mathematical formula:
    ∂L/∂x[i] = softmax[i] * (∂L/∂y[i] - sum_j(∂L/∂y[j] * softmax[j]))

    This can be vectorized as:
    grad_x = softmax * (grad_y - sum(grad_y * softmax, keepdims=True))

    TODO: Implement gradient computation for softmax activation.

    APPROACH:
    1. Extract input tensor from self.inputs
    2. If tensor requires gradients:
       - Compute sum term: np.sum(grad_output * self.output.data, axis=self.dim, keepdims=True)
       - Compute gradient: self.output.data * (grad_output - sum_term)
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
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        softmax = self.output.data
        # Compute sum(grad_output * softmax) along the softmax dimension
        sum_term = np.sum(grad_output * softmax, axis=self.dim, keepdims=True)

        # Softmax gradient: softmax * (grad_output - sum_term)
        grad_x = softmax * (grad_output - sum_term)

        return (grad_x,)
    return (None,)
    ### END SOLUTION

# %% [markdown]
"""
### GELUFunction.backward: Gradient Rules for the Transformer Activation

GELU is the activation inside every transformer MLP you will build in Module 13.
Unlike ReLU it is smooth everywhere, so its gradient is defined at every point,
including zero.

**Mathematical Principle (the sigmoid form Module 02 built):**
```
z = a·s            where s = σ(1.702·a) and σ is Module 02's sigmoid

∂z/∂a = s + a·1.702·s·(1 - s)
        └┘   └── how the gate itself moves with a ──┘
        how much passes
```

**Why two terms and not one:**
```
ReLU's gate is a step: it is either open or shut, so only the first term exists.
GELU's gate is smooth, so changing a changes BOTH what passes through and how far
open the gate is. Both effects carry gradient.
```

**The systems consequence**: GELU costs an exponential and several multiplies per
element where ReLU costs one comparison. Module 17 will fuse the whole expression
into a single pass over memory for exactly that reason.
"""

# %% nbgrader={"grade": false, "grade_id": "gelu-backward", "solution": true}
#| exporti
@method_of(GELUFunction)
def backward(self, grad_output):
    """
    Gradient computation for GELU activation.

    GELU: f(x) = x * Φ(x) where Φ is the CDF of the standard normal.
    Module 02 implements the sigmoid form gelu(x) ≈ x * σ(1.702x), so this
    backward differentiates that same expression.

    **Key Insight:** GELU is smooth, so negative inputs still receive a small
    gradient, unlike ReLU's hard zero.

    TODO: Implement gradient computation for GELU activation.

    APPROACH:
    1. Extract input tensor from self.inputs
    2. If tensor requires gradients:
       - Compute s = σ(1.702 * x) with Module 02's stable sigmoid
       - Product rule: gelu_grad = s + x * 1.702 * s * (1 - s)
       - Multiply by grad_output
    3. Else return (None,)

    HINTS:
    - SigmoidFunction().forward(1.702 * x) gives a stable σ without rewriting it
    - The two terms are "what passes" and "how the gate moves"; both carry gradient
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        x = tensor.data
        # forward: gelu(x) = x * sig(1.702x)   (Module 02)
        # d/dx [x * sig(1.702x)] = sig(1.702x) + x * 1.702 * sig(1.702x) * (1 - sig(1.702x))
        sig = SigmoidFunction().forward(1.702 * x)   # Module 02's stable sigmoid
        gelu_grad = sig + x * 1.702 * sig * (1.0 - sig)

        return (grad_output * gelu_grad,)
    return (None,)
    ### END SOLUTION

# %% [markdown]
"""
### MSEFunction.backward: Gradient Rules for Mean Squared Error

Loss functions are where the backward pass begins. MSE has the gentlest gradient
of the two you will build, and the most obvious.

**Mathematical Principle:**
```
If L = (1/n)·Σ(pred - target)², then ∂L/∂pred = (2/n)·(pred - target)
```

**Why the gradient is just the scaled error:**
```
prediction too high  -> (pred - target) > 0 -> gradient positive -> step down
prediction too low   -> (pred - target) < 0 -> gradient negative -> step up
prediction exact     -> gradient 0          -> no update
```

**The systems consequence**: the gradient is proportional to the error, so a
single wildly wrong prediction produces a wildly large gradient. That is why MSE
is sensitive to outliers, and why the training loop in Module 08 will need gradient
clipping.
"""

# %% nbgrader={"grade": false, "grade_id": "mse-backward", "solution": true}
#| exporti
@method_of(MSEFunction)
def backward(self, grad_output):
    """
    Gradient computation for Mean Squared Error Loss.

    MSE: L = mean((predictions - targets)²)
    Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N

    Compute gradient for MSE loss.

    TODO: Implement gradient computation for Mean Squared Error loss.

    APPROACH:
    1. Extract predictions tensor from self.inputs
    2. If predictions requires gradients:
       - Compute difference: predictions.data - targets.data
       - Apply MSE derivative: 2 * difference / N
       - Multiply by grad_output: grad * grad_output
       - Return (result, None): targets carry no gradient
    3. Else return (None, None)

    EXAMPLE:
    >>> predictions = Tensor([2.0, 3.0], requires_grad=True)
    >>> targets = Tensor([1.0, 2.0])
    >>> loss = MSE(predictions, targets)  # (1² + 1²)/2 = 1.0
    >>> # During backward: grad_output = 1
    >>> # grad = 2 * ([2, 3] - [1, 2]) / 2 = [1, 1]

    HINTS:
    - MSE derivative: ∂MSE/∂pred = 2 * (pred - target) / N
    - N = np.size(targets.data) (total number of elements)
    - Multiply by grad_output for chain rule
    """
    ### BEGIN SOLUTION
    predictions, targets = self.inputs

    if isinstance(predictions, Tensor) and predictions.requires_grad:
        # Gradient: 2 * (predictions - targets) / N
        num_samples = np.size(targets.data)
        grad = 2.0 * (predictions.data - targets.data) / num_samples

        return grad * grad_output, None
    return None, None
    ### END SOLUTION

# %% [markdown]
"""
### BinaryCrossEntropyFunction.backward: Gradient Rules for Binary Cross-Entropy

Binary cross-entropy is the loss for yes/no predictions. Its gradient looks
alarming written out, then collapses into something remarkably clean.

**Mathematical Principle:**
```
If L = -[y·log(p) + (1-y)·log(1-p)], then
    ∂L/∂p = (p - y) / (p·(1 - p))
```

**And when p came from a sigmoid, the pieces cancel:**
```
∂L/∂logit = ∂L/∂p · ∂p/∂logit
          = (p - y)/(p(1-p))  ·  p(1-p)
          = p - y
```

The same clean form as MSE, arrived at by a very different route. This
cancellation is why sigmoid and BCE are always paired.

**The systems consequence**: the un-cancelled form divides by p·(1-p), which goes
to zero as the model becomes confident. A confident wrong prediction produces a
gradient that overflows. Real frameworks fuse sigmoid and BCE into one operation
so the cancellation happens before any division does.
"""

# %% nbgrader={"grade": false, "grade_id": "bce-backward", "solution": true}
#| exporti
@method_of(BinaryCrossEntropyFunction)
def backward(self, grad_output):
    """
    Gradient computation for Binary Cross-Entropy Loss.

    BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)

    Compute gradient for BCE loss.

    TODO: Implement gradient computation for Binary Cross-Entropy loss.

    APPROACH:
    1. Extract predictions tensor from self.inputs
    2. If predictions requires gradients:
       - Clip predictions: p = np.clip(predictions.data, eps, 1-eps)
       - Get targets: y = targets.data
       - Apply BCE derivative: (p - y) / (p * (1-p) * N)
       - Multiply by grad_output
       - Return (result, None): targets carry no gradient
    3. Else return (None, None)

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
    predictions, targets = self.inputs

    if isinstance(predictions, Tensor) and predictions.requires_grad:
        eps = EPSILON
        p = np.clip(predictions.data, eps, 1 - eps)
        y = targets.data
        num_samples = np.size(targets.data)

        # Gradient: (p - y) / (p * (1-p) * N)
        grad = (p - y) / (p * (1 - p) * num_samples)

        return grad * grad_output, None
    return None, None
    ### END SOLUTION

# %% [markdown]
"""
### Softmax Probabilities, Reused from Module 02

CrossEntropyFunction.backward needs the softmax probabilities of the logits as a
plain array. Module 02 already solved the hard part, subtracting the row maximum
before exponentiating so large logits cannot overflow, and its `SoftmaxFunction`
does its arithmetic on NumPy arrays. So this helper is one line: call that
operation's `forward` on the array rather than deriving the trick a second time.
"""

# %% nbgrader={"grade": false, "grade_id": "stable-softmax-helper", "solution": false}
#| export
def _stable_softmax(logits_data):
    """Softmax over the last axis of a (batch, classes) array, via Module 02's numerically stable operation."""
    return SoftmaxFunction().forward(logits_data)

# %% [markdown]
"""
### 🧪 Unit Test: Stable Softmax Helper

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
### One-Hot Encoding

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
### 🧪 Unit Test: One-Hot Encoding Helper

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
### CrossEntropyFunction.backward: Gradient Rules for Cross-Entropy Loss

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
#| exporti
@method_of(CrossEntropyFunction)
def backward(self, grad_output):
    """
    Gradient computation for Cross-Entropy Loss.

    CrossEntropy: L = -mean(log_softmax(logits)[targets])

    The gradient with respect to logits is remarkably elegant:
    ∂L/∂logits = (softmax(logits) - one_hot(targets)) / N

    This is one of the most beautiful results in machine learning:
    - The gradient is simply the difference between predictions and targets
    - It naturally scales with how wrong we are
    - It's numerically stable when computed via softmax

    Compute gradient for cross-entropy loss.

    Uses helper functions for each sub-computation:
    - _stable_softmax(): Converts logits to probabilities (numerically stable)
    - _one_hot_encode(): Converts target indices to one-hot vectors

    TODO: Implement gradient computation for Cross-Entropy loss.

    APPROACH:
    1. Extract logits tensor from self.inputs
    2. If logits requires gradients:
       - Compute softmax using _stable_softmax(logits.data)
       - Encode targets using _one_hot_encode(targets, batch_size, num_classes)
       - Apply CE derivative: (softmax - one_hot) / batch_size
       - Multiply by grad_output
       - Return (result, None): targets carry no gradient
    3. Else return (None, None)

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
    logits, targets = self.inputs

    if isinstance(logits, Tensor) and logits.requires_grad:
        batch_size, num_classes = logits.data.shape[0], logits.data.shape[1]
        softmax = _stable_softmax(logits.data)
        one_hot = _one_hot_encode(targets.data.astype(int), batch_size, num_classes)

        # Gradient: (softmax - one_hot) / batch_size
        grad = (softmax - one_hot) / batch_size

        return grad * grad_output, None
    return None, None
    ### END SOLUTION

# %% [markdown]
"""
### LogSoftmax.backward: Gradient Rules for Log-Softmax

`log_softmax` from Module 04 is an operation too, and models that call it directly (rather than through `CrossEntropyLoss`) need its gradient.

**Mathematical Rule:** If y = log_softmax(x) along an axis, then
∂L/∂x = grad_y - softmax(x) * sum(grad_y) along that axis, and softmax(x) = exp(y).
"""

# %% nbgrader={"grade": false, "grade_id": "log-softmax-backward", "solution": true}
#| exporti
@method_of(LogSoftmax)
def backward(self, grad_output):
    """
    Gradient computation for log-softmax.

    TODO: Implement grad_output - exp(output) * sum(grad_output) along self.dim.

    APPROACH:
    1. softmax = np.exp(self.output.data)  (apply() saved the log-softmax output on the node)
    2. total = np.sum(grad_output, axis=self.dim, keepdims=True)
    3. Return (grad_output - softmax * total,)
    """
    ### BEGIN SOLUTION
    tensor, = self.inputs

    if isinstance(tensor, Tensor) and tensor.requires_grad:
        softmax = np.exp(self.output.data)
        total = np.sum(grad_output, axis=self.dim, keepdims=True)
        return grad_output - softmax * total,
    return None,
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "no-grad-context", "solution": false}
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


# %% [markdown]
"""
### Walking the Graph in the Right Order

We have every gradient rule. What remains is the traversal: given the output,
visit the graph and hand each tensor its gradient. The obvious approach is
recursion -- compute a tensor's gradient, then immediately recurse into its
parents. That works, right up until a tensor is used twice.

```
        x ──► y ──┬──► loss = y * y
                  └──►
```

Here `y` feeds the multiply twice, so the true gradient at `y` is the SUM of
what both edges send back. Naive recursion reaches `y` on the first edge and
descends the whole subtree below it before the second edge has contributed
anything. So `x` is updated from a partial gradient. Worse, when the second
edge arrives it descends that same subtree all over again -- work doubles at
every reused node, which is exponential on a graph with several of them.

The fix is a **topological order**: an ordering of the graph in which every
consumer of a tensor appears before the tensor itself.

```
   build order (DFS post-order)    reversed = topological order
   [x, y, loss]                    [loss, y, x]
                                     │     │   └─ visited last, gradient complete
                                     │     └───── both edges have contributed
                                     └─────────── the seed
```

Process tensors in that order and each one is visited exactly once, at the
moment its gradient is complete. Correct, and linear in the size of the graph
rather than exponential.

This is the piece that makes reverse-mode AD practical, and it is why PyTorch,
JAX, and every other framework sort before they walk. `backward()` below builds
the order with a depth-first post-order traversal, reverses it, then makes a
single pass -- accumulating into each `.grad` and passing each parent its share.

One consequence worth noticing: the graph can only be freed *after* the whole
walk. Releasing each node as you pass it would drop exactly the second
contribution the sort exists to collect.
"""

# %% [markdown]
"""
### Completing apply() and Adding backward()

Module 01 left two slots open. `Function.apply()` runs an operation but forgets it, and `Tensor.backward()` raises. The cell below fills both in:

1. **apply() records the graph** - After running forward(), the output remembers the operation that produced it (`_grad_fn`) whenever an input has requires_grad=True
2. **Adding backward() method** - Implements reverse-mode automatic differentiation
3. **Maintaining compatibility** - Tensor's operators were already routed through apply() in Module 01, so all existing code continues to work unchanged

**The Pattern:**
```
Module 01: x + y → Add.apply(x, y) → result
Module 06: x + y → Add.apply(x, y) → result that remembers Add (if requires_grad=True)
```

This is how PyTorch's `torch.autograd.Function` works - clean, modern, and educational.
"""

# %% nbgrader={"grade": false, "grade_id": "apply-and-backward", "solution": false}
#| exporti
@method_of(Function)
def __repr__(self):
    """The name PyTorch shows for a grad_fn, e.g. <AddBackward> or <Conv2dBackward>."""
    name = type(self).__name__
    if name.endswith("Function"):
        name = name[: -len("Function")]
    return f"<{name}Backward>"


@method_of(Function)
@classmethod
def apply(cls, *inputs, **params):
    """
    Run the operation and, when a gradient is wanted, remember it on the output.

    This replaces the Module 01 version, which computed the result and forgot
    the operation. Two things are new. The node keeps its output (several
    backward() rules need it), and when any input has requires_grad=True and
    tracking is on, the output records the node in `_grad_fn`. The chain of
    `_grad_fn` links is the computation graph.
    """
    node = cls(*inputs, **params)
    out = Tensor(node.forward(*[t.data for t in inputs]))
    node.output = out
    if _GRAD_TRACKING_ENABLED and any(t.requires_grad for t in inputs):
        out.requires_grad = True
        out._grad_fn = node
    return out


@method_of(Tensor)
def backward(self, gradient=None, retain_graph=False):
    """
    Compute gradients via backpropagation.

    This is the key method that makes training possible!
    It implements reverse-mode automatic differentiation.

    **Algorithm:**
    1. Build a topological order of the graph, so every tensor is visited
       only after all of the operations that consumed it
    2. Seed the output tensor with the incoming gradient
    3. Walk that order once, accumulating into each tensor's `.grad` and
       handing each parent its share
    4. Release the graph once, at the end (unless retain_graph=True)

    **Why the topological order matters.** A tensor can feed more than one
    operation -- a residual connection, a reused activation, or something as
    small as `loss = y * y`. Its true gradient is the SUM of what every
    consumer sends back, so it must not propagate until all of them have
    contributed. Walking the graph in topological order guarantees exactly
    that, and visits every tensor exactly once.

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
    y.backward()   # Computes gradients for x, then releases the graph
    print(x.grad)  # [3.0]

    # A reused tensor accumulates from every consumer:
    h = x * 2
    (h * h).backward()   # dL/dx = 2*(2x)*2 = 8*x
    ```
    """

    # Only compute gradients if required
    if not self.requires_grad:
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

    if isinstance(gradient, Tensor):
        gradient = gradient.data

    if self._grad_fn is None and getattr(self, "_graph_released", False):
        raise RuntimeError(
            "Trying to backward through the graph a second time. The graph was released "
            "after the first backward(); pass retain_graph=True to keep it."
        )

    # ---- Step 1: topological sort -------------------------------------
    # Depth-first from the output, appending each tensor only AFTER its
    # parents. Reversing that post-order gives an order in which every
    # consumer of a tensor appears before the tensor itself.
    topo_order = []
    seen = set()
    # Iterative depth-first post-order (a recursive visit would hit Python's
    # recursion limit on a few thousand chained operations).
    stack = [(self, False)]
    while stack:
        tensor, expanded = stack.pop()
        if expanded:
            topo_order.append(tensor)
            continue
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        stack.append((tensor, True))
        fn = tensor._grad_fn
        if fn is not None:
            for parent in fn.inputs:
                if isinstance(parent, Tensor):
                    stack.append((parent, False))
    topo_order.reverse()

    # ---- Step 2: seed the output --------------------------------------
    # pending[id(tensor)] holds the gradient accumulated from the consumers
    # visited so far. Keyed by id() because Tensors are not hashable.
    pending = {id(self): gradient}

    # ---- Step 3: one pass, in topological order ------------------------
    for tensor in topo_order:
        grad = pending.get(id(tensor))
        if grad is None:
            continue

        if not tensor.requires_grad:
            continue

        # Accumulate into this tensor's .grad
        if tensor.grad is None:
            tensor.grad = np.zeros_like(tensor.data)
        grad = _reduce_broadcast_grad(grad, tensor.grad.shape)
        if np.shape(grad) != tensor.grad.shape:
            raise ValueError(f"gradient of shape {np.shape(grad)} does not match tensor of shape {tensor.grad.shape}")
        tensor.grad += grad

        # Hand each parent its share
        fn = tensor._grad_fn
        if fn is None:
            continue
        parent_grads = fn.backward(grad)
        for parent, parent_grad in zip(fn.inputs, parent_grads):
            if not isinstance(parent, Tensor) or parent_grad is None:
                continue
            if not parent.requires_grad:
                continue
            parent_grad = _reduce_broadcast_grad(parent_grad, parent.data.shape)
            if np.shape(parent_grad) != parent.data.shape:
                raise ValueError(
                    f"{fn!r} returned a gradient of shape {np.shape(parent_grad)} for an input of shape {parent.data.shape}"
                )
            if id(parent) in pending:
                pending[id(parent)] = pending[id(parent)] + parent_grad
            else:
                pending[id(parent)] = parent_grad

    # ---- Step 4: release the graph, once ------------------------------
    # The graph holds references to every intermediate tensor, so without
    # this, memory grows with each training step. Releasing per-node DURING
    # the walk would be a bug: a tensor with two consumers would lose the
    # second contribution.
    if not retain_graph:
        for tensor in topo_order:
            if tensor._grad_fn is not None:
                tensor._graph_released = True
            tensor._grad_fn = None


@method_of(Tensor)
def zero_grad(self):
    """
    Reset gradients to zero.

    Call this before each backward pass to prevent gradient accumulation
    from previous iterations.
    """
    self.grad = None

# %% [markdown]
"""
### In-Place Operations Break Autograd

**THIS IS THE MOST COMMON SILENT FAILURE IN TINYTORCH!**

### Critical Rule: Never Modify Tensors In-Place When requires_grad=True

**WRONG ❌ - This Corrupts the Gradient Graph:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x * x
x.data[0] = 999  # ❌ Mul.backward will read the corrupted x
y.backward()     # ❌ x.grad[0] comes out as 1998, not 2
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
# ❌ WRONG - touching a parameter BETWEEN forward and backward
W = Tensor([[0.5, 0.3]], requires_grad=True)
y = x.matmul(W.transpose())
loss = compute_loss(y, target)
W.data *= 0.9            # ❌ the recorded graph still points at W
loss.backward()          # ❌ MatMul.backward now reads the modified W

# ✅ CORRECT - update AFTER backward, once the graph has been released
W = Tensor([[0.5, 0.3]], requires_grad=True)
y = x.matmul(W.transpose())
loss = compute_loss(y, target)
loss.backward()
W.data -= 0.01 * W.grad  # ✅ exactly what Module 07's optimizers will do
```

### How to Debug In-Place Corruption

If your gradients look wrong or you get mysterious errors:

1. **Search your code** for `.data[` assignments
2. **Search for** in-place operators: `+=`, `-=`, `*=`, `/=` on `.data`
3. **Check custom functions** that modify tensors
4. **Verify** parameter updates run after backward(), never between forward and backward

### Why PyTorch Has torch.no_grad()

PyTorch switches gradient tracking off wherever a computation must not be recorded:
parameter updates, and evaluation passes that would otherwise save every forward
tensor. TinyTorch has the same switch:

```python
from tinytorch.core.autograd import no_grad
with no_grad():
    logits = model(x)   # forward only: no graph, no saved tensors
```

Module 07's optimizers will not need it: they write `param.data` after `backward()`
has released the graph, so nothing is recording. Reach for `no_grad()` in
evaluation loops, where the saved forward tensors would only cost memory.

**How it works**: `no_grad()` sets a global flag that all tracked operations check.
When the flag is off, operations skip graph construction entirely -- the result tensor
will have `requires_grad=False` regardless of its inputs.

### Memory Impact

**Question**: "Why not update `.data` between forward and backward and save a pass?"

**Answer**: The recorded graph holds references to the tensors it saw during the
forward pass. Change one of them and backward() differentiates a computation that
never happened. Correctness > premature perf.

**Bottom Line**: If a tensor has `requires_grad=True`, treat it as **immutable**. Always create new tensors instead of modifying in-place.

---
"""

# %% [markdown]
"""
#### 🧪 Unit Test: Tensor Autograd Enhancement

This test validates our enhanced Tensor class computes gradients correctly.

**What we're testing**: Gradient computation and chain rule implementation
**Why it matters**: This is the core of automatic differentiation
**Expected**: Correct gradients for various operations and computation graphs
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-autograd", "locked": true, "points": 20}
def test_unit_tensor_autograd():
    """🧪 Test Tensor autograd enhancement."""
    print("🧪 Unit Test: Tensor Autograd Enhancement...")

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
#### 🧪 Unit Test: Gradients Through a Reused Tensor

This test validates the topological traversal: a tensor consumed by more than
one operation must accumulate from every consumer before it propagates.

**What we're testing**: Gradients are correct when an intermediate tensor is used twice
**Why it matters**: Residual connections, attention, and any `y * y` hit this path;
a traversal that visits a node once per edge silently halves the gradient
**Expected**: Analytic gradients, with the default `retain_graph=False`
"""

# %% nbgrader={"grade": true, "grade_id": "test-reused-tensor-gradients", "locked": true, "points": 10}
def test_unit_reused_tensor_gradients():
    """🧪 Test gradient accumulation through a tensor with multiple consumers."""
    print("🧪 Unit Test: Reused Tensor Gradients...")

    # loss = (x @ W)^2  ->  dL/dW = 2(xW)x = 24,  dL/dx = 2(xW)W = 36
    x = Tensor(np.array([[2.0]]), requires_grad=True)
    W = Tensor(np.array([[3.0]]), requires_grad=True)
    y = x.matmul(W)
    (y * y).backward()
    assert np.allclose(W.grad, 24.0), f"Expected dL/dW = 24, got {W.grad}"
    assert np.allclose(x.grad, 36.0), f"Expected dL/dx = 36, got {x.grad}"
    print("   ✅ y = x@W, loss = y*y: both gradients correct")

    # out = z + z  ->  dout/dW = 2x = 4
    x2 = Tensor(np.array([[2.0]]), requires_grad=True)
    W2 = Tensor(np.array([[3.0]]), requires_grad=True)
    z = x2.matmul(W2)
    (z + z).backward()
    assert np.allclose(W2.grad, 4.0), f"Expected dout/dW = 4, got {W2.grad}"
    print("   ✅ z = x@W, out = z+z: gradient accumulates from both edges")

    # A tensor reused at several depths: each level doubles the gradient
    x3 = Tensor(np.array([[1.0]]), requires_grad=True)
    W3 = Tensor(np.array([[1.0]]), requires_grad=True)
    h = x3.matmul(W3)
    for _ in range(8):
        h = h + h
    h.backward()
    assert np.allclose(W3.grad, 2 ** 8), f"Expected 2^8 = 256, got {W3.grad}"
    print("   ✅ eight reuse levels: gradient is 2^8, computed in one pass")

    print("✅ Reused-tensor gradients work correctly!")

if __name__ == "__main__":
    test_unit_reused_tensor_gradients()

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
        x_no_grad = Tensor(rng.standard_normal(shape))
        base_memory = x_no_grad.data.nbytes

        # With gradient tracking
        x_with_grad = Tensor(rng.standard_normal(shape), requires_grad=True)
        y = x_with_grad * 2  # Simple operation that builds graph
        z = y + 1

        # Estimate graph overhead: saved tensors in grad_fn
        graph_overhead = 0
        if hasattr(z, '_grad_fn') and z._grad_fn is not None:
            for tensor in z._grad_fn.inputs:
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
    test_unit_reduce_broadcast_grad()
    test_unit_function_classes()
    test_unit_broadcast_gradients()
    test_unit_tensor_autograd()
    test_unit_reused_tensor_gradients()

    print("\nRunning integration scenarios...")

    # Test 1: Multi-layer computation graph
    print("🧪 Integration Test: Multi-layer Neural Network...")

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
    print("🧪 Integration Test: Gradient Accumulation...")

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
    print("🧪 Integration Test: Complex Operations...")

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

    # Test 4: the reductions and views every model uses
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True)
    (x.mean() + x.max(axis=1).sum() + x.contiguous().sum()).backward()
    assert np.allclose(x.grad, np.full((2, 3), 1 / 6) + np.array([[0, 1, 0], [0, 0, 1]]) + 1)
    print("✅ Mean, max, and copy gradients work!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 06")

# %%
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
- **Implemented backward() for every operation** (Add, Mul, Matmul, Sum, ...) with correct gradients
- **Completed apply()** so every operation records itself in the graph
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
