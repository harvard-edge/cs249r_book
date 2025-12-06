---
title: "Autograd"
description: "Build the automatic differentiation engine that powers neural network training"
difficulty: "⭐⭐⭐⭐"
time_estimate: "8-10 hours"
prerequisites: ["01_tensor", "02_activations", "03_layers", "04_losses"]
next_steps: ["06_optimizers"]
learning_objectives:
  - "Understand computational graph construction and gradient flow in reverse-mode autodiff"
  - "Implement Function base class with operation-specific gradient computation rules"
  - "Enhance Tensor class with backward() method for automatic gradient tracking"
  - "Analyze memory overhead of computation graphs and gradient accumulation strategies"
  - "Connect implementation to PyTorch's torch.autograd.Function architecture"
---

# 05. Autograd

**FOUNDATION TIER** | Difficulty: ⭐⭐⭐⭐ (4/4) | Time: 8-10 hours

## Overview

Build the automatic differentiation engine that makes neural network training possible. This module implements reverse-mode autodiff by enhancing the existing Tensor class with gradient tracking capabilities and creating Function classes that encode gradient computation rules for each operation. You'll implement the mathematical foundation that transforms TinyTorch from a static computation library into a dynamic, trainable ML framework where calling backward() on any tensor automatically computes gradients throughout the entire computation graph.

```{mermaid}
graph BT
    x["x<br/>Tensor<br/>(input)"] --> mul["*<br/>MulBackward"]
    w["w<br/>Tensor<br/>(weight)"] --> mul
    mul --> add["+<br/>AddBackward"]
    b["b<br/>Tensor<br/>(bias)"] --> add
    add --> relu["ReLU<br/>ReLUBackward"]
    relu --> loss["Loss<br/>Function"]

    loss -.backward().-> relu
    relu -.∂L/∂relu.-> add
    add -.∂L/∂add.-> mul
    add -.∂L/∂b.-> b
    mul -.∂L/∂x.-> x
    mul -.∂L/∂w.-> w

    style loss fill:#ffcdd2
    style x fill:#c5e1a5
    style w fill:#c5e1a5
    style b fill:#c5e1a5
    style mul fill:#e1f5fe
    style add fill:#e1f5fe
    style relu fill:#e1f5fe
```

**Computational Graph Example**: Forward pass (solid arrows) builds the graph, backward pass (dotted arrows) propagates gradients.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand computational graph construction**: Learn how autodiff systems dynamically build directed acyclic graphs during forward pass that track operation dependencies for gradient flow
- **Implement Function base class with gradient rules**: Create the Function architecture where each operation (AddBackward, MulBackward, MatmulBackward) implements its specific chain rule derivative computation
- **Enhance Tensor class with backward() method**: Add gradient tracking attributes (requires_grad, grad, _grad_fn) and implement reverse-mode differentiation that traverses computation graphs
- **Analyze memory overhead and accumulation**: Understand how computation graphs store intermediate values, when gradients accumulate vs. reset, and memory-computation trade-offs in gradient checkpointing
- **Connect to PyTorch's autograd architecture**: Recognize how your Function classes mirror torch.autograd.Function and understand the enhanced Tensor approach vs. deprecated Variable wrapper pattern

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement Function base class and operation-specific gradient functions (AddBackward, MulBackward, MatmulBackward, SumBackward), enhance Tensor class with backward() method, create enable_autograd() that activates gradient tracking
2. **Use**: Apply automatic differentiation to mathematical expressions, compute gradients for neural network parameters (weights and biases), verify gradient correctness against manual chain rule calculations
3. **Reflect**: How does computation graph memory scale with network depth? Why does backward pass take 2-3x forward pass time despite similar operations? When does gradient accumulation help vs. hurt training?

## Implementation Guide

### Function Base Class - Foundation of Gradient Computation
```python
from tinytorch.core.tensor import Tensor

class Function:
    """
    Base class for differentiable operations.

    Each operation (add, multiply, matmul) inherits from Function
    and implements the apply() method that computes gradients.
    """

    def __init__(self, *tensors):
        """Store input tensors needed for backward pass."""
        self.saved_tensors = tensors
        self.next_functions = []

        # Build computation graph connections
        for t in tensors:
            if isinstance(t, Tensor) and t.requires_grad:
                if getattr(t, '_grad_fn', None) is not None:
                    self.next_functions.append(t._grad_fn)

    def apply(self, grad_output):
        """
        Compute gradients for inputs using chain rule.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of gradients for each input tensor
        """
        raise NotImplementedError("Each Function must implement apply()")

# Usage: Every operation creates a Function subclass
# that remembers inputs and knows how to compute gradients
```

### AddBackward - Gradient Rules for Addition
```python
class AddBackward(Function):
    """
    Gradient computation for tensor addition.

    Mathematical Rule: If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1
    Gradient flows unchanged to both inputs.
    """

    def apply(self, grad_output):
        """Addition distributes gradients equally to both inputs."""
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output  # ∂(a+b)/∂a = 1

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output  # ∂(a+b)/∂b = 1

        return grad_a, grad_b

# Example: z = x + y computes dz/dx = 1, dz/dy = 1
```

### MulBackward - Gradient Rules for Multiplication
```python
class MulBackward(Function):
    """
    Gradient computation for element-wise multiplication.

    Mathematical Rule: If z = a * b, then ∂z/∂a = b and ∂z/∂b = a
    Each input's gradient equals grad_output times the OTHER input.
    """

    def apply(self, grad_output):
        """Product rule: gradient = grad_output * other_input."""
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output * b.data  # ∂(a*b)/∂a = b

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data  # ∂(a*b)/∂b = a

        return grad_a, grad_b

# Example: z = x * y computes dz/dx = y, dz/dy = x
```

### MatmulBackward - Gradient Rules for Matrix Multiplication
```python
class MatmulBackward(Function):
    """
    Gradient computation for matrix multiplication.

    Mathematical Rule: If Z = A @ B, then:
    - ∂Z/∂A = grad_Z @ B.T
    - ∂Z/∂B = A.T @ grad_Z

    Dimension check: A(m×k) @ B(k×n) = Z(m×n)
    Backward: grad_Z(m×n) @ B.T(n×k) = grad_A(m×k) ✓
              A.T(k×m) @ grad_Z(m×n) = grad_B(k×n) ✓
    """

    def apply(self, grad_output):
        """Matrix multiplication gradients involve transposing inputs."""
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            # ∂(A@B)/∂A = grad_output @ B.T
            b_T = np.swapaxes(b.data, -2, -1)
            grad_a = np.matmul(grad_output, b_T)

        if isinstance(b, Tensor) and b.requires_grad:
            # ∂(A@B)/∂B = A.T @ grad_output
            a_T = np.swapaxes(a.data, -2, -1)
            grad_b = np.matmul(a_T, grad_output)

        return grad_a, grad_b

# Core operation for neural network weight gradients
```

---

**✓ CHECKPOINT 1: Computational Graph Construction Complete**

You've implemented the Function base class and gradient rules for core operations:
- ✅ Function base class with apply() method
- ✅ AddBackward, MulBackward, MatmulBackward, SumBackward
- ✅ Understanding of chain rule for each operation

**What you can do now**: Build computation graphs during forward pass that track operation dependencies.

**Next milestone**: Enhance Tensor class to automatically call these Functions during backward pass.

**Progress**: ~40% through Module 05 (~3-4 hours) | Still to come: Tensor.backward() implementation (~4-6 hours)

---

### Enhanced Tensor with backward() Method
```python
def enable_autograd():
    """
    Enhance Tensor class with automatic differentiation capabilities.

    This function monkey-patches Tensor operations to track gradients:
    - Replaces __add__, __mul__, matmul with gradient-tracking versions
    - Adds backward() method for reverse-mode differentiation
    - Adds zero_grad() method for resetting gradients
    """

    def backward(self, gradient=None):
        """
        Compute gradients via reverse-mode autodiff.

        Traverses computation graph backwards, applying chain rule
        at each operation to propagate gradients to all inputs.
        """
        if not self.requires_grad:
            return

        # Initialize gradient for scalar outputs
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError("backward() requires gradient for non-scalars")

        # Accumulate gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += gradient

        # Propagate through computation graph
        grad_fn = getattr(self, '_grad_fn', None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)

            # Recursively call backward on parent tensors
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad)

    # Install backward() method on Tensor class
    Tensor.backward = backward

# Usage: enable_autograd() activates gradients globally
```

### Complete Neural Network Example
```python
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.tensor import Tensor

enable_autograd()  # Activate gradient tracking

# Forward pass builds computation graph automatically
x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
W1 = Tensor([[0.5, 0.3], [0.2, 0.4], [0.1, 0.6]], requires_grad=True)
b1 = Tensor([[0.1, 0.2]], requires_grad=True)

# Each operation stores its Function for backward pass
h1 = x.matmul(W1) + b1  # h1._grad_fn = AddBackward
                         # h1 contains MatmulBackward + AddBackward

W2 = Tensor([[0.3], [0.5]], requires_grad=True)
output = h1.matmul(W2)   # output._grad_fn = MatmulBackward
loss = (output ** 2).sum()  # loss._grad_fn = SumBackward

# Backward pass traverses graph in reverse, computing all gradients
loss.backward()

# All parameters now have gradients
print(f"x.grad shape: {x.grad.shape}")    # (1, 3)
print(f"W1.grad shape: {W1.grad.shape}")  # (3, 2)
print(f"b1.grad shape: {b1.grad.shape}")  # (1, 2)
print(f"W2.grad shape: {W2.grad.shape}")  # (2, 1)
```

---

**✓ CHECKPOINT 2: Automatic Differentiation Working**

You've completed the core autograd implementation:
- ✅ Function classes with gradient computation rules
- ✅ Enhanced Tensor with backward() method
- ✅ Computational graph traversal in reverse order
- ✅ Gradient accumulation and propagation

**What you can do now**: Train any neural network by calling loss.backward() to compute all parameter gradients automatically.

**Next milestone**: Apply autograd to complete networks in Module 06 (Optimizers) and Module 07 (Training).

**Progress**: ~80% through Module 05 (~7-8 hours) | Still to come: Testing & systems analysis (~1-2 hours)

---

## Getting Started

### Prerequisites
Ensure you understand the mathematical building blocks:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test tensor
tito test activations
tito test layers
tito test losses
```

### Development Workflow
1. **Open the development file**: `modules/05_autograd/autograd.py`
2. **Implement Function base class**: Create gradient computation foundation with saved_tensors and apply() method
3. **Build operation Functions**: Implement AddBackward, MulBackward, SubBackward, DivBackward, MatmulBackward gradient rules
4. **Add backward() to Tensor**: Implement reverse-mode differentiation with gradient accumulation and graph traversal
5. **Create enable_autograd()**: Monkey-patch Tensor operations to track gradients and build computation graphs
6. **Extend to activations and losses**: Add ReLUBackward, SigmoidBackward, MSEBackward, CrossEntropyBackward gradient functions
7. **Export and verify**: `tito module complete 05 && tito test autograd`

## Testing

### Comprehensive Test Suite
Run the full test suite to verify mathematical correctness:

```bash
# TinyTorch CLI (recommended)
tito test autograd

# Direct pytest execution
python -m pytest tests/05_autograd/ -v

# Run specific test categories
python -m pytest tests/05_autograd/test_gradient_flow.py -v
python -m pytest tests/05_autograd/test_batched_matmul_backward.py -v
```

### Test Coverage Areas
- ✅ **Function Classes**: Verify AddBackward, MulBackward, MatmulBackward compute correct gradients according to mathematical definitions
- ✅ **Backward Pass**: Test gradient flow through multi-layer computation graphs with multiple operation types
- ✅ **Chain Rule Application**: Ensure composite functions (f(g(x))) correctly apply chain rule: df/dx = (df/dg) × (dg/dx)
- ✅ **Gradient Accumulation**: Verify gradients accumulate correctly when multiple paths lead to same tensor
- ✅ **Broadcasting Gradients**: Test gradient unbroadcasting when operations involve tensors of different shapes
- ✅ **Neural Network Integration**: Validate seamless gradient computation through layers, activations, and loss functions

### Inline Testing & Mathematical Verification
The module includes comprehensive mathematical validation:
```python
# Example inline test output
🔬 Unit Test: Function Classes...
✅ AddBackward gradient computation correct
✅ MulBackward gradient computation correct
✅ MatmulBackward gradient computation correct
✅ SumBackward gradient computation correct
📈 Progress: Function Classes ✓

# Mathematical verification with known derivatives
🔬 Unit Test: Tensor Autograd Enhancement...
✅ Simple gradient: d(3x+1)/dx = 3 ✓
✅ Matrix multiplication gradients match analytical solution ✓
✅ Multi-operation chain rule application correct ✓
✅ Gradient accumulation works correctly ✓
📈 Progress: Autograd Enhancement ✓

# Integration test
🧪 Integration Test: Multi-layer Neural Network...
✅ Forward pass builds computation graph correctly
✅ Backward pass computes gradients for all parameters
✅ Gradient shapes match parameter shapes
✅ Complex operations (matmul + add + mul + sum) work correctly
```

### Manual Testing Examples
```python
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.tensor import Tensor

enable_autograd()

# Test 1: Power rule - d(x^2)/dx = 2x
x = Tensor([3.0], requires_grad=True)
y = x * x  # y = x²
y.backward()
print(f"d(x²)/dx at x=3: {x.grad}")  # Should be 6.0 ✓

# Test 2: Product rule - d(uv)/dx = u'v + uv'
x = Tensor([2.0], requires_grad=True)
u = x * x      # u = x², du/dx = 2x
v = x * x * x  # v = x³, dv/dx = 3x²
y = u * v      # y = x⁵, dy/dx = 5x⁴
y.backward()
print(f"d(x⁵)/dx at x=2: {x.grad}")  # Should be 80.0 ✓

# Test 3: Chain rule - d(f(g(x)))/dx = f'(g(x)) × g'(x)
x = Tensor([2.0], requires_grad=True)
g = x * x           # g(x) = x², g'(x) = 2x
f = g + g + g       # f(g) = 3g, f'(g) = 3
f.backward()
# df/dx = f'(g) × g'(x) = 3 × 2x = 6x = 12
print(f"d(3x²)/dx at x=2: {x.grad}")  # Should be 12.0 ✓

# Test 4: Gradient accumulation in multi-path graphs
x = Tensor([1.0], requires_grad=True)
y1 = x + x  # Path 1: dy1/dx = 1 + 1 = 2
y2 = x * 3  # Path 2: dy2/dx = 3
z = y1 + y2 # z = (x+x) + (3x) = 5x, dz/dx = 5
z.backward()
print(f"dz/dx with multiple paths: {x.grad}")  # Should be 5.0 ✓
```

## Systems Thinking Questions

### Computational Graph Memory and Construction
- **Graph Building**: How do operations dynamically construct the computational graph during forward pass? What data structures represent the graph?
- **Memory Overhead**: Each Function stores saved_tensors for backward pass. For a ResNet-50 with 50 layers, estimate memory overhead relative to parameters
- **Graph Lifetime**: When is the computation graph built? When is it freed? What happens if you call backward() twice without recreating the graph?
- **Dynamic vs Static Graphs**: PyTorch builds graphs dynamically (define-by-run) while TensorFlow 1.x used static graphs (define-then-run). What are the trade-offs for debugging, memory, and compilation?

### Reverse-Mode vs Forward-Mode Autodiff
- **Computational Complexity**: For function f: ℝⁿ → ℝᵐ, forward-mode costs O(n) passes, reverse-mode costs O(m) passes. Why do neural networks always use reverse-mode?
- **Neural Network Case**: For loss: ℝᴺ → ℝ¹ where N=millions of parameters and m=1, what's the speedup of reverse-mode vs forward-mode?
- **Jacobian Computation**: Forward-mode computes Jacobian-vector products (JVP), reverse-mode computes vector-Jacobian products (VJP). When does each matter?
- **Second-Order Derivatives**: Computing Hessians (gradients of gradients) for Newton's method requires running autodiff twice. What's the memory cost?

### Gradient Accumulation and Memory Management
- **Intermediate Value Storage**: Backward pass requires values from forward pass (saved_tensors). For 100-layer ResNet, what percentage of memory is computation graph vs parameters?
- **Gradient Checkpointing**: Trade computation for memory by recomputing forward pass values during backward. When does this make sense? What's the time-memory trade-off?
- **Gradient Accumulation**: Processing batch as 4 mini-batches with gradient accumulation uses less memory than single large batch. Why? Does it change training dynamics?
- **In-Place Operations**: `x += y` can corrupt gradients by overwriting values needed for backward pass. How do frameworks detect and prevent this?

### Real-World Applications
- **Deep Learning Training**: Every neural network from ResNets to GPT-4 relies on automatic differentiation for computing weight gradients during backpropagation
- **Probabilistic Programming**: Bayesian inference frameworks (Pyro, Stan) use autodiff to compute gradients of log-probability with respect to latent variables
- **Robotics and Control**: Trajectory optimization uses autodiff to compute gradients of cost functions with respect to control inputs for gradient-based planning
- **Physics Simulations**: Differentiable physics engines use autodiff for inverse problems like inferring material properties from observed motion

### How Your Implementation Maps to PyTorch

**What you just built:**
```python
# Your TinyTorch autograd implementation
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import AddBackward, MulBackward

# Forward pass with gradient tracking
x = Tensor([[1.0, 2.0]], requires_grad=True)
w = Tensor([[0.5], [0.7]], requires_grad=True)
y = x.matmul(w)  # Builds computation graph
loss = y.mean()

# Backward pass computes gradients
loss.backward()  # YOUR implementation traverses graph
print(x.grad)  # Gradients you computed
print(w.grad)
```

**How PyTorch does it:**
```python
# PyTorch equivalent
import torch

# Forward pass with gradient tracking
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
w = torch.tensor([[0.5], [0.7]], requires_grad=True)
y = x @ w  # Builds computation graph (same concept)
loss = y.mean()

# Backward pass computes gradients
loss.backward()  # PyTorch autograd engine
print(x.grad)  # Same gradient values
print(w.grad)
```

**Key Insight**: Your `Function` classes (AddBackward, MulBackward, MatmulBackward) implement the **exact same gradient computation rules** that PyTorch uses internally. When you call `loss.backward()`, both implementations traverse the computation graph in reverse topological order, applying the chain rule via each Function's backward method.

**What's the SAME?**
- **Computational graph architecture**: Tensor operations create Function nodes
- **Gradient computation**: Chain rule via reverse-mode autodiff
- **API design**: `requires_grad`, `.backward()`, `.grad` attribute
- **Function pattern**: `forward()` computes output, `backward()` computes gradients
- **Tensor enhancement**: Gradients stored directly in Tensor (modern PyTorch style, not Variable wrapper)

**What's different in production PyTorch?**
- **Backend**: C++/CUDA implementation ~100-1000× faster
- **Memory optimization**: Graph nodes pooled and reused across iterations
- **Optimized gradients**: Hand-tuned gradient formulas (e.g., fused operations)
- **Advanced features**: Higher-order gradients, gradient checkpointing, JIT compilation

**Why this matters**: When you debug PyTorch training and encounter `RuntimeError: element 0 of tensors does not require grad`, you understand this is checking the computation graph structure you implemented. When gradients are `None`, you know backward() hasn't been called or the tensor isn't connected to the loss—concepts from YOUR implementation.

**Production usage example**:
```python
# PyTorch production code (after TinyTorch)
import torch
import torch.nn as nn

model = nn.Linear(784, 10)  # Uses torch.Tensor with requires_grad=True
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop - same workflow you built
output = model(input)  # Forward pass builds graph
loss = nn.CrossEntropyLoss()(output, target)
loss.backward()  # Backward pass (YOUR implementation's logic)
optimizer.step()  # Update using .grad (YOUR gradients)
```

After implementing autograd yourself, you understand that `loss.backward()` traverses the computation graph you built during forward pass, calling each operation's gradient function (AddBackward, MatmulBackward, etc.) in reverse order—exactly like your implementation.

### Mathematical Foundations
- **Chain Rule**: ∂f/∂x = (∂f/∂u)(∂u/∂x) for composite functions f(u(x)) - the mathematical foundation of backpropagation
- **Computational Graphs as DAGs**: Directed acyclic graphs where nodes are operations and edges are data dependencies enable topological ordering for backward pass
- **Jacobians and Matrix Calculus**: For vector-valued functions, gradients are Jacobian matrices. Matrix multiplication gradient rules come from Jacobian chain rule
- **Dual Numbers**: Alternative autodiff implementation using numbers with infinitesimals: a + bε where ε² = 0

### Performance Characteristics
- **Time Complexity**: Backward pass takes roughly 2-3x forward pass time (not 1x!) because matmul gradients need two matmuls (grad_x = grad_y @ W.T, grad_W = x.T @ grad_y)
- **Space Complexity**: Computation graph memory scales with number of operations in forward pass, typically 1-2x parameter memory for deep networks
- **Numerical Stability**: Gradients can vanish (→0) or explode (→∞) in deep networks. What causes this? How do residual connections and layer normalization help?
- **Sparse Gradients**: Embedding layers produce sparse gradients (most entries zero). Specialized gradient accumulation saves memory

```{admonition} Systems Reality Check
:class: tip

**Production Context**: PyTorch's autograd engine processes billions of gradient computations per second using optimized C++ gradient functions, memory pooling, and compiled graph perf. Your Python implementation demonstrates the mathematical principles but runs ~100-1000x slower.

**Performance Note**: For ResNet-50 (25M parameters), the computational graph stores ~100MB of intermediate activations during forward pass. Gradient checkpointing reduces this to ~10MB by recomputing activations, trading 30% extra computation for 90% memory savings - critical for training larger models on limited GPU memory.

**Architecture Evolution**: PyTorch originally used separate Variable wrapper but merged it into Tensor in v0.4.0 (2018) for cleaner API. Your implementation follows this modern enhanced-Tensor approach, not the deprecated Variable pattern.
```

## Ready to Build?

You're about to implement the mathematical foundation that makes modern AI possible. Automatic differentiation is the invisible engine powering every neural network, from simple classifiers to GPT and diffusion models. Before autodiff, researchers manually derived gradient formulas for each layer and loss function - tedious, error-prone, and severely limiting research progress. Automatic differentiation changed everything.

Understanding autodiff from first principles will give you deep insight into how deep learning really works. You'll implement the Function base class that encodes gradient rules, enhance the Tensor class with backward() that traverses computation graphs, and see why reverse-mode autodiff enables efficient training of billion-parameter models. This is where mathematics meets software engineering to create something truly powerful.

The enhanced Tensor approach you'll build mirrors modern PyTorch (post-v0.4) where gradients are native Tensor capabilities, not external wrappers. You'll understand why computation graphs consume memory proportional to network depth, why backward pass takes 2-3x forward pass time, and why gradient checkpointing trades computation for memory. These insights are critical for training large models efficiently.

Take your time with each Function class, verify gradients match manual chain rule calculations, and enjoy building the heart of machine learning. This module transforms TinyTorch from a static math library into a trainable ML framework - the moment everything comes alive.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/05_autograd/autograd.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/05_autograd/autograd.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/05_autograd/autograd.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

```

---

<div class="prev-next-area">
<a class="left-prev" href="../modules/04_losses/ABOUT.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../modules/06_optimizers/ABOUT.html" title="next page">Next Module →</a>
</div>
