# Module 06: Autograd

:::{admonition} Module Info
:class: note

**FOUNDATION TIER** | Difficulty: ●●●○ | Time: 6-8 hours | Prerequisites: 01-05

**Prerequisites: Modules 01-05** means you need:
- Tensor operations (matmul, broadcasting, reductions)
- Activation functions (understanding non-linearity)
- Neural network layers (what gradients flow through)
- Loss functions (the "why" behind gradients)
- DataLoader for efficient batch processing

If you can compute a forward pass through a neural network manually and understand why we need to minimize loss, you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F06_autograd%2F06_autograd.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/06_autograd/06_autograd.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/06_autograd.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Autograd is the gradient engine that makes neural networks learn. Every modern deep learning framework—PyTorch, TensorFlow, JAX—has automatic differentiation at its core. Without autograd, training a neural network would require deriving and coding gradients by hand for every parameter in every layer. For a network with millions of parameters, this is impossible.

In this module, you'll build reverse-mode automatic differentiation from scratch. Your autograd system will track computation graphs during the forward pass, then flow gradients backward through every operation using the chain rule. By the end, calling `loss.backward()` will automatically compute gradients for every parameter in your network, just like PyTorch.

This is the most conceptually challenging module in the Foundation tier, but it unlocks everything that follows: optimizers, training loops, and the ability to learn from data.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** the Function base class that enables gradient computation for all operations
- **Build** computation graphs that track dependencies between tensors during forward pass
- **Master** the chain rule by implementing backward passes for arithmetic, matrix multiplication, and reductions
- **Understand** memory trade-offs between storing intermediate values and recomputing forward passes
- **Connect** your autograd implementation to PyTorch's design patterns and production optimizations
```

## What You'll Build

```{mermaid}
:align: center
:caption: Autograd System
flowchart LR
    subgraph "Autograd System"
        A["Function<br/>Base Class"]
        B["Operation Functions<br/>Add, Mul, Matmul"]
        C["Backward Pass<br/>Gradient Flow"]
        D["Computation Graph<br/>Tracking"]
        E["enable_autograd()<br/>Global Activation"]
    end

    A --> B --> C --> D --> E

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `Function` base class | Storing inputs for backward pass |
| 2 | `AddBackward`, `MulBackward`, `MatmulBackward` | Operation-specific gradient rules |
| 3 | `backward()` method on Tensor | Reverse-mode differentiation |
| 4 | `enable_autograd()` enhancement | Monkey-patching operations for gradient tracking |
| 5 | Integration tests | Multi-layer gradient flow |

**The pattern you'll enable:**
```python
# Automatic gradient computation
x = Tensor([2.0], requires_grad=True)
y = x * 3 + 1  # y = 3x + 1
y.backward()   # Computes dy/dx = 3 automatically
print(x.grad)  # [3.0]
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Higher-order derivatives (gradients of gradients)—PyTorch supports this with `create_graph=True`
- Dynamic computation graphs—your graphs are built during forward pass only
- GPU kernel fusion—PyTorch's JIT compiler optimizes backward pass operations
- Checkpointing for memory efficiency—that's an advanced optimization technique

**You are building the core gradient engine.** Advanced optimizations come in production frameworks.

## API Reference

This section documents the autograd components you'll build. These integrate with the existing Tensor class from Module 01.

### Function Base Class

```python
Function(*tensors)
```

Base class for all differentiable operations. Every operation (addition, multiplication, etc.) inherits from Function and implements gradient computation rules.

### Core Function Classes

| Class | Purpose | Gradient Rule |
|-------|---------|---------------|
| `AddBackward` | Addition gradients | ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1 |
| `SubBackward` | Subtraction gradients | ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1 |
| `MulBackward` | Multiplication gradients | ∂(a*b)/∂a = b, ∂(a*b)/∂b = a |
| `DivBackward` | Division gradients | ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b² |
| `MatmulBackward` | Matrix multiplication gradients | ∂(A@B)/∂A = grad@B.T, ∂(A@B)/∂B = A.T@grad |
| `SumBackward` | Reduction gradients | ∂sum(a)/∂a[i] = 1 for all i |
| `ReshapeBackward` | Shape manipulation | ∂(X.reshape(...))/∂X = grad.reshape(X.shape) |
| `TransposeBackward` | Transpose gradients | ∂(X.T)/∂X = grad.T |

**Additional Backward Classes:** The implementation includes backward functions for activations (`ReLUBackward`, `SigmoidBackward`, `SoftmaxBackward`, `GELUBackward`), losses (`MSEBackward`, `BCEBackward`, `CrossEntropyBackward`), and other operations (`PermuteBackward`, `EmbeddingBackward`, `SliceBackward`). These follow the same pattern as the core classes above.

### Enhanced Tensor Methods

Your implementation adds these methods to the Tensor class:

| Method | Signature | Description |
|--------|-----------|-------------|
| `backward` | `backward(gradient=None) -> None` | Compute gradients via backpropagation |
| `zero_grad` | `zero_grad() -> None` | Reset gradients to None |

### Global Activation

| Function | Signature | Description |
|----------|-----------|-------------|
| `enable_autograd` | `enable_autograd(quiet=False) -> None` | Activate gradient tracking globally |

## Core Concepts

This section covers the fundamental ideas behind automatic differentiation. Understanding these concepts deeply will help you debug gradient issues in any framework, not just TinyTorch.

### Computation Graphs

A computation graph is a directed acyclic graph (DAG) where nodes represent tensors and edges represent operations. When you write `y = x * 3 + 1`, you're implicitly building a graph with three nodes (x, intermediate result, y) and two operations (multiply, add).

Autograd systems build these graphs during the forward pass by recording each operation. Every tensor created by an operation stores a reference to the function that created it. This reference is the key to gradient flow: when you call `backward()`, the system traverses the graph in reverse, applying the chain rule at each node.

Here's how a simple computation builds a graph:

```
Forward Pass:  x → [Mul(*3)] → temp → [Add(+1)] → y
Backward Pass: grad_x ← [MulBackward] ← grad_temp ← [AddBackward] ← grad_y
```

Each operation stores its inputs because backward pass needs them to compute gradients. For multiplication `z = a * b`, the gradient with respect to `a` is `grad_z * b`, so we must save `b` during forward pass. This is the core memory trade-off in autograd: storing intermediate values uses memory, but enables fast backward passes.

Your implementation tracks graphs with the `_grad_fn` attribute:

```python
class AddBackward(Function):
    """Gradient computation for addition."""

    def __init__(self, a, b):
        """Store inputs needed for backward pass."""
        self.saved_tensors = (a, b)

    def apply(self, grad_output):
        """Compute gradients for both inputs."""
        return grad_output, grad_output  # Addition distributes gradients equally
```

When you compute `z = x + y`, your enhanced Tensor class automatically creates an AddBackward instance and attaches it to `z`:

```python
result = x.data + y.data
result_tensor = Tensor(result)
result_tensor._grad_fn = AddBackward(x, y)  # Track operation
```

This simple pattern enables arbitrarily complex computation graphs.

### The Chain Rule

The chain rule is the mathematical foundation of backpropagation. For composite functions, the derivative of the output with respect to any input equals the product of derivatives along the path connecting them.

Mathematically, if `z = f(g(x))`, then `dz/dx = (dz/dg) * (dg/dx)`. In computation graphs with multiple paths, gradients from all paths accumulate. This is gradient accumulation, and it's why shared parameters (like embedding tables used multiple times) correctly receive gradients from all their uses.

Consider this computation: `loss = (x * W + b)²`

```
Forward:  x → [Mul(W)] → z1 → [Add(b)] → z2 → [Square] → loss

Backward chain rule:
  ∂loss/∂z2 = 2*z2              (square backward)
  ∂loss/∂z1 = ∂loss/∂z2 * 1     (addition backward)
  ∂loss/∂x  = ∂loss/∂z1 * W     (multiplication backward)
```

Each backward function multiplies the incoming gradient by the local derivative. Here's how your MulBackward implements this:

```python
class MulBackward(Function):
    """Gradient computation for element-wise multiplication."""

    def apply(self, grad_output):
        """
        For z = a * b:
        ∂z/∂a = b → grad_a = grad_output * b
        ∂z/∂b = a → grad_b = grad_output * a

        Uses vectorized element-wise multiplication (NumPy broadcasting).
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if a.requires_grad:
            grad_a = grad_output * b.data  # Vectorized element-wise multiplication

        if b.requires_grad:
            grad_b = grad_output * a.data  # NumPy handles broadcasting automatically

        return grad_a, grad_b
```

The elegance is that each operation only knows its own derivative. The chain rule connects them all. NumPy's vectorized operations handle all element-wise computations efficiently without explicit loops.

### Backward Pass Implementation

The backward pass traverses the computation graph in reverse topological order, computing gradients for each tensor. Your `backward()` method implements this as a recursive tree walk:

```python
def backward(self, gradient=None):
    """Compute gradients via backpropagation."""
    if not self.requires_grad:
        return

    # Initialize gradient for scalar outputs
    if gradient is None:
        if self.data.size == 1:
            gradient = np.ones_like(self.data)
        else:
            raise ValueError("backward() requires gradient for non-scalar tensors")

    # Accumulate gradient (vectorized NumPy operation)
    if self.grad is None:
        self.grad = np.zeros_like(self.data)
    self.grad += gradient

    # Propagate to parent tensors
    if hasattr(self, '_grad_fn') and self._grad_fn is not None:
        grads = self._grad_fn.apply(gradient)  # Compute input gradients using vectorized ops

        for tensor, grad in zip(self._grad_fn.saved_tensors, grads):
            if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                tensor.backward(grad)  # Recursive call
```

The recursion naturally handles arbitrarily deep networks. For a 100-layer network, calling `loss.backward()` triggers 100 recursive calls, one per layer, flowing gradients from output to input. Note that while the graph traversal uses recursion, the gradient computations within each `apply()` method use vectorized NumPy operations for efficiency.

The `gradient` parameter deserves attention. For scalar losses (the typical case), you call `loss.backward()` without arguments, and the method initializes the gradient to 1.0. This makes sense: `∂loss/∂loss = 1`. For non-scalar tensors, you must provide the gradient from the next layer explicitly.

### Gradient Accumulation

Gradient accumulation is both a feature and a potential bug. When you call `backward()` multiple times on the same tensor, gradients add together. This is intentional: it enables mini-batch gradient descent and gradient checkpointing.

Consider processing a large batch in smaller chunks to fit in memory:

```python
# Large batch (doesn't fit in memory)
for mini_batch in split_batch(large_batch, chunks=4):
    loss = model(mini_batch)
    loss.backward()  # Gradients accumulate in model parameters

# Now gradients equal the sum over the entire large batch
optimizer.step()
model.zero_grad()  # Reset for next iteration
```

Without gradient accumulation, you'd need to store all mini-batch gradients and sum them manually. With accumulation, the autograd system handles it automatically.

But accumulation becomes a bug if you forget to call `zero_grad()` between iterations:

```python
# WRONG: Gradients accumulate across iterations
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients keep adding!
    optimizer.step()  # Updates use accumulated gradients from all previous batches

# CORRECT: Zero gradients after each update
for batch in dataloader:
    model.zero_grad()  # Reset gradients
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

Your `zero_grad()` implementation is simple but crucial:

```python
def zero_grad(self):
    """Reset gradients to None."""
    self.grad = None
```

Setting to None instead of zeros saves memory: NumPy doesn't allocate arrays until you accumulate the first gradient.

### Memory Management in Autograd

Autograd's memory footprint comes from two sources: stored intermediate tensors and gradient storage. For a forward pass through an N-layer network, you store roughly N intermediate activations. During backward pass, you store gradients for every parameter.

Consider a simple linear layer: `y = x @ W + b`

**Forward pass stores:**
- x (needed for computing grad_W = x.T @ grad_y)
- W (needed for computing grad_x = grad_y @ W.T)

**Backward pass allocates:**
- grad_x (same shape as x)
- grad_W (same shape as W)
- grad_b (same shape as b)

For a batch of 32 samples through a (512, 768) linear layer, the memory breakdown is:

```
Forward storage:
  x: 32 × 512 × 4 bytes = 64 KB
  W: 512 × 768 × 4 bytes = 1,572 KB

Backward storage:
  grad_x: 32 × 512 × 4 bytes = 64 KB
  grad_W: 512 × 768 × 4 bytes = 1,572 KB
  grad_b: 768 × 4 bytes = 3 KB

Total: ~3.3 MB for one layer (2× parameter size + activation size)
```

Multiply by network depth and you see why memory limits batch size. A 100-layer transformer stores 100× the activations, which can easily exceed GPU memory.

Production frameworks mitigate this with gradient checkpointing: they discard intermediate activations during forward pass and recompute them during backward pass. This trades compute (recomputing activations) for memory (not storing them). Your implementation doesn't do this—it's an advanced optimization—but understanding the trade-off is essential.

The implementation shows this memory overhead clearly in the MatmulBackward class:

```python
class MatmulBackward(Function):
    """
    Gradient computation for matrix multiplication.

    For Z = A @ B:
    - Must store A and B during forward pass
    - Backward computes: grad_A = grad_Z @ B.T and grad_B = A.T @ grad_Z
    - Uses vectorized NumPy operations (np.matmul, np.swapaxes)
    """

    def apply(self, grad_output):
        a, b = self.saved_tensors  # Retrieved from memory
        grad_a = grad_b = None

        if a.requires_grad:
            # Vectorized transpose and matmul (no explicit loops)
            b_T = np.swapaxes(b.data, -2, -1)
            grad_a = np.matmul(grad_output, b_T)

        if b.requires_grad:
            # Vectorized operations for efficiency
            a_T = np.swapaxes(a.data, -2, -1)
            grad_b = np.matmul(a_T, grad_output)

        return grad_a, grad_b
```

Notice that both `a` and `b` must be saved during forward pass. For large matrices, this storage cost dominates memory usage. All gradient computations use vectorized NumPy operations, which are implemented in optimized C/Fortran code under the hood—no explicit Python loops are needed.

## Production Context

### Your Implementation vs. PyTorch

Your autograd system and PyTorch's share the same design: computation graphs built during forward pass, reverse-mode differentiation during backward pass, and gradient accumulation in parameter tensors. The differences are in scale and optimization.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Graph Building** | Python objects, `_grad_fn` attribute | C++ objects, compiled graph |
| **Memory** | Stores all intermediates | Gradient checkpointing, memory pools |
| **Speed** | Pure Python, NumPy backend | C++/CUDA, fused kernels |
| **Operations** | 10 backward functions | 2000+ optimized backward functions |
| **Debugging** | Direct Python inspection | `torch.autograd.profiler`, graph visualization |

### Code Comparison

The following comparison shows identical conceptual patterns in TinyTorch and PyTorch. The APIs mirror each other because both implement the same autograd algorithm.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch import Tensor

# Create tensors with gradient tracking
x = Tensor([[1.0, 2.0]], requires_grad=True)
W = Tensor([[3.0], [4.0]], requires_grad=True)

# Forward pass builds computation graph
y = x.matmul(W)  # y = x @ W
loss = (y * y).sum()  # loss = sum(y²)

# Backward pass computes gradients
loss.backward()

# Access gradients
print(f"x.grad: {x.grad}")  # ∂loss/∂x
print(f"W.grad: {W.grad}")  # ∂loss/∂W
```
````

````{tab-item} ⚡ PyTorch
```python
import torch

# Create tensors with gradient tracking
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
W = torch.tensor([[3.0], [4.0]], requires_grad=True)

# Forward pass builds computation graph
y = x @ W  # PyTorch uses @ operator
loss = (y * y).sum()

# Backward pass computes gradients
loss.backward()

# Access gradients
print(f"x.grad: {x.grad}")
print(f"W.grad: {W.grad}")
```
````
`````

Let's walk through the comparison line by line:

- **Line 3-4 (Tensor creation)**: Both frameworks use `requires_grad=True` to enable gradient tracking. This is an opt-in design: most tensors (data, labels) don't need gradients, only parameters do.
- **Line 7-8 (Forward pass)**: Operations automatically build computation graphs. TinyTorch uses `.matmul()` method; PyTorch supports both `.matmul()` and the `@` operator.
- **Line 11 (Backward pass)**: Single method call triggers reverse-mode differentiation through the entire graph.
- **Line 14-15 (Gradient access)**: Both store gradients in the `.grad` attribute. Gradients have the same shape as the original tensor.

```{tip} What's Identical

Computation graph construction, chain rule implementation, and gradient accumulation semantics. When you debug PyTorch autograd issues, you're debugging the same algorithm you implemented here.
```

### Why Autograd Matters at Scale

To appreciate why automatic differentiation is essential, consider the scale of modern networks:

- **GPT-3**: 175 billion parameters = **175,000,000,000 gradients** to compute per training step
- **Training time**: Each backward pass takes roughly **2× forward pass time** (gradients require 2 matmuls per forward matmul)
- **Memory**: Storing computation graphs for a transformer can require **10× model size** in GPU memory

Manual gradient derivation becomes impossible at this scale. Even for a 3-layer MLP with 1 million parameters, manually coding gradients would take weeks and inevitably contain bugs. Autograd makes training tractable by automating the most error-prone part of deep learning.

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for autograd's performance characteristics and design decisions.

**Q1: Computation Graph Memory**

A 5-layer MLP processes a batch of 64 samples. Each layer stores its input activation for backward pass. Layer dimensions are: 784 → 512 → 256 → 128 → 10. How much memory (in MB) is used to store activations for one batch?

```{admonition} Answer
:class: dropdown

Layer 1 input: 64 × 784 × 4 bytes = 200 KB
Layer 2 input: 64 × 512 × 4 bytes = 131 KB
Layer 3 input: 64 × 256 × 4 bytes = 66 KB
Layer 4 input: 64 × 128 × 4 bytes = 33 KB
Layer 5 input: 64 × 10 × 4 bytes = 3 KB

**Total: ~433 KB ≈ 0.43 MB**

This is per forward pass! A 100-layer transformer would store 100× this amount, which is why gradient checkpointing trades compute for memory by recomputing activations during backward pass.
```

**Q2: Backward Pass Complexity**

A forward pass through a linear layer `y = x @ W` (where x is 32×512 and W is 512×256) takes 8ms. How long will the backward pass take?

```{admonition} Answer
:class: dropdown

Forward: 1 matmul (x @ W)

Backward: 2 matmuls
  - grad_x = grad_y @ W.T (32×256 @ 256×512)
  - grad_W = x.T @ grad_y (512×32 @ 32×256)

**Backward takes ~2× forward time ≈ 16ms**

This is why training (forward + backward) takes roughly 3× inference time. GPU parallelism and kernel fusion can reduce this, but the fundamental 2:1 ratio remains.
```

**Q3: Gradient Accumulation Memory**

You have 16GB GPU memory and a model with 1B parameters (float32). How much memory is available for activations and gradients during training?

```{admonition} Answer
:class: dropdown

Model parameters: 1B × 4 bytes = 4 GB
Gradients: 1B × 4 bytes = 4 GB
Optimizer state (Adam): 1B × 8 bytes = 8 GB (momentum + variance)

**Total framework overhead: 16 GB**

**Available for activations: 0 GB** - you've already exceeded memory!

This is why large models use gradient accumulation across multiple forward passes before updating parameters, or gradient checkpointing to reduce activation memory. The "2× parameter size" rule (parameters + gradients) is a minimum; optimizers add more overhead.
```

**Q4: requires_grad Performance**

A typical training batch has: 32 images (input), 10M parameter tensors (weights), 50 intermediate activation tensors. If requires_grad defaults to True for all tensors, how many tensors unnecessarily track gradients?

```{admonition} Answer
:class: dropdown

Tensors that NEED gradients:
- Parameters: 10M tensors ✓

Tensors that DON'T need gradients:
- Input images: 32 tensors (no gradient needed for data)
- Intermediate activations: 50 tensors (needed for backward but not updated)

**32 input tensors unnecessarily track gradients** if requires_grad defaults to True.

This is why PyTorch defaults requires_grad=False for new tensors and requires explicit opt-in for parameters. For image inputs with 32×3×224×224 = 4.8M values each, tracking gradients wastes 4.8M × 4 bytes = 19 MB per image × 32 = 608 MB for the batch!
```

**Q5: Graph Retention**

You forget to call `zero_grad()` before each training iteration. After 10 iterations, how do the gradients compare to correct training?

```{admonition} Answer
:class: dropdown

**Gradients accumulate across all 10 iterations.**

If correct gradient for iteration i is `g_i`, your accumulated gradient is:
`grad = g_1 + g_2 + g_3 + ... + g_10`

**Effects:**
1. **Magnitude**: Gradients are ~10× larger than they should be
2. **Direction**: The sum of 10 different gradients, which may not point toward the loss minimum
3. **Learning**: Parameter updates use the wrong direction and wrong magnitude
4. **Result**: Training diverges or oscillates instead of converging

**Bottom line**: Always call `zero_grad()` at the start of each iteration (or after `optimizer.step()`).
```

## Further Reading

For students who want to understand the academic foundations and mathematical underpinnings of automatic differentiation:

### Seminal Papers

- **Automatic Differentiation in Machine Learning: a Survey** - Baydin et al. (2018). Comprehensive survey of AD techniques, covering forward-mode, reverse-mode, and mixed-mode differentiation. Essential reading for understanding autograd theory. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)

- **Automatic Differentiation of Algorithms** - Griewank (1989). The foundational work on reverse-mode AD that underlies all modern deep learning frameworks. Introduces the mathematical formalism for gradient computation via the chain rule. [Computational Optimization and Applications](https://doi.org/10.1007/BF00139316)

- **PyTorch: An Imperative Style, High-Performance Deep Learning Library** - Paszke et al. (2019). Describes PyTorch's autograd implementation and design philosophy. Shows how imperative programming (define-by-run) enables dynamic computation graphs. [NeurIPS 2019](https://arxiv.org/abs/1912.01703)

### Additional Resources

- **Textbook**: "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 6 covers backpropagation and computational graphs with excellent visualizations
- **Tutorial**: [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) - Stanford's visual explanation of gradient flow through computation graphs
- **Documentation**: [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html) - Official guide to PyTorch's autograd implementation details

## What's Next

```{seealso} Coming Up: Module 07 - Optimizers

Implement SGD, Adam, and other optimization algorithms that use your autograd gradients to update parameters and train neural networks. You'll complete the training loop and make your networks learn from data.
```

**Preview - How Your Autograd Gets Used in Future Modules:**

| Module | What It Does | Your Autograd In Action |
|--------|--------------|------------------------|
| **07: Optimizers** | Update parameters using gradients | `optimizer.step()` uses `param.grad` computed by backward() |
| **08: Training** | Complete training loops | `loss.backward()` → `optimizer.step()` → repeat |
| **12: Attention** | Multi-head self-attention | Gradients flow through Q, K, V projections automatically |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/06_autograd/06_autograd.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/06_autograd/06_autograd.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
