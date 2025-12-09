---
title: "Optimizers"
description: "Gradient-based parameter optimization algorithms - SGD, Adam, and AdamW"
difficulty: "⭐⭐⭐⭐"
time_estimate: "6-8 hours"
prerequisites: ["tensor", "autograd"]
next_steps: ["training"]
learning_objectives:
  - "Understand optimization theory and convergence dynamics in neural network training"
  - "Implement SGD, momentum, and Adam optimizers from mathematical foundations"
  - "Design learning rate scheduling strategies for stable convergence"
  - "Analyze memory vs convergence trade-offs across optimization algorithms"
  - "Connect optimizer design to PyTorch's torch.optim implementation patterns"
---

# 06. Optimizers

**FOUNDATION TIER** | Difficulty: ⭐⭐⭐⭐ (4/4) | Time: 6-8 hours

## Overview

Welcome to the Optimizers module! You'll implement the learning algorithms that power every neural network—transforming gradients into intelligent parameter updates that enable models to learn from data. This module builds the optimization foundation used across all modern deep learning frameworks.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand optimization dynamics**: Master convergence behavior, learning rate sensitivity, and how gradients guide parameter updates in high-dimensional loss landscapes
- **Implement core optimization algorithms**: Build SGD, momentum, Adam, and AdamW optimizers from mathematical first principles
- **Analyze memory-convergence trade-offs**: Understand why Adam uses 3x memory but converges faster than SGD on many problems
- **Master adaptive learning rates**: See how Adam's per-parameter learning rates handle different gradient scales automatically
- **Connect to production frameworks**: Understand how your implementations mirror PyTorch's torch.optim.SGD and torch.optim.Adam design patterns

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement SGD with momentum, Adam optimizer with adaptive learning rates, and AdamW with decoupled weight decay from mathematical foundations
2. **Use**: Apply optimization algorithms to train neural networks on real classification and regression tasks
3. **Reflect**: Why does Adam converge faster initially but SGD often achieves better final test accuracy? What's the memory cost of adaptive learning rates?

## Getting Started

### Prerequisites

Ensure you understand the mathematical foundations:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test tensor
tito test autograd
```

**Required Background:**
- **Tensor Operations**: Understanding parameter storage and update mechanics
- **Automatic Differentiation**: Gradients computed via backpropagation
- **Calculus**: Derivatives, gradient descent, chain rule
- **Linear Algebra**: Vector operations, element-wise operations

### Development Workflow

1. **Open the development file**: `modules/06_optimizers/optimizers_dev.ipynb`
2. **Implement Optimizer base class**: Start with parameter management and zero_grad interface
3. **Build SGD with momentum**: Add velocity accumulation for smoother convergence
4. **Create Adam optimizer**: Implement adaptive learning rates with moment estimation and bias correction
5. **Add AdamW optimizer**: Build decoupled weight decay for proper regularization
6. **Export and verify**: `tito module complete 06 && tito test optimizers`

**Development Tips:**
- Test each optimizer on simple quadratic functions (f(x) = x²) where you can verify analytical convergence
- Compare convergence speed between SGD and Adam on the same problem
- Visualize loss curves to understand optimization dynamics
- Check momentum/moment buffers are properly initialized and updated
- Compare Adam vs AdamW to see the effect of decoupled weight decay

## Implementation Guide

### Core Optimization Algorithms

```python
# Base optimizer class with parameter management
class Optimizer:
    """Base class defining optimizer interface."""
    def __init__(self, params: List[Tensor]):
        self.params = list(params)
        self.step_count = 0

    def zero_grad(self):
        """Clear gradients from all parameters."""
        for param in self.params:
            param.grad = None

    def step(self):
        """Update parameters - implemented by subclasses."""
        raise NotImplementedError

# SGD with momentum for accelerated convergence
sgd = SGD(parameters=[w1, w2, bias], lr=0.01, momentum=0.9)
sgd.zero_grad()  # Clear previous gradients
loss.backward()  # Compute new gradients via autograd
sgd.step()       # Update parameters with momentum

# Adam optimizer with adaptive learning rates
adam = Adam(parameters=[w1, w2, bias], lr=0.001, betas=(0.9, 0.999))
adam.zero_grad()
loss.backward()
adam.step()      # Adaptive updates per parameter

# AdamW with decoupled weight decay
adamw = AdamW(parameters=[w1, w2, bias], lr=0.001, weight_decay=0.01)
adamw.zero_grad()
loss.backward()
adamw.step()     # Adam + proper regularization
```

### SGD with Momentum Implementation

```python
class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum.

    Momentum physics: velocity accumulates gradients over time,
    smoothing noisy updates and accelerating in consistent directions.
    """
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize momentum buffers (created lazily)
        self.momentum_buffers = [None for _ in self.params]

    def step(self):
        """Update parameters using momentum: v = βv + ∇L, θ = θ - αv"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update momentum buffer
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = np.zeros_like(param.data)

                # Update velocity: v_t = β*v_{t-1} + grad
                self.momentum_buffers[i] = (self.momentum * self.momentum_buffers[i]
                                           + grad)
                grad = self.momentum_buffers[i]

            # Update parameter: θ_t = θ_{t-1} - α*v_t
            param.data = param.data - self.lr * grad

        self.step_count += 1
```

### Adam Optimizer Implementation

```python
class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates.

    Combines momentum (first moment) with RMSprop-style adaptive rates
    (second moment) for robust optimization across different scales.
    """
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates (3x memory vs SGD)
        self.m_buffers = [None for _ in self.params]  # First moment
        self.v_buffers = [None for _ in self.params]  # Second moment

    def step(self):
        """Update parameters with adaptive learning rates"""
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            # Apply weight decay (Adam's approach - has issues)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Initialize buffers if needed
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)

            # Update biased first moment: m_t = β1*m_{t-1} + (1-β1)*grad
            self.m_buffers[i] = (self.beta1 * self.m_buffers[i]
                                + (1 - self.beta1) * grad)

            # Update biased second moment: v_t = β2*v_{t-1} + (1-β2)*grad²
            self.v_buffers[i] = (self.beta2 * self.v_buffers[i]
                                + (1 - self.beta2) * (grad ** 2))

            # Bias correction (critical for early training steps)
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            # Adaptive parameter update: θ = θ - α*m_hat/(√v_hat + ε)
            param.data = (param.data - self.lr * m_hat
                         / (np.sqrt(v_hat) + self.eps))
```

### AdamW Implementation (Decoupled Weight Decay)

```python
class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay.

    AdamW fixes Adam's weight decay bug by applying regularization
    directly to parameters, separate from gradient-based updates.
    """
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (same as Adam)
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]

    def step(self):
        """Perform AdamW update with decoupled weight decay"""
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient (NOT modified by weight decay - key difference!)
            grad = param.grad

            # Initialize buffers if needed
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)

            # Update moments using pure gradients
            self.m_buffers[i] = (self.beta1 * self.m_buffers[i]
                                + (1 - self.beta1) * grad)
            self.v_buffers[i] = (self.beta2 * self.v_buffers[i]
                                + (1 - self.beta2) * (grad ** 2))

            # Compute bias correction
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            # Apply gradient-based update
            param.data = (param.data - self.lr * m_hat
                         / (np.sqrt(v_hat) + self.eps))

            # Apply decoupled weight decay (after gradient update!)
            if self.weight_decay != 0:
                param.data = param.data * (1 - self.lr * self.weight_decay)
```

### Complete Training Integration

```python
# Modern training workflow combining all components
from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import SGD, Adam, AdamW

# Model setup (from previous modules)
model = Sequential([
    Linear(784, 128), ReLU(),
    Linear(128, 64), ReLU(),
    Linear(64, 10)
])

# Optimization setup
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_inputs, batch_targets in dataloader:
        # Forward pass
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update parameters

        epoch_loss += loss.data

    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
```

## Common Pitfalls

### Learning Rate Too Large Causing Divergence

**Problem**: Overly aggressive learning rates cause parameters to overshoot optimal values, leading to exploding loss or NaN values.

**Solution**: Start with conservative learning rates and gradually increase if training is stable:

```python
# ❌ Wrong - learning rate too large
optimizer = SGD(parameters, lr=1.0)  # May cause divergence
# Loss: 10.5 → 45.2 → 203.8 → NaN (exploded!)

# ✅ Correct - conservative learning rate
optimizer = SGD(parameters, lr=0.01)  # Stable convergence
# Loss: 10.5 → 9.8 → 9.2 → 8.7 (steady improvement)

# Rule of thumb: Start with lr=0.001 for Adam, lr=0.01 for SGD
```

### Forgetting zero_grad() Before backward()

**Problem**: Gradients accumulate across iterations unless explicitly cleared, causing incorrect parameter updates.

**Solution**: Always call `optimizer.zero_grad()` at the beginning of each training iteration:

```python
# ❌ Wrong - gradients accumulate
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()  # Gradients keep adding up!
        optimizer.step()

# ✅ Correct - clear gradients each iteration
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()  # Clear old gradients
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Momentum Value Out of Range

**Problem**: Setting momentum outside [0, 1) causes instability or no momentum effect.

**Solution**: Use standard momentum values: 0.9 for most cases, 0.99 for very smooth gradients:

```python
# ❌ Wrong - momentum >= 1.0 causes divergence
optimizer = SGD(parameters, lr=0.01, momentum=1.2)  # Invalid!

# ❌ Wrong - momentum < 0 makes no physical sense
optimizer = SGD(parameters, lr=0.01, momentum=-0.5)

# ✅ Correct - standard momentum values
optimizer = SGD(parameters, lr=0.01, momentum=0.9)   # Recommended
optimizer = SGD(parameters, lr=0.01, momentum=0.99)  # Very smooth
```

### Adam vs AdamW Weight Decay Confusion

**Problem**: Using Adam's weight decay incorrectly scales regularization with adaptive learning rates, causing inconsistent regularization strength across parameters.

**Solution**: Use AdamW for proper decoupled weight decay when regularization is needed:

```python
# ❌ Wrong - Adam's weight decay couples with adaptive LR
optimizer = Adam(parameters, lr=0.001, weight_decay=0.01)
# Weight decay effect varies per parameter based on gradient history!

# ✅ Correct - AdamW decouples weight decay from adaptive LR
optimizer = AdamW(parameters, lr=0.001, weight_decay=0.01)
# Weight decay applied uniformly: param *= (1 - lr * wd)
```

### Incorrect Bias Correction in Adam

**Problem**: Forgetting bias correction in early training steps causes incorrect moment estimates when buffers start at zero.

**Solution**: Always apply bias correction as `m_hat = m / (1 - beta1^t)` and `v_hat = v / (1 - beta2^t)`:

```python
# ❌ Wrong - no bias correction
m_hat = self.m_buffers[i]  # Biased toward zero initially!
v_hat = self.v_buffers[i]

# ✅ Correct - bias correction
bias_correction1 = 1 - self.beta1 ** self.step_count
bias_correction2 = 1 - self.beta2 ** self.step_count
m_hat = self.m_buffers[i] / bias_correction1
v_hat = self.v_buffers[i] / bias_correction2
```

## Testing

### Comprehensive Test Suite

Run the full test suite to verify optimization algorithm correctness:

```bash
# TinyTorch CLI (recommended)
tito test optimizers

# Direct pytest execution
python -m pytest tests/ -k optimizers -v

# Test specific optimizer
python -m pytest tests/test_optimizers.py::test_adam_convergence -v
```

### Test Coverage Areas

- **Algorithm Implementation**: Verify Optimizer base, SGD, Adam, and AdamW compute mathematically correct parameter updates
- **Mathematical Correctness**: Test against analytical solutions for convex optimization problems (quadratic functions)
- **State Management**: Ensure proper momentum and moment estimation tracking across training steps
- **Memory Efficiency**: Verify buffer initialization and memory usage patterns
- **Training Integration**: Test optimizers in complete neural network training workflows with real data

### Inline Testing & Convergence Analysis

The module includes comprehensive mathematical validation and convergence visualization:

```python
# Example inline test output
🔬 Unit Test: Base Optimizer...
✅ Parameter validation working correctly
✅ zero_grad clears all gradients properly
✅ Error handling for non-gradient parameters
📈 Progress: Base Optimizer ✓

# SGD with momentum validation
🔬 Unit Test: SGD with momentum...
✅ Parameter updates follow momentum equation v_t = βv_{t-1} + ∇L
✅ Velocity accumulation working correctly
✅ Weight decay applied properly
✅ Momentum accelerates convergence vs vanilla SGD
📈 Progress: SGD with Momentum ✓

# Adam optimizer validation
🔬 Unit Test: Adam optimizer...
✅ First moment estimation (m_t) computed correctly
✅ Second moment estimation (v_t) computed correctly
✅ Bias correction applied properly (critical for early steps)
✅ Adaptive learning rates working per parameter
✅ Convergence faster than SGD on ill-conditioned problem
📈 Progress: Adam Optimizer ✓

# AdamW decoupled weight decay validation
🔬 Unit Test: AdamW optimizer...
✅ Weight decay decoupled from gradient updates
✅ Results differ from Adam (proving proper implementation)
✅ Regularization consistent across gradient scales
✅ With zero weight decay, matches Adam behavior
📈 Progress: AdamW Optimizer ✓
```

### Manual Testing Examples

```python
from tinytorch.core.optimizers import SGD, Adam, AdamW
from tinytorch.core.tensor import Tensor

# Test 1: SGD convergence on simple quadratic
print("Test 1: SGD on f(x) = x²")
x = Tensor([10.0], requires_grad=True)
sgd = SGD([x], lr=0.1, momentum=0.9)

for step in range(100):
    sgd.zero_grad()
    loss = (x ** 2).sum()  # Minimize f(x) = x², minimum at x=0
    loss.backward()
    sgd.step()

    if step % 10 == 0:
        print(f"Step {step}: x = {x.data[0]:.6f}, loss = {loss.data:.6f}")
# Expected: x should converge to 0

# Test 2: Adam on multidimensional optimization
print("\nTest 2: Adam on f(x,y) = x² + y²")
params = Tensor([5.0, -3.0], requires_grad=True)
adam = Adam([params], lr=0.1)

for step in range(50):
    adam.zero_grad()
    loss = (params ** 2).sum()  # Minimize ||x||²
    loss.backward()
    adam.step()

    if step % 10 == 0:
        print(f"Step {step}: params = {params.data}, loss = {loss.data:.6f}")
# Expected: Both parameters converge to 0

# Test 3: Compare SGD vs Adam vs AdamW convergence
print("\nTest 3: Optimizer comparison")
x_sgd = Tensor([10.0], requires_grad=True)
x_adam = Tensor([10.0], requires_grad=True)
x_adamw = Tensor([10.0], requires_grad=True)

sgd = SGD([x_sgd], lr=0.01, momentum=0.9)
adam = Adam([x_adam], lr=0.01)
adamw = AdamW([x_adamw], lr=0.01, weight_decay=0.01)

for step in range(20):
    # SGD update
    sgd.zero_grad()
    loss_sgd = (x_sgd ** 2).sum()
    loss_sgd.backward()
    sgd.step()

    # Adam update
    adam.zero_grad()
    loss_adam = (x_adam ** 2).sum()
    loss_adam.backward()
    adam.step()

    # AdamW update
    adamw.zero_grad()
    loss_adamw = (x_adamw ** 2).sum()
    loss_adamw.backward()
    adamw.step()

    if step % 5 == 0:
        print(f"Step {step}: SGD={x_sgd.data[0]:.6f}, Adam={x_adam.data[0]:.6f}, AdamW={x_adamw.data[0]:.6f}")
# Expected: Adam/AdamW converge faster initially
```

## Production Context

### Your Implementation vs. Production Frameworks

Understanding what you're building vs. what production frameworks provide:

| Feature | Your Optimizers (Module 06) | PyTorch torch.optim | TensorFlow tf.keras.optimizers |
|---------|---------------------------|---------------------|-------------------------------|
| **Backend** | NumPy (CPU-only) | C++/CUDA (CPU/GPU) | C++/CUDA/XLA |
| **SGD** | Momentum + weight decay | ✅ `optim.SGD()` | ✅ `SGD()` |
| **Adam** | First/second moments + bias correction | ✅ `optim.Adam()` | ✅ `Adam()` |
| **AdamW** | Decoupled weight decay | ✅ `optim.AdamW()` | ✅ `AdamW()` |
| **Learning Rate Scheduling** | ❌ Not implemented | ✅ `lr_scheduler.*` | ✅ `schedules.*` |
| **Gradient Clipping** | ❌ Manual | ✅ `clip_grad_norm_()` | ✅ `clipnorm` parameter |
| **Mixed Precision** | ❌ FP32 only | ✅ AMP integration | ✅ `loss_scale` |
| **Fused Optimizers** | ❌ Separate ops | ✅ CUDA fused kernels | ✅ XLA fusion |
| **Distributed Training** | ❌ Single device | ✅ DDP, FSDP support | ✅ `tf.distribute` |

**Educational Focus**: Your implementations prioritize clarity of optimization algorithms. Production frameworks add performance optimizations (fused kernels, mixed precision) and distributed training support while maintaining the same core update rules.

### Side-by-Side Code Comparison

**Your implementation:**
```python
from tinytorch.core.optimizers import SGD, Adam, AdamW
from tinytorch.core.tensor import Tensor

# Create model parameters
w1 = Tensor(np.random.randn(784, 256), requires_grad=True)
b1 = Tensor(np.zeros(256), requires_grad=True)
w2 = Tensor(np.random.randn(256, 10), requires_grad=True)
b2 = Tensor(np.zeros(10), requires_grad=True)

# Initialize optimizer with all parameters
optimizer = AdamW([w1, b1, w2, b2], lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear gradients
    loss = forward_pass(x, w1, b1, w2, b2)
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters
```

**Equivalent PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create model (parameters managed automatically)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Initialize optimizer (gets parameters from model)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop (same workflow!)
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear gradients
    loss = forward_pass(model, x)
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters
```

**Key Differences:**
1. **Parameter Management**: PyTorch's `nn.Module` automatically collects parameters. You manually pass parameter lists to optimizers.
2. **Fused Operations**: PyTorch uses CUDA fused kernels that combine parameter update operations (read gradient + update parameter + apply weight decay) into single GPU kernel for efficiency.
3. **Learning Rate Scheduling**: PyTorch provides `lr_scheduler` classes that automatically adjust learning rates during training (cosine annealing, step decay, etc.).
4. **Mixed Precision**: PyTorch's AMP (Automatic Mixed Precision) integrates with optimizers to scale gradients, preventing underflow in FP16 training.

### Real-World Production Usage

**OpenAI GPT-3 Training**: Uses AdamW optimizer with β1=0.9, β2=0.95, weight_decay=0.1 across 175B parameters distributed across thousands of GPUs. Learning rate warmup (0→max over 375M tokens) followed by cosine decay is critical for stable training at scale.

**Google BERT Pre-training**: Uses Adam (now typically AdamW) with β1=0.9, β2=0.999, learning rate=1e-4 with warmup over 10K steps. Gradient accumulation over 8-64 mini-batches enables effective batch sizes of 256-8192 despite GPU memory constraints.

**Meta ResNet ImageNet**: Primarily uses SGD with momentum=0.9, weight_decay=1e-4, learning rate starting at 0.1 with step decay (÷10 at epochs 30, 60, 90). Momentum SGD empirically outperforms Adam on computer vision tasks despite slower convergence.

**Tesla Autopilot**: On-device training uses modified SGD with aggressive gradient clipping (prevents outlier frames from destabilizing models) and smaller momentum (0.5-0.7) for faster adaptation to distribution shifts in real-time deployment.

**Hugging Face Transformers**: Default training uses AdamW with β1=0.9, β2=0.999, ε=1e-8, weight_decay=0.01. Learning rate warmup over 500 steps then linear decay prevents divergence in early training when parameters are randomly initialized.

### Performance Characteristics at Scale

**Memory Overhead per Optimizer:**
- **SGD**: 2× parameter memory (params + momentum buffers)
- **Adam**: 3× parameter memory (params + first moment + second moment)
- **AdamW**: 3× parameter memory (same as Adam, decoupling is algorithmic not memory-based)
- For GPT-3 (175B params in FP16): Adam requires 175B×2bytes×3 = 1.05TB memory just for optimizer state!

**Computational Cost per Step:**
- **SGD with momentum**: 2 operations per parameter (update momentum, update parameter)
- **Adam**: 7 operations per parameter (update m, update v, bias correction ×2, sqrt, division, parameter update)
- **Gradient clipping overhead**: Additional norm computation across all parameters before optimizer step
- For 1B parameter model: Adam requires ~7B operations per step vs SGD's ~2B (3.5× cost)

**Distributed Training Optimization:**
- Gradient synchronization (all-reduce) happens BEFORE optimizer step
- Each GPU computes local gradients via autograd, then averages gradients across devices
- Optimizer state (momentum, Adam moments) is NOT synchronized—each GPU maintains independent optimizer state
- For 8 GPUs training GPT: 8× autograd computation, 1× gradient synchronization, 8× independent optimizer updates

**Learning Rate Scheduling Impact:**
- Cosine annealing schedule: `lr_t = lr_max × 0.5 × (1 + cos(πt/T))` provides smooth decay without abrupt steps
- Warmup prevents early divergence when parameters are random: `lr_t = t/warmup_steps × lr_max` for initial steps
- Production systems typically use warmup (500-10K steps) + cosine decay (to ~0.1× initial LR)
- Learning rate is THE most important hyperparameter—wrong schedule can prevent convergence entirely

### How Your Implementation Maps to PyTorch

**What you just built:**
```python
# Your AdamW implementation
class AdamW(Optimizer):
    def step(self):
        self.step_count += 1
        for i, param in enumerate(self.params):
            # Update moments
            self.m_buffers[i] = beta1 * self.m_buffers[i] + (1 - beta1) * grad
            self.v_buffers[i] = beta2 * self.v_buffers[i] + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m_buffers[i] / (1 - beta1 ** self.step_count)
            v_hat = self.v_buffers[i] / (1 - beta2 ** self.step_count)

            # Adam update
            param.data -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Decoupled weight decay
            param.data *= (1 - lr * weight_decay)
```

**How PyTorch does it:**
```python
# PyTorch C++ implementation (simplified)
# torch/optim/adamw.py calls into C++
void AdamW::step() {
    for (auto& param : parameters) {
        // Update moments (vectorized CUDA kernel)
        m[i] = beta1 * m[i] + (1 - beta1) * grad;
        v[i] = beta2 * v[i] + (1 - beta2) * (grad * grad);

        // Bias correction
        m_hat = m[i] / (1 - pow(beta1, step));
        v_hat = v[i] / (1 - pow(beta2, step));

        // Fused update kernel (combines 3 operations)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)  // Adam update
                      - lr * wd * param;                    // Weight decay
    }
}
```

**Key Insight**: Your implementation uses the **exact same update equations** as PyTorch. The difference is execution (NumPy CPU vs CUDA GPU kernels) and optimization level (separate operations vs fused kernels), not the fundamental optimization algorithm.

## Systems Thinking Questions

### Real-World Applications

- **Large Language Models**: GPT and BERT training relies on AdamW optimizer for stable convergence across billions of parameters with varying gradient scales and proper regularization
- **Computer Vision**: ResNet and Vision Transformer training typically uses SGD with momentum for best final test accuracy despite slower initial convergence
- **Recommendation Systems**: Online learning systems use adaptive optimizers like Adam for continuous model updates with non-stationary data distributions
- **Reinforcement Learning**: Policy gradient methods depend heavily on careful optimizer choice and learning rate tuning due to high variance gradients

### Optimization Theory Foundations

- **Gradient Descent**: Update rule θ_{t+1} = θ_t - α∇L(θ_t) where α is learning rate controlling step size in steepest descent direction
- **Momentum**: Velocity accumulation v_{t+1} = βv_t + ∇L(θ_t), then θ_{t+1} = θ_t - αv_{t+1} smooths noisy gradients and accelerates convergence
- **Adam**: Combines momentum (first moment m_t) with adaptive learning rates (second moment v_t), includes bias correction for early training steps
- **AdamW**: Decouples weight decay from gradient updates: applies gradient update first, then weight decay, fixing Adam's regularization bug

### Performance Characteristics

- **SGD Memory**: O(2n) memory for n parameters (params + momentum buffers), most memory-efficient optimizer with momentum
- **Adam Memory**: O(3n) memory due to first and second moment buffers (params + m_buffers + v_buffers), 1.5x SGD cost
- **Convergence Speed**: Adam often converges faster initially due to adaptive rates, especially with sparse gradients or varying scales
- **Final Performance**: SGD with momentum often achieves better test accuracy on computer vision tasks despite slower convergence
- **Learning Rate Sensitivity**: Adam/AdamW are more robust to learning rate choice than vanilla SGD, making them popular for transformer training
- **Computational Cost**: Adam requires ~1.5x more computation per step (moment updates + bias correction + sqrt operations) than SGD

### Critical Thinking: Memory vs Convergence Trade-offs

**Reflection Question**: Why does Adam use 3x the memory of parameter-only storage (and 1.5x SGD), and when is this trade-off worth it?

**Key Insights:**
- **Memory Cost**: Adam stores parameter data + first moment (momentum) + second moment (variance) for every parameter
- **Adaptive Benefit**: Per-parameter learning rates handle different gradient scales automatically
- **Use Case**: Transformers benefit from Adam (varying embedding vs attention scales), CNNs often prefer SGD (more uniform scales)
- **Production Decision**: Memory-constrained systems (mobile, edge devices) may prefer SGD despite slower convergence
- **Training Time**: Faster convergence can save GPU hours, offsetting memory cost in cloud training scenarios

**Reflection Question**: Why does SGD with momentum often achieve better test accuracy than Adam on vision tasks, despite slower training?

**Key Insights:**
- **Generalization**: SGD explores flatter minima that generalize better to test data
- **Overfitting**: Adam's fast convergence may lead to sharper minima with worse generalization
- **Learning Rate Schedule**: Careful learning rate decay with SGD achieves better final performance
- **Task Dependency**: Effect is strongest on CNNs, less pronounced on transformers
- **Modern Practice**: AdamW with proper weight decay often bridges this gap

**Reflection Question**: How does AdamW's decoupled weight decay fix Adam's regularization bug?

**Key Insights:**
- **Adam Bug**: Adds weight decay to gradients, so adaptive learning rates affect regularization strength inconsistently
- **AdamW Fix**: Applies weight decay directly to parameters after gradient update, decoupling optimization from regularization
- **Consistency**: Weight decay effect is now uniform across parameters regardless of gradient magnitudes
- **Production Impact**: AdamW is now preferred over Adam in most modern training pipelines (BERT, GPT-3, etc.)

## Ready to Build?

You're about to implement the algorithms that enable all of modern deep learning! Every neural network—from the image classifiers in your phone to GPT-4—depends on the optimization algorithms you're building in this module.

Understanding these algorithms from first principles will transform how you think about training. When you implement momentum physics and see how velocity accumulation smooths noisy gradients, when you build Adam's adaptive learning rates and understand why they help with varying parameter scales, when you create AdamW and see how decoupled weight decay fixes Adam's bug—you'll develop deep intuition for why some training configurations work and others fail.

Take your time with the mathematics. Test your optimizers on simple quadratic functions where you can verify convergence analytically. Compare SGD vs Adam vs AdamW on the same problem to see their different behaviors. Visualize loss curves to understand optimization dynamics. Monitor memory usage to see the trade-offs. This hands-on experience will make you a better practitioner who can debug training failures, tune hyperparameters effectively, and make informed decisions about optimizer choice in production systems. Enjoy building the intelligence behind intelligent systems!

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/06_optimizers/optimizers_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/06_optimizers/optimizers_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/06_optimizers/optimizers_dev.ipynb
:class-header: bg-light

Browse the Jupyter notebook and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

```

---

<div class="prev-next-area">
<a class="left-prev" href="../modules/05_autograd_ABOUT.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../modules/07_training_ABOUT.html" title="next page">Next Module →</a>
</div>
