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
# Module 07: Optimizers - Sophisticated Learning Algorithms

Welcome to Module 07! You'll build optimizers that enable neural networks to learn from gradients using sophisticated algorithms.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor with gradients (Modules 01-06)
**You'll Build**: SGD, Adam, and AdamW optimizers with sophisticated momentum and adaptive learning
**You'll Enable**: Modern optimization algorithms that power state-of-the-art neural networks

**Connection Map**:
```
Gradients → Optimizers → Training
(Module 06)  (Module 07)  (Module 08)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement SGD with momentum for stable gradient descent
2. Build Adam optimizer with adaptive learning rates
3. Create AdamW optimizer with decoupled weight decay
4. Understand memory and computational trade-offs in optimization algorithms

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/07_optimizers/optimizers_dev.py`
**Building Side:** Code exports to `tinytorch.core.optimizers`

```python
# How to use this module:
from tinytorch.core.optimizers import SGD, Adam, AdamW
```

**Why this matters:**
- **Learning:** Complete optimization system for modern neural network training
- **Production:** Proper organization like PyTorch's torch.optim with all optimization algorithms together
- **Consistency:** All optimization logic and parameter updating in core.optimizers
- **Integration:** Works seamlessly with gradients from Module 06 for complete training capability
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.optimizers
#| export

import numpy as np
from typing import List, Union, Optional, Dict, Any

# Import Tensor from Module 01 (now with gradient support from Module 06)
from tinytorch.core.tensor import Tensor

# Enable autograd to add gradient tracking to Tensor
# This module depends on Module 06 (Autograd) being available
from tinytorch.core.autograd import enable_autograd
enable_autograd()

# Constants for optimizer defaults
DEFAULT_LEARNING_RATE_SGD = 0.01  # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001  # Default learning rate for Adam/AdamW
DEFAULT_MOMENTUM = 0.9  # Default momentum for SGD
DEFAULT_BETA1 = 0.9  # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999  # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8  # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01  # Default weight decay for AdamW

# %% [markdown]
"""
## 💡 Introduction: What are Optimizers?

Optimizers are the engines that drive neural network learning. They take gradients computed from your loss function and use them to update model parameters toward better solutions. Think of optimization as navigating a complex landscape where you're trying to find the lowest valley (minimum loss).

### The Optimization Challenge

Imagine you're hiking in dense fog, trying to reach the bottom of a valley. You can only feel the slope under your feet (the gradient), but you can't see where you're going. Different optimization strategies are like different hiking approaches:

```
Loss Landscape (2D visualization):
       🏔️
      /  \\
   🚶 /    \\
    /      \\
   /   🎯   \\  ← Global minimum (goal)
  /          \\
 🏔️          🏔️

Challenge: Navigate to 🎯 using only local slope information!
```

### Our Optimizer Toolkit

**SGD (Stochastic Gradient Descent)**
- Strategy: Always step downhill
- Problem: Can get stuck oscillating in narrow valleys
- Solution: Add momentum to "coast" through oscillations

**Adam (Adaptive Moment Estimation)**
- Strategy: Adapt step size for each parameter individually
- Advantage: Different learning rates for different dimensions
- Key Insight: Some directions need big steps, others need small steps

**AdamW (Adam with Weight Decay)**
- Strategy: Adam + proper regularization
- Fix: Separates optimization from regularization
- Result: Better generalization and training stability

### The Mathematics Behind Movement

At its core, optimization follows: **θ_new = θ_old - α * direction**

Where:
- `θ` = parameters (your position in the landscape)
- `α` = step size (learning rate)
- `direction` = where to step (gradient-based)

But sophisticated optimizers do much more than basic gradient descent!
"""

# %% [markdown]
"""
## 📐 Foundations: Mathematical Background

### Understanding Momentum: The Physics of Optimization

Momentum in optimization works like momentum in physics. A ball rolling down a hill doesn't immediately change direction when it hits a small bump - it has momentum that carries it forward.

```
Without Momentum (SGD):           With Momentum:
     ↓                                ↘️
  ←  •  →  ← oscillation           →  •  → smooth path
     ↑                                ↙️

Narrow valley problem:            Momentum solution:
|\\     /|                        |\\     /|
| \\ • / | ← ping-pong             | \\ •→/ | ← smoother
|  \\ /  |   motion                |  \\ /  |   descent
|   ●   |                        |   ●   |
```

**SGD with Momentum Formula:**
```
velocity = β * previous_velocity + (1-β) * current_gradient
parameter = parameter - learning_rate * velocity

Where β ≈ 0.9 means "90% memory of previous direction"
```

### Adam: Adaptive Learning for Each Parameter

Adam solves a key problem: different parameters need different learning rates. Imagine adjusting the focus and zoom on a camera - you need fine control for focus but coarse control for zoom.

```
Parameter Landscape (2 dimensions):

   param2
     ^
     |
   😞|    steep gradient
     |    (needs small steps)
     |
  ---+--●--→ param1
     |     \\
     |      \\ gentle gradient
     |       \\ (needs big steps)

Adam Solution: Automatic step size per parameter!
```

**Adam's Two-Memory System:**

1. **First Moment (m)**: "Which direction am I usually going?"
   - `m = β₁ * old_m + (1-β₁) * gradient`
   - Like momentum, but for direction

2. **Second Moment (v)**: "How big are my gradients usually?"
   - `v = β₂ * old_v + (1-β₂) * gradient²`
   - Tracks gradient magnitude

3. **Adaptive Update**:
   - `step_size = m / √v`
   - Big gradients → smaller steps
   - Small gradients → relatively bigger steps

### AdamW: Fixing Weight Decay

Adam has a subtle bug in how it applies weight decay (regularization). AdamW fixes this:

```
Adam (incorrect):               AdamW (correct):
gradient += weight_decay * param    [compute gradient update]
update_param_with_gradient()        param -= learning_rate * gradient_update
                                   param *= (1 - weight_decay)  ← separate!

Why it matters:
- Adam: Weight decay affected by adaptive learning rates
- AdamW: Weight decay is consistent regardless of gradients
```
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Optimizers

Now we'll implement each optimizer step by step, following the pattern: understand the algorithm → implement it → test it immediately. Each optimizer builds on the foundation of the previous one.

### Implementation Strategy

```
Optimizer Base Class
    ↓
SGD (foundation algorithm)
    ↓
SGD + Momentum (reduce oscillations)
    ↓
Adam (adaptive learning rates)
    ↓
AdamW (proper weight decay)
```
"""

# %% nbgrader={"grade": false, "grade_id": "optimizer-base", "solution": true}
#| export
class Optimizer:
    """
    Base class for all optimizers.

    This class defines the common interface that all optimizers must implement:
    - zero_grad(): Clear gradients from parameters
    - step(): Update parameters based on gradients
    """

    def __init__(self, params: List[Tensor]):
        """
        Initialize optimizer with parameters to optimize.

        TODO: Set up the parameter list for optimization

        APPROACH:
        1. Store parameters as a list for iteration
        2. Validate that all parameters require gradients
        3. Initialize step counter for algorithms that need it

        EXAMPLE:
        >>> linear = Linear(784, 128)
        >>> optimizer = SGD(linear.parameters(), lr=0.01)

        HINT: Store parameters for iteration during optimization steps
        """
        ### BEGIN SOLUTION
        # Validate and store parameters
        if not isinstance(params, list):
            params = list(params)

        # Store parameters - gradient tracking is handled by autograd module
        self.params = params
        self.step_count = 0  # For algorithms that need step counting
        ### END SOLUTION

    def zero_grad(self):
        """
        Clear gradients from all parameters.

        TODO: Reset all parameter gradients to None

        APPROACH:
        1. Iterate through all parameters
        2. Set each parameter's grad to None

        EXAMPLE:
        >>> optimizer.zero_grad()  # Clears all gradients
        >>> assert param.grad is None for param in optimizer.params

        WHY: Gradients accumulate by default, so we need to clear them between batches
        """
        ### BEGIN SOLUTION
        for param in self.params:
            param.grad = None
        ### END SOLUTION

    def step(self):
        """
        Update parameters based on gradients.

        This is abstract - each optimizer implements its own update rule.
        """
        raise NotImplementedError("Subclasses must implement step()")

# %% [markdown]
"""
### 🔬 Unit Test: Base Optimizer
This test validates our base Optimizer class works correctly.
**What we're testing**: Parameter validation and zero_grad functionality
**Why it matters**: Foundation for all specific optimizer implementations
**Expected**: Proper parameter storage and gradient clearing
"""

# %% nbgrader={"grade": true, "grade_id": "test-optimizer-base", "locked": true, "points": 10}
def test_unit_optimizer_base():
    """🔬 Test base Optimizer functionality."""
    print("🔬 Unit Test: Base Optimizer...")

    # Create test parameters
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    # Add some gradients
    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([[0.3, 0.4], [0.5, 0.6]])

    # Create optimizer
    optimizer = Optimizer([param1, param2])

    # Test parameter storage
    assert len(optimizer.params) == 2
    assert optimizer.params[0] is param1
    assert optimizer.params[1] is param2
    assert optimizer.step_count == 0

    # Test zero_grad
    optimizer.zero_grad()
    assert param1.grad is None
    assert param2.grad is None

    # Test that optimizer accepts any tensor (no validation required)
    # Gradient tracking is handled by the autograd module
    regular_param = Tensor([1.0])
    opt = Optimizer([regular_param])
    assert len(opt.params) == 1

    print("✅ Base Optimizer works correctly!")

if __name__ == "__main__":
    test_unit_optimizer_base()

# %% [markdown]
"""
## 🏗️ SGD - Stochastic Gradient Descent

SGD is the foundation of neural network perf. It implements the simple but powerful idea: "move in the direction opposite to the gradient."

### Why SGD Works

Gradients point uphill (toward higher loss). To minimize loss, we go downhill:

```
Loss Surface (side view):

    Loss
     ^
     |
  📈 |     current position
     |    /
     |   • ← you are here
     |  / \\
     | /   \\ gradient points uphill
     |/     \\
     ●-------\\--→ parameters
      \\        \\
       \\        ↘️ SGD steps downhill
        \\        (opposite to gradient)
         \\⭐ ← goal (minimum loss)
```

### The Oscillation Problem

Pure SGD can get trapped oscillating in narrow valleys:

```
Narrow valley (top view):
  \\     /
   \\   /   ← steep sides
    \\ /
  4← • →2  ← SGD bounces back and forth
    / \\
   1   3   instead of going down the valley
  /     \\
 ●       \\
 goal     \\
```

### Momentum Solution

Momentum remembers the direction you were going and continues in that direction:

```
With momentum:
  \\     /
   \\   /
    \\ /
     •  ← smooth path down the valley
    / ↓
   /   ↓
  ●    ↓  momentum carries us through oscillations
 goal
```

**Implementation:** SGD keeps a "velocity" buffer that accumulates momentum.
"""

# %% nbgrader={"grade": false, "grade_id": "sgd-optimizer", "solution": true}
#| export
class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.

    SGD is the foundational optimization algorithm that moves parameters
    in the direction opposite to gradients. With momentum, it remembers
    previous updates to reduce oscillations and accelerate convergence.
    """

    def __init__(self, params: List[Tensor], lr: float = DEFAULT_LEARNING_RATE_SGD, momentum: float = 0.0, weight_decay: float = 0.0):
        """
        Initialize SGD optimizer.

        TODO: Set up SGD with momentum and weight decay

        APPROACH:
        1. Call parent constructor to set up parameters
        2. Store learning rate, momentum, and weight decay
        3. Initialize momentum buffers for each parameter

        EXAMPLE:
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        HINTS:
        - Momentum buffers should be initialized as None
        - They'll be created lazily on first step
        """
        ### BEGIN SOLUTION
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize momentum buffers (created lazily)
        self.momentum_buffers = [None for _ in self.params]
        ### END SOLUTION

    def has_momentum(self) -> bool:
        """
        Check if this optimizer uses momentum.

        This explicit API method replaces the need for hasattr() checks
        in checkpointing code (Module 08).

        Returns:
            bool: True if momentum is enabled (momentum > 0), False otherwise

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> optimizer.has_momentum()
            True
        """
        return self.momentum > 0

    def get_momentum_state(self) -> Optional[List]:
        """
        Get momentum buffers for checkpointing.

        This explicit API method provides safe access to momentum buffers
        without using hasattr(), making the API contract clear.

        Returns:
            Optional[List]: List of momentum buffers if momentum is enabled,
                          None otherwise

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> optimizer.step()  # Initialize buffers
            >>> state = optimizer.get_momentum_state()
            >>> # Later: optimizer.set_momentum_state(state)
        """
        if not self.has_momentum():
            return None
        return [buf.copy() if buf is not None else None
                for buf in self.momentum_buffers]

    def set_momentum_state(self, state: Optional[List]) -> None:
        """
        Restore momentum buffers from checkpointing.

        This explicit API method provides safe restoration of momentum state
        without using hasattr().

        Args:
            state: List of momentum buffers or None

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> state = optimizer.get_momentum_state()
            >>> # Training interruption...
            >>> new_optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> new_optimizer.set_momentum_state(state)
        """
        if state is None or not self.has_momentum():
            return

        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"State length {len(state)} doesn't match "
                f"optimizer parameters {len(self.momentum_buffers)}"
            )

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffers[i] = buf.copy()

    def step(self):
        """
        Perform SGD update step with momentum.

        TODO: Implement SGD parameter update with momentum

        APPROACH:
        1. For each parameter with gradients:
           a. Apply weight decay if specified
           b. Update momentum buffer
           c. Update parameter using momentum

        FORMULA:
        - With weight decay: grad = grad + weight_decay * param
        - Momentum: v = momentum * v_prev + grad
        - Update: param = param - lr * v

        HINTS:
        - Skip parameters without gradients
        - Initialize momentum buffers on first use
        - Use in-place operations to save memory
        """
        ### BEGIN SOLUTION
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient data - grad can be Tensor or numpy array
            grad = param.grad
            # Handle both Tensor (with .data) and numpy array (from autograd) cases
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                # grad is already a numpy array from autograd
                grad_data = grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            # Update momentum buffer
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    # Initialize momentum buffer
                    self.momentum_buffers[i] = np.zeros_like(param.data)

                # Update momentum: v = momentum * v_prev + grad
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] + grad_data
                grad_data = self.momentum_buffers[i]

            # Update parameter: param = param - lr * grad
            param.data = param.data - self.lr * grad_data

        # Increment step counter
        self.step_count += 1
        ### END SOLUTION

# %% [markdown]
"""
### 🔬 Unit Test: SGD Optimizer
This test validates our SGD implementation works correctly.
**What we're testing**: SGD updates with and without momentum
**Why it matters**: Core optimization algorithm used in neural network training
**Expected**: Correct parameter updates following SGD formulas
"""

# %% nbgrader={"grade": true, "grade_id": "test-sgd", "locked": true, "points": 15}
def test_unit_sgd_optimizer():
    """🔬 Test SGD optimizer implementation."""
    print("🔬 Unit Test: SGD Optimizer...")

    # Test basic SGD without momentum
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = SGD([param], lr=0.1)
    original_data = param.data.copy()

    optimizer.step()

    # Expected: param = param - lr * grad = [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    expected = original_data - 0.1 * param.grad.data
    assert np.allclose(param.data, expected)
    assert optimizer.step_count == 1

    # Test SGD with momentum
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    param2.grad = Tensor([0.1, 0.2])

    optimizer_momentum = SGD([param2], lr=0.1, momentum=0.9)

    # First step: v = 0.9 * 0 + [0.1, 0.2] = [0.1, 0.2]
    optimizer_momentum.step()
    expected_first = np.array([1.0, 2.0]) - 0.1 * np.array([0.1, 0.2])
    assert np.allclose(param2.data, expected_first)

    # Second step with same gradient
    param2.grad = Tensor([0.1, 0.2])
    optimizer_momentum.step()
    # v = 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
    expected_momentum = np.array([0.19, 0.38])
    expected_second = expected_first - 0.1 * expected_momentum
    assert np.allclose(param2.data, expected_second, rtol=1e-5)

    # Test weight decay
    param3 = Tensor([1.0, 2.0], requires_grad=True)
    param3.grad = Tensor([0.1, 0.2])

    optimizer_wd = SGD([param3], lr=0.1, weight_decay=0.01)
    optimizer_wd.step()

    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    expected_wd = np.array([1.0, 2.0]) - 0.1 * np.array([0.11, 0.22])
    assert np.allclose(param3.data, expected_wd)

    print("✅ SGD optimizer works correctly!")

if __name__ == "__main__":
    test_unit_sgd_optimizer()

# %% [markdown]
"""
## 🏗️ Adam - Adaptive Moment Estimation

Adam solves a fundamental problem with SGD: different parameters often need different learning rates. Think of tuning a complex system where some knobs need gentle adjustments and others need bold changes.

### The Parameter Scaling Problem

Consider a neural network with both embedding weights and output weights:

```
Parameter Sensitivity Landscape:

  output_weight                 embedding_weight
       ↑                              ↑
       |                              |
    😱 |  steep cliff                 |  🐌 gentle slope
       |  (needs tiny steps)          |  (needs big steps)
       |                              |
    ━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━→

Same learning rate = disaster!
• Small LR: output weights learn fast, embeddings crawl
• Large LR: embeddings learn well, output weights explode
```

### Adam's Adaptive Solution

Adam automatically adjusts learning rates by tracking two statistics:

```
1. MOMENTUM (first moment): "Which way am I usually going?"
   m = 0.9 * old_direction + 0.1 * current_gradient

   Visualization:
   old: →→→→
   new:     ↗️
   m:   →→→↗️  (weighted average)

2. SCALE (second moment): "How big are my steps usually?"
   v = 0.999 * old_scale + 0.001 * (current_gradient)²

   Big gradients → bigger v → smaller effective steps
   Small gradients → smaller v → bigger effective steps

3. ADAPTIVE UPDATE:
   step = momentum / √scale
   param = param - learning_rate * step
```

### Bias Correction: The Cold Start Problem

Adam starts with m=0 and v=0, which creates a bias toward zero initially:

```
Without bias correction:    With bias correction:

Step 1: m = 0.9*0 + 0.1*g    Step 1: m̂ = m / (1-0.9¹) = m / 0.1
       = 0.1*g (too small!)           = g (correct!)

Step 2: m = 0.9*0.1*g + 0.1*g Step 2: m̂ = m / (1-0.9²) = m / 0.19
       = 0.19*g (still small)         ≈ g (better!)
```

**Key Insight:** Adam is like having an automatic transmission that adjusts gear ratios for each parameter individually.
"""

# %% nbgrader={"grade": false, "grade_id": "adam-optimizer", "solution": true}
#| export
class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates.

    Adam computes individual adaptive learning rates for different parameters
    from estimates of first and second moments of the gradients.
    This makes it effective for problems with sparse gradients or noisy data.
    """

    def __init__(self, params: List[Tensor], lr: float = DEFAULT_LEARNING_RATE_ADAM, betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2), eps: float = DEFAULT_EPS, weight_decay: float = 0.0):
        """
        Initialize Adam optimizer.

        TODO: Set up Adam with adaptive learning rates

        APPROACH:
        1. Call parent constructor
        2. Store hyperparameters (lr, betas, eps, weight_decay)
        3. Initialize first and second moment buffers

        PARAMETERS:
        - lr: Learning rate (default: 0.001)
        - betas: Coefficients for computing running averages (default: (0.9, 0.999))
        - eps: Small constant for numerical stability (default: 1e-8)
        - weight_decay: L2 penalty coefficient (default: 0.0)

        EXAMPLE:
        >>> optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        """
        ### BEGIN SOLUTION
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (created lazily)
        self.m_buffers = [None for _ in self.params]  # First moment (mean)
        self.v_buffers = [None for _ in self.params]  # Second moment (variance)
        ### END SOLUTION

    def step(self):
        """
        Perform Adam update step.

        TODO: Implement Adam parameter update with adaptive learning rates

        APPROACH:
        1. For each parameter with gradients:
           a. Apply weight decay if specified
           b. Update first moment estimate (momentum of gradient)
           c. Update second moment estimate (momentum of squared gradient)
           d. Compute bias-corrected moments
           e. Update parameter using adaptive learning rate

        FORMULAS:
        - m_t = β₁ * m_{t-1} + (1-β₁) * g_t
        - v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
        - m̂_t = m_t / (1-β₁^t)
        - v̂_t = v_t / (1-β₂^t)
        - θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

        HINTS:
        - Initialize buffers as zeros on first use
        - Use step_count for bias correction
        - Square gradients element-wise for second moment
        """
        ### BEGIN SOLUTION
        # Increment step counter first (needed for bias correction)
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient data - grad can be Tensor or numpy array
            grad = param.grad
            # Handle both Tensor (with .data) and numpy array (from autograd) cases
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                # grad is already a numpy array from autograd
                grad_data = grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            # Initialize buffers if needed
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)

            # Update biased first moment estimate
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data

            # Update biased second moment estimate
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

            # Compute bias correction
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            # Compute bias-corrected moments
            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            # Update parameter
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        ### END SOLUTION

# %% [markdown]
"""
### 🔬 Unit Test: Adam Optimizer
This test validates our Adam implementation works correctly.
**What we're testing**: Adam updates with adaptive learning rates and bias correction
**Why it matters**: Most popular optimizer for modern neural networks
**Expected**: Correct parameter updates following Adam formulas
"""

# %% nbgrader={"grade": true, "grade_id": "test-adam", "locked": true, "points": 20}
def test_unit_adam_optimizer():
    """🔬 Test Adam optimizer implementation."""
    print("🔬 Unit Test: Adam Optimizer...")

    # Test basic Adam functionality
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    original_data = param.data.copy()

    # First step
    optimizer.step()

    # Manually compute expected values
    grad = np.array([0.1, 0.2])

    # First moment: m = 0.9 * 0 + 0.1 * grad = 0.1 * grad
    m = 0.1 * grad

    # Second moment: v = 0.999 * 0 + 0.001 * grad^2 = 0.001 * grad^2
    v = 0.001 * (grad ** 2)

    # Bias correction
    bias_correction1 = 1 - 0.9 ** 1  # = 0.1
    bias_correction2 = 1 - 0.999 ** 1  # = 0.001

    m_hat = m / bias_correction1  # = grad
    v_hat = v / bias_correction2  # = grad^2

    # Update
    expected = original_data - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)

    assert np.allclose(param.data, expected, rtol=1e-6)
    assert optimizer.step_count == 1

    # Test second step to verify moment accumulation
    param.grad = Tensor([0.1, 0.2])
    optimizer.step()

    # Should have updated moments
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None
    assert optimizer.step_count == 2

    # Test with weight decay
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    param2.grad = Tensor([0.1, 0.2])

    optimizer_wd = Adam([param2], lr=0.01, weight_decay=0.01)
    optimizer_wd.step()

    # Weight decay should modify the effective gradient
    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    # The exact computation is complex, but we can verify parameter changed
    assert not np.array_equal(param2.data, np.array([1.0, 2.0]))

    print("✅ Adam optimizer works correctly!")

if __name__ == "__main__":
    test_unit_adam_optimizer()

# %% [markdown]
"""
## 🏗️ AdamW - Adam with Decoupled Weight Decay

AdamW fixes a subtle but important bug in Adam's weight decay implementation. The bug affects how regularization interacts with adaptive learning rates.

### The Adam Weight Decay Bug

In standard Adam, weight decay is added to gradients before the adaptive scaling:

```
Adam's approach (problematic):
1. gradient = computed_gradient + weight_decay * parameter
2. m = β₁ * m + (1-β₁) * gradient
3. v = β₂ * v + (1-β₂) * gradient²
4. step = m / √v
5. parameter = parameter - learning_rate * step

Problem: Weight decay gets "adapted" by the learning rate scaling!
```

### Why This Matters

Weight decay should be a consistent regularization force, but Adam makes it inconsistent:

```
Parameter Update Comparison:

Large gradients → small adaptive LR → weak weight decay effect
Small gradients → large adaptive LR → strong weight decay effect

This is backwards! We want consistent regularization.
```

### AdamW's Fix: Decoupled Weight Decay

AdamW separates gradient-based updates from weight decay:

```
AdamW's approach (correct):
1. m = β₁ * m + (1-β₁) * pure_gradient  ← NO weight decay here
2. v = β₂ * v + (1-β₂) * pure_gradient²
3. step = m / √v
4. parameter = parameter - learning_rate * step        ← gradient update
5. parameter = parameter * (1 - weight_decay_rate)    ← separate decay

Result: Consistent regularization independent of gradient magnitudes!
```

### Visual Comparison

```
Adam weight decay:               AdamW weight decay:

gradient ──┐                    gradient ──→ adaptive ──→ param
           ├─→ adaptive ──→ param                  update
weight ────┘   scaling
decay
                                weight ─────────→ param
                                decay           shrinkage

Coupled (inconsistent)          Decoupled (consistent)
```

**Key Insight:** AdamW treats optimization and regularization as separate, independent processes, leading to better training dynamics and generalization.
"""

# %% nbgrader={"grade": false, "grade_id": "adamw-optimizer", "solution": true}
#| export
class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    AdamW fixes a bug in Adam's weight decay implementation by decoupling
    weight decay from the gradient-based update. This leads to better
    regularization and is the preferred version for most applications.
    """

    def __init__(self, params: List[Tensor], lr: float = DEFAULT_LEARNING_RATE_ADAM, betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2), eps: float = DEFAULT_EPS, weight_decay: float = DEFAULT_WEIGHT_DECAY_ADAMW):
        """
        Initialize AdamW optimizer.

        TODO: Set up AdamW with decoupled weight decay

        APPROACH:
        1. Call parent constructor
        2. Store hyperparameters (note higher default weight_decay)
        3. Initialize moment buffers like Adam

        KEY DIFFERENCE from Adam:
        - Weight decay is applied directly to parameters, not added to gradients
        - This provides better regularization behavior

        EXAMPLE:
        >>> optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        """
        ### BEGIN SOLUTION
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (same as Adam)
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]
        ### END SOLUTION

    def step(self):
        """
        Perform AdamW update step with decoupled weight decay.

        TODO: Implement AdamW parameter update

        APPROACH:
        1. For each parameter with gradients:
           a. Update moments using gradients (NOT modified by weight decay)
           b. Compute bias-corrected moments
           c. Apply gradient-based update
           d. Apply weight decay directly to parameters

        KEY DIFFERENCE from Adam:
        - Weight decay: θ_t = θ_t - lr * weight_decay * θ_t (applied after gradient update)
        - NOT: grad = grad + weight_decay * param (Adam's incorrect approach)

        FORMULAS:
        - Same moment updates as Adam (using unmodified gradients)
        - Gradient update: θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
        - Weight decay: θ_t = θ_t * (1 - lr * weight_decay)

        HINT: Apply weight decay after gradient update for proper decoupling
        """
        ### BEGIN SOLUTION
        # Increment step counter first
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient data - grad can be Tensor or numpy array
            grad = param.grad
            # Handle both Tensor (with .data) and numpy array (from autograd) cases
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                # grad is already a numpy array from autograd
                grad_data = grad

            # Initialize buffers if needed
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)

            # Update moments using pure gradients
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

            # Compute bias correction
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            # Compute bias-corrected moments
            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            # Apply gradient-based update
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Apply decoupled weight decay
            if self.weight_decay != 0:
                param.data = param.data * (1 - self.lr * self.weight_decay)
        ### END SOLUTION

# %% [markdown]
"""
### 🔬 Unit Test: AdamW Optimizer
This test validates our AdamW implementation with decoupled weight decay.
**What we're testing**: AdamW updates with proper weight decay decoupling
**Why it matters**: State-of-the-art optimizer for transformer models
**Expected**: Correct separation of gradient updates and weight decay
"""

# %% nbgrader={"grade": true, "grade_id": "test-adamw", "locked": true, "points": 20}
def test_unit_adamw_optimizer():
    """🔬 Test AdamW optimizer implementation."""
    print("🔬 Unit Test: AdamW Optimizer...")

    # Test AdamW vs Adam difference in weight decay
    # Create identical parameters for comparison
    param_adam = Tensor([1.0, 2.0], requires_grad=True)
    param_adamw = Tensor([1.0, 2.0], requires_grad=True)

    param_adam.grad = Tensor([0.1, 0.2])
    param_adamw.grad = Tensor([0.1, 0.2])

    # Create optimizers with same settings
    adam = Adam([param_adam], lr=0.01, weight_decay=0.01)
    adamw = AdamW([param_adamw], lr=0.01, weight_decay=0.01)

    # Take one step
    adam.step()
    adamw.step()

    # Results should be different due to weight decay implementation
    assert not np.allclose(param_adam.data, param_adamw.data, rtol=1e-6)

    # Test AdamW basic functionality
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = AdamW([param], lr=0.01, weight_decay=0.01)
    original_data = param.data.copy()

    optimizer.step()

    # Parameter should have changed
    assert not np.array_equal(param.data, original_data)
    assert optimizer.step_count == 1

    # Test that moment buffers are created
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None

    # Test zero weight decay behaves like Adam
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([1.0, 2.0], requires_grad=True)

    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([0.1, 0.2])

    adam_no_wd = Adam([param1], lr=0.01, weight_decay=0.0)
    adamw_no_wd = AdamW([param2], lr=0.01, weight_decay=0.0)

    adam_no_wd.step()
    adamw_no_wd.step()

    # Should be very similar (within numerical precision)
    assert np.allclose(param1.data, param2.data, rtol=1e-10)

    print("✅ AdamW optimizer works correctly!")

if __name__ == "__main__":
    test_unit_adamw_optimizer()

# %% [markdown]
"""
## 🔧 Integration: Bringing It Together

Now let's see how our optimizers perform in realistic scenarios. We'll compare their behavior on the same optimization problem to understand their different characteristics.

### Optimizer Behavior Comparison

Each optimizer takes a different approach to the same problem:

```
Optimization Problem: Find minimum of f(x) = x²

SGD approach:        Adam approach:        AdamW approach:
  ↓                    ↓                     ↓
 x ──→ minimize       x ──→ minimize       x ──→ minimize
  ↑                    ↑                     ↑
fixed LR           adaptive LR          adaptive LR + decay
```
"""


# %% [markdown]
"""
## 📊 Systems Analysis: Optimizer Performance and Memory

Different optimizers have very different resource requirements. Understanding these trade-offs is crucial for production ML systems.

### Memory Usage Patterns

```
Optimizer Memory Requirements (per parameter):

SGD:           Adam/AdamW:
┌────────┐     ┌────────┐
│ param  │     │ param  │
├────────┤     ├────────┤
│momentum│     │   m    │ ← first moment
└────────┘     ├────────┤
               │   v    │ ← second moment
               └────────┘

2× memory       3× memory
```

### Computational Complexity

```
Per-step Operations:

SGD:                     Adam:
• 1 multiplication       • 3 multiplications
• 1 addition            • 4 additions
• 1 subtraction         • 1 subtraction
                        • 1 square root
                        • 1 division

O(n) simple ops         O(n) complex ops
```
"""

# %% nbgrader={"grade": false, "grade_id": "optimizer-analysis", "solution": true}
def analyze_optimizer_memory_usage():
    """📊 Analyze memory usage of different optimizers."""
    print("📊 Analyzing Optimizer Memory Usage...")

    # Create test parameters of different sizes
    param_sizes = [1000, 10000, 100000]  # 1K, 10K, 100K parameters

    print("Optimizer Memory Analysis (per parameter tensor):")
    print("=" * 60)
    print(f"{'Size':<10} {'SGD':<10} {'Adam':<10} {'AdamW':<10} {'Ratio':<10}")
    print("-" * 60)

    for size in param_sizes:
        # Create parameter
        param = Tensor(np.random.randn(size), requires_grad=True)
        param.grad = Tensor(np.random.randn(size))

        # SGD memory (parameter + momentum buffer)
        sgd = SGD([param], momentum=0.9)
        sgd.step()  # Initialize buffers
        sgd_memory = size * 2  # param + momentum buffer

        # Adam memory (parameter + 2 moment buffers)
        param_adam = Tensor(np.random.randn(size), requires_grad=True)
        param_adam.grad = Tensor(np.random.randn(size))
        adam = Adam([param_adam])
        adam.step()  # Initialize buffers
        adam_memory = size * 3  # param + m_buffer + v_buffer

        # AdamW memory (same as Adam)
        adamw_memory = adam_memory

        # Memory ratio (Adam/SGD)
        ratio = adam_memory / sgd_memory

        print(f"{size:<10} {sgd_memory:<10} {adam_memory:<10} {adamw_memory:<10} {ratio:.1f}x")

    print("\n💡 Key Insights:")
    print("- SGD: 2× parameter memory (momentum buffer)")
    print("- Adam/AdamW: 3× parameter memory (two moment buffers)")
    print("- Memory scales linearly with model size")
    print("- Trade-off: More memory for better convergence")

# %% nbgrader={"grade": false, "grade_id": "optimizer-convergence", "solution": true}
def analyze_optimizer_convergence_behavior():
    """📊 Analyze convergence behavior of different optimizers."""
    print("📊 Analyzing Optimizer Convergence Behavior...")

    # Simulate optimization of a quadratic function: f(x) = 0.5 * x^2
    # Optimal solution: x* = 0, gradient = x

    def quadratic_loss(x):
        """Simple quadratic function for optimization testing."""
        return 0.5 * (x ** 2).sum()

    def compute_gradient(x):
        """Gradient of quadratic function: df/dx = x."""
        return x.copy()

    # Starting point
    x_start = np.array([5.0, -3.0, 2.0])  # Far from optimum [0, 0, 0]

    # Test different optimizers
    optimizers_to_test = [
        ("SGD", SGD, {"lr": 0.1}),
        ("SGD+Momentum", SGD, {"lr": 0.1, "momentum": 0.9}),
        ("Adam", Adam, {"lr": 0.1}),
        ("AdamW", AdamW, {"lr": 0.1, "weight_decay": 0.01})
    ]

    print("Convergence Analysis (quadratic function f(x) = 0.5 * x²):")
    print("=" * 70)
    print(f"{'Optimizer':<15} {'Step 0':<12} {'Step 5':<12} {'Step 10':<12} {'Final Loss':<12}")
    print("-" * 70)

    for name, optimizer_class, kwargs in optimizers_to_test:
        # Reset parameter
        param = Tensor(x_start.copy(), requires_grad=True)
        optimizer = optimizer_class([param], **kwargs)

        losses = []

        # Run optimization for 10 steps
        for step in range(11):
            # Compute loss and gradient
            loss = quadratic_loss(param.data)
            param.grad = Tensor(compute_gradient(param.data))

            losses.append(loss)

            # Update parameters
            if step < 10:  # Don't update after last evaluation
                optimizer.step()
                optimizer.zero_grad()

        # Format results
        step0 = f"{losses[0]:.6f}"
        step5 = f"{losses[5]:.6f}"
        step10 = f"{losses[10]:.6f}"
        final = f"{losses[10]:.6f}"

        print(f"{name:<15} {step0:<12} {step5:<12} {step10:<12} {final:<12}")

    print("\n💡 Key Insights:")
    print("- SGD: Steady progress but can be slow")
    print("- SGD+Momentum: Faster convergence, less oscillation")
    print("- Adam: Adaptive rates help with different parameter scales")
    print("- AdamW: Similar to Adam with regularization effects")

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
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_optimizer_base()
    test_unit_sgd_optimizer()
    test_unit_adam_optimizer()
    test_unit_adamw_optimizer()

    print("\nRunning integration scenarios...")

    # Test realistic neural network optimization scenario
    print("🔬 Integration Test: Multi-layer Network Optimization...")

    # Import components from TinyTorch package (previous modules must be completed and exported)
    from tinytorch.core.layers import Linear
    from tinytorch.core.activations import ReLU
    from tinytorch.core.losses import MSELoss

    # Create parameters for a 2-layer network
    # Layer 1: 3 inputs -> 4 hidden
    W1 = Tensor(np.random.randn(3, 4) * 0.1, requires_grad=True)
    b1 = Tensor(np.zeros(4), requires_grad=True)

    # Layer 2: 4 hidden -> 2 outputs
    W2 = Tensor(np.random.randn(4, 2) * 0.1, requires_grad=True)
    b2 = Tensor(np.zeros(2), requires_grad=True)

    params = [W1, b1, W2, b2]

    # Add realistic gradients
    W1.grad = Tensor(np.random.randn(3, 4) * 0.01)
    b1.grad = Tensor(np.random.randn(4) * 0.01)
    W2.grad = Tensor(np.random.randn(4, 2) * 0.01)
    b2.grad = Tensor(np.random.randn(2) * 0.01)

    # Test all optimizers on same network
    optimizers = [
        SGD(params, lr=0.01, momentum=0.9),
        Adam([p for p in params], lr=0.001),  # Fresh param list for Adam
        AdamW([p for p in params], lr=0.001, weight_decay=0.01)  # Fresh param list for AdamW
    ]

    # Save original parameter values
    original_params = [p.data.copy() for p in params]

    # Test SGD
    optimizers[0].step()
    sgd_params = [p.data.copy() for p in params]

    # Restore parameters and test Adam
    for i, p in enumerate(params):
        p.data = original_params[i].copy()
        # Re-add gradients since they may have been modified
        if i == 0:
            p.grad = Tensor(np.random.randn(3, 4) * 0.01)
        elif i == 1:
            p.grad = Tensor(np.random.randn(4) * 0.01)
        elif i == 2:
            p.grad = Tensor(np.random.randn(4, 2) * 0.01)
        else:
            p.grad = Tensor(np.random.randn(2) * 0.01)

    # Update parameter references for Adam
    optimizers[1].params = params
    optimizers[1].step()
    adam_params = [p.data.copy() for p in params]

    # Restore parameters and test AdamW
    for i, p in enumerate(params):
        p.data = original_params[i].copy()
        # Re-add gradients
        if i == 0:
            p.grad = Tensor(np.random.randn(3, 4) * 0.01)
        elif i == 1:
            p.grad = Tensor(np.random.randn(4) * 0.01)
        elif i == 2:
            p.grad = Tensor(np.random.randn(4, 2) * 0.01)
        else:
            p.grad = Tensor(np.random.randn(2) * 0.01)

    # Update parameter references for AdamW
    optimizers[2].params = params
    optimizers[2].step()
    adamw_params = [p.data.copy() for p in params]

    # Verify parameters changed differently for each optimizer
    for i in range(len(params)):
        # Parameters should be different from original
        assert not np.array_equal(sgd_params[i], original_params[i])
        assert not np.array_equal(adam_params[i], original_params[i])
        assert not np.array_equal(adamw_params[i], original_params[i])

        # Different optimizers should produce different results
        assert not np.allclose(sgd_params[i], adam_params[i], rtol=1e-6)

    print("✅ Multi-layer network optimization works!")

    # Test optimizer state management
    print("🔬 Integration Test: Optimizer State Management...")

    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = Adam([param], lr=0.001)

    # First step should initialize buffers
    optimizer.step()
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None
    assert optimizer.step_count == 1

    # Zero grad should clear gradients but preserve optimizer state
    optimizer.zero_grad()
    assert param.grad is None
    assert optimizer.m_buffers[0] is not None  # State preserved
    assert optimizer.step_count == 1  # Step count preserved

    print("✅ Optimizer state management works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 07_optimizers")

# %% [markdown]
"""
## 🤔 ML Systems Thinking

Now that your optimizers work, let's explore the systems trade-offs between them. Every optimizer choice affects memory usage, convergence speed, and training stability.

### Questions to Consider

**Q1: Memory vs Performance**
You've implemented SGD (2× memory) and Adam (3× memory). For a model with 10 billion parameters at float32 (4 bytes each):
- How much total memory does each optimizer require?
- At what model size does Adam's extra 50% memory overhead become prohibitive?
- What real-world constraints might force you to choose SGD over Adam?

**Q2: Learning Rate Sensitivity**
SGD uses a fixed learning rate for all parameters, while Adam adapts per-parameter:
- Why might Adam converge faster on problems with parameters at different scales?
- When might SGD's uniform learning rate actually be an advantage?
- How does momentum in SGD relate to Adam's first moment estimation?

**Q3: Optimizer State Management**
Adam and AdamW maintain momentum buffers (m, v) that persist across training steps:
- What happens to these buffers when you checkpoint during training?
- If you resume training with different hyperparameters, should you restore the old buffers?
- How does optimizer state affect distributed training across multiple GPUs?

**Q4: Weight Decay Trade-offs**
AdamW decouples weight decay from gradient updates:
- Why does Adam's coupled weight decay behave inconsistently?
- In what scenarios would AdamW's consistent regularization matter most?
- How does weight decay interact with learning rate schedules?

### Systems Implications

**Memory Hierarchy:**
```
Model Size: 1B parameters (4GB)
├─ SGD:     8GB total (4GB params + 4GB momentum)
├─ Adam:    12GB total (4GB params + 4GB m + 4GB v)
└─ Impact:  May not fit in GPU memory, forcing:
            • Smaller batch sizes
            • Model parallelism
            • Optimizer state sharding (ZeRO optimization)
```

**Convergence Patterns:**
- **SGD + Momentum:** Steady progress, may need learning rate tuning
- **Adam:** Fast initial convergence, may overfit without proper regularization
- **AdamW:** Adam's speed + better generalization, standard for transformers

**Production Considerations:**
- **Training cost:** Adam's extra memory means fewer models per GPU
- **Hyperparameter tuning:** SGD more sensitive to learning rate choice
- **Model generalization:** AdamW often generalizes better than Adam
- **Checkpoint size:** Adam checkpoints are 1.5× larger than SGD

### Performance Analysis

Our earlier analysis functions revealed:
- `analyze_optimizer_memory_usage()`: Adam requires exactly 1.5× SGD's memory
- `analyze_optimizer_convergence_behavior()`: Adam often converges in fewer steps

**The Key Insight:**
Optimizer choice is a systems trade-off between:
- **Memory budget** (can you afford 3× parameter memory?)
- **Convergence speed** (how many training steps can you afford?)
- **Generalization quality** (does your model perform well on unseen data?)

There's no universally best optimizer—only the right choice for your constraints!
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Optimizers Update Weights

**What you built:** Optimization algorithms (SGD, Adam) that update neural network weights.

**Why it matters:** Gradients tell us which direction reduces the loss, but someone has to
actually move the weights. That's what optimizers do! SGD takes simple steps, while Adam
adapts the learning rate for each parameter—like having a personal trainer for each weight.

In the next module, you'll combine optimizers with a training loop to actually train networks!
"""

# %%
def demo_optimizers():
    """🎯 See optimizers update weights."""
    print("🎯 AHA MOMENT: Optimizers Update Weights")
    print("=" * 45)

    # Create a parameter with a gradient
    weight = Tensor(np.array([5.0]), requires_grad=True)
    weight.grad = np.array([1.0])  # Gradient pointing "uphill"

    print(f"Initial weight: {weight.data[0]:.2f}")
    print(f"Gradient:       {weight.grad[0]:.2f} (pointing uphill)")

    # SGD takes a step in the opposite direction
    optimizer = SGD([weight], lr=0.5)
    optimizer.step()

    print(f"\nAfter SGD step: {weight.data[0]:.2f}")
    print(f"Moved: {5.0 - weight.data[0]:.2f} (opposite to gradient)")

    print("\n✨ Optimizer moves weights to reduce loss!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_optimizers()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Optimizers

Congratulations! You've built sophisticated optimization algorithms that power modern neural network training!

### Key Accomplishments
- Built SGD optimizer with momentum for stable gradient descent and oscillation reduction
- Implemented Adam optimizer with adaptive learning rates and bias correction for different parameter scales
- Created AdamW optimizer with decoupled weight decay for proper regularization
- Analyzed memory trade-offs: SGD (2×), Adam/AdamW (3× parameter memory)
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your optimizer implementations enable sophisticated neural network training! With gradients from Module 06 and optimizers from Module 07, you're ready to build complete training loops.

Export with: `tito module complete 07_optimizers`

**Next**: Module 08 will add training loops, learning rate scheduling, and checkpointing for complete end-to-end neural network training!
"""
