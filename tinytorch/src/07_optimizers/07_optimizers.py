# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
r"""
# Module 07: Optimizers - Turning Gradients into Updates

Welcome to Module 07! You'll build the optimizers that turn the gradients Module 06 computes into parameter updates.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor with autograd tape tracking (Modules 01–06)  
**You'll Build**: SGD, Adam, and AdamW optimizers with velocity momentum, adaptive per-parameter scaling, and decoupled weight decay  
**You'll Enable**: The parameter update engines driving Module 08's training loop and every downstream architecture

<div align="center">
  <img src="optimizer_blueprint.svg" alt="Optimizers Blueprint: You Are Here" width="380px">
</div>

$$\underbrace{\text{Modules 01–06}}_{\text{Tensor and Autograd Tape}} \longrightarrow \underbrace{\mathbf{\text{Optimizers}}}_{\mathbf{\text{Mod 07 (Active)}}} \longrightarrow \underbrace{\text{Training Loop}}_{\text{Mod 08}} \longrightarrow \underbrace{\text{Scale and Architecture}}_{\text{Mods 09–20}}$$

## 🎯 Learning Objectives
By the end of this module, you will:
1. **SGD with Velocity Momentum**: Implement physical momentum accumulation to dampen high-curvature ravine oscillations.
2. **Adaptive Moments (Adam)**: Construct running first ($\mathbf{m}$) and second ($\mathbf{v}$) moment estimators with early-step bias correction.
3. **Decoupled Weight Decay (AdamW)**: Separate analytical $L_2$ gradient penalty from adaptive step scaling to restore scale-invariant regularization.
4. **Systems & Memory Economics**: Profile the 16-byte-per-parameter optimizer memory footprint that dictates hardware training limits.

## 📦 Where This Code Lives in the Final Package

<div align="center">
  <img src="optimizer_margin_source.svg" alt="Source Code Mapping" width="220px">
</div>

**Learning Side:** You work in `modules/07_optimizers/optimizers.ipynb`  
**Building Side:** Code exports to `tinytorch.core.optimizers`

```python
# How to use this module:
from tinytorch.core.optimizers import SGD, Adam, AdamW
```

**Why this matters:**
- **Framework Parity**: Matches PyTorch's `torch.optim` class layout, so `SGD`, `Adam`, and `AdamW` are constructed and stepped the same way. TinyTorch simplifies in two places worth knowing about: parameters live in one flat list rather than `torch.optim`'s per-group `param_groups`, and there is no `state_dict()` / `load_state_dict()`, so Module 08 will checkpoint the buffers directly through `get_momentum_state()`.
- **Production Decoupling**: Isolate update mathematical mechanics from model execution graphs and training state machines.
- **Loss Navigation**: Transform raw instantaneous gradients into robust descent trajectories across non-convex loss surfaces.
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01 and 06 must be complete
- **Module 01 (`Tensor`)**: Provides model parameter tensors with contiguous data buffers.
- **Module 06 (`Autograd`)**: Writes analytical gradients into `param.grad` via reverse topological traversal.

| Component | Upstream Origin | Role in Module 07 | Downstream Target |
| :--- | :--- | :--- | :--- |
| **`Tensor`** | Module 01 (`core.tensor`) | Model parameter instances carrying weight buffers | Consumed by optimizers |
| **`param.grad`** | Module 06 (`core.autograd`) | Instantaneous gradient vectors $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ | Read by optimizer `step()` |
| **`method_of`** | Module 06 (`core.autograd`) | Method decorator attaching step logic to optimizer classes | Implementation cleanly modularized |
| **Optimizers** | Module 07 (`core.optimizers`) | State buffers ($\mathbf{v}, \mathbf{m}$) and the rebinding of `param.data` to the updated array | Wired into Module 08 Training Loop |

$$\mathbf{w} \in \mathbb{R}^D \xrightarrow{\text{Forward (Mod 01)}} \mathcal{L} \xrightarrow{\text{Backward (Mod 06)}} \mathbf{g} = \nabla_{\mathbf{w}} \mathcal{L} \xrightarrow{\text{Step (Mod 07)}} \mathbf{w}' = \mathbf{w} - \eta \cdot \mathbf{u}(\mathbf{g}) \xrightarrow{\text{Epoch (Mod 08)}} \text{Trained Model}$$

Optimizers are the operational step that transforms gradients into learning. Module 08 will integrate them into an end-to-end training loop.
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.optimizers
#| export

import numpy as np
rng = np.random.default_rng(7)
from typing import List, Optional

from tinytorch.core.tensor import Tensor
# Importing Module 06 completes every Tensor operation with its backward half,
# which is what puts a gradient into param.grad for an optimizer to read.
import tinytorch.core.autograd
from tinytorch.core.autograd import method_of  # attach a method to a class, as in Module 06

# Constants for optimizer defaults
DEFAULT_LEARNING_RATE_SGD = 0.01  # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001  # Default learning rate for Adam/AdamW
DEFAULT_BETA1 = 0.9  # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999  # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8  # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01  # Default weight decay for AdamW

# %% [markdown]
r"""
## 💡 Introduction: What are Optimizers?

Optimizers are the numerical engines that drive neural network learning. They take analytical gradients computed by Module 06's autograd engine and update model parameters toward loss minima. In high-dimensional deep learning, loss landscapes are rarely isotropic bowls. They feature ill-conditioned ravines, saddle points, and sharp cliffs.

<div align="center">
  <img src="ravine_optimization.svg" alt="Ill-Conditioned Ravine and 16-Byte Optimizer Memory Rule" width="680px">
</div>

### The Ill-Conditioned Ravine Challenge

Consider an anisotropic quadratic bowl with condition number $\kappa = 100$:

$$\mathcal{L}(w_1, w_2) = 50 w_1^2 + 0.5 w_2^2 \implies \mathbf{H} = \begin{bmatrix} 100 & 0 \\ 0 & 1 \end{bmatrix}, \quad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = 100$$

- **Steep Axis ($w_1$)**: The gradient $\nabla_{w_1} \mathcal{L} = 100 w_1$ is massive, causing vanilla gradient descent to oscillate violently across the valley walls.
- **Gentle Axis ($w_2$)**: The gradient $\nabla_{w_2} \mathcal{L} = w_2$ is tiny, causing progress along the valley floor to stall.

Every optimization algorithm in this module implements a specialized strategy to resolve this curvature imbalance under the universal update template:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \cdot \mathbf{u}(\mathbf{g}_t, \mathbf{s}_t)$$

where $\boldsymbol{\theta}$ represents model parameters, $\alpha$ is the learning rate, $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}$ is the current gradient, and $\mathbf{s}_t$ denotes internal optimizer state buffers.
"""

# %% [markdown]
r"""
## 📐 Foundations: Mathematical Background

<div align="center">
  <img src="optimizer_update_pipeline.svg" alt="Optimizer Update Pipeline: SGD vs Adam vs AdamW" width="680px">
</div>

### 1. Understanding Momentum: The Physics of Optimization

In physical systems, a rolling marble possesses inertia: when traveling through a narrow ravine, alternating sideways forces cancel out while forward momentum accumulates along the valley floor.

$$\begin{aligned}
\mathbf{v}_t &= \beta \mathbf{v}_{t-1} + \mathbf{g}_t \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \alpha \mathbf{v}_t
\end{aligned}$$

With momentum coefficient $\beta \approx 0.9$, the effective step size along consistent gradient directions scales by $\frac{1}{1 - \beta} \approx 10\times$, while high-frequency transverse oscillations cancel.

### 2. Adam: Adaptive Moment Estimation

Adam maintains running exponential moving averages (EMA) of both the gradient direction (first moment $\mathbf{m}$) and uncentered variance (second moment $\mathbf{v}$):

$$\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad & \hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2, \quad & \hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t}
\end{aligned}$$

The parameter update scales coordinates inversely by their empirical standard deviation:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

- **High-Curvature Coordinates**: Large $v_i \implies \sqrt{v_i}$ dampens step size, preventing explosive divergence.
- **Low-Curvature Coordinates**: Small $v_i \implies \sqrt{v_i}$ amplifies step size, accelerating escape from flat plateaus.

### 3. AdamW: Decoupled Weight Decay

When standard $L_2$ regularization $\frac{1}{2} \lambda \|\boldsymbol{\theta}\|^2$ is folded into the loss, its gradient $\lambda \boldsymbol{\theta}$ is divided by $\sqrt{\mathbf{v}}$. Parameters with large historical gradients receive suppressed regularization, while parameters with small gradients are over-penalized.

AdamW restores scale-invariant regularization by separating parameter decay from gradient adaptation:

$$\boldsymbol{\theta}_{t+1} = \underbrace{(1 - \alpha \lambda) \boldsymbol{\theta}_t}_{\text{decoupled weight shrinkage}} - \underbrace{\frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t}_{\text{pure gradient adaptive step}}$$

| Algorithm | State Buffers | Update Direction $\mathbf{u}_t$ | Regularization Mechanism |
| :--- | :--- | :--- | :--- |
| **SGD** | None | $\mathbf{g}_t$ | Coupled gradient penalty $\mathbf{g} + \lambda \boldsymbol{\theta}$ |
| **SGD + Momentum** | Velocity $\mathbf{v} \in \mathbb{R}^D$ | $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$ | Coupled gradient penalty $\mathbf{g} + \lambda \boldsymbol{\theta}$ |
| **Adam** | Moments $\mathbf{m}, \mathbf{v} \in \mathbb{R}^D$ | $\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$ | Coupled gradient penalty $\mathbf{g} + \lambda \boldsymbol{\theta}$ |
| **AdamW** | Moments $\mathbf{m}, \mathbf{v} \in \mathbb{R}^D$ | $\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$ | **Decoupled**: $\boldsymbol{\theta} \leftarrow (1 - \alpha \lambda) \boldsymbol{\theta}$ |
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Building Optimizers

We construct each optimizer incrementally, establishing the base class contract before specializing into first-order and adaptive update rules:

| Component | Scope | Core Responsibility |
| :--- | :--- | :--- |
| **`Optimizer` (Base)** | Interface & Plumbing | Holds parameter references, implements `zero_grad()`, unpacks gradient buffers |
| **`SGD`** | Classical Descent | First-order gradient updates with velocity accumulation buffer |
| **`Adam`** | Adaptive Moments | First-moment direction EMA, second-moment magnitude EMA, bias corrections |
| **`AdamW`** | Modern Standard | Decoupled parameter decay applied directly to weights before adaptive step |
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
        2. Reject a duplicated parameter: the same tensor listed twice would get
           two state slots and be stepped twice per update
        3. Give each parameter a grad attribute (set to None) only if it has
           none yet, preserving existing requires_grad configuration
        4. Initialize step counter for algorithms that need it

        EXAMPLE:
        >>> linear = Linear(784, 128)
        >>> optimizer = SGD(linear.parameters(), lr=0.01)

        HINTS:
        - Store parameters for iteration during optimization steps
        - Constructing an optimizer must not clear a gradient a parameter is
          already carrying. Someone may have run backward() before building the
          optimizer, and PyTorch does not zero gradients at construction either.
          Only set grad when the attribute is missing.
        """
        ### BEGIN SOLUTION role="scaffold"
        # Own the list so later caller edits cannot misalign parameters and state.
        self.params = list(params)
        if len({id(param) for param in self.params}) != len(self.params):
            raise ValueError("Optimizer parameters must not contain duplicates")

        # Parameters track gradients through layer definitions. Do NOT reset param.grad:
        # a caller may have run backward() before building the optimizer.
        for param in self.params:
            if not hasattr(param, 'grad'):
                param.grad = None
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
        >>> assert all(param.grad is None for param in optimizer.params)

        HINTS:
        - Gradients accumulate by default, so they must be cleared between batches
        """
        ### BEGIN SOLUTION role="scaffold"
        for param in self.params:
            param.grad = None
        ### END SOLUTION

    def step(self):
        """
        Update parameters based on gradients.

        This is abstract. Each optimizer implements its own update rule.
        """
        raise NotImplementedError(
            f"Abstract method step() not implemented\n"
            f"  ❌ {self.__class__.__name__} inherits from Optimizer but doesn't define step()\n"
            f"  💡 Each optimizer must implement its own update rule (SGD, Adam, etc.)\n"
            f"  🔧 Override step() in your optimizer subclass:\n"
            f"      def step(self):\n"
            f"          for param in self.params:\n"
            f"              if param.grad is not None:\n"
            f"                  param.data -= self.lr * self._extract_gradient(param)"
        )

# %% [markdown]
r"""
### Gradient Extraction: Handling Tensor vs NumPy Gradients

Module 06's `backward()` writes a bare NumPy ndarray into `param.grad`. However, during isolated unit testing or custom training loops, gradients may be initialized directly as `Tensor` instances (holding an internal `.data` buffer).

To ensure deterministic numeric updates across all algorithms, `_extract_gradient()` acts as an idempotent unwrapping contract:

| Gradient Representation | Type Signature | Extraction Logic | Normalized Return |
| :--- | :--- | :--- | :--- |
| **Autograd Engine Output** | `np.ndarray` | Direct pass-through | `np.ndarray` |
| **Explicit Tensor Wrapper** | `Tensor` | Extract buffer via `grad.data` | `np.ndarray` |

This helper is attached directly to the base `Optimizer` class via `method_of(Optimizer)` so that `SGD`, `Adam`, and `AdamW` share unified type handling.
"""

# %% nbgrader={"grade": false, "grade_id": "extract-gradient", "solution": true}
#| exporti
@method_of(Optimizer)
def _extract_gradient(self, param: Tensor) -> np.ndarray:
    """
    Extract gradient data as a NumPy array from a parameter.

    Module 06's backward() stores a raw NumPy array in param.grad; a gradient
    set by hand may be a Tensor (with a .data attribute). This helper
    normalizes both cases to a plain NumPy array for optimizer math.

    TODO: Return the gradient's underlying NumPy array

    APPROACH:
    1. Get param.grad
    2. If it's a Tensor, return its .data attribute
    3. If it's already a NumPy array, return it directly

    EXAMPLE:
    >>> param = Tensor([1.0, 2.0], requires_grad=True)
    >>> param.grad = Tensor([0.1, 0.2])
    >>> optimizer._extract_gradient(param)
    array([0.1, 0.2])

    HINT: Use isinstance(grad, Tensor) to check the type
    """
    ### BEGIN SOLUTION role="scaffold"
    grad = param.grad
    if isinstance(grad, Tensor):
        return grad.data
    else:
        return grad
    ### END SOLUTION


# %% [markdown]
"""
### 🧪 Unit Test: Gradient Extraction

This test validates that `_extract_gradient` correctly handles both
Tensor-wrapped gradients and raw NumPy array gradients.

**What we're testing**: Gradient normalization across storage formats
**Why it matters**: Every optimizer needs raw NumPy data for update math
**Expected**: NumPy array output regardless of input format
"""

# %% nbgrader={"grade": true, "grade_id": "test-extract-gradient", "locked": true, "points": 5}
def test_unit_extract_gradient():
    """🧪 Test _extract_gradient handles Tensor and ndarray gradients."""
    print("🧪 Unit Test: Gradient Extraction...")

    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Optimizer([param])

    # Case 1: Gradient is a Tensor
    param.grad = Tensor([0.1, 0.2])
    grad_data = optimizer._extract_gradient(param)
    assert isinstance(grad_data, np.ndarray), "Should return ndarray from Tensor grad"
    assert np.allclose(grad_data, [0.1, 0.2])

    # Case 2: Gradient is a raw NumPy array
    param.grad = np.array([0.3, 0.4])
    grad_data = optimizer._extract_gradient(param)
    assert isinstance(grad_data, np.ndarray), "Should return ndarray from ndarray grad"
    assert np.allclose(grad_data, [0.3, 0.4])

    # Case 3: Multi-dimensional gradient
    param_2d = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    opt_2d = Optimizer([param_2d])
    param_2d.grad = Tensor([[0.1, 0.2], [0.3, 0.4]])
    grad_data_2d = opt_2d._extract_gradient(param_2d)
    assert grad_data_2d.shape == (2, 2)
    assert np.allclose(grad_data_2d, [[0.1, 0.2], [0.3, 0.4]])

    print("✅ Gradient extraction works correctly!")

if __name__ == "__main__":
    test_unit_extract_gradient()

# %% [markdown]
"""
### 🧪 Unit Test: Base Optimizer

This test validates our base Optimizer class works correctly.

**What we're testing**: Parameter validation and zero_grad functionality
**Why it matters**: Foundation for all specific optimizer implementations
**Expected**: Proper parameter storage and gradient clearing
"""

# %% nbgrader={"grade": true, "grade_id": "test-optimizer-base", "locked": true, "points": 10}
def test_unit_optimizer_base():
    """🧪 Test base Optimizer functionality."""
    print("🧪 Unit Test: Base Optimizer...")

    # Create test parameters
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    # Create the optimizer
    optimizer = Optimizer([param1, param2])

    # Test parameter storage
    assert len(optimizer.params) == 2
    assert optimizer.params[0] is param1
    assert optimizer.params[1] is param2
    assert optimizer.step_count == 0

    # Add gradients AFTER creating optimizer to test zero_grad properly
    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([[0.3, 0.4], [0.5, 0.6]])

    # Test zero_grad
    optimizer.zero_grad()
    assert param1.grad is None
    assert param2.grad is None

    # An optimizer preserves parameter requires_grad settings (so frozen layers
    # remain frozen). What the constructor does reject is a duplicate parameter.
    regular_param = Tensor([1.0], requires_grad=True)
    opt = Optimizer([regular_param])
    assert len(opt.params) == 1
    assert regular_param.requires_grad

    frozen_param = Tensor([2.0], requires_grad=False)
    opt_frozen = Optimizer([frozen_param])
    assert len(opt_frozen.params) == 1
    assert not frozen_param.requires_grad

    try:
        Optimizer([regular_param, regular_param])
        assert False, "Optimizer should reject a duplicated parameter"
    except ValueError:
        pass

    print("✅ Base Optimizer works correctly!")

if __name__ == "__main__":
    test_unit_optimizer_base()

# %% [markdown]
r"""
## 🏗️ SGD: Stochastic Gradient Descent

SGD is the foundation of neural network optimization. It implements the principle of steepest descent: stepping in the direction opposite to the analytical gradient.

### Why SGD Works: The Geometry of Steepest Descent

The gradient vector $\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$ defines the direction of greatest local rate of increase (steepest ascent). To minimize the scalar loss objective $\mathcal{L}$, parameter updates move opposite to this vector:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) = \left[ \frac{\partial \mathcal{L}}{\partial \theta_1}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_D} \right]^T \implies \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

### The Oscillation Problem in Ill-Conditioned Ravines

In deep architectures, loss surfaces rarely form isotropic spherical bowls. Instead, they form elongated, ill-conditioned ravines where the Hessian eigenvalues diverge ($\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$):

$$\mathbf{g}_t = \mathbf{g}_t^{\text{transverse}} + \mathbf{g}_t^{\text{floor}}, \quad \text{where } \|\mathbf{g}_t^{\text{transverse}}\| \gg \|\mathbf{g}_t^{\text{floor}}\|$$

Because the transverse walls are steep, gradient descent steps overshoot and bounce back and forth across opposing valley walls ($\mathbf{g}_{t+1}^{\text{transverse}} \approx -\mathbf{g}_t^{\text{transverse}}$). This wastes kinetic energy in high-frequency oscillations while making negligible forward progress along the gentle valley floor.

### The Momentum Solution: Physical Inertia (Polyak Heavy-Ball)

Momentum introduces physical inertia into the update dynamics. Instead of taking steps proportional to instantaneous gradients, SGD maintains a running velocity accumulator $\mathbf{v}_t$:

$$\begin{aligned}
\mathbf{v}_t &= \beta \mathbf{v}_{t-1} + \mathbf{g}_t \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \alpha \mathbf{v}_t
\end{aligned}$$

Unrolling the velocity recurrence reveals how momentum acts as a directional low-pass filter:

$$\mathbf{v}_t = \sum_{\tau=0}^t \beta^{t-\tau} \mathbf{g}_\tau$$

- **Transverse Components**: Alternating gradient signs cancel out across successive steps ($\sum \beta^{t-\tau} \mathbf{g}_\tau^{\text{transverse}} \to \mathbf{0}$).
- **Floor Components**: Consistent gradients accumulate constructively, accelerating along the valley floor up to a terminal steady-state multiplier of $\frac{1}{1 - \beta} \approx 10\times$ (for $\beta = 0.9$).

| Metric / Property | Vanilla SGD ($\beta = 0$) | SGD with Momentum ($\beta = 0.9$) |
| :--- | :--- | :--- |
| **Effective Velocity** | $\mathbf{v}_t = \mathbf{g}_t$ | $\mathbf{v}_t = 0.9 \mathbf{v}_{t-1} + \mathbf{g}_t$ |
| **Steady-State Step Multiplier** | $1.0\times$ | $\frac{1}{1 - \beta} = 10.0\times$ |
| **Ravine Trajectory** | Transverse zig-zag oscillation | Filtered low-pass forward acceleration |
| **State Buffer Memory** | $0\text{ bytes}$ (stateless) | $4\text{ bytes/param}$ (one `momentum_buffers` entry) |
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
        ### BEGIN SOLUTION role="scaffold"
        for name, value in (("lr", lr), ("momentum", momentum), ("weight_decay", weight_decay)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
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

        Module 08's Trainer will call this before saving a checkpoint, to know
        whether there is optimizer state to store.

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

        Module 08's Trainer will store the returned list in the checkpoint file.
        The buffers are copied, so later steps do not alter the saved state.

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

        Module 08's Trainer will call this when it resumes from a checkpoint, so a
        resumed run continues with the velocity it had, not from zero.

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
                f"Momentum state length mismatch\n"
                f"  ❌ State has {len(state)} buffers, but optimizer has {len(self.momentum_buffers)} parameters\n"
                f"  💡 Checkpoint was saved with a different model architecture or parameter count\n"
                f"  🔧 Ensure you're loading state into an optimizer with the same number of parameters:\n"
                f"      # Check parameter counts match before restoring\n"
                f"      assert len(saved_state) == len(optimizer.params)"
            )

        # Validate every buffer before replacing any state. Broadcasting a wrong
        # shape during step() could otherwise change the parameter's data shape.
        for param, buf in zip(self.params, state):
            if buf is not None and (not isinstance(buf, np.ndarray) or buf.shape != param.data.shape):
                raise ValueError("Momentum buffer shape must match its parameter")
        self.momentum_buffers = [None if buf is None else buf.copy() for buf in state]

    def step(self):
        """
        Perform SGD update step with momentum.

        TODO: Implement SGD parameter update by composing helpers

        APPROACH:
        1. For each parameter with gradients:
           a. Extract gradient using self._extract_gradient(param)
           b. Apply weight decay if specified
           c. Update momentum buffer
           d. Update parameter using momentum

        FORMULA:
        - With weight decay: grad = grad + weight_decay * param
        - Momentum: v = momentum * v_prev + grad
        - Update: param = param - lr * v

        HINTS:
        - Skip parameters without gradients
        - Use self._extract_gradient() from the base class
        - Initialize momentum buffers on first use
        """
        ### BEGIN SOLUTION
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

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
### 🧪 Unit Test: SGD Optimizer

This test validates our SGD implementation works correctly.

**What we're testing**: SGD updates with and without momentum
**Why it matters**: Core optimization algorithm used in neural network training
**Expected**: Correct parameter updates following SGD formulas
"""

# %% nbgrader={"grade": true, "grade_id": "test-sgd", "locked": true, "points": 15}
def test_unit_sgd_optimizer():
    """🧪 Test SGD optimizer implementation."""
    print("🧪 Unit Test: SGD Optimizer...")

    # Test basic SGD without momentum
    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = SGD([param], lr=0.1)
    # Set the gradient by hand (no forward/backward in a unit test)
    param.grad = Tensor([0.1, 0.2])
    original_data = param.data.copy()

    optimizer.step()

    # Expected: param = param - lr * grad = [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    expected = original_data - 0.1 * np.array([0.1, 0.2])
    assert np.allclose(param.data, expected)
    assert optimizer.step_count == 1

    # Test SGD with momentum
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    optimizer_momentum = SGD([param2], lr=0.1, momentum=0.9)
    # Set gradient AFTER creating optimizer
    param2.grad = Tensor([0.1, 0.2])

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
    optimizer_wd = SGD([param3], lr=0.1, weight_decay=0.01)
    # Set gradient AFTER creating optimizer
    param3.grad = Tensor([0.1, 0.2])
    optimizer_wd.step()

    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    expected_wd = np.array([1.0, 2.0]) - 0.1 * np.array([0.11, 0.22])
    assert np.allclose(param3.data, expected_wd)

    print("✅ SGD optimizer works correctly!")

if __name__ == "__main__":
    test_unit_sgd_optimizer()

# %% [markdown]
r"""
## 🏗️ Adam: Adaptive Moment Estimation

Adam (Adaptive Moment Estimation) solves a fundamental failure mode of SGD: in deep networks, different parameters require radically different effective step sizes.

### The Parameter Sensitivity Dilemma

Consider a neural network where early feature layers receive attenuated backpropagated gradients ($\sim 10^{-4}$), while the classification head receives direct, large-magnitude error signals ($\sim 10^{-1}$). A single global scalar learning rate $\alpha$ produces an impossible trade-off:
- **Small $\alpha$**: Output weights converge smoothly, but early layers remain virtually frozen.
- **Large $\alpha$**: Early layers learn effectively, but output layer weights oscillate wildly or explode.

### Adam's Dual-Momentum Formulation

Adam dynamically normalizes each coordinate's update step by tracking two running statistics via Exponential Moving Averages (EMA):

$$\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t && \text{(First Moment: Directional Velocity)} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^{\odot 2} && \text{(Second Moment: Coordinate Energy / Variance)}
\end{aligned}$$

With standard defaults $\beta_1 = 0.9$ and $\beta_2 = 0.999$, $\mathbf{m}_t$ averages over $\sim 10$ recent gradients while $\mathbf{v}_t$ averages energy over $\sim 1000$ recent steps.

### Bias Correction: Resolving the Cold-Start Problem

Because buffers are initialized at zero ($\mathbf{m}_0 = \mathbf{0}, \mathbf{v}_0 = \mathbf{0}$), early-step running moments are heavily biased toward zero. Unrolling the recurrence for stationary expectation $\mathbb{E}[\mathbf{g}_i] \approx \mathbb{E}[\mathbf{g}_t]$:

$$\mathbb{E}[\mathbf{m}_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i\right] = (1 - \beta_1^t) \mathbb{E}[\mathbf{g}_t] \implies \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$

$$\mathbb{E}[\mathbf{v}_t] = \mathbb{E}\left[(1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} \mathbf{g}_i^{\odot 2}\right] = (1 - \beta_2^t) \mathbb{E}[\mathbf{g}_t^{\odot 2}] \implies \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

The bias correction divisor $1 - \beta^t$ starts near zero and asymptotes to $1.0$ as $t \to \infty$, ensuring mathematically unbiased estimators from the very first step:

| Step $t$ | Raw Accumulator $\mathbf{m}_t$ | Bias Divisor $(1 - \beta_1^t)$ | Unbiased Estimate $\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$ | Effective Step Scaling |
| :--- | :--- | :--- | :--- | :--- |
| $t = 1$ | $0.100 \cdot \mathbf{g}$ | $1 - 0.900 = 0.100$ | $\frac{0.100 \mathbf{g}}{0.100} = 1.000 \cdot \mathbf{g}$ | $100\%$ true gradient signal |
| $t = 2$ | $0.190 \cdot \mathbf{g}$ | $1 - 0.810 = 0.190$ | $\frac{0.190 \mathbf{g}}{0.190} = 1.000 \cdot \mathbf{g}$ | $100\%$ true gradient signal |
| $t = 3$ | $0.271 \cdot \mathbf{g}$ | $1 - 0.729 = 0.271$ | $\frac{0.271 \mathbf{g}}{0.271} = 1.000 \cdot \mathbf{g}$ | $100\%$ true gradient signal |

### The Scale-Invariant Update Step

Dividing directional momentum by the root-mean-square energy yields coordinate-wise scale invariance:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

Parameters with consistently large gradients are scaled down, while parameters with faint gradients are boosted, allowing all network layers to learn concurrently at rate $\sim \alpha$.
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
        ### BEGIN SOLUTION role="scaffold"
        for name, value in (("lr", lr), ("eps", eps), ("weight_decay", weight_decay)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(betas) != 2 or any(not np.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two finite values in [0, 1)")
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (created lazily)
        self.m_buffers = [None for _ in self.params]  # First moment (mean)
        self.update_counts = [0 for _ in self.params]  # Age of each parameter's moments
        self.v_buffers = [None for _ in self.params]  # Second moment (variance)
        ### END SOLUTION

# %% [markdown]
r"""
### Moment Updates: EMA and Bias Correction

Adam tracks two running statistics per parameter: a first moment ($\mathbf{m}$, exponentially decaying average of past gradients) and a second uncentered moment ($\mathbf{v}$, exponentially decaying average of squared gradients). Because buffers are initialized to zero, both statistics are biased toward zero in early iterations and require step-dependent scaling corrections:

$$\begin{aligned}
\text{Raw First Moment (EMA):} \quad & \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \\
\text{Raw Second Moment (EMA):} \quad & \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \\
\text{Bias-Corrected Estimates:} \quad & \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
\end{aligned}$$

| Moment Buffer | Statistic Modeled | Default Parameter | Bias Correction Factor | Asymptotic Behavior ($t \to \infty$) |
| :--- | :--- | :--- | :--- | :--- |
| **First Moment $\mathbf{m}$** | Mean direction | $\beta_1 = 0.9$ | $1 - \beta_1^t$ | $1 - 0.9^t \to 1.0$ |
| **Second Moment $\mathbf{v}$** | Uncentered variance | $\beta_2 = 0.999$ | $1 - \beta_2^t$ | $1 - 0.999^t \to 1.0$ |

This helper isolates the moment tracking and bias correction mathematics so that `step()` cleanly composes extraction, moment evaluation, and coordinate updates.
"""

# %% nbgrader={"grade": false, "grade_id": "adam-update-moments", "solution": true}
#| export
@method_of(Adam)
def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
    """
    Update first and second moment estimates with bias correction.

    Computes the exponential moving averages of the gradient (first moment)
    and the squared gradient (second moment), then applies bias correction
    to counteract the zero-initialization bias in early training steps.

    TODO: Update moment buffers and return bias-corrected estimates

    APPROACH:
    1. Initialize m and v buffers to zeros if this is the first call
    2. Update first moment: m = beta1 * m + (1 - beta1) * grad
    3. Update second moment: v = beta2 * v + (1 - beta2) * grad^2
    4. Compute bias corrections using this parameter's update count
    5. Return bias-corrected m_hat and v_hat

    EXAMPLE:
    >>> m_hat, v_hat = self._update_moments(0, np.array([0.1, 0.2]))
    >>> # m_hat ≈ grad (after bias correction at step 1)
    >>> # v_hat ≈ grad^2 (after bias correction at step 1)

    HINTS:
    - Increment self.update_counts[i] here: only a parameter with a gradient ages
    - Bias correction denominators: (1 - beta^t) approach 1 as t grows
    """
    ### BEGIN SOLUTION
    # Initialize buffers if needed
    if self.m_buffers[i] is None:
        self.m_buffers[i] = np.zeros_like(grad_data)
        self.v_buffers[i] = np.zeros_like(grad_data)

    # Update biased first moment estimate
    self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data

    # Update biased second moment estimate
    self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

    # A parameter with grad=None skipped this update. Its moments are younger
    # than the optimizer's global step count, so correct using their own age.
    self.update_counts[i] += 1
    bias_correction1 = 1 - self.beta1 ** self.update_counts[i]
    bias_correction2 = 1 - self.beta2 ** self.update_counts[i]

    # Compute bias-corrected moments
    m_hat = self.m_buffers[i] / bias_correction1
    v_hat = self.v_buffers[i] / bias_correction2

    return m_hat, v_hat
    ### END SOLUTION


# %% [markdown]
"""
### 🧪 Unit Test: Adam Moment Updates

This test validates that `_update_moments` correctly computes exponential
moving averages and bias correction for Adam's first and second moments.

**What we're testing**: EMA computation and bias correction math
**Why it matters**: Incorrect moments produce wrong adaptive learning rates
**Expected**: Bias-corrected moments match hand-calculated values
"""

# %% nbgrader={"grade": true, "grade_id": "test-adam-update-moments", "locked": true, "points": 10}
def test_unit_adam_update_moments():
    """🧪 Test Adam _update_moments computes correct EMA and bias correction."""
    print("🧪 Unit Test: Adam Moment Updates...")

    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    grad = np.array([0.1, 0.2])

    # _update_moments ages this parameter itself through update_counts, so the
    # helper is called directly; the global step_count does not drive bias correction.
    m_hat, v_hat = optimizer._update_moments(0, grad)

    # Manual calculation for step 1:
    # m = 0.9 * 0 + 0.1 * [0.1, 0.2] = [0.01, 0.02]
    # v = 0.999 * 0 + 0.001 * [0.01, 0.04] = [0.00001, 0.00004]
    # bias_correction1 = 1 - 0.9^1 = 0.1
    # bias_correction2 = 1 - 0.999^1 = 0.001
    # m_hat = [0.01, 0.02] / 0.1 = [0.1, 0.2]  (= grad, as expected!)
    # v_hat = [0.00001, 0.00004] / 0.001 = [0.01, 0.04]  (= grad^2)

    assert np.allclose(m_hat, grad), f"m_hat should equal grad at step 1, got {m_hat}"
    assert np.allclose(v_hat, grad ** 2), f"v_hat should equal grad^2 at step 1, got {v_hat}"

    # Second update of the same parameter: update_counts[0] becomes 2
    m_hat2, v_hat2 = optimizer._update_moments(0, grad)

    # Moments should still be close to grad (converging to true mean)
    assert m_hat2 is not None
    assert v_hat2 is not None
    # Buffers should be updated in-place
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None

    print("✅ Adam moment updates work correctly!")

if __name__ == "__main__":
    test_unit_adam_update_moments()

# %% [markdown]
r"""
### Adam Step: Composing Gradient Extraction, Moments, and Update

The Adam `step()` pipeline executes a sequence of localized numerical transformations across each parameter:

| Step Phase | Mathematical Operation | Systems Invariant |
| :--- | :--- | :--- |
| **1. Extraction** | $\mathbf{g} = \text{unwrap}(\text{param.grad})$ | Obtains contiguous float32 NumPy buffer |
| **2. Coupled Decay** | $\mathbf{g} \leftarrow \mathbf{g} + \lambda \boldsymbol{\theta}$ | Coupled $L_2$ penalty added into a fresh array, leaving `param.grad` untouched |
| **3. Moments** | $(\hat{\mathbf{m}}, \hat{\mathbf{v}}) = \text{EMA}(\mathbf{g})$ | Bias-corrected first and second moment updates |
| **4. Mutation** | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}$ | Elementwise update that rebinds `param.data` to a new array |

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$
"""

# %% nbgrader={"grade": false, "grade_id": "adam-step", "solution": true}
#| exporti
@method_of(Adam)
def step(self):
    """
    Perform Adam update step by composing helpers.

    TODO: Implement Adam parameter update using _extract_gradient and _update_moments

    APPROACH:
    1. Increment the global step_count (each moment helper tracks its own age)
    2. For each parameter with gradients:
       a. Extract gradient with self._extract_gradient(param)
       b. Apply weight decay to gradient if specified
       c. Update moments with self._update_moments(i, grad_data)
       d. Update parameter: param -= lr * m_hat / (sqrt(v_hat) + eps)

    FORMULAS:
    - θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

    HINTS:
    - step_count counts optimizer calls; update_counts counts each parameter's updates
    - _update_moments returns (m_hat, v_hat) tuple
    - Weight decay modifies grad_data before moment update
    """
    ### BEGIN SOLUTION
    # Count optimizer calls; each parameter separately counts its moment updates.
    self.step_count += 1

    for i, param in enumerate(self.params):
        if param.grad is None:
            continue

        # Extract gradient using shared helper
        grad_data = self._extract_gradient(param)

        # Apply weight decay to a copy, so param.grad is never mutated
        if self.weight_decay != 0:
            grad_data = grad_data + self.weight_decay * param.data

        # Update moments and get bias-corrected estimates
        m_hat, v_hat = self._update_moments(i, grad_data)

        # Update parameter
        param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    ### END SOLUTION


# %% [markdown]
"""
### 🧪 Unit Test: Adam Optimizer

This test validates our Adam implementation works correctly.

**What we're testing**: Adam updates with adaptive learning rates and bias correction
**Why it matters**: Most popular optimizer for modern neural networks
**Expected**: Correct parameter updates following Adam formulas
"""

# %% nbgrader={"grade": true, "grade_id": "test-adam", "locked": true, "points": 20}
def test_unit_adam_optimizer():
    """🧪 Test Adam optimizer implementation."""
    print("🧪 Unit Test: Adam Optimizer...")

    # Test basic Adam functionality
    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    # Set the gradient by hand (no forward/backward in a unit test)
    param.grad = Tensor([0.1, 0.2])
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
    optimizer_wd = Adam([param2], lr=0.01, weight_decay=0.01)
    # Set gradient AFTER creating optimizer
    param2.grad = Tensor([0.1, 0.2])
    optimizer_wd.step()

    # Weight decay should modify the effective gradient
    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    # The exact computation is complex, but we can verify parameter changed
    assert not np.array_equal(param2.data, np.array([1.0, 2.0]))

    print("✅ Adam optimizer works correctly!")

if __name__ == "__main__":
    test_unit_adam_optimizer()

# %% [markdown]
r"""
## 🏗️ AdamW: Adam with Decoupled Weight Decay

AdamW (Loshchilov & Hutter, 2017) resolves a fundamental flaw in how adaptive optimizers historically handled weight decay. In standard SGD, $L_2$ regularization and weight decay are mathematically identical; in adaptive algorithms like Adam, they diverge catastrophically.

### The Mathematical Divergence: SGD vs Adam

In classical SGD, adding an $L_2$ regularization penalty $\frac{\lambda}{2} \|\boldsymbol{\theta}\|_2^2$ to the loss function yields:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{reg}} = \mathbf{g}_t + \lambda \boldsymbol{\theta}_t \implies \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha (\mathbf{g}_t + \lambda \boldsymbol{\theta}_t) = (1 - \alpha \lambda) \boldsymbol{\theta}_t - \alpha \mathbf{g}_t$$

Each parameter is shrunk uniformly by factor $(1 - \alpha \lambda)$ at every step, independent of gradient scale.

### The Inversion Pathology in Standard Adam

Standard Adam implements "coupled" weight decay by adding $\lambda \boldsymbol{\theta}_t$ directly to the gradient vector $\mathbf{g}_t^{\text{coupled}} = \mathbf{g}_t + \lambda \boldsymbol{\theta}_t$ before computing moment buffers:

$$\Delta \boldsymbol{\theta}_t = -\alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \approx -\alpha \frac{\mathbf{g}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \frac{\alpha \lambda}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \boldsymbol{\theta}_t$$

Notice the denominator $\sqrt{\hat{\mathbf{v}}_t} + \epsilon$:
- **Frequent or High-Magnitude Gradients**: $\hat{\mathbf{v}}_t$ is large $\implies$ the effective weight decay $\frac{\alpha \lambda}{\sqrt{\hat{\mathbf{v}}_t}}$ is severely suppressed! Active parameters receive virtually no regularization.
- **Sparse or Low-Magnitude Gradients**: $\hat{\mathbf{v}}_t$ is small $\implies$ the effective weight decay is amplified! Rare feature parameters are shrunk aggressively toward zero.

This behavior is completely backwards from principled statistical regularization.

### AdamW's Decoupled Solution

AdamW decouples weight decay from adaptive gradient scaling. Moments are accumulated exclusively on pure task gradients $\mathbf{g}_t$, while shrinkage is applied directly to the parameter buffer:

$$\boldsymbol{\theta}_{t+1} = \underbrace{(1 - \alpha \lambda) \boldsymbol{\theta}_t}_{\text{Uniform Parameter Shrinkage}} - \underbrace{\alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}}_{\text{Adaptive Gradient Step}}$$

| Property | Coupled Regularization (Adam) | Decoupled Weight Decay (AdamW) |
| :--- | :--- | :--- |
| **Gradient Fed to Moments** | $\mathbf{g}_t + \lambda \boldsymbol{\theta}_t$ (contaminated) | $\mathbf{g}_t$ (pure loss gradient) |
| **Coordinate Decay Rate** | $\frac{\alpha \lambda}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$ (inversely scaled by variance) | $\alpha \lambda$ (constant and uniform across all weights) |
| **Frequent / Active Features** | Under-regularized (large $\hat{v} \implies$ negligible decay) | Consistently regularized |
| **Sparse / Inactive Features** | Over-regularized (small $\hat{v} \implies$ massive decay) | Consistently regularized |
| **Hyperparameter Coupling** | Optimal $\lambda$ depends nonlinearly on $\alpha$ | Optimal $\lambda$ is largely decoupled from $\alpha$ |
"""

# %% nbgrader={"grade": false, "grade_id": "adamw-optimizer", "solution": true}
#| export
class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    AdamW applies weight decay directly to the parameters instead of folding
    it into the gradient, so the decay is not rescaled by the adaptive step.
    This gives cleaner regularization and is the preferred choice in practice.
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
        ### BEGIN SOLUTION role="scaffold"
        for name, value in (("lr", lr), ("eps", eps), ("weight_decay", weight_decay)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(betas) != 2 or any(not np.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two finite values in [0, 1)")
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (same as Adam)
        self.m_buffers = [None for _ in self.params]
        self.update_counts = [0 for _ in self.params]  # Age of each parameter's moments
        self.v_buffers = [None for _ in self.params]
        ### END SOLUTION

# %% [markdown]
r"""
### AdamW Moment Updates: Pure Gradients without Decay Pollution

AdamW employs the exact same Exponential Moving Average (EMA) and bias correction mathematics as Adam. However, the architectural context differs fundamentally: AdamW feeds **pure objective gradients** $\mathbf{g}_t$ into `_update_moments()`, ensuring that parameter decay never distorts the coordinate variance estimates:

$$\mathbf{g}_t \xrightarrow{\text{pure gradient}} \_ \text{update\_moments}(i, \mathbf{g}_t) \implies \begin{cases} \hat{\mathbf{m}}_t = \frac{\beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t}{1 - \beta_1^t} \\ \hat{\mathbf{v}}_t = \frac{\beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^{\odot 2}}{1 - \beta_2^t} \end{cases}$$

Parameter shrinkage is subsequently applied as an independent transformation directly to the parameter buffer.
"""

# %% nbgrader={"grade": false, "grade_id": "adamw-update-moments", "solution": true}
#| export
@method_of(AdamW)
def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
    """
    Update first and second moment estimates with bias correction for AdamW.

    Identical math to Adam's _update_moments: EMA of gradient and squared
    gradient, with bias correction. The key difference is in how step()
    calls this -- AdamW passes pure gradients without weight decay mixed in.

    TODO: Update moment buffers and return bias-corrected estimates

    APPROACH:
    1. Initialize m and v buffers to zeros if this is the first call
    2. Update first moment: m = beta1 * m + (1 - beta1) * grad
    3. Update second moment: v = beta2 * v + (1 - beta2) * grad^2
    4. Compute bias corrections using this parameter's update count
    5. Return bias-corrected m_hat and v_hat

    EXAMPLE:
    >>> m_hat, v_hat = self._update_moments(0, np.array([0.1, 0.2]))

    HINT: Same math as Adam -- the decoupling happens in step(), not here
    """
    ### BEGIN SOLUTION role="scaffold"
    # Initialize buffers if needed
    if self.m_buffers[i] is None:
        self.m_buffers[i] = np.zeros_like(grad_data)
        self.v_buffers[i] = np.zeros_like(grad_data)

    # Update biased first moment estimate
    self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data

    # Update biased second moment estimate
    self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

    # A parameter with grad=None skipped this update. Its moments are younger
    # than the optimizer's global step count, so correct using their own age.
    self.update_counts[i] += 1
    bias_correction1 = 1 - self.beta1 ** self.update_counts[i]
    bias_correction2 = 1 - self.beta2 ** self.update_counts[i]

    # Compute bias-corrected moments
    m_hat = self.m_buffers[i] / bias_correction1
    v_hat = self.v_buffers[i] / bias_correction2

    return m_hat, v_hat
    ### END SOLUTION


# %% [markdown]
"""
### 🧪 Unit Test: AdamW Moment Updates

This test validates that AdamW's `_update_moments` computes the same EMA
and bias correction as Adam (the math is identical; the decoupling
difference is in how `step()` uses these results).

**What we're testing**: EMA computation for AdamW
**Why it matters**: Correct moments are needed for adaptive learning rates
**Expected**: Same bias-corrected values as Adam for identical inputs
"""

# %% nbgrader={"grade": true, "grade_id": "test-adamw-update-moments", "locked": true, "points": 10}
def test_unit_adamw_update_moments():
    """🧪 Test AdamW _update_moments produces correct bias-corrected moments."""
    print("🧪 Unit Test: AdamW Moment Updates...")

    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = AdamW([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    grad = np.array([0.1, 0.2])

    # First update of this parameter; _update_moments tracks the age itself
    m_hat, v_hat = optimizer._update_moments(0, grad)

    # At step 1, bias-corrected m_hat should equal the gradient
    assert np.allclose(m_hat, grad), f"m_hat should equal grad at step 1, got {m_hat}"
    assert np.allclose(v_hat, grad ** 2), f"v_hat should equal grad^2 at step 1, got {v_hat}"

    # Verify buffers were initialized
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None

    # Verify AdamW and Adam produce same moment values for same input
    param_adam = Tensor([1.0, 2.0], requires_grad=True)
    adam_opt = Adam([param_adam], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    # Reset AdamW buffers for fair comparison
    param_adamw = Tensor([1.0, 2.0], requires_grad=True)
    adamw_opt = AdamW([param_adamw], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    m_adam, v_adam = adam_opt._update_moments(0, grad)
    m_adamw, v_adamw = adamw_opt._update_moments(0, grad)

    assert np.allclose(m_adam, m_adamw), "Adam and AdamW moments should be identical"
    assert np.allclose(v_adam, v_adamw), "Adam and AdamW moments should be identical"

    print("✅ AdamW moment updates work correctly!")

if __name__ == "__main__":
    test_unit_adamw_update_moments()

# %% [markdown]
r"""
### AdamW Step: Decoupled Weight Decay Composition

AdamW's `step()` composes the unified gradient extraction and moment update helpers into an atomic four-phase pipeline per parameter tensor:

| Phase | Operation | Mathematical Transformation | Implementation Vector |
| :--- | :--- | :--- | :--- |
| **1. Unpack** | Gradient Extraction | $\mathbf{g}_t \leftarrow \text{param.grad}$ | `grad_data = self._extract_gradient(param)` |
| **2. Moments** | Variance Tracking | $(\hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t) \leftarrow \text{EMA}(\mathbf{g}_t)$ | `m_hat, v_hat = self._update_moments(i, grad_data)` |
| **3. Shrinkage** | Decoupled Weight Decay | $\boldsymbol{\theta} \leftarrow (1 - \alpha \lambda) \boldsymbol{\theta}$ | `param.data = param.data * (1 - self.lr * self.weight_decay)` |
| **4. Descent** | Adaptive Normalization | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$ | `param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)` |

Applying parameter shrinkage in Phase 3 prior to the adaptive descent in Phase 4 ensures that regularization remains completely orthogonal to gradient conditioning.
"""

# %% nbgrader={"grade": false, "grade_id": "adamw-step", "solution": true}
#| exporti
@method_of(AdamW)
def step(self):
    """
    Perform AdamW update step by composing helpers with decoupled weight decay.

    TODO: Implement AdamW parameter update using _extract_gradient and _update_moments

    APPROACH:
    1. Increment the global step_count (each moment helper tracks its own age)
    2. For each parameter with gradients:
       a. Extract gradient with self._extract_gradient(param)
       b. Update moments with self._update_moments(i, grad_data) -- pure gradient
       c. Decay the old weight: param *= (1 - lr * weight_decay)
       d. Apply gradient update: param -= lr * m_hat / (sqrt(v_hat) + eps)

    KEY DIFFERENCE from Adam:
    - NO weight decay added to gradient before moment update
    - Weight decay applied to the old parameter BEFORE gradient update

    HINTS:
    - Do NOT modify grad_data with weight decay (that is Adam's coupled form)
    - Apply decay as a multiplicative factor on param.data
    """
    ### BEGIN SOLUTION role="scaffold"
    # Increment step counter first
    self.step_count += 1

    for i, param in enumerate(self.params):
        if param.grad is None:
            continue

        # Extract gradient using shared helper
        grad_data = self._extract_gradient(param)

        # Update moments using PURE gradients (no weight decay mixed in)
        m_hat, v_hat = self._update_moments(i, grad_data)

        # Decay the old weight, not the adaptive update we are about to add.
        if self.weight_decay != 0:
            param.data = param.data * (1 - self.lr * self.weight_decay)

        # Apply gradient-based update independently of decay.
        param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    ### END SOLUTION


# %% [markdown]
"""
### 🧪 Unit Test: AdamW Optimizer

This test validates our AdamW implementation with decoupled weight decay.

**What we're testing**: AdamW updates with proper weight decay decoupling
**Why it matters**: State-of-the-art optimizer for modern neural networks
**Expected**: Correct separation of gradient updates and weight decay
"""

# %% nbgrader={"grade": true, "grade_id": "test-adamw", "locked": true, "points": 20}
def test_unit_adamw_optimizer():
    """🧪 Test AdamW optimizer implementation."""
    print("🧪 Unit Test: AdamW Optimizer...")

    # Test AdamW vs Adam difference in weight decay
    # Create identical parameters for comparison
    param_adam = Tensor([1.0, 2.0], requires_grad=True)
    param_adamw = Tensor([1.0, 2.0], requires_grad=True)

    # Create optimizers with same settings
    adam = Adam([param_adam], lr=0.01, weight_decay=0.01)
    adamw = AdamW([param_adamw], lr=0.01, weight_decay=0.01)

    # Set the gradients by hand
    param_adam.grad = Tensor([0.1, 0.2])
    param_adamw.grad = Tensor([0.1, 0.2])

    # Take one step
    adam.step()
    adamw.step()

    # Results should be different due to weight decay implementation
    assert not np.allclose(param_adam.data, param_adamw.data, rtol=1e-6)

    # Test AdamW basic functionality
    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = AdamW([param], lr=0.01, weight_decay=0.01)
    # Set gradient AFTER creating optimizer
    param.grad = Tensor([0.1, 0.2])
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

    adam_no_wd = Adam([param1], lr=0.01, weight_decay=0.0)
    adamw_no_wd = AdamW([param2], lr=0.01, weight_decay=0.0)

    # Set gradients AFTER creating optimizers
    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([0.1, 0.2])

    adam_no_wd.step()
    adamw_no_wd.step()

    # Should be very similar (within numerical precision)
    assert np.allclose(param1.data, param2.data, rtol=1e-10)

    # Decay acts on the old weight, independently of the adaptive step.
    p = Tensor([2.0])
    exact = AdamW([p], lr=0.1, betas=(0.0, 0.0), weight_decay=0.5)
    p.grad = np.array([1.0], dtype=np.float32)
    exact.step()
    assert np.allclose(p.data, [1.8], atol=1e-7), "Decay must not shrink the new gradient update"

    print("✅ AdamW optimizer works correctly!")

if __name__ == "__main__":
    test_unit_adamw_optimizer()

# %% [markdown]
"""
### Checkpointing Adam's Moments

Module 08's Trainer will save optimizer state through three small methods that SGD
already has: `has_momentum()`, `get_momentum_state()`, and `set_momentum_state()`.
Adam and AdamW carry two buffers per parameter instead of one, so they answer the
same three questions with (m, v) pairs. These methods save and restore **buffers
only**, not a complete optimizer checkpoint. Module 08's Trainer will also save
the per-parameter `update_counts`, global `step_count`, and hyperparameters.
The moments and their update counts must resume together: restoring only the
buffers gives the wrong bias correction on the next step.
"""

# %% nbgrader={"grade": false, "grade_id": "adam-checkpoint-state", "solution": false}
#| exporti
def _adam_has_momentum(self) -> bool:
    """Adam always keeps moment buffers, so there is always state to checkpoint."""
    return True

def _adam_get_momentum_state(self) -> List:
    """Copy only (m, v) buffers; Trainer also saves counts and hyperparameters."""
    return [
        (None if m is None else m.copy(), None if v is None else v.copy())
        for m, v in zip(self.m_buffers, self.v_buffers)
    ]

def _adam_set_momentum_state(self, state: Optional[List]) -> None:
    """Restore only buffers; the caller must also restore update_counts to resume."""
    if state is None:
        return
    if len(state) != len(self.m_buffers):
        raise ValueError(
            f"Optimizer state mismatch: state has {len(state)} entries, "
            f"optimizer has {len(self.m_buffers)} parameters"
        )
    # Check the complete state before mutating either list of moment buffers.
    for param, pair in zip(self.params, state):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Adam state entries must be (m, v) pairs")
        m, v = pair
        if (m is None) != (v is None):
            raise ValueError("Adam moment buffers must both be initialized or both be None")
        for buf in pair:
            if buf is not None and (not isinstance(buf, np.ndarray) or buf.shape != param.data.shape):
                raise ValueError("Adam moment buffer shape must match its parameter")
    restored_m = [None if m is None else m.copy() for m, _ in state]
    restored_v = [None if v is None else v.copy() for _, v in state]
    self.m_buffers, self.v_buffers = restored_m, restored_v

for _cls in (Adam, AdamW):
    _cls.has_momentum = _adam_has_momentum
    _cls.get_momentum_state = _adam_get_momentum_state
    _cls.set_momentum_state = _adam_set_momentum_state

# %% [markdown]
"""
### 🧪 Unit Test: Adam Checkpoint State

**What we're testing**: get_momentum_state / set_momentum_state round trip for Adam and AdamW
**Why it matters**: Module 08's Trainer will restore optimizer state from checkpoints
**Expected**: Restored buffers equal the saved ones
"""

# %% nbgrader={"grade": true, "grade_id": "test-adam-checkpoint-state", "locked": true, "points": 5}
def test_unit_adam_checkpoint_state():
    """🧪 Test that Adam's moments survive a get/set round trip."""
    print("🧪 Unit Test: Adam checkpoint state...")
    W = Tensor(rng.standard_normal((3, 2)))
    optimizer = Adam([W], lr=0.001)
    W.grad = Tensor(rng.standard_normal((3, 2)))
    optimizer.step()

    assert optimizer.has_momentum()
    state = optimizer.get_momentum_state()
    assert len(state) == 1 and state[0][0] is not None and state[0][1] is not None

    fresh = AdamW([W], lr=0.001)
    fresh.set_momentum_state(state)
    assert np.allclose(fresh.m_buffers[0], optimizer.m_buffers[0])
    assert np.allclose(fresh.v_buffers[0], optimizer.v_buffers[0])
    print("✅ Adam checkpoint state works correctly!")

if __name__ == "__main__":
    test_unit_adam_checkpoint_state()

# %% [markdown]
r"""
## 🔧 Integration: Bringing It Together

All four update rules answer the same `step()` call, so a training loop can swap one for another without changing a line. Each embodies a distinct geometric approach to resolving loss curvature, and the two Systems Analysis cells that follow measure the two costs that difference carries, resident memory and progress on an ill-conditioned landscape.

| Algorithm | Trajectory Dynamic | Effective Step Scaling | Regularization Coupling |
| :--- | :--- | :--- | :--- |
| **SGD** | Steepest descent along local slope | Global scalar $\alpha$ | None (or coupled $L_2$ gradient penalty) |
| **SGD + Momentum** | Heavy-ball physical inertia | Accumulated velocity $\frac{\alpha}{1 - \beta}$ | None |
| **Adam** | Directional & variance EMA tracking | Coordinate-wise $\frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$ | Coupled to gradient before moment estimation |
| **AdamW** | Directional & variance EMA tracking | Coordinate-wise $\frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$ | **Decoupled**: Direct parameter shrinkage $(1 - \alpha \lambda)$ |
"""


# %% [markdown]
r"""
## 📊 Systems Analysis: Optimizer Performance and Memory

In production machine learning systems, optimizer state buffers often dominate the total resident memory of hardware accelerators. Understanding the arithmetic and memory bandwidth trade-offs is essential for scaling models to large architectures.

<div align="center">
  <img src="optimizer_margin_memory.svg" alt="AdamW State: 16 Bytes per Parameter" width="320px">
</div>

### Memory Usage Patterns: The 16-Byte Parameter Rule

For single-precision (FP32) training, each model parameter incurs substantial secondary state allocations.

**What the multipliers count.** Every $\times$ figure in this module is quoted against the 4 bytes of the weight itself, and it counts the entire resident training footprint: the weight, the gradient buffer the backward pass fills, and any optimizer state. That is why stateless SGD is already $2\times$ before a single byte of optimizer state exists.

| Buffer Role | Precision & Size | Resident Lifespan | SGD ($\beta=0$) | SGD + Momentum | Adam / AdamW |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Weight ($\boldsymbol{\theta}$)** | FP32 (4 Bytes) | Permanent | 4 Bytes | 4 Bytes | 4 Bytes |
| **Loss Gradient ($\mathbf{g}$)** | FP32 (4 Bytes) | Transient (backward pass) | 4 Bytes | 4 Bytes | 4 Bytes |
| **First Moment Buffer ($\mathbf{m}$)** | FP32 (4 Bytes) | Permanent state | 0 Bytes | 4 Bytes (`momentum_buffers`) | 4 Bytes (`m_buffers`) |
| **Second Moment Buffer ($\mathbf{v}$)** | FP32 (4 Bytes) | Permanent state | 0 Bytes | 0 Bytes | 4 Bytes (`v_buffers`) |
| **Total Resident Memory** | — | — | **8 B / param ($2\times$)** | **12 B / param ($3\times$)** | **16 B / param ($4\times$)** |

> **Systems Takeaway**: Training a 7-billion parameter language model in FP32 requires $7 \times 10^9 \times 16\text{ Bytes} = 112\text{ GB}$ of memory solely for weights, gradients, and AdamW moments, before allocating a single byte for activation tensors or KV caches!

### Computational Complexity & Memory Bandwidth

Because optimizer steps execute element-wise over the entire parameter set, their execution speed is bound by memory bandwidth rather than compute throughput (FLOP/s):

| Optimizer | Arithmetic FLOPs / Parameter | Dominant Vector Math Kernel | Memory Bandwidth (Bytes / Param) |
| :--- | :--- | :--- | :--- |
| **Vanilla SGD** | 2 FLOPs ($\boldsymbol{\theta} - \alpha \mathbf{g}$) | Scaled axpy | 8 B Read ($\boldsymbol{\theta}, \mathbf{g}$) + 4 B Write ($\boldsymbol{\theta}$) = 12 B |
| **SGD + Momentum** | 4 FLOPs ($\mathbf{v} \leftarrow \beta \mathbf{v} + \mathbf{g}; \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \mathbf{v}$) | Compound axpy | 12 B Read ($\boldsymbol{\theta}, \mathbf{g}, \mathbf{v}$) + 8 B Write ($\mathbf{v}, \boldsymbol{\theta}$) = 20 B |
| **Adam / AdamW** | ~14 FLOPs (2 EMA, 2 bias div, sqrt, div, decay, axpy) | Fused point-wise kernel | 16 B Read ($\boldsymbol{\theta}, \mathbf{g}, \mathbf{m}, \mathbf{v}$) + 12 B Write ($\mathbf{m}, \mathbf{v}, \boldsymbol{\theta}$) = 28 B |
"""

# %% nbgrader={"grade": false, "grade_id": "optimizer-analysis", "solution": false}
def analyze_optimizer_memory_usage():
    """📊 Analyze memory usage of different optimizers."""
    print("📊 Analyzing Optimizer Memory Usage...")

    # Create test parameters of different sizes
    param_sizes = [1000, 10000, 100000]  # 1K, 10K, 100K parameters

    def resident_bytes(optimizer, *buffer_lists):
        """Weight + gradient + optimizer state, read straight off ndarray.nbytes."""
        total = 0
        for param in optimizer.params:
            total += param.data.nbytes
            grad = param.grad
            total += (grad.data if isinstance(grad, Tensor) else grad).nbytes
        for buffers in buffer_lists:
            total += sum(buf.nbytes for buf in buffers if buf is not None)
        return total

    def measure(optimizer_class, kwargs, size):
        """Build the optimizer, take one real step, then weigh what it retains."""
        param = Tensor(rng.standard_normal(size), requires_grad=True)
        optimizer = optimizer_class([param], **kwargs)
        # Set gradient AFTER creating optimizer
        param.grad = Tensor(rng.standard_normal(size))
        optimizer.step()  # A state buffer only exists after the first step
        if isinstance(optimizer, SGD):
            return resident_bytes(optimizer, optimizer.momentum_buffers)
        return resident_bytes(optimizer, optimizer.m_buffers, optimizer.v_buffers)

    configs = [
        ("SGD", SGD, {"lr": 0.01}),
        ("SGD+Mom", SGD, {"lr": 0.01, "momentum": 0.9}),
        ("Adam", Adam, {}),
        ("AdamW", AdamW, {}),
    ]

    print("Resident FP32 training memory, measured with ndarray.nbytes:")
    print("=" * 74)
    print(f"{'Params':<10}{'SGD':>13}{'SGD+Mom':>13}{'Adam':>13}{'AdamW':>13}")
    print("-" * 74)

    per_param = {}
    for size in param_sizes:
        totals = [measure(cls, kwargs, size) for _, cls, kwargs in configs]
        for (name, _, _), total in zip(configs, totals):
            per_param[name] = total // size
        print(f"{size:<10}" + "".join(f"{total / 1000:>10.1f} kB" for total in totals))

    print("-" * 74)
    print(f"{'B / param':<10}" + "".join(f"{per_param[name]:>11} B" for name, _, _ in configs))

    weight_bytes = 4  # FP32
    print("\n💡 Key Insights:")
    print(f"- Measured per parameter: SGD {per_param['SGD']} B ({per_param['SGD'] // weight_bytes}x), "
          f"SGD+momentum {per_param['SGD+Mom']} B ({per_param['SGD+Mom'] // weight_bytes}x), "
          f"Adam/AdamW {per_param['Adam']} B ({per_param['Adam'] // weight_bytes}x)")
    print("- Each multiplier counts the 4-byte weight plus the 4-byte gradient plus")
    print("  optimizer state, so stateless SGD is already 2x with no state at all")
    print(f"- Adam's two moment buffers add {per_param['Adam'] - per_param['SGD']} B, "
          f"which doubles SGD's {per_param['SGD']} B rather than adding half of it")
    print("- Memory scales linearly with parameter count, so the per-parameter constant")
    print("  is the number to budget hardware against")


if __name__ == "__main__":
    analyze_optimizer_memory_usage()

# %% nbgrader={"grade": false, "grade_id": "optimizer-convergence", "solution": false}
def analyze_optimizer_convergence_behavior():
    """📊 Analyze convergence behavior of different optimizers."""
    print("📊 Analyzing Optimizer Convergence Behavior...")

    # The Introduction's ill-conditioned bowl, L(w) = 50*w1^2 + 0.5*w2^2, whose
    # Hessian is diag(100, 1) and whose condition number is therefore 100. An
    # isotropic bowl would make every optimizer look alike, because the curvature
    # gap is the whole thing these algorithms exist to handle.
    curvature = np.array([100.0, 1.0])

    def quadratic_loss(w):
        """Ill-conditioned quadratic: 0.5 * (100*w1^2 + 1*w2^2)."""
        return 0.5 * float((curvature * w ** 2).sum())

    def compute_gradient(w):
        """Gradient of the ill-conditioned quadratic: [100*w1, w2]."""
        return curvature * w

    # Starting point
    w_start = np.array([1.0, 1.0])  # Both axes equally far from the optimum [0, 0]
    steps = 50

    # The learning rates differ on purpose. SGD's is capped by the STEEP axis: the
    # w1 factor is 1 - lr*100, so lr = 0.02 already fails to contract and lr = 0.1
    # gives -9 and diverges. Adam's step size is bounded by lr rather than by
    # curvature, so it can run at the lr that destroys SGD.
    optimizers_to_test = [
        ("SGD", SGD, {"lr": 0.01}),
        ("SGD+Momentum", SGD, {"lr": 0.01, "momentum": 0.9}),
        ("Adam", Adam, {"lr": 0.1}),
        ("AdamW", AdamW, {"lr": 0.1, "weight_decay": 0.01})
    ]

    print(f"Convergence on L(w) = 50*w1^2 + 0.5*w2^2 (kappa = 100), {steps} steps:")
    print("=" * 78)
    print(f"{'Optimizer':<15}{'lr':<8}{'Step 0':<13}{'Step 10':<13}{'Step 25':<13}{'Step 50':<13}")
    print("-" * 78)

    endpoints = []
    for name, optimizer_class, kwargs in optimizers_to_test:
        # Reset parameter
        param = Tensor(w_start.copy(), requires_grad=True)
        optimizer = optimizer_class([param], **kwargs)

        losses = []

        # Run optimization
        for step in range(steps + 1):
            # Compute loss and gradient
            losses.append(quadratic_loss(param.data))
            param.grad = Tensor(compute_gradient(param.data))

            # Update parameters
            if step < steps:  # Don't update after the last evaluation
                optimizer.step()
                optimizer.zero_grad()

        endpoints.append((name, param.data.copy()))
        print(f"{name:<15}{kwargs['lr']:<8}"
              f"{losses[0]:<13.6f}{losses[10]:<13.6f}{losses[25]:<13.6f}{losses[steps]:<13.6f}")

    print("-" * 78)
    print("Distance still to travel on each axis (steep w1, gentle w2):")
    for name, w in endpoints:
        print(f"  {name:<15} |w1| = {abs(w[0]):.4f}    |w2| = {abs(w[1]):.4f}")

    print("\n💡 Key Insights:")
    print("- SGD: lr=0.01 is exactly right for the steep axis (1 - lr*100 = 0) and 100x")
    print("  too small for the gentle one, so |w2| crawls from 1.0 only down to 0.605")
    print("- SGD+Momentum: the 1/(1-beta) = 10x amplification finally moves the gentle")
    print("  axis down to 0.058, but it re-excites the steep axis SGD had already")
    print("  solved, which is why its step-10 and step-25 losses are WORSE than SGD's")
    print("- Adam: per-coordinate normalization drives w1 and w2 along the same path to")
    print("  float32 rounding despite the 100x curvature gap, at an lr that breaks SGD")
    print("- AdamW: the shrink factor here is 1 - lr*lambda = 0.999 per step, so it")
    print("  tracks Adam closely; weight decay buys generalization, not convergence")


if __name__ == "__main__":
    analyze_optimizer_convergence_behavior()

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
    test_unit_extract_gradient()
    test_unit_optimizer_base()
    test_unit_sgd_optimizer()
    test_unit_adam_update_moments()
    test_unit_adam_optimizer()
    test_unit_adamw_update_moments()
    test_unit_adamw_optimizer()
    test_unit_adam_checkpoint_state()

    print("\nRunning integration scenarios...")

    # Drive all three optimizers over one shared set of parameter tensors
    print("🧪 Integration Test: Shared Parameters Across Three Optimizers...")

    # The gradients below are synthesized, not produced by a forward and backward
    # pass. This module's contract is the update rule alone; Module 08 will feed
    # these same optimizers from a real Linear, ReLU, and MSELoss training loop.

    # Tensors shaped like a 3 -> 4 -> 2 network
    # Layer 1: 3 inputs -> 4 hidden
    W1 = Tensor(rng.standard_normal((3, 4)) * 0.1, requires_grad=True)
    b1 = Tensor(np.zeros(4), requires_grad=True)

    # Layer 2: 4 hidden -> 2 outputs
    W2 = Tensor(rng.standard_normal((4, 2)) * 0.1, requires_grad=True)
    b2 = Tensor(np.zeros(2), requires_grad=True)

    params = [W1, b1, W2, b2]

    # Test all optimizers on the same network
    optimizers = [
        SGD(params, lr=0.01, momentum=0.9),
        Adam(params, lr=0.001),
        AdamW(params, lr=0.001, weight_decay=0.01),
    ]

    def set_gradients():
        for p in params:
            p.grad = Tensor(rng.standard_normal(p.shape) * 0.01)

    # Save original parameter values
    original_params = [p.data.copy() for p in params]

    # Each optimizer starts from the same weights with fresh gradients
    results = []
    for optimizer in optimizers:
        for p, original in zip(params, original_params):
            p.data = original.copy()
        set_gradients()
        optimizer.step()
        results.append([p.data.copy() for p in params])
    sgd_params, adam_params, adamw_params = results

    # Verify parameters changed differently for each optimizer
    for i in range(len(params)):
        # Parameters should be different from original
        assert not np.array_equal(sgd_params[i], original_params[i])
        assert not np.array_equal(adam_params[i], original_params[i])
        assert not np.array_equal(adamw_params[i], original_params[i])

        # Different optimizers should produce different results
        assert not np.allclose(sgd_params[i], adam_params[i], rtol=1e-6)

    print("✅ Shared-parameter optimization works!")

    # Test optimizer state management
    print("🧪 Integration Test: Optimizer State Management...")

    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Adam([param], lr=0.001)
    # Set gradient AFTER creating optimizer
    param.grad = Tensor([0.1, 0.2])

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
    print("Run: tito module complete 07")

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of optimizer operations and their systems implications:

### Question 1: Memory vs Performance
**Question**: You've implemented SGD (8 bytes per parameter, 2x) and Adam (16 bytes per parameter, 4x), counting weight plus gradient plus optimizer state. For a model with 10 billion parameters at float32 (4 bytes each):

**Consider**:
- How much total memory does each optimizer require?
- On one 80 GB accelerator, what is the largest model each optimizer can hold? Adam's extra 8 bytes per parameter is the whole question.
- What real-world constraints might force you to choose SGD over Adam?

**Calculate**:
- Parameters: 10 x 10^9
- Bytes per float32: 4
- SGD memory (2x, 8 B/param): ___________GB
- Adam memory (4x, 16 B/param): ___________GB

---

### Question 2: Learning Rate Sensitivity
**Question**: SGD uses a fixed learning rate for all parameters, while Adam adapts per-parameter.

**Consider**:
- Why might Adam converge faster on problems with parameters at different scales?
- When might SGD's uniform learning rate actually be an advantage?
- How does momentum in SGD relate to Adam's first moment estimation?

**Real-world context**: Deep neural networks often have early layers with very different gradient magnitudes than later layers. Adam's adaptive rates help balance these naturally.

---

### Question 3: Optimizer State Management
**Question**: Adam and AdamW maintain momentum buffers (m, v) that persist across training steps.

**Consider**:
- What happens to these buffers when you checkpoint during training?
- If you resume training with different hyperparameters, should you restore the old buffers?
- How does optimizer state affect training when you restart from a checkpoint?

**Think about**:
- Checkpoint size: Adam stores 2 additional tensors per parameter
- Resume behavior: Warm vs cold restart implications
- Resume behavior: What happens to momentum buffers if you change learning rate mid-training?

---

### Question 4: Weight Decay Trade-offs
**Question**: AdamW decouples weight decay from gradient updates.

**Consider**:
- Why does Adam's coupled weight decay behave inconsistently?
- In what scenarios would AdamW's consistent regularization matter most?
- How does weight decay interact with learning rate schedules?

**Real-world context**: The AdamW paper showed that proper decoupling leads to better generalization on ImageNet and language models.

---

### Question 5: Memory Requirements at Production Scale
**Question**: For training a GPT-scale model with 1 billion parameters, calculate the memory requirements:

**Calculate**:
- Parameters: 1 x 10^9
- Bytes per float32: 4
- Parameter memory: ___________GB

**With Adam optimizer (4x memory, 16 B/param)**:
- Total: ___________GB

**Real-world implications**:
- Why do we need multiple GPUs for training large models?
- Why is understanding optimizer memory crucial for choosing hardware?
- When would you choose SGD over Adam despite slower convergence?

---

### Bonus Question: Optimization Analysis

**Scenario**: You're training a deep neural network and observing the following:
- Loss decreases rapidly for first 1000 steps
- Loss plateaus between steps 1000-5000
- Loss suddenly increases at step 5000

**Questions**:
1. What might cause the plateau? How would momentum help?
2. What might cause the sudden increase? Is this an optimizer issue?
3. How would you diagnose whether this is a learning rate problem vs. data problem?
4. Would switching from Adam to AdamW help in this scenario?

**Key insight**: Optimization is not just about algorithms. It is about understanding the interaction between data, model architecture, and training dynamics.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Optimizers Update Weights

**What you built:** Optimization algorithms (SGD, Adam) that update neural network weights.

**Why it matters:** Gradients tell us which direction reduces the loss, but someone has to
actually move the weights. That's what optimizers do! SGD takes simple steps, while Adam
adapts the learning rate for each parameter, like having a personal trainer for each weight.

In the next module, you'll combine optimizers with a training loop to actually train networks!
"""

# %%
def demo_optimizers():
    """🎯 See optimizers update weights."""
    print("🎯 AHA MOMENT: Optimizers Update Weights")
    print("=" * 45)

    # Create a parameter with a gradient
    weight = Tensor(np.array([5.0]), requires_grad=True)

    # SGD takes a step in the opposite direction
    optimizer = SGD([weight], lr=0.5)
    # Set the gradient by hand so the demo needs no forward or backward pass
    weight.grad = np.array([1.0])  # Gradient pointing "uphill"

    print(f"Initial weight: {weight.data[0]:.2f}")
    print(f"Gradient:       {weight.grad[0]:.2f} (pointing uphill)")

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
- **Built SGD optimizer** with momentum for stable gradient descent and oscillation reduction
- **Implemented Adam optimizer** with adaptive learning rates and bias correction for different parameter scales
- **Created AdamW optimizer** with decoupled weight decay for proper regularization
- **Analyzed memory trade-offs**: 8 bytes per parameter for SGD (2x), 12 with momentum (3x), 16 for Adam/AdamW (4x)
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: counting weight plus gradient plus optimizer state, plain SGD needs 2x the parameter bytes, SGD with momentum 3x, and Adam/AdamW 4x, which is the 16 bytes per parameter in FP32
- **Adaptive learning**: Adam automatically adjusts step sizes per parameter for faster convergence
- **Weight decay coupling**: AdamW fixes Adam's inconsistent regularization by decoupling weight decay
- **State management**: Optimizer buffers must be checkpointed for training resume

### Ready for Next Steps
Your optimizer implementations enable sophisticated neural network training! With gradients from Module 06 and optimizers from Module 07, you're ready to build complete training loops.

Export with: `tito module complete 07`

**Next**: Module 08 will add training loops, learning rate scheduling, and checkpointing for complete end-to-end neural network training!
"""
