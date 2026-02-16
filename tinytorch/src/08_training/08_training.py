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
# Module 08: Training - Complete Learning Loops

Welcome to Module 08! You're about to build the complete training infrastructure that brings neural networks to life through end-to-end learning.

## 🔗 Prerequisites & Progress
**You've Built**: Tensors, activations, layers, losses, DataLoader, gradients, and optimizers
**You'll Build**: Complete training loops with checkpointing, scheduling, and gradient management
**You'll Enable**: Full model training pipeline for the MLP milestone

**Connection Map**:
```
DataLoader → Autograd → Optimizers → Training → Convolutions
(Module 05)  (Module 06)  (Module 07)  (Module 08)  (Module 09)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement a complete Trainer class with train/eval modes
2. Build learning rate scheduling and gradient clipping
3. Create checkpointing for model persistence
4. Test training loops with immediate validation
5. Understand gradient accumulation patterns

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/08_training/training_dev.py`
**Building Side:** Code exports to `tinytorch.core.training`

```python
# How to use this module:
from tinytorch.core.training import Trainer, CosineSchedule, clip_grad_norm
```

**Why this matters:**
- **Learning:** Complete training system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's training infrastructure with all training components together
- **Consistency:** All training operations and scheduling functionality in core.training
- **Integration:** Works seamlessly with optimizers and losses for complete learning pipelines
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.training
#| export

import numpy as np
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import sys
import os

# Import dependencies from other modules
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, AdamW

# Enable autograd for gradient tracking (required for training)
from tinytorch.core.autograd import enable_autograd
enable_autograd()

# Constants for learning rate scheduling defaults
DEFAULT_MAX_LR = 0.1  # Default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01  # Default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS = 100  # Default total epochs for learning rate schedule

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-07 must be working

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `pickle` (for checkpoint serialization)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` - Tensor class from Module 01
- `tinytorch.core.layers` - Linear layer from Module 03
- `tinytorch.core.losses` - Loss functions from Module 04
- `tinytorch.core.autograd` - Gradient tracking from Module 06
- `tinytorch.core.optimizers` - SGD, AdamW from Module 07

**Dependency Flow**:
```
Tensor → Layers → Losses → Autograd → Optimizers → Training
(01)     (03)     (04)     (06)       (07)         (08)
```

Students completing this module will have built a complete training
infrastructure that orchestrates all previous components.
"""

# %% [markdown]
"""
## 💡 Introduction: What is Training?

Training is where the magic happens - it's the process that transforms a randomly initialized neural network into an intelligent system that can solve problems. Think of training as teaching: you show the model examples, it makes predictions, you measure how wrong it is, and then you adjust its parameters to do better next time.

The training process follows a consistent pattern across all machine learning:

1. **Forward Pass**: Input flows through the model to produce predictions
2. **Loss Calculation**: Compare predictions to true answers
3. **Backward Pass**: Compute gradients showing how to improve
4. **Parameter Update**: Adjust model weights using an optimizer
5. **Repeat**: Continue until the model learns the pattern

But production training systems need much more than this basic loop. They need learning rate scheduling (starting fast, slowing down), gradient clipping (preventing exploding gradients), checkpointing (saving progress), and evaluation modes (testing without learning).

**What we're building today:**
- A complete `Trainer` class that orchestrates the entire learning process
- Learning rate scheduling that adapts during training
- Gradient clipping that prevents training instability
- Checkpointing system for saving and resuming training
- Train/eval modes for proper model behavior
"""

# %% [markdown]
"""
## 📐 Foundations: Mathematical Background

### Training Loop Mathematics

The core training loop implements gradient descent with sophisticated improvements:

**Basic Update Rule:**
```
θ(t+1) = θ(t) - η ∇L(θ(t))
```
Where θ are parameters, η is learning rate, and ∇L is the loss gradient.

**Learning Rate Scheduling:**
For cosine annealing over T epochs:
```
η(t) = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2
```

**Gradient Clipping:**
When ||∇L|| > max_norm, rescale:
```
∇L ← ∇L * max_norm / ||∇L||
```

**Gradient Accumulation:**
For effective batch size B_eff = accumulation_steps * B_actual:
```
∇L_accumulated = (1/accumulation_steps) * Σ ∇L_batch_i
```

### Train vs Eval Modes

Some layers behave differently during training vs inference:
- Some layers behave differently (e.g., dropout is active during training but disabled during inference)
- **Gradient computation**: Enabled during training, disabled during evaluation for efficiency

This mode switching is crucial for proper model behavior and performance.
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Training Infrastructure

Now let's implement the complete training system. We'll build each component step by step: learning rate scheduling, gradient utilities, and finally the complete Trainer class.

Each component will follow the pattern: **Explanation → Implementation → Test** so you understand what you're building before you build it.
"""

# %% [markdown]
"""
### 🏗️ Learning Rate Scheduling: Adaptive Training Speed

Learning rate scheduling is like adjusting your driving speed based on road conditions. You start fast on the highway (high learning rate for quick progress), then slow down in neighborhoods (low learning rate for fine-tuning).

#### Why Cosine Scheduling Works

Cosine annealing follows a smooth curve that provides:
- **Aggressive learning initially** - Fast convergence when far from optimum
- **Gradual slowdown** - Stable convergence as you approach the solution
- **Smooth transitions** - No sudden learning rate drops that shock the model

#### The Mathematics

Cosine annealing uses the cosine function to smoothly transition from max_lr to min_lr:

```
Learning Rate Schedule:

max_lr ┌─\
       │   \
       │     \
       │       \
       │         \
min_lr └───────────\────────
       0    25    50   75  100 epochs

Formula: lr = min_lr + (max_lr - min_lr) * (1 + cos(π * epoch / total_epochs)) / 2
```

This creates a natural learning curve that adapts training speed to the optimization landscape.
"""

# %% nbgrader={"grade": false, "grade_id": "scheduler", "locked": false, "solution": true}
#| export
class CosineSchedule:
    """
    Cosine annealing learning rate schedule.

    Starts at max_lr, decreases following a cosine curve to min_lr over T epochs.
    This provides aggressive learning initially, then fine-tuning at the end.

    TODO: Implement cosine annealing schedule

    APPROACH:
    1. Store max_lr, min_lr, and total_epochs
    2. In get_lr(), compute cosine factor: (1 + cos(π * epoch / total_epochs)) / 2
    3. Interpolate: min_lr + (max_lr - min_lr) * cosine_factor

    EXAMPLE:
    >>> schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
    >>> print(schedule.get_lr(0))    # Start: 0.1
    >>> print(schedule.get_lr(50))   # Middle: ~0.055
    >>> print(schedule.get_lr(100))  # End: 0.01

    HINT: Use np.cos() and np.pi for the cosine calculation
    """
    ### BEGIN SOLUTION
    def __init__(self, max_lr: float = DEFAULT_MAX_LR, min_lr: float = DEFAULT_MIN_LR, total_epochs: int = DEFAULT_TOTAL_EPOCHS):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: CosineSchedule

This test validates our learning rate scheduling implementation.

**What we're testing**: Cosine annealing produces correct learning rates
**Why it matters**: Proper scheduling often makes the difference between convergence and failure
**Expected**: Smooth decrease from max_lr to min_lr following cosine curve
"""

# %% nbgrader={"grade": true, "grade_id": "test_scheduler", "locked": true, "points": 10}
def test_unit_cosine_schedule():
    """🧪 Test CosineSchedule implementation."""
    print("🧪 Unit Test: CosineSchedule...")

    # Test basic schedule
    schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)

    # Test start, middle, and end
    lr_start = schedule.get_lr(0)
    lr_middle = schedule.get_lr(50)
    lr_end = schedule.get_lr(100)

    print(f"Learning rate at epoch 0: {lr_start:.4f}")
    print(f"Learning rate at epoch 50: {lr_middle:.4f}")
    print(f"Learning rate at epoch 100: {lr_end:.4f}")

    # Validate behavior
    assert abs(lr_start - 0.1) < 1e-6, f"Expected 0.1 at start, got {lr_start}"
    assert abs(lr_end - 0.01) < 1e-6, f"Expected 0.01 at end, got {lr_end}"
    assert 0.01 < lr_middle < 0.1, f"Middle LR should be between min and max, got {lr_middle}"

    # Test monotonic decrease in first half
    lr_quarter = schedule.get_lr(25)
    assert lr_quarter > lr_middle, "LR should decrease monotonically in first half"

    print("✅ CosineSchedule works correctly!")

if __name__ == "__main__":
    test_unit_cosine_schedule()

# %% [markdown]
"""
### 🏗️ Gradient Clipping: Preventing Training Explosions

Gradient clipping is like having a speed governor on your car - it prevents dangerous situations where gradients become so large they destroy training progress.

#### The Problem: Exploding Gradients

During training, gradients can sometimes become extremely large, causing:
- **Parameter updates that are too big** - Model jumps far from the optimal solution
- **Numerical instability** - Values become NaN or infinite
- **Training collapse** - Model performance suddenly degrades

#### The Solution: Global Norm Clipping

Instead of clipping each gradient individually, we compute the global norm across all parameters and scale uniformly:

```
Gradient Clipping Process:

1. Compute Global Norm:
   total_norm = √(sum of all gradient squares)

2. Check if Clipping Needed:
   if total_norm > max_norm:
       clip_coefficient = max_norm / total_norm

3. Scale All Gradients:
   for each gradient:
       gradient *= clip_coefficient

Visualization:
Original Gradients:  [100, 200, 50] → norm = 230
With max_norm=1.0:   [0.43, 0.87, 0.22] → norm = 1.0
```

This preserves the relative magnitudes while preventing explosion.
"""

# %% nbgrader={"grade": false, "grade_id": "gradient_clipping", "locked": false, "solution": true}
#| export

def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    """
    Clip gradients by global norm to prevent exploding gradients.

    This is crucial for training stability, especially with RNNs and deep networks.
    Instead of clipping each gradient individually, we compute the global norm
    across all parameters and scale uniformly if needed.

    TODO: Implement gradient clipping by global norm

    APPROACH:
    1. Compute total norm: sqrt(sum of squared gradients across all parameters)
    2. If total_norm > max_norm, compute clip_coef = max_norm / total_norm
    3. Scale all gradients by clip_coef: grad *= clip_coef
    4. Return the original norm for monitoring

    EXAMPLE:
    >>> params = [Tensor([1, 2, 3], requires_grad=True)]
    >>> params[0].grad = Tensor([10, 20, 30])  # Large gradients
    >>> original_norm = clip_grad_norm(params, max_norm=1.0)
    >>> print(f"Clipped norm: {np.linalg.norm(params[0].grad.data):.2f}")  # Should be ≤ 1.0

    HINTS:
    - Use np.linalg.norm() to compute norms
    - Only clip if total_norm > max_norm
    - Modify gradients in-place for efficiency
    """
    ### BEGIN SOLUTION
    if not parameters:
        return 0.0

    # Collect all gradients and compute global norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            # Handle both Tensor gradients and numpy array gradients
            if isinstance(param.grad, np.ndarray):
                grad_data = param.grad
            else:
                # Trust that Tensor has .data attribute
                grad_data = param.grad.data
            total_norm += np.sum(grad_data ** 2)

    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                # Handle both Tensor gradients and numpy array gradients
                if isinstance(param.grad, np.ndarray):
                    param.grad = param.grad * clip_coef
                else:
                    # Trust that Tensor has .data attribute
                    param.grad.data = param.grad.data * clip_coef

    return float(total_norm)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Gradient Clipping

This test validates our gradient clipping implementation.

**What we're testing**: Global norm clipping properly rescales large gradients
**Why it matters**: Prevents exploding gradients that can destroy training
**Expected**: Gradients scaled down when norm exceeds threshold
"""

# %% nbgrader={"grade": true, "grade_id": "test_clipping", "locked": true, "points": 10}
def test_unit_clip_grad_norm():
    """🧪 Test clip_grad_norm implementation."""
    print("🧪 Unit Test: Gradient Clipping...")

    # Use real Tensor from Module 01
    import sys
    # Tensor already imported at module level

    # Test case 1: Large gradients that need clipping
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param1.grad = np.array([3.0, 4.0])  # norm = 5.0

    param2 = Tensor([3.0, 4.0], requires_grad=True)
    param2.grad = np.array([6.0, 8.0])  # norm = 10.0

    params = [param1, param2]
    # Total norm = sqrt(5² + 10²) = sqrt(125) ≈ 11.18

    original_norm = clip_grad_norm(params, max_norm=1.0)

    # Check original norm was large
    assert original_norm > 1.0, f"Original norm should be > 1.0, got {original_norm}"

    # Check gradients were clipped
    new_norm = 0.0
    for param in params:
        if isinstance(param.grad, np.ndarray):
            grad_data = param.grad
        else:
            # Trust that Tensor has .data attribute
            grad_data = param.grad.data
        new_norm += np.sum(grad_data ** 2)
    new_norm = np.sqrt(new_norm)

    print(f"Original norm: {original_norm:.2f}")
    print(f"Clipped norm: {new_norm:.2f}")

    assert abs(new_norm - 1.0) < 1e-6, f"Clipped norm should be 1.0, got {new_norm}"

    # Test case 2: Small gradients that don't need clipping
    small_param = Tensor([1.0, 2.0], requires_grad=True)
    small_param.grad = np.array([0.1, 0.2])
    small_params = [small_param]
    original_small = clip_grad_norm(small_params, max_norm=1.0)

    assert original_small < 1.0, "Small gradients shouldn't be clipped"

    print("✅ Gradient clipping works correctly!")

if __name__ == "__main__":
    test_unit_clip_grad_norm()

# %% [markdown]
"""
### 🏗️ The Trainer Class: Orchestrating Complete Training

The Trainer class coordinates all the components you've built (model, optimizer, loss
function, scheduler) into a unified training system. You will implement each method
one at a time, testing as you go.

#### Trainer Architecture Overview

```
Trainer Components:
┌────────────────────────────────────────────────┐
│  Trainer                                       │
│  ├── __init__       → Store components, state  │
│  ├── train_epoch    → Forward/backward loop    │
│  ├── evaluate       → Forward only, metrics    │
│  ├── save_checkpoint → Serialize to disk       │
│  └── load_checkpoint → Restore from disk       │
│                                                │
│  Private helpers (provided):                   │
│  ├── _get_model_state / _set_model_state       │
│  ├── _get_optimizer_state / _set_optimizer_state│
│  └── _get_scheduler_state / _set_scheduler_state│
└────────────────────────────────────────────────┘
```

You will implement the five public methods. The private serialization helpers
are provided because they are pickle plumbing, not training concepts.
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-class-def", "locked": true, "solution": false}
#| export
class Trainer:
    """
    Complete training orchestrator for neural networks.

    Handles the full training lifecycle: forward pass, loss computation,
    backward pass, optimization, scheduling, checkpointing, and evaluation.

    This is the central class that brings together all the components
    you've built in previous modules.
    """

    # ── Private serialization helpers (provided, not student-implemented) ──
    # These are pickle plumbing for checkpoint save/load. The training
    # concepts you'll implement are in the public methods below.

    def _get_model_state(self):
        """Extract model parameters for checkpointing."""
        return {i: param.data.copy() for i, param in enumerate(self.model.parameters())}

    def _set_model_state(self, state):
        """Restore model parameters from checkpoint."""
        for i, param in enumerate(self.model.parameters()):
            if i in state:
                param.data = state[i].copy()

    def _get_optimizer_state(self):
        """Extract optimizer state for checkpointing."""
        state = {}
        state['lr'] = self.optimizer.lr
        if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
            momentum_state = self.optimizer.get_momentum_state()
            if momentum_state is not None:
                state['momentum_buffers'] = momentum_state
        return state

    def _set_optimizer_state(self, state):
        """Restore optimizer state from checkpoint."""
        if 'lr' in state:
            self.optimizer.lr = state['lr']
        if 'momentum_buffers' in state:
            if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
                self.optimizer.set_momentum_state(state['momentum_buffers'])

    def _get_scheduler_state(self):
        """Extract scheduler state for checkpointing."""
        if self.scheduler is None:
            return None
        return {
            'max_lr': getattr(self.scheduler, 'max_lr', None),
            'min_lr': getattr(self.scheduler, 'min_lr', None),
            'total_epochs': getattr(self.scheduler, 'total_epochs', None)
        }

    def _set_scheduler_state(self, state):
        """Restore scheduler state from checkpoint."""
        if state is None or self.scheduler is None:
            return
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)

# %% [markdown]
"""
### 🏗️ Trainer.__init__ - Setting Up the Training System

The constructor stores all training components and initializes tracking state.
Think of it as assembling the instruments before the orchestra plays.

```
Trainer State After __init__:
┌─────────────────────────────────────┐
│  Components:                        │
│    model       → Neural network     │
│    optimizer   → Parameter updater  │
│    loss_fn     → Error measure      │
│    scheduler   → LR adjuster (opt)  │
│    grad_clip_norm → Stability (opt) │
│                                     │
│  State:                             │
│    epoch = 0                        │
│    step = 0                         │
│    training_mode = True             │
│                                     │
│  History:                           │
│    train_loss = []                  │
│    eval_loss = []                   │
│    learning_rates = []              │
└─────────────────────────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-init", "locked": false, "solution": true}
#| export
def trainer_init(self, model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None):
    """
    Initialize trainer with model and training components.

    Args:
        model: Neural network to train (must have .forward() and .parameters())
        optimizer: Parameter update strategy (SGD, Adam, etc.)
        loss_fn: Loss function (CrossEntropy, MSE, etc.)
        scheduler: Optional learning rate scheduler (e.g., CosineSchedule)
        grad_clip_norm: Optional max gradient norm for clipping (float)

    TODO: Store all components and initialize training state

    APPROACH:
    1. Store model, optimizer, loss_fn, scheduler, and grad_clip_norm as attributes
    2. Initialize epoch=0, step=0, training_mode=True
    3. Create history dict with keys: 'train_loss', 'eval_loss', 'learning_rates'
       (each mapping to an empty list)

    EXAMPLE:
    >>> trainer = Trainer(model, optimizer, MSELoss())
    >>> print(trainer.epoch)     # 0
    >>> print(trainer.step)      # 0
    >>> print(trainer.history)   # {'train_loss': [], 'eval_loss': [], 'learning_rates': []}

    HINT: This is straightforward assignment. The key insight is WHAT state
    a training system needs to track across epochs.
    """
    ### BEGIN SOLUTION
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.scheduler = scheduler
    self.grad_clip_norm = grad_clip_norm

    # Training state
    self.epoch = 0
    self.step = 0
    self.training_mode = True

    # History tracking
    self.history = {
        'train_loss': [],
        'eval_loss': [],
        'learning_rates': []
    }
    ### END SOLUTION

Trainer.__init__ = trainer_init

# %% [markdown]
"""
### 🧪 Unit Test: Trainer.__init__

**What we're testing**: Trainer stores all components and initializes state correctly
**Why it matters**: Every training run depends on proper initialization
**Expected**: All attributes set, counters at zero, empty history
"""

# %% nbgrader={"grade": true, "grade_id": "test-trainer-init", "locked": true, "points": 5}
def test_unit_trainer_init():
    """🧪 Test Trainer.__init__ implementation."""
    print("🧪 Unit Test: Trainer.__init__...")

    class DummyModel:
        def __init__(self):
            self.training = True
        def forward(self, x):
            return x
        def parameters(self):
            return []

    model = DummyModel()
    optimizer = SGD([], lr=0.01)
    loss_fn = MSELoss()
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)

    trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)

    # Verify components stored
    assert trainer.model is model, "Model not stored"
    assert trainer.optimizer is optimizer, "Optimizer not stored"
    assert trainer.loss_fn is loss_fn, "Loss function not stored"
    assert trainer.scheduler is scheduler, "Scheduler not stored"
    assert trainer.grad_clip_norm == 1.0, "Grad clip norm not stored"

    # Verify state initialization
    assert trainer.epoch == 0, f"Expected epoch=0, got {trainer.epoch}"
    assert trainer.step == 0, f"Expected step=0, got {trainer.step}"
    assert trainer.training_mode is True, "Expected training_mode=True"

    # Verify history
    assert 'train_loss' in trainer.history, "Missing train_loss in history"
    assert 'eval_loss' in trainer.history, "Missing eval_loss in history"
    assert 'learning_rates' in trainer.history, "Missing learning_rates in history"
    assert len(trainer.history['train_loss']) == 0, "train_loss should be empty"

    # Test without optional args
    trainer2 = Trainer(model, optimizer, loss_fn)
    assert trainer2.scheduler is None, "Scheduler should default to None"
    assert trainer2.grad_clip_norm is None, "Grad clip should default to None"

    print("✅ Trainer.__init__ works correctly!")

if __name__ == "__main__":
    test_unit_trainer_init()

# %% [markdown]
"""
### 🏗️ Trainer.train_epoch - The Core Learning Loop

This is the heart of training. Each epoch iterates through the dataset, performing
the forward-backward-update cycle that drives learning.

```
Training Loop Flow:
┌─────────────────────────────────────────────┐
│  for each batch in dataloader:              │
│    outputs = model.forward(inputs)     →    │
│    loss = loss_fn(outputs, targets)    →    │
│    loss.backward(grad)                →    │
│    optimizer.step()                   →    │
│    optimizer.zero_grad()                    │
│  scheduler.get_lr(epoch)                    │
└─────────────────────────────────────────────┘
```

With gradient accumulation, the update step happens every N batches instead
of every batch, enabling larger effective batch sizes without more memory.

We'll build this in three pieces: process a single batch, perform an optimizer
update, then compose them into the full epoch loop.
"""

# %% [markdown]
"""
#### Step 1: Process a Single Batch

The inner loop body: run forward pass, compute loss, and run backward pass
with scaled gradients for accumulation.
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-process-batch", "locked": false, "solution": true}
#| export
def _trainer_process_batch(self, inputs, targets, accumulation_steps):
    """
    Process one batch: forward pass, loss computation, backward pass.

    Args:
        inputs: Input tensor for this batch
        targets: Target tensor for this batch
        accumulation_steps: Number of batches per optimizer update (for scaling)

    Returns:
        Scaled loss value (float) for accumulation tracking

    TODO: Implement the forward-backward cycle for a single batch

    APPROACH:
    1. Forward pass: model.forward(inputs)
    2. Compute loss: loss_fn.forward(outputs, targets)
    3. Scale loss by 1/accumulation_steps
    4. Backward pass with scaled gradient

    HINT: scaled_gradient = np.ones_like(loss.data) / accumulation_steps
    """
    ### BEGIN SOLUTION
    # Forward pass
    outputs = self.model.forward(inputs)
    loss = self.loss_fn.forward(outputs, targets)

    # Scale loss for accumulation
    scaled_loss = loss.data / accumulation_steps

    # Backward pass with scaled gradient
    scaled_gradient = np.ones_like(loss.data) / accumulation_steps
    loss.backward(scaled_gradient)

    return float(scaled_loss)
    ### END SOLUTION

Trainer._process_batch = _trainer_process_batch

# %% [markdown]
"""
#### Step 2: Perform Optimizer Update

When enough gradients have accumulated, clip them (if configured),
step the optimizer, and reset gradients for the next accumulation window.
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-optimizer-update", "locked": false, "solution": true}
#| export
def _trainer_optimizer_update(self):
    """
    Clip gradients (if enabled) and step the optimizer.

    TODO: Implement the gradient clip + optimizer step + zero_grad cycle

    APPROACH:
    1. If grad_clip_norm is set, call clip_grad_norm on model parameters
    2. Call optimizer.step() to update weights
    3. Call optimizer.zero_grad() to reset gradients
    """
    ### BEGIN SOLUTION
    if self.grad_clip_norm is not None:
        params = self.model.parameters()
        clip_grad_norm(params, self.grad_clip_norm)

    self.optimizer.step()
    self.optimizer.zero_grad()
    ### END SOLUTION

Trainer._optimizer_update = _trainer_optimizer_update

# %% [markdown]
"""
#### Step 3: Compose the Full Epoch Loop

Now combine `_process_batch` and `_optimizer_update` into the complete
training epoch with accumulation, scheduling, and history tracking.
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-train-epoch", "locked": false, "solution": true}
#| export
def trainer_train_epoch(self, dataloader, accumulation_steps=1):
    """
    Train for one epoch through the dataset.

    Args:
        dataloader: Iterable yielding (inputs, targets) batches
        accumulation_steps: Number of batches to accumulate before update

    Returns:
        Average loss for the epoch (float)

    TODO: Compose _process_batch and _optimizer_update into the epoch loop

    APPROACH:
    1. Set model.training = True and self.training_mode = True
    2. Loop over batches, calling self._process_batch for each
    3. Every accumulation_steps batches, call self._optimizer_update
    4. Handle remaining accumulated gradients after the loop
    5. Record average loss, update scheduler, increment epoch

    HINT: Check (batch_idx + 1) % accumulation_steps == 0 for update timing
    """
    ### BEGIN SOLUTION
    self.model.training = True
    self.training_mode = True

    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        accumulated_loss += self._process_batch(inputs, targets, accumulation_steps)

        # Update parameters every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            self._optimizer_update()
            total_loss += accumulated_loss
            accumulated_loss = 0.0
            num_batches += 1
            self.step += 1

    # Handle remaining accumulated gradients
    if accumulated_loss > 0:
        self._optimizer_update()
        total_loss += accumulated_loss
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    self.history['train_loss'].append(avg_loss)

    # Update scheduler
    if self.scheduler is not None:
        current_lr = self.scheduler.get_lr(self.epoch)
        self.optimizer.lr = current_lr
        self.history['learning_rates'].append(current_lr)

    self.epoch += 1
    return avg_loss
    ### END SOLUTION

Trainer.train_epoch = trainer_train_epoch

# %% [markdown]
"""
### 🧪 Unit Test: Trainer._process_batch

**What we're testing**: A single forward-backward pass returns a scaled loss value
**Why it matters**: This is the atomic unit of training — if one batch doesn't work, nothing will
**Expected**: Returns a float loss, model parameters have gradients after the call
"""

# %% nbgrader={"grade": true, "grade_id": "test-trainer-process-batch", "locked": true, "points": 5}
def test_unit_trainer_process_batch():
    """🧪 Test Trainer._process_batch implementation."""
    print("🧪 Unit Test: Trainer._process_batch...")

    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSELoss()
    trainer = Trainer(model, optimizer, loss_fn)

    inputs = Tensor([[1.0, 0.5]])
    targets = Tensor([[2.0]])

    scaled_loss = trainer._process_batch(inputs, targets, accumulation_steps=1)

    # Should return a float
    assert isinstance(scaled_loss, float), f"Expected float, got {type(scaled_loss)}"
    # Loss should be non-negative
    assert scaled_loss >= 0, f"Expected non-negative loss, got {scaled_loss}"

    print("✅ Trainer._process_batch works correctly!")

if __name__ == "__main__":
    test_unit_trainer_process_batch()

# %% [markdown]
"""
### 🧪 Unit Test: Trainer._optimizer_update

**What we're testing**: Gradient clipping + optimizer step + zero_grad cycle
**Why it matters**: Incorrect update logic causes training to diverge or stall
**Expected**: Parameters change after update, gradients are zeroed
"""

# %% nbgrader={"grade": true, "grade_id": "test-trainer-optimizer-update", "locked": true, "points": 5}
def test_unit_trainer_optimizer_update():
    """🧪 Test Trainer._optimizer_update implementation."""
    print("🧪 Unit Test: Trainer._optimizer_update...")

    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSELoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Do a forward-backward to create gradients
    inputs = Tensor([[1.0, 0.5]])
    targets = Tensor([[2.0]])
    trainer._process_batch(inputs, targets, accumulation_steps=1)

    # Record params before update
    params_before = [p.data.copy() for p in model.parameters()]

    # Perform update
    trainer._optimizer_update()

    # Parameters should have changed
    params_after = [p.data for p in model.parameters()]
    changed = any(not np.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Parameters should change after optimizer update"

    print("✅ Trainer._optimizer_update works correctly!")

if __name__ == "__main__":
    test_unit_trainer_optimizer_update()

# %% [markdown]
"""
### 🧪 Unit Test: Trainer.train_epoch

**What we're testing**: The core training loop processes batches and updates parameters
**Why it matters**: This is the single most important function in any ML system
**Expected**: Loss is computed, epoch increments, history records loss
"""

# %% nbgrader={"grade": true, "grade_id": "test-trainer-train-epoch", "locked": true, "points": 15}
def test_unit_trainer_train_epoch():
    """🧪 Test Trainer.train_epoch implementation."""
    print("🧪 Unit Test: Trainer.train_epoch...")

    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSELoss()

    trainer = Trainer(model, optimizer, loss_fn)

    dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
        (Tensor([[0.5, 1.0]]), Tensor([[1.5]]))
    ]

    # Train one epoch
    loss = trainer.train_epoch(dataloader)

    # Verify return type
    assert isinstance(loss, (float, np.floating)), f"Expected float loss, got {type(loss)}"

    # Verify epoch incremented
    assert trainer.epoch == 1, f"Expected epoch=1, got {trainer.epoch}"

    # Verify history recorded
    assert len(trainer.history['train_loss']) == 1, "Should have 1 loss recorded"

    # Verify model was in training mode
    assert trainer.training_mode is True, "Should be in training mode"

    # Train another epoch - loss should still be a valid number
    loss2 = trainer.train_epoch(dataloader)
    assert isinstance(loss2, (float, np.floating)), f"Second epoch loss should be float"
    assert trainer.epoch == 2, f"Expected epoch=2, got {trainer.epoch}"
    assert len(trainer.history['train_loss']) == 2, "Should have 2 losses recorded"

    # Test with scheduler
    model2 = SimpleModel()
    optimizer2 = SGD(model2.parameters(), lr=0.1)
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)
    trainer2 = Trainer(model2, optimizer2, loss_fn, scheduler=scheduler)

    trainer2.train_epoch(dataloader)
    assert len(trainer2.history['learning_rates']) == 1, "Should record LR with scheduler"

    # Test with gradient clipping
    model3 = SimpleModel()
    optimizer3 = SGD(model3.parameters(), lr=0.01)
    trainer3 = Trainer(model3, optimizer3, loss_fn, grad_clip_norm=0.5)

    loss3 = trainer3.train_epoch(dataloader)
    assert isinstance(loss3, (float, np.floating)), "Training with grad clip should work"

    print(f"  Epoch 1 loss: {loss:.4f}")
    print(f"  Epoch 2 loss: {loss2:.4f}")
    print("✅ Trainer.train_epoch works correctly!")

if __name__ == "__main__":
    test_unit_trainer_train_epoch()

# %% [markdown]
"""
### 🏗️ Trainer.evaluate - Measuring Model Performance

Evaluation runs the model in inference mode: forward pass only, no gradient
updates. This tells you how well the model generalizes to data it hasn't
trained on.

```
Evaluation Flow:
┌──────────────────────────────────────┐
│  model.training = False              │
│                                      │
│  for each batch in dataloader:       │
│    outputs = model.forward(inputs)   │
│    loss = loss_fn(outputs, targets)  │
│    accumulate loss + accuracy        │
│                                      │
│  return avg_loss, accuracy           │
└──────────────────────────────────────┘
```

Key difference from training: no backward pass, no optimizer step,
no gradient clipping.
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-evaluate", "locked": false, "solution": true}
#| export
def trainer_evaluate(self, dataloader):
    """
    Evaluate model on dataset without updating parameters.

    Args:
        dataloader: Iterable yielding (inputs, targets) batches

    Returns:
        Tuple of (average_loss, accuracy)

    TODO: Implement evaluation loop (forward pass only, no gradient updates)

    APPROACH:
    1. Set model.training = False and self.training_mode = False
    2. For each batch: forward pass only, accumulate loss
    3. For classification: compute accuracy from argmax predictions
    4. Record average loss in self.history['eval_loss']
    5. Return (avg_loss, accuracy)

    EXAMPLE:
    >>> eval_loss, accuracy = trainer.evaluate(test_data)
    >>> print(f"Eval loss: {eval_loss:.4f}, Accuracy: {accuracy:.2%}")

    HINTS:
    - For multi-class: predictions = np.argmax(outputs.data, axis=1)
    - Handle both integer targets and one-hot targets
    - accuracy = correct / total if total > 0 else 0.0
    """
    ### BEGIN SOLUTION
    self.model.training = False
    self.training_mode = False

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for inputs, targets in dataloader:
        # Forward pass only
        outputs = self.model.forward(inputs)
        loss = self.loss_fn.forward(outputs, targets)

        total_loss += loss.data
        num_batches += 1

        # Calculate accuracy (for classification)
        if len(outputs.data.shape) > 1:  # Multi-class
            predictions = np.argmax(outputs.data, axis=1)
            if len(targets.data.shape) == 1:  # Integer targets
                correct += np.sum(predictions == targets.data)
            else:  # One-hot targets
                correct += np.sum(predictions == np.argmax(targets.data, axis=1))
            total += len(predictions)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    self.history['eval_loss'].append(avg_loss)

    return avg_loss, accuracy
    ### END SOLUTION

Trainer.evaluate = trainer_evaluate

# %% [markdown]
"""
### 🧪 Unit Test: Trainer.evaluate

**What we're testing**: Evaluation computes loss and accuracy without modifying the model
**Why it matters**: Proper evaluation prevents overfitting and validates generalization
**Expected**: Returns valid loss and accuracy, model set to eval mode
"""

# %% nbgrader={"grade": true, "grade_id": "test-trainer-evaluate", "locked": true, "points": 10}
def test_unit_trainer_evaluate():
    """🧪 Test Trainer.evaluate implementation."""
    print("🧪 Unit Test: Trainer.evaluate...")

    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSELoss()

    trainer = Trainer(model, optimizer, loss_fn)

    dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
        (Tensor([[0.5, 1.0]]), Tensor([[1.5]]))
    ]

    eval_loss, accuracy = trainer.evaluate(dataloader)

    # Verify return types
    assert isinstance(eval_loss, (float, np.floating)), f"Expected float eval_loss, got {type(eval_loss)}"
    assert isinstance(accuracy, (float, np.floating)), f"Expected float accuracy, got {type(accuracy)}"

    # Verify model was set to eval mode
    assert trainer.training_mode is False, "Should be in eval mode after evaluate()"
    assert model.training is False, "Model should be in eval mode"

    # Verify history recorded
    assert len(trainer.history['eval_loss']) == 1, "Should have 1 eval loss recorded"

    # Verify loss is a reasonable number (not NaN or inf)
    assert np.isfinite(eval_loss), f"Eval loss should be finite, got {eval_loss}"

    print(f"  Eval loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")
    print("✅ Trainer.evaluate works correctly!")

if __name__ == "__main__":
    test_unit_trainer_evaluate()

# %% [markdown]
"""
### 🏗️ Trainer.save_checkpoint - Persisting Training State

Checkpointing saves everything needed to resume training later: model weights,
optimizer state, scheduler state, epoch count, and training history. This is
essential for long training runs that may be interrupted.

```
Checkpoint Contents:
┌───────────────────────────────────┐
│  checkpoint.pkl                   │
│  ├── epoch: 42                    │
│  ├── step: 1680                   │
│  ├── model_state: {weights...}    │
│  ├── optimizer_state: {lr, mom..} │
│  ├── scheduler_state: {lr range}  │
│  ├── history: {losses, lrs...}    │
│  └── training_mode: True          │
└───────────────────────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-save-checkpoint", "locked": false, "solution": true}
#| export
def trainer_save_checkpoint(self, path: str):
    """
    Save complete training state for resumption.

    Args:
        path: File path to save checkpoint (.pkl)

    TODO: Serialize all training state to disk using pickle

    APPROACH:
    1. Build a checkpoint dict with keys: epoch, step, model_state,
       optimizer_state, scheduler_state, history, training_mode
    2. Use self._get_model_state(), self._get_optimizer_state(),
       self._get_scheduler_state() to extract component states
    3. Create parent directory if needed: Path(path).parent.mkdir(parents=True, exist_ok=True)
    4. Write with pickle.dump()

    EXAMPLE:
    >>> trainer.save_checkpoint('/tmp/checkpoint.pkl')
    >>> # Later: trainer.load_checkpoint('/tmp/checkpoint.pkl')

    HINT: The private _get_*_state() helpers are already provided.
    """
    ### BEGIN SOLUTION
    checkpoint = {
        'epoch': self.epoch,
        'step': self.step,
        'model_state': self._get_model_state(),
        'optimizer_state': self._get_optimizer_state(),
        'scheduler_state': self._get_scheduler_state(),
        'history': self.history,
        'training_mode': self.training_mode
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    ### END SOLUTION

Trainer.save_checkpoint = trainer_save_checkpoint

# %% [markdown]
"""
### 🧪 Unit Test: Trainer.save_checkpoint

**What we're testing**: Checkpoint file is created and contains all required state
**Why it matters**: Lost training progress on a long run is costly
**Expected**: File created on disk with correct contents
"""

# %% nbgrader={"grade": true, "grade_id": "test-trainer-save-checkpoint", "locked": true, "points": 5}
def test_unit_trainer_save_checkpoint():
    """🧪 Test Trainer.save_checkpoint implementation."""
    print("🧪 Unit Test: Trainer.save_checkpoint...")

    class DummyModel:
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = DummyModel()
    optimizer = SGD(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, MSELoss())

    # Set some state to verify it persists
    trainer.epoch = 5
    trainer.step = 100
    trainer.history['train_loss'].append(0.5)

    checkpoint_path = "/tmp/test_save_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    # Verify file exists
    assert os.path.exists(checkpoint_path), "Checkpoint file should exist"

    # Verify contents
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    assert checkpoint['epoch'] == 5, f"Expected epoch=5, got {checkpoint['epoch']}"
    assert checkpoint['step'] == 100, f"Expected step=100, got {checkpoint['step']}"
    assert 'model_state' in checkpoint, "Missing model_state"
    assert 'optimizer_state' in checkpoint, "Missing optimizer_state"
    assert 'history' in checkpoint, "Missing history"
    assert len(checkpoint['history']['train_loss']) == 1, "History should have 1 entry"

    # Clean up
    os.remove(checkpoint_path)

    print("✅ Trainer.save_checkpoint works correctly!")

if __name__ == "__main__":
    test_unit_trainer_save_checkpoint()

# %% [markdown]
"""
### 🏗️ Trainer.load_checkpoint - Resuming Training

Loading a checkpoint restores the exact training state so you can continue
where you left off. This means restoring epoch count, optimizer state
(including momentum buffers), and the full training history.

```
Load Flow:
checkpoint.pkl ──→ pickle.load() ──→ restore epoch, step
                                  ──→ restore model weights
                                  ──→ restore optimizer state
                                  ──→ restore scheduler state
                                  ──→ restore history
```
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-load-checkpoint", "locked": false, "solution": true}
#| export
def trainer_load_checkpoint(self, path: str):
    """
    Load training state from checkpoint.

    Args:
        path: File path to load checkpoint from (.pkl)

    TODO: Deserialize training state from disk and restore all components

    APPROACH:
    1. Open and unpickle the checkpoint file
    2. Restore self.epoch, self.step, self.history, self.training_mode
    3. Call self._set_model_state() if 'model_state' in checkpoint
    4. Call self._set_optimizer_state() if 'optimizer_state' in checkpoint
    5. Call self._set_scheduler_state() if 'scheduler_state' in checkpoint

    EXAMPLE:
    >>> trainer.save_checkpoint('/tmp/checkpoint.pkl')
    >>> trainer.epoch = 999  # Some change
    >>> trainer.load_checkpoint('/tmp/checkpoint.pkl')
    >>> print(trainer.epoch)  # Restored to original value

    HINT: The private _set_*_state() helpers are already provided.
    """
    ### BEGIN SOLUTION
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    self.epoch = checkpoint['epoch']
    self.step = checkpoint['step']
    self.history = checkpoint['history']
    self.training_mode = checkpoint['training_mode']

    # Restore states
    if 'model_state' in checkpoint:
        self._set_model_state(checkpoint['model_state'])
    if 'optimizer_state' in checkpoint:
        self._set_optimizer_state(checkpoint['optimizer_state'])
    if 'scheduler_state' in checkpoint:
        self._set_scheduler_state(checkpoint['scheduler_state'])
    ### END SOLUTION

Trainer.load_checkpoint = trainer_load_checkpoint

# %% [markdown]
"""
### 🧪 Unit Test: Trainer.load_checkpoint

**What we're testing**: Checkpoint loading restores exact training state
**Why it matters**: Resuming training must produce the same result as uninterrupted training
**Expected**: All state (epoch, step, model weights, history) restored correctly
"""

# %% nbgrader={"grade": true, "grade_id": "test-trainer-load-checkpoint", "locked": true, "points": 5}
def test_unit_trainer_load_checkpoint():
    """🧪 Test Trainer.load_checkpoint implementation."""
    print("🧪 Unit Test: Trainer.load_checkpoint...")

    class DummyModel:
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = DummyModel()
    optimizer = SGD(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, MSELoss())

    # Set distinctive state
    trainer.epoch = 7
    trainer.step = 200
    trainer.history['train_loss'].extend([0.9, 0.7, 0.5])
    trainer.training_mode = False

    # Save original model weights for comparison
    original_weights = model.parameters()[0].data.copy()

    checkpoint_path = "/tmp/test_load_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    # Corrupt state
    trainer.epoch = 999
    trainer.step = 0
    trainer.history = {'train_loss': [], 'eval_loss': [], 'learning_rates': []}
    trainer.training_mode = True

    # Restore
    trainer.load_checkpoint(checkpoint_path)

    # Verify restoration
    assert trainer.epoch == 7, f"Expected epoch=7, got {trainer.epoch}"
    assert trainer.step == 200, f"Expected step=200, got {trainer.step}"
    assert trainer.training_mode is False, "training_mode should be restored to False"
    assert len(trainer.history['train_loss']) == 3, "History should have 3 entries"
    assert trainer.history['train_loss'] == [0.9, 0.7, 0.5], "History values should match"

    # Verify model weights restored
    restored_weights = model.parameters()[0].data
    assert np.allclose(restored_weights, original_weights), "Model weights should be restored"

    # Clean up
    os.remove(checkpoint_path)

    print("✅ Trainer.load_checkpoint works correctly!")

if __name__ == "__main__":
    test_unit_trainer_load_checkpoint()

# %% [markdown]
"""
## 🔧 Integration: Complete Training Example

Now let's create a complete training example that demonstrates how all the components work together. This integration shows the full power of our training infrastructure.

### Building a Complete Training Pipeline

```
Training Pipeline Architecture:

Model Creation
      ↓
Optimizer Setup (with parameters)
      ↓
Loss Function Selection
      ↓
Learning Rate Scheduler
      ↓
Trainer Initialization
      ↓
Training Loop (multiple epochs)
      ↓
Evaluation & Checkpointing
```

This example brings together everything you've built in Modules 01-07.
"""

# %% nbgrader={"grade": false, "grade_id": "integration_example", "solution": true}
def demonstrate_complete_training_pipeline():
    """
    Complete end-to-end training example using all components.

    This demonstrates how Trainer, scheduler, gradient clipping,
    and checkpointing work together in a real training scenario.
    """
    print("🏗️ Building Complete Training Pipeline...")
    print("=" * 60)

    # Step 1: Create model using REAL Linear layer
    class SimpleNN:
        def __init__(self):
            self.layer1 = Linear(3, 5)
            self.layer2 = Linear(5, 2)
            self.training = True

        def forward(self, x):
            x = self.layer1.forward(x)
            # Simple ReLU-like activation (max with 0)
            x = Tensor(np.maximum(0, x.data))
            x = self.layer2.forward(x)
            return x

        def parameters(self):
            return self.layer1.parameters() + self.layer2.parameters()

    print("✓ Model created: 3 → 5 → 2 network")

    # Step 2: Create optimizer
    model = SimpleNN()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    print("✓ Optimizer: SGD with momentum")

    # Step 3: Create loss function
    loss_fn = MSELoss()
    print("✓ Loss function: MSE")

    # Step 4: Create scheduler
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=5)
    print("✓ Scheduler: Cosine annealing (0.1 → 0.001)")

    # Step 5: Create trainer with gradient clipping
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=1.0
    )
    print("✓ Trainer initialized with gradient clipping")

    # Step 6: Create synthetic training data
    train_data = [
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2))),
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2))),
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2)))
    ]
    print("✓ Training data: 3 batches of 4 samples")

    # Step 7: Train for multiple epochs
    print("\n🚀 Starting Training...")
    print("-" * 60)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Learning Rate':<15}")
    print("-" * 60)

    for epoch in range(3):
        loss = trainer.train_epoch(train_data)
        lr = scheduler.get_lr(epoch)
        print(f"{epoch:<8} {loss:<12.6f} {lr:<15.6f}")

    # Step 8: Save checkpoint
    checkpoint_path = "/tmp/training_example_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved: {checkpoint_path}")

    # Step 9: Evaluate
    eval_loss, accuracy = trainer.evaluate(train_data)
    print(f"✓ Evaluation - Loss: {eval_loss:.6f}, Accuracy: {accuracy:.6f}")

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n" + "=" * 60)
    print("✅ Complete training pipeline executed successfully!")
    print("\n💡 This pipeline demonstrates:")
    print("   • Model → Optimizer → Loss → Scheduler → Trainer integration")
    print("   • Training loop with scheduling and gradient clipping")
    print("   • Checkpointing for training persistence")
    print("   • Evaluation mode for model assessment")

# %% [markdown]
"""
## 📊 Systems Analysis: Training Performance and Memory

Training systems have significant resource requirements. Understanding memory usage, checkpoint sizes, and training overhead helps optimize production ML pipelines.

### Training Memory Breakdown

```
Training Memory Requirements:

Forward Pass Memory:
┌─────────────────┐
│ Activations     │ ← Stored for backward pass
├─────────────────┤
│ Model Params    │ ← Network weights
└─────────────────┘

Backward Pass Memory:
┌─────────────────┐
│ Gradients       │ ← Same size as params
├─────────────────┤
│ Optimizer State │ ← 2-3× params (momentum, Adam buffers)
└─────────────────┘

Checkpoint Memory:
┌─────────────────┐
│ Model State     │ ← Full parameter snapshot
├─────────────────┤
│ Optimizer State │ ← All momentum/Adam buffers
├─────────────────┤
│ Training Meta   │ ← Epoch, history, scheduler
└─────────────────┘

Total Training Memory ≈ 5-6× Model Parameters
```

### Key Systems Insights

**Gradient Accumulation Trade-off:**
- Effective batch size = accumulation_steps × actual_batch_size
- Memory: Fixed (only 1 batch in memory at a time)
- Time: Increases linearly with accumulation steps
- Use case: Large models that don't fit with desired batch size

**Checkpoint Size:**
- Base model: 1× parameters
- With optimizer (Adam): ~3× parameters
- With full history: Additional metadata
- Compression: Pickle overhead ~10-20%
"""

# %%
def analyze_training_memory():
    """📊 Analyze memory overhead of training components."""
    print("📊 Analyzing Training Memory Overhead...")

    # Create models of different sizes
    model_sizes = [
        ("Small", 100),    # 100 parameters
        ("Medium", 1000),  # 1K parameters
        ("Large", 10000)   # 10K parameters
    ]

    print("\nTraining Memory Analysis:")
    print("=" * 90)
    print(f"{'Model':<10} {'Params':<10} {'Gradients':<12} {'SGD State':<12} {'Adam State':<12} {'Total':<10}")
    print("-" * 90)

    for name, param_count in model_sizes:
        # Base memory: parameters
        param_memory = param_count * 4  # 4 bytes per float32

        # Gradients: same as parameters
        grad_memory = param_count * 4

        # SGD optimizer state: momentum buffer
        sgd_memory = param_count * 4

        # Adam optimizer state: 2 buffers (m and v)
        adam_memory = param_count * 2 * 4

        # Total with Adam (worst case)
        total_memory = param_memory + grad_memory + adam_memory

        # Convert to human-readable
        def format_memory(bytes):
            if bytes < 1024:
                return f"{bytes}B"
            elif bytes < 1024 * 1024:
                return f"{bytes/1024:.1f}KB"
            else:
                return f"{bytes/(1024*1024):.1f}MB"

        print(f"{name:<10} {format_memory(param_memory):<10} "
              f"{format_memory(grad_memory):<12} {format_memory(sgd_memory):<12} "
              f"{format_memory(adam_memory):<12} {format_memory(total_memory):<10}")

    print("\n💡 Key Insights:")
    print("- Training memory = Parameters + Gradients + Optimizer State")
    print("- SGD: 3× parameter memory (params + grads + momentum)")
    print("- Adam: 4× parameter memory (params + grads + 2 moment buffers)")
    print("- Gradient accumulation reduces memory but increases training time")

# %%
def analyze_checkpoint_overhead():
    """📊 Analyze checkpoint size and overhead."""
    print("\n📊 Analyzing Checkpoint Overhead...")

    # Create a simple model
    class TinyModel:
        def __init__(self, size):
            self.layer = Linear(size, size)
            self.training = True

        def forward(self, x):
            return self.layer.forward(x)

        def parameters(self):
            return self.layer.parameters()

    sizes = [10, 50, 100]

    print("\nCheckpoint Size Analysis:")
    print("=" * 70)
    print(f"{'Model Size':<12} {'Raw Params':<15} {'Checkpoint':<15} {'Overhead':<10}")
    print("-" * 70)

    import pickle
    import sys

    for size in sizes:
        # Create model and trainer
        model = TinyModel(size)
        optimizer = SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, MSELoss())

        # Estimate raw parameter size
        param_count = size * size + size  # W + b
        raw_size = param_count * 4  # 4 bytes per float32

        # Create checkpoint and measure size
        checkpoint_path = f"/tmp/checkpoint_test_{size}.pkl"
        trainer.save_checkpoint(checkpoint_path)

        import os
        checkpoint_size = os.path.getsize(checkpoint_path)
        overhead = (checkpoint_size / raw_size - 1) * 100

        # Clean up
        os.remove(checkpoint_path)

        def format_size(bytes):
            if bytes < 1024:
                return f"{bytes}B"
            return f"{bytes/1024:.1f}KB"

        print(f"{size}×{size:<8} {format_size(raw_size):<15} "
              f"{format_size(checkpoint_size):<15} {overhead:.1f}%")

    print("\n💡 Key Insights:")
    print("- Checkpoints include model state + optimizer state + training metadata")
    print("- Pickle serialization adds 10-30% overhead")
    print("- Adam optimizer doubles checkpoint size vs SGD")
    print("- Use checkpoint frequency wisely in production (memory vs fault tolerance)")

# Run the systems analysis
if __name__ == "__main__":
    analyze_training_memory()
    analyze_checkpoint_overhead()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test_module", "locked": true, "points": 20}
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
    test_unit_cosine_schedule()
    test_unit_clip_grad_norm()
    test_unit_trainer_init()
    test_unit_trainer_process_batch()
    test_unit_trainer_optimizer_update()
    test_unit_trainer_train_epoch()
    test_unit_trainer_evaluate()
    test_unit_trainer_save_checkpoint()
    test_unit_trainer_load_checkpoint()

    print("\nRunning integration scenarios...")

    # Test complete training pipeline integration with REAL components
    print("🧪 Integration Test: Complete Training Pipeline...")

    # Use REAL components from previous modules (already imported at module level)

    # Create a simple model using REAL Linear layer
    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)  # Real Linear from Module 03
            self.training = True

        def forward(self, x):
            return self.layer.forward(x)

        def parameters(self):
            return self.layer.parameters()

    # Create integrated system with REAL components
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)  # Real SGD from Module 07
    loss_fn = MSELoss()  # Real MSELoss from Module 04
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=0.5
    )

    # Test data using REAL Tensors
    data = [
        (Tensor([[1.0, 0.5]]), Tensor([[0.8]])),
        (Tensor([[0.5, 1.0]]), Tensor([[0.2]]))
    ]

    # Test training
    initial_loss = trainer.train_epoch(data)
    assert isinstance(initial_loss, (float, np.floating)), "Training should return float loss"
    assert trainer.epoch == 1, "Epoch should increment"

    # Test evaluation
    eval_loss, accuracy = trainer.evaluate(data)
    assert isinstance(eval_loss, (float, np.floating)), "Evaluation should return float loss"
    assert isinstance(accuracy, (float, np.floating)), "Evaluation should return float accuracy"

    # Test scheduling
    lr_epoch_0 = scheduler.get_lr(0)
    lr_epoch_1 = scheduler.get_lr(1)
    assert lr_epoch_0 > lr_epoch_1, "Learning rate should decrease"

    # Test gradient clipping with large gradients using real Tensor
    large_param = Tensor([1.0, 2.0], requires_grad=True)
    large_param.grad = np.array([100.0, 200.0])
    large_params = [large_param]

    original_norm = clip_grad_norm(large_params, max_norm=1.0)
    assert original_norm > 1.0, "Original norm should be large"

    if isinstance(large_params[0].grad, np.ndarray):
        grad_data = large_params[0].grad
    else:
        # Trust that Tensor has .data attribute
        grad_data = large_params[0].grad.data
    new_norm = np.linalg.norm(grad_data)
    assert abs(new_norm - 1.0) < 1e-6, "Clipped norm should equal max_norm"

    # Test checkpointing
    checkpoint_path = "/tmp/integration_test_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    original_epoch = trainer.epoch
    trainer.epoch = 999
    trainer.load_checkpoint(checkpoint_path)

    assert trainer.epoch == original_epoch, "Checkpoint should restore state"

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("✅ End-to-end training pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 08")

if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of training systems and their implications:

### 1. Memory Trade-offs
**Question**: If you have a model with 1 million parameters and use Adam optimizer, what's the total training memory required?

**Consider**:
- Parameters: 1M parameters at 4 bytes each
- Gradients: Same size as parameters
- Adam state: 2 buffers (momentum and variance) per parameter
- How does gradient accumulation help when you want batch_size=128 but only batch_size=32 fits?

---

### 2. Gradient Clipping
**Question**: Why do we clip gradients by *global norm* rather than clipping each gradient independently?

**Consider**:
- What happens if each parameter's gradient is clipped to 1.0 separately?
- How does global norm preserve the gradient direction?
- What does it signal when gradients consistently exceed max_norm?

---

### 3. Learning Rate Scheduling
**Question**: Why does cosine annealing start with high learning rate and end with low learning rate?

**Consider**:
- What phase of optimization benefits from large vs. small updates?
- Compare: fixed lr=0.1 vs cosine schedule (0.1 to 0.01)
- When might a fixed learning rate actually be better?

---

### 4. Checkpointing Strategy
**Question**: You're training for 100 epochs with 1GB checkpoints. How often should you save?

**Consider**:
- Disk space: 100 checkpoints = 100GB
- Recovery time: If training crashes at epoch 95, how much work is lost?
- What information MUST be in a checkpoint to resume training exactly?

---

### 5. Train vs Eval Modes
**Question**: Why is it crucial to set model.training = False during evaluation?

**Consider**:
- What layers might behave differently in training vs eval? (Think about dropout.)
- What would happen if you forgot to zero gradients between training steps?
- How does gradient accumulation intentionally exploit not zeroing?

**The answers reveal deep understanding of training systems!**
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Training Just Works

**What you built:** A complete training infrastructure with Trainer, schedulers, and checkpoints.

**Why it matters:** You've assembled all the pieces: tensors → layers → losses → autograd →
optimizers → training loop. This is the complete ML training pipeline! The Trainer orchestrates
forward pass, loss computation, backward pass, and weight updates—just like PyTorch Lightning.

In the milestones, you'll use this training infrastructure to train real models on real data!
"""

# %%
def demo_training():
    """🎯 See the training loop in action."""
    print("🎯 AHA MOMENT: Training Just Works")
    print("=" * 45)

    # Simple linear regression: learn y = 2x + 1
    np.random.seed(42)
    X = Tensor(np.random.randn(20, 1))
    y = Tensor(X.data * 2 + 1)  # True relationship

    # Simple model: one weight, one bias
    w = Tensor(np.array([[0.0]]), requires_grad=True)
    b = Tensor(np.array([0.0]), requires_grad=True)

    optimizer = SGD([w, b], lr=0.1)
    loss_fn = MSELoss()

    print("Learning y = 2x + 1:")
    for epoch in range(5):
        # Forward
        pred = X.matmul(w) + b
        loss = loss_fn(pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Epoch {epoch+1}: w={w.data[0,0]:.2f}, b={b.data[0]:.2f}, loss={float(loss.data):.4f}")

    print(f"\nLearned: y = {w.data[0,0]:.1f}x + {b.data[0]:.1f}")
    print("Target:  y = 2.0x + 1.0")

    print("\n✨ Your training loop learned the pattern!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_training()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Training

Congratulations! You've built the complete training infrastructure that orchestrates neural network learning!

### Key Accomplishments
- **Built a complete Trainer class** with training/evaluation loops and gradient management
- **Implemented CosineSchedule** for adaptive learning rate management
- **Created clip_grad_norm** for training stability via global norm clipping
- **Added checkpointing** for training persistence and resumption
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: Training requires 4-6x model size (params + grads + optimizer state)
- **Gradient accumulation**: Trades time for memory, enabling larger effective batch sizes
- **Checkpoint overhead**: Pickle adds 10-30% overhead, optimizer state doubles size
- **Scheduling behavior**: Cosine annealing balances aggressive initial learning with fine-tuning

Export with: `tito module complete 08`

**Next**: Module 09 will add convolution operations for spatial neural network processing!
"""
