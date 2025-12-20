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

# %% nbgrader={"grade": false, "grade_id": "imports", "locked": false, "solution": false}
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

# Constants for learning rate scheduling defaults
DEFAULT_MAX_LR = 0.1  # Default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01  # Default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS = 100  # Default total epochs for learning rate schedule

# %% [markdown]
"""
## 💡 Introduction - What is Training?

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
## 📐 Foundations - Mathematical Background

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

Many layers behave differently during training vs inference:
- **Dropout**: Active during training, disabled during evaluation
- **BatchNorm**: Updates statistics during training, uses fixed statistics during evaluation
- **Gradient computation**: Enabled during training, disabled during evaluation for efficiency

This mode switching is crucial for proper model behavior and performance.
"""

# %% [markdown]
"""
## 🏗️ Implementation - Building Training Infrastructure

Now let's implement the complete training system. We'll build each component step by step: learning rate scheduling, gradient utilities, and finally the complete Trainer class.

Each component will follow the pattern: **Explanation → Implementation → Test** so you understand what you're building before you build it.
"""

# %% [markdown]
"""
### Learning Rate Scheduling - Adaptive Training Speed

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
    """🔬 Test CosineSchedule implementation."""
    print("🔬 Unit Test: CosineSchedule...")

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
### Gradient Clipping - Preventing Training Explosions

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
    """🔬 Test clip_grad_norm implementation."""
    print("🔬 Unit Test: Gradient Clipping...")

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
### The Trainer Class - Orchestrating Complete Training

The Trainer class is like a conductor orchestrating a symphony - it coordinates all the components (model, optimizer, loss function, scheduler) to create beautiful music (successful training).

#### Training Loop Architecture

The training loop follows a consistent pattern across all machine learning:

```
Training Loop Structure:

for epoch in range(num_epochs):
    ┌─────────────────── TRAINING PHASE ───────────────────┐
    │                                                       │
    │  for batch in dataloader:                             │
    │      ┌─── Forward Pass ───────┐                       │
    │      │ 1. input → model       │                       │
    │      │ 2. predictions         │                       │
    │      └────────────────────────┘                       │
    │               ↓                                       │
    │      ┌─── Loss Computation ───┐                       │
    │      │ 3. loss = loss_fn()    │                       │
    │      └────────────────────────┘                       │
    │               ↓                                       │
    │      ┌─── Backward Pass ──────┐                       │
    │      │ 4. loss.backward()     │                       │
    │      │ 5. gradients           │                       │
    │      └────────────────────────┘                       │
    │               ↓                                       │
    │      ┌─── Parameter Update ───┐                       │
    │      │ 6. optimizer.step()    │                       │
    │      │ 7. zero gradients      │                       │
    │      └────────────────────────┘                       │
    └───────────────────────────────────────────────────────┘
             ↓
    ┌─── Learning Rate Update ───┐
    │ 8. scheduler.step()        │
    └────────────────────────────┘
```

#### Key Features

- **Train/Eval Modes**: Different behavior during training vs evaluation
- **Gradient Accumulation**: Effective larger batch sizes with limited memory
- **Checkpointing**: Save/resume training state for long experiments
- **Progress Tracking**: Monitor loss, learning rate, and other metrics
"""

# %% nbgrader={"grade": false, "grade_id": "trainer_class", "locked": false, "solution": true}
#| export
class Trainer:
    """
    Complete training orchestrator for neural networks.

    Handles the full training lifecycle: forward pass, loss computation,
    backward pass, optimization, scheduling, checkpointing, and evaluation.

    This is the central class that brings together all the components
    you've built in previous modules.

    TODO: Implement complete Trainer class

    APPROACH:
    1. __init__(): Store model, optimizer, loss_fn, scheduler, and grad_clip_norm
    2. train_epoch(): Loop through dataloader, forward → loss → backward → step
    3. evaluate(): Similar loop but set model.training=False, no grad updates
    4. save/load_checkpoint(): Use pickle to persist/restore all training state

    EXAMPLE:
    >>> model = SimpleModel()
    >>> optimizer = SGD(model.parameters(), lr=0.01)
    >>> trainer = Trainer(model, optimizer, MSELoss())
    >>> # Training data: list of (input, target) tuples
    >>> data = [(Tensor([[1.0]]), Tensor([[2.0]]))]
    >>> loss = trainer.train_epoch(data)
    >>> eval_loss, accuracy = trainer.evaluate(data)
    >>> trainer.save_checkpoint('/tmp/checkpoint.pkl')

    HINTS:
    - In train_epoch(), set model.training = True at start
    - For gradient accumulation, scale loss by 1/accumulation_steps
    - Use clip_grad_norm() before optimizer.step() if grad_clip_norm is set
    - Update scheduler after each epoch: optimizer.lr = scheduler.get_lr(epoch)
    - In evaluate(), set model.training = False and don't update gradients
    - Checkpoints should include: epoch, step, model state, optimizer state, scheduler state, history
    """
    ### BEGIN SOLUTION
    def __init__(self, model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None):
        """
        Initialize trainer with model and training components.

        Args:
            model: Neural network to train
            optimizer: Parameter update strategy (SGD, Adam, etc.)
            loss_fn: Loss function (CrossEntropy, MSE, etc.)
            scheduler: Optional learning rate scheduler
            grad_clip_norm: Optional gradient clipping threshold
        """
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

    def train_epoch(self, dataloader, accumulation_steps=1):
        """
        Train for one epoch through the dataset.

        Args:
            dataloader: Iterable yielding (inputs, targets) batches
            accumulation_steps: Number of batches to accumulate before update

        Returns:
            Average loss for the epoch
        """
        self.model.training = True
        self.training_mode = True

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            # Scale loss for accumulation
            scaled_loss = loss.data / accumulation_steps
            accumulated_loss += scaled_loss

            # Backward pass
            loss.backward()

            # Update parameters every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_norm is not None:
                    params = self.model.parameters()
                    clip_grad_norm(params, self.grad_clip_norm)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                self.step += 1

        # Handle remaining accumulated gradients
        if accumulated_loss > 0:
            if self.grad_clip_norm is not None:
                params = self.model.parameters()
                clip_grad_norm(params, self.grad_clip_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulated_loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)

        # Update scheduler
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            # Update optimizer learning rate (trust it has lr attribute)
            self.optimizer.lr = current_lr
            self.history['learning_rates'].append(current_lr)

        self.epoch += 1
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluate model on dataset without updating parameters.

        Args:
            dataloader: Iterable yielding (inputs, targets) batches

        Returns:
            Average loss and accuracy
        """
        self.model.training = False
        self.training_mode = False

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            # Forward pass only
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            total_loss += loss.data

            # Calculate accuracy (for classification)
            # Trust that Tensors have .data attribute
            if len(outputs.data.shape) > 1:  # Multi-class
                predictions = np.argmax(outputs.data, axis=1)
                if len(targets.data.shape) == 1:  # Integer targets
                    correct += np.sum(predictions == targets.data)
                else:  # One-hot targets
                    correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                total += len(predictions)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history['eval_loss'].append(avg_loss)

        return avg_loss, accuracy

    def save_checkpoint(self, path: str):
        """
        Save complete training state for resumption.

        Args:
            path: File path to save checkpoint
        """
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

    def load_checkpoint(self, path: str):
        """
        Load training state from checkpoint.

        Args:
            path: File path to load checkpoint from
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        self.training_mode = checkpoint['training_mode']

        # Restore states (simplified for educational purposes)
        if 'model_state' in checkpoint:
            self._set_model_state(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self._set_optimizer_state(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self._set_scheduler_state(checkpoint['scheduler_state'])

    def _get_model_state(self):
        """Extract model parameters for checkpointing."""
        # Trust model has parameters() method
        return {i: param.data.copy() for i, param in enumerate(self.model.parameters())}

    def _set_model_state(self, state):
        """Restore model parameters from checkpoint."""
        # Trust model has parameters() method
        for i, param in enumerate(self.model.parameters()):
            if i in state:
                param.data = state[i].copy()

    def _get_optimizer_state(self):
        """Extract optimizer state for checkpointing."""
        state = {}
        # Trust optimizer has lr attribute (from Modules 06)
        state['lr'] = self.optimizer.lr
        # Use explicit API for momentum state (Module 07)
        # All optimizers with momentum support have get_momentum_state() method
        if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
            momentum_state = self.optimizer.get_momentum_state()
            if momentum_state is not None:
                state['momentum_buffers'] = momentum_state
        return state

    def _set_optimizer_state(self, state):
        """Restore optimizer state from checkpoint."""
        if 'lr' in state:
            # Trust optimizer has lr attribute (from Modules 06)
            self.optimizer.lr = state['lr']
        # Use explicit API for momentum state (Module 07)
        # All optimizers with momentum support have set_momentum_state() method
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
        # Educational Note: hasattr() is legitimate here because:
        # 1. Schedulers are user-extensible with custom attributes
        # 2. State dict may have keys from different scheduler types
        # 3. We safely skip attributes that don't exist on current scheduler
        # This is duck-typing for polymorphic checkpoint restoration
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Trainer Class
This test validates our complete training system.
**What we're testing**: Trainer orchestrates training loop correctly
**Why it matters**: This is the backbone that enables all neural network training
**Expected**: Training reduces loss, evaluation works, checkpointing preserves state
"""

# %% nbgrader={"grade": true, "grade_id": "test_trainer", "locked": true, "points": 15}
def test_unit_trainer():
    """🔬 Test Trainer implementation."""
    print("🔬 Unit Test: Trainer...")

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

    # Create trainer with REAL components
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)  # Real SGD from Module 07
    loss_fn = MSELoss()  # Real MSELoss from Module 04
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)

    trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)

    # Test training
    print("Testing training epoch...")
    # Use real Tensors for data
    dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
        (Tensor([[0.5, 1.0]]), Tensor([[1.5]]))
    ]

    loss = trainer.train_epoch(dataloader)
    assert isinstance(loss, (float, np.floating)), f"Expected float loss, got {type(loss)}"
    assert trainer.epoch == 1, f"Expected epoch 1, got {trainer.epoch}"

    # Test evaluation
    print("Testing evaluation...")
    eval_loss, accuracy = trainer.evaluate(dataloader)
    assert isinstance(eval_loss, (float, np.floating)), f"Expected float eval_loss, got {type(eval_loss)}"
    assert isinstance(accuracy, (float, np.floating)), f"Expected float accuracy, got {type(accuracy)}"

    # Test checkpointing
    print("Testing checkpointing...")
    checkpoint_path = "/tmp/test_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    # Modify trainer state
    original_epoch = trainer.epoch
    trainer.epoch = 999

    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)
    assert trainer.epoch == original_epoch, f"Checkpoint didn't restore epoch correctly"

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"✅ Trainer works correctly! Final loss: {loss:.4f}")

if __name__ == "__main__":
    test_unit_trainer()

# %% [markdown]
"""
## 🔧 Integration - Complete Training Example

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
## 📊 Systems Analysis - Training Performance and Memory

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

# %% nbgrader={"grade": false, "grade_id": "analyze_training_memory", "solution": true}
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

# %% nbgrader={"grade": false, "grade_id": "analyze_checkpoint_overhead", "solution": true}
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
    test_unit_trainer()

    print("\nRunning integration scenarios...")

    # Test complete training pipeline integration with REAL components
    print("🔬 Integration Test: Complete Training Pipeline...")

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
    print("Run: tito module complete 07")

if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Before we complete this module, let's reflect on the systems aspects of training infrastructure.

Use only knowledge from Modules 01-07 to answer these questions:

**1. Memory Trade-offs:**
   - If you have a model with 1 million parameters and use Adam optimizer, estimate the total training memory required (parameters + gradients + optimizer state).
   - How does gradient accumulation help when you want batch_size=128 but can only fit batch_size=32 in memory?

**2. Gradient Clipping:**
   - Why do we clip gradients by *global norm* rather than clipping each gradient independently?
   - What happens to training if gradients consistently exceed max_norm? What does this signal?

**3. Learning Rate Scheduling:**
   - Why does cosine annealing start with high learning rate and end with low learning rate?
   - Compare training with fixed lr=0.1 vs cosine schedule (0.1 → 0.01). When might fixed LR be better?

**4. Checkpointing Strategy:**
   - You're training for 100 epochs. Checkpoints are large (1GB each). How often should you save checkpoints?
   - What information MUST be in a checkpoint to resume training exactly where you left off?

**5. Train vs Eval Modes:**
   - Why is it crucial to set model.training = False during evaluation?
   - What would happen if you forgot to zero gradients between training steps?

**Think about these questions. The answers reveal deep understanding of training systems!**
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

Congratulations! You've built a complete training infrastructure that orchestrates the entire machine learning training process!

### Key Accomplishments
- Built Trainer class with complete training/evaluation loops
- Implemented CosineSchedule for adaptive learning rate management
- Created clip_grad_norm for training stability and gradient management
- Added comprehensive checkpointing for training persistence
- Analyzed training memory overhead and checkpoint costs
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Gained
Through hands-on implementation and analysis, you discovered:
- **Training memory is 4-6× model size**: Parameters + gradients + optimizer buffers
- **Gradient accumulation trades time for memory**: Enables larger effective batch sizes
- **Checkpoints include full training state**: Model + optimizer + scheduler + metadata
- **Learning rate scheduling improves convergence**: Cosine annealing balances speed and stability
- **Gradient clipping prevents instability**: Global norm preserves gradient direction

### Real-World Context
Your training infrastructure mirrors production ML systems:
- **PyTorch Lightning Trainer**: Similar architecture with training/eval loops
- **Hugging Face Transformers**: Uses same checkpoint patterns
- **Production Training**: All major ML frameworks use gradient clipping and scheduling

### Ready for Next Steps
Your training implementation enables sophisticated model training with proper scheduling, stability controls, and state management.

**Export with:** `tito module complete 07`

**Next**: Module 09 (Convolutions) will add spatial neural network operations, enabling CNN architectures for computer vision!

**🎓 You now understand the complete training infrastructure that powers modern ML systems!**
"""
