# Module 08: Training

:::{admonition} Module Info
:class: note

**FOUNDATION TIER** | Difficulty: ●●○○ | Time: 5-7 hours | Prerequisites: 01-07

By completing Modules 01-07, you've built all the fundamental components: tensors, activations, layers, losses, dataloader, autograd, and optimizers. Each piece works perfectly in isolation, but real machine learning requires orchestrating these components into a cohesive training process. This module provides that orchestration.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F08_training%2F08_training.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/08_training/08_training.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/08_training.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Training is where all your previous work comes together. You've built tensors that can store data, layers that transform inputs, loss functions that measure error, autograd that computes gradients, and optimizers that update parameters. But these components don't connect themselves. The training loop is the conductor that orchestrates this symphony: forward passes flow data through layers, loss functions measure mistakes, backward passes compute gradients, and optimizers improve parameters. Repeat this cycle thousands of times and your randomly initialized network learns to solve problems.

Production training systems need more than this basic loop. Learning rates should start high for rapid progress, then decay for stable convergence. Gradients sometimes explode and need clipping. Long training runs require checkpointing to survive crashes. Models need separate train and evaluation modes. This module builds all of this infrastructure into a complete Trainer class that mirrors PyTorch Lightning and Hugging Face training systems.

By the end, you'll have a production-grade training infrastructure ready for the MLP milestone.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** a complete Trainer class orchestrating forward pass, loss computation, backward pass, and parameter updates
- **Master** learning rate scheduling with cosine annealing that adapts training speed over time
- **Understand** gradient clipping by global norm that prevents training instability
- **Build** checkpointing systems that save and restore complete training state for fault tolerance
- **Analyze** training memory overhead (4-6× model size) and checkpoint storage costs
```

## What You'll Build

```{mermaid}
:align: center
:caption: Training Infrastructure
flowchart TD
    subgraph "Training Infrastructure"
        A["CosineSchedule<br/>Adaptive learning rate"]
        B["clip_grad_norm()<br/>Gradient stability"]
        C["Trainer<br/>Complete orchestration"]
        D["Checkpointing<br/>State persistence"]
    end

    subgraph "Training Loop"
        E["Forward Pass"] --> F["Loss Computation"]
        F --> G["Backward Pass"]
        G --> H["Gradient Clipping"]
        H --> I["Parameter Update"]
        I --> J["LR Schedule"]
        J --> E
    end

    A --> C
    B --> C
    C --> E
    C --> D

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `CosineSchedule` class | Learning rate annealing (fast → slow) |
| 2 | `clip_grad_norm()` function | Global gradient clipping for stability |
| 3 | `Trainer.train_epoch()` | Complete training loop with scheduling |
| 4 | `Trainer.evaluate()` | Evaluation mode without gradient updates |
| 5 | `Trainer.save/load_checkpoint()` | Training state persistence |

**The pattern you'll enable:**
```python
# Complete training pipeline (modules 01-07 working together)
trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)
for epoch in range(100):
    train_loss = trainer.train_epoch(train_data)
    eval_loss, accuracy = trainer.evaluate(val_data)
    trainer.save_checkpoint(f"checkpoint_{epoch}.pkl")
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Distributed training across multiple GPUs (PyTorch uses `DistributedDataParallel`)
- Mixed precision training (PyTorch Automatic Mixed Precision requires specialized tensor types)
- Advanced schedulers like warmup or cyclical learning rates (production frameworks offer dozens of variants)

**You are building the core training orchestration.** Spatial operations for computer vision come next.

## API Reference

This section provides a quick reference for the training infrastructure you'll build. Use this while implementing to understand expected signatures and behavior.

### CosineSchedule

```python
CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
```

Cosine annealing learning rate schedule that smoothly decreases from `max_lr` to `min_lr` over `total_epochs`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_lr` | `get_lr(epoch: int) -> float` | Returns learning rate for given epoch |

### Gradient Clipping

```python
clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float
```

Clips gradients by global norm to prevent exploding gradients. Returns original norm for monitoring.

### Trainer

```python
Trainer(model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None)
```

Orchestrates complete training lifecycle with forward pass, loss computation, backward pass, optimization, and checkpointing.

#### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `train_epoch` | `train_epoch(dataloader, accumulation_steps=1) -> float` | Train for one epoch, returns average loss |
| `evaluate` | `evaluate(dataloader) -> Tuple[float, float]` | Evaluate model, returns (loss, accuracy) |
| `save_checkpoint` | `save_checkpoint(path: str) -> None` | Save complete training state |
| `load_checkpoint` | `load_checkpoint(path: str) -> None` | Restore training state from file |

## Core Concepts

This section covers the fundamental ideas behind production training systems. These patterns apply to every ML framework and understanding them deeply will serve you throughout your career.

### The Training Loop

The training loop is a simple pattern repeated thousands of times: push data through the model (forward pass), measure how wrong it is (loss), compute how to improve (backward pass), and update parameters (optimizer step). This cycle transforms random weights into intelligent systems.

Here's the complete training loop from your Trainer implementation:

```python
def train_epoch(self, dataloader, accumulation_steps=1):
    """Train for one epoch through the dataset."""
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
        self.optimizer.lr = current_lr
        self.history['learning_rates'].append(current_lr)

    self.epoch += 1
    return avg_loss
```

Each iteration processes one batch: the model transforms inputs into predictions, the loss function compares predictions to targets, backward pass computes gradients, gradient clipping prevents instability, and the optimizer updates parameters. The accumulated loss divided by batch count gives average training loss for monitoring convergence.

The `accumulation_steps` parameter enables a clever memory trick: if you want an effective batch size of 128 but can only fit 32 samples in GPU memory, set `accumulation_steps=4`. Gradients accumulate across 4 batches before the optimizer step, creating the same update as processing all 128 samples at once.

### Epochs and Iterations

Training operates on two timescales: iterations (single batch updates) and epochs (complete passes through the dataset). Understanding this hierarchy helps you reason about training progress and resource requirements.

An iteration processes one batch: forward pass, backward pass, optimizer step. If your dataset has 10,000 samples and batch size is 32, one epoch requires 313 iterations (10,000 ÷ 32, rounded up). Training a model to convergence typically requires dozens or hundreds of epochs, meaning tens of thousands of iterations.

The mathematics is straightforward but the implications are significant. Training ImageNet with 1.2 million images, batch size 256, for 90 epochs requires 421,875 iterations (1,200,000 ÷ 256 × 90). At 250ms per iteration, that's 29 hours of compute. Understanding this arithmetic helps you estimate training costs and debug slow convergence.

Your Trainer tracks both: `self.step` counts total iterations across all epochs, while `self.epoch` counts how many complete dataset passes you've completed. Schedulers typically operate on epoch boundaries (learning rate changes each epoch), while monitoring systems track loss per iteration.

### Train vs Eval Modes

Neural networks behave differently during training versus evaluation. Layers like dropout randomly zero activations during training (for regularization) but keep all activations during evaluation. Batch normalization computes running statistics during training but uses fixed statistics during evaluation. Your Trainer needs to signal which mode the model is in.

The pattern is simple: set `model.training = True` before training, set `model.training = False` before evaluation. This boolean flag propagates through layers, changing their behavior:

```python
def evaluate(self, dataloader):
    """Evaluate model without updating parameters."""
    self.model.training = False
    self.training_mode = False

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        # Forward pass only (no backward!)
        outputs = self.model.forward(inputs)
        loss = self.loss_fn.forward(outputs, targets)

        total_loss += loss.data

        # Calculate accuracy (for classification)
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
```

Notice what's missing: no `loss.backward()`, no `optimizer.step()`, no gradient updates. Evaluation measures current model performance without changing parameters. This separation is crucial: if you accidentally left `training = True` during evaluation, dropout would randomly zero activations, giving you noisy accuracy measurements that don't reflect true model quality.

### Learning Rate Scheduling

Learning rate scheduling adapts training speed over time. Early in training, when parameters are far from optimal, high learning rates enable rapid progress. Late in training, when approaching a good solution, low learning rates enable stable convergence without overshooting. Fixed learning rates force you to choose between fast early progress and stable late convergence. Scheduling gives you both.

Cosine annealing uses the cosine function to smoothly transition from maximum to minimum learning rate:

```python
def get_lr(self, epoch: int) -> float:
    """Get learning rate for current epoch."""
    if epoch >= self.total_epochs:
        return self.min_lr

    # Cosine annealing formula
    cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
    return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
```

The mathematics creates a smooth curve. At epoch 0, `np.cos(0) = 1`, so `cosine_factor = (1+1)/2 = 1.0`, giving `max_lr`. At the final epoch, `np.cos(π) = -1`, so `cosine_factor = (1-1)/2 = 0.0`, giving `min_lr`. Between these extremes, the cosine function creates a smooth descent.

Visualizing the schedule for `max_lr=0.1`, `min_lr=0.01`, `total_epochs=100`:

```
Epoch   0:  0.100 (aggressive learning)
Epoch  25:  0.085 (still fast)
Epoch  50:  0.055 (slowing down)
Epoch  75:  0.025 (fine-tuning)
Epoch 100:  0.010 (stable convergence)
```

Your Trainer applies the schedule automatically after each epoch:

```python
if self.scheduler is not None:
    current_lr = self.scheduler.get_lr(self.epoch)
    self.optimizer.lr = current_lr
```

This updates the optimizer's learning rate before the next epoch begins, creating adaptive training speed without manual intervention.

### Gradient Clipping

Gradient clipping prevents exploding gradients that destroy training progress. During backpropagation, gradients sometimes become extremely large (thousands or even infinity), causing parameter updates that jump far from the optimal solution or create numerical overflow (NaN values). Clipping rescales large gradients to a safe maximum while preserving their direction.

The key insight is clipping by global norm rather than individual gradients. Computing the norm across all parameters `√(Σ g²)` and scaling uniformly preserves the relative magnitudes between different parameters:

```python
def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    """Clip gradients by global norm to prevent exploding gradients."""
    # Compute global norm across all parameters
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            grad_data = param.grad if isinstance(param.grad, np.ndarray) else param.grad.data
            total_norm += np.sum(grad_data ** 2)

    total_norm = np.sqrt(total_norm)

    # Scale all gradients if norm exceeds threshold
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                if isinstance(param.grad, np.ndarray):
                    param.grad = param.grad * clip_coef
                else:
                    param.grad.data = param.grad.data * clip_coef

    return float(total_norm)
```

Consider gradients `[100, 200, 50]` with global norm `√(100² + 200² + 50²) = 230`. With `max_norm=1.0`, we compute `clip_coef = 1.0 / 230 = 0.00435` and scale all gradients: `[0.435, 0.870, 0.217]`. The new norm is exactly 1.0, but the relative magnitudes are preserved (the second gradient is still twice the first).

This uniform scaling is crucial. If we clipped each gradient independently to 1.0, we'd get `[1.0, 1.0, 1.0]`, destroying the information that the second parameter needs larger updates than the first. Global norm clipping prevents explosions while respecting the gradient's message about relative importance.

### Checkpointing

Checkpointing saves complete training state to disk, enabling fault tolerance and experimentation. Training runs take hours or days. Hardware fails. You want to try different hyperparameters after epoch 50. Checkpoints make all of this possible by capturing everything needed to resume training exactly where you left off.

A complete checkpoint includes:

```python
def save_checkpoint(self, path: str):
    """Save complete training state for resumption."""
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
```

Model state is straightforward: copy all parameter tensors. Optimizer state is more subtle: SGD with momentum stores velocity buffers (one per parameter), Adam stores two moment buffers (first and second moments). Scheduler state captures current learning rate progression. Training metadata includes epoch counter and loss history.

Loading reverses the process:

```python
def load_checkpoint(self, path: str):
    """Restore training state from checkpoint."""
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
```

After loading, training resumes as if the interruption never happened. The next `train_epoch()` call starts at the correct epoch, uses the correct learning rate, and continues optimizing from the exact parameter values where you stopped.

### Computational Complexity

Training complexity depends on model architecture and dataset size. For a simple fully connected network with L layers of size d, each forward pass is O(d² × L) (matrix multiplications dominate). Backward pass has the same complexity (automatic differentiation revisits each operation). With N training samples and batch size B, one epoch requires N/B iterations.

Total training cost for E epochs:

```
Time per iteration:    O(d² × L) × 2     (forward + backward)
Iterations per epoch:  N / B
Total iterations:      (N / B) × E
Total complexity:      O((N × E × d² × L) / B)
```

Real numbers make this concrete. Training a 2-layer network (d=512) on 10,000 samples (batch size 32) for 100 epochs:

```
d² × L = 512² × 2 = 524,288 operations per sample
Batch operations = 524,288 × 32 = 16.8 million ops
Iterations per epoch = 10,000 / 32 = 313
Total iterations = 313 × 100 = 31,300
Total operations = 31,300 × 16.8M = 525 billion operations
```

At 1 billion operations per second (typical CPU), that's 525 seconds (9 minutes). This arithmetic explains why GPUs matter: a GPU at 1 trillion ops/second (1000× faster) completes this in 0.5 seconds.

Memory complexity is simpler but just as important:

| Component | Memory |
|-----------|--------|
| Model parameters | d² × L × 4 bytes (float32) |
| Gradients | Same as parameters |
| Optimizer state (SGD) | Same as parameters (momentum) |
| Optimizer state (Adam) | 2× parameters (two moments) |
| Activations | d × B × L × 4 bytes |

Total training memory is typically 4-6× model size, depending on optimizer. This explains GPU memory constraints: a 1GB model requires 4-6GB GPU memory for training, limiting batch size when memory is scarce.

## Production Context

### Your Implementation vs. PyTorch

Your Trainer class and PyTorch's training infrastructure (Lightning, Hugging Face Trainer) share the same architectural patterns. The differences lie in scale: production frameworks support distributed training, mixed precision, complex schedulers, and dozens of callbacks. But the core loop is identical.

| Feature | Your Implementation | PyTorch / Lightning |
|---------|---------------------|---------------------|
| **Training Loop** | Manual forward/backward/step | Same pattern, with callbacks |
| **Schedulers** | Cosine annealing | 20+ schedulers (warmup, cyclic, etc.) |
| **Gradient Clipping** | Global norm clipping | Same algorithm, GPU-optimized |
| **Checkpointing** | Pickle-based state saving | Same concept, optimized formats |
| **Distributed Training** | ✗ Single device | ✓ Multi-GPU, multi-node |
| **Mixed Precision** | ✗ FP32 only | ✓ Automatic FP16/BF16 |

### Code Comparison

The following comparison shows equivalent training pipelines in TinyTorch and PyTorch. Notice how the conceptual flow is identical: create model, optimizer, loss, trainer, then loop through epochs.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch import Trainer, CosineSchedule, SGD, MSELoss

# Setup
model = MyModel()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
trainer = Trainer(model, optimizer, MSELoss(), scheduler, grad_clip_norm=1.0)

# Training loop
for epoch in range(100):
    train_loss = trainer.train_epoch(train_data)
    eval_loss, acc = trainer.evaluate(val_data)

    if epoch % 10 == 0:
        trainer.save_checkpoint(f"ckpt_{epoch}.pkl")
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import Trainer

# Setup (nearly identical!)
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)
trainer = Trainer(max_epochs=100, gradient_clip_val=1.0)

# Training (abstracted by Lightning)
trainer.fit(model, train_dataloader, val_dataloader)
# Lightning handles the loop, checkpointing, and callbacks automatically
```
````
`````

Let's walk through the key similarities and differences:

- **Line 1-2 (Imports)**: TinyTorch exposes classes directly; PyTorch uses module hierarchy. Same concepts, different organization.
- **Line 4-5 (Model and Optimizer)**: Identical pattern. Both frameworks pass model parameters to optimizer for tracking.
- **Line 6 (Scheduler)**: TinyTorch uses `CosineSchedule` class; PyTorch uses `CosineAnnealingLR`. Same mathematics (cosine annealing), same purpose.
- **Line 7 (Trainer Setup)**: TinyTorch takes explicit model, optimizer, loss, and scheduler; PyTorch Lightning abstracts these into the model definition. Both support gradient clipping.
- **Line 9-13 (Training Loop)**: TinyTorch makes the epoch loop explicit; Lightning hides it inside `trainer.fit()`. Under the hood, Lightning runs the exact same loop you implemented.
- **Checkpointing**: TinyTorch requires manual `save_checkpoint()` calls; Lightning checkpoints automatically based on validation metrics.

```{tip} What's Identical

The core training loop pattern: forward pass → loss → backward → gradient clipping → optimizer step → learning rate scheduling. When debugging PyTorch training, you'll understand exactly what's happening because you built it yourself.
```

### Why Training Infrastructure Matters at Scale

To appreciate the engineering behind training systems, consider production scale:

- **GPT-3 training**: 175 billion parameters, trained on 300 billion tokens, cost ~$4.6 million in compute time. A single checkpoint is **350 GB** (larger than most hard drives). Checkpoint frequency must balance fault tolerance against storage costs.

- **ImageNet training**: 1.2 million images, 90 epochs standard. At 250ms per iteration (batch size 256), that's **29 hours** on one GPU. Learning rate scheduling is the difference between 75% accuracy (poor) and 76.5% accuracy (state-of-the-art).

- **Training instability**: Without gradient clipping, 1 in 50 training runs randomly diverges (gradients explode, model outputs NaN, all progress lost). Production systems can't tolerate 2% failure rates when runs cost thousands of dollars.

The infrastructure you built handles these challenges at educational scale. The same patterns scale to production: checkpointing every N epochs, cosine schedules for stable convergence, gradient clipping for reliability.

## Check Your Understanding

Test yourself with these systems thinking questions. They build intuition for the performance characteristics and trade-offs you'll encounter in production ML.

**Q1: Training Memory Calculation**

You have a model with 10 million parameters (float32) and use Adam optimizer. Estimate total training memory required: parameters + gradients + optimizer state. Then compare with SGD optimizer.

```{admonition} Answer
:class: dropdown

**Adam optimizer:**
- Parameters: 10M × 4 bytes = **40 MB**
- Gradients: 10M × 4 bytes = **40 MB**
- Adam state (two moments): 10M × 2 × 4 bytes = **80 MB**
- **Total: 160 MB** (4× parameter size)

**SGD with momentum:**
- Parameters: 10M × 4 bytes = **40 MB**
- Gradients: 10M × 4 bytes = **40 MB**
- Momentum buffer: 10M × 4 bytes = **40 MB**
- **Total: 120 MB** (3× parameter size)

**Key insight:** Optimizer choice affects memory by 33%. For large models near GPU memory limits, SGD may be the only option.
```

**Q2: Gradient Accumulation Trade-off**

You want batch size 128 but your GPU can only fit 32 samples. You use gradient accumulation with `accumulation_steps=4`. How does this affect:
(a) Memory usage?
(b) Training time?
(c) Gradient noise?

```{admonition} Answer
:class: dropdown

**(a) Memory:** No change. Only one batch (32 samples) in GPU memory at a time. Gradients accumulate in parameter `.grad` buffers which already exist.

**(b) Training time:** **4× slower per update**. You process 4 batches sequentially (forward + backward) before optimizer step. Total iterations stays the same, but wall-clock time increases linearly with accumulation steps.

**(c) Gradient noise:** **Reduced** (same as true batch_size=128). Averaging gradients over 128 samples gives more accurate gradient estimate than 32 samples, leading to more stable training.

**Trade-off summary:** Gradient accumulation exchanges compute time for effective batch size when memory is limited. You get better gradients (less noise) but slower training (more time per update).
```

**Q3: Learning Rate Schedule Analysis**

Training with fixed `lr=0.1` converges quickly initially but oscillates around the optimum, never quite reaching it. Training with cosine schedule (0.1 → 0.01) converges slower initially but reaches better final accuracy. Explain why, and suggest when fixed LR might be better.

```{admonition} Answer
:class: dropdown

**Why fixed LR oscillates:**
High learning rate (0.1) enables large parameter updates. Early in training (far from optimum), large updates accelerate convergence. Near the optimum, large updates overshoot, causing oscillation: update jumps past the optimum, then jumps back, repeatedly.

**Why cosine schedule reaches better accuracy:**
Starting high (0.1) provides fast early progress. Gradual decay (0.1 → 0.01) allows the model to take progressively smaller steps as it approaches the optimum. By the final epochs, lr=0.01 enables fine-tuning without overshooting.

**When fixed LR is better:**
- **Short training runs** (< 10 epochs): Scheduling overhead not worth it
- **Learning rate tuning**: Finding optimal LR is easier with fixed values
- **Transfer learning**: When fine-tuning pre-trained models, fixed low LR (0.001) often works best

**Rule of thumb:** For training from scratch over 50+ epochs, scheduling almost always improves final accuracy by 1-3%.
```

**Q4: Checkpoint Storage Strategy**

You're training for 100 epochs. Each checkpoint is 1 GB. Checkpointing every epoch creates 100 GB of storage. Checkpointing every 10 epochs risks losing 10 epochs of work if training crashes. Design a checkpointing strategy that balances fault tolerance and storage costs.

```{admonition} Answer
:class: dropdown

**Strategy: Keep last N + best + milestones**

1. **Keep last N=3 checkpoints** (rolling window): `epoch_98.pkl`, `epoch_99.pkl`, `epoch_100.pkl` (3 GB)
2. **Keep best checkpoint** (lowest validation loss): `best_epoch_72.pkl` (1 GB)
3. **Keep milestone checkpoints** (every 25 epochs): `epoch_25.pkl`, `epoch_50.pkl`, `epoch_75.pkl` (3 GB)

**Total storage: 7 GB** (vs 100 GB for every epoch)

**Fault tolerance:**
- Last 3 checkpoints: Lose at most 1 epoch of work
- Best checkpoint: Can always restart from best validation performance
- Milestones: Can restart experiments from quarter-points

**Implementation:**
```python
if epoch % 25 == 0:  # Milestone
    save_checkpoint(f"milestone_epoch_{epoch}.pkl")
elif epoch >= last_3_start:  # Last 3
    save_checkpoint(f"recent_epoch_{epoch}.pkl")
if is_best_validation:  # Best
    save_checkpoint(f"best_epoch_{epoch}.pkl")
```

**Production systems** use this strategy plus cloud storage for off-site backup.
```

**Q5: Global Norm Clipping Analysis**

Two training runs: (A) clips each gradient individually to max 1.0, (B) clips by global norm (max_norm=1.0). Both encounter gradients `[50, 100, 5]` with global norm `√(50² + 100² + 5²) ≈ 112`. What are the clipped gradients in each case? Which preserves gradient direction better?

```{admonition} Answer
:class: dropdown

**(A) Individual clipping** (clip each to max 1.0):
- Original: `[50, 100, 5]`
- Clipped: `[1.0, 1.0, 1.0]`
- **Result:** All parameters get equal updates (destroys relative importance information)

**(B) Global norm clipping** (scale uniformly):
- Original: `[50, 100, 5]`, global norm ≈ 112
- Scale factor: `1.0 / 112 ≈ 0.0089`
- Clipped: `[0.45, 0.89, 0.04]`
- New global norm: **1.0** (exactly max_norm)
- **Result:** Relative magnitudes preserved (second parameter still gets 2× update of first)

**Why (B) is better:**
Gradients encode relative importance: parameter 2 needs larger updates than parameter 1. Global norm clipping prevents explosion while respecting this information. Individual clipping destroys it, effectively treating all parameters as equally important.

**Verification:** `√(0.45² + 0.89² + 0.04²) ≈ 1.0` ✓
```

## Further Reading

For students who want to understand the academic foundations and advanced training techniques:

### Seminal Papers

- **Cyclical Learning Rates for Training Neural Networks** - Smith (2017). Introduced cyclical learning rate schedules and the learning rate finder technique. Cosine annealing is a variant of these ideas. [arXiv:1506.01186](https://arxiv.org/abs/1506.01186)

- **On the Difficulty of Training Recurrent Neural Networks** - Pascanu et al. (2013). Analyzed the exploding and vanishing gradient problem, introducing gradient clipping as a solution. The global norm clipping you implemented comes from this work. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)

- **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** - Goyal et al. (2017). Showed how to scale batch size and learning rate together, introducing linear warmup and gradient accumulation techniques for distributed training. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

### Additional Resources

- **PyTorch Lightning Documentation**: [Training Loop Documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) - See how production frameworks implement the same training patterns you built
- **Weights & Biases Tutorial**: "Hyperparameter Tuning" - Excellent guide on learning rate scheduling and gradient clipping in practice

## What's Next

```{seealso} Coming Up: Module 09 - Convolutions

Implement Conv2d, MaxPool2d, and Flatten layers to build convolutional neural networks. Your Trainer will orchestrate training CNNs on image datasets, enabling the CNN milestone.
```

**Preview - How Your Training Infrastructure Gets Used:**

| Module | What It Does | Your Trainer In Action |
|--------|--------------|------------------------|
| **09: Convolutions** | Convolutional layers for images | Train CNNs with same `trainer.train_epoch()` loop |
| **Milestone: MLP** | Complete MNIST digit recognition | `trainer` orchestrates full training pipeline |
| **Milestone: CNN** | Complete CIFAR-10 classification | Train vision models with your training infrastructure |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/08_training/08_training.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/08_training/08_training.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
