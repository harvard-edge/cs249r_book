---
title: "Training"
description: "Complete training loops with scheduling, gradient clipping, and checkpointing"
difficulty: "⭐⭐⭐⭐"
time_estimate: "6-8 hours"
prerequisites: ["tensor", "activations", "layers", "losses", "autograd", "optimizers"]
next_steps: ["dataloader"]
learning_objectives:
  - "Implement complete Trainer class orchestrating forward/backward passes, loss computation, and optimization"
  - "Build CosineSchedule for adaptive learning rate management during training"
  - "Create gradient clipping utilities to prevent exploding gradients and training instability"
  - "Design checkpointing system for saving and resuming training state"
  - "Understand memory overhead, gradient accumulation, and train/eval mode switching"
---

# 07. Training

**FOUNDATION TIER** | Difficulty: ⭐⭐⭐⭐ (4/4) | Time: 6-8 hours

## Overview

Build the complete training infrastructure that orchestrates neural network learning end-to-end. This capstone module of the Foundation tier brings together all previous components—tensors, layers, losses, gradients, and optimizers—into production-ready training loops with learning rate scheduling, gradient clipping, and model checkpointing. You'll create the same training patterns that power PyTorch, TensorFlow, and every production ML system.

## Learning Objectives

By the end of this module, you will be able to:

- **Implement complete Trainer class**: Orchestrate forward passes, loss computation, backpropagation, and parameter updates into cohesive training loops with train/eval mode switching
- **Build CosineSchedule for adaptive learning rates**: Create learning rate schedulers that start fast for quick convergence, then slow down for fine-tuning, following cosine annealing curves
- **Create gradient clipping utilities**: Implement global norm gradient clipping to prevent exploding gradients and training instability in deep networks
- **Design checkpointing system**: Build save/load functionality that preserves complete training state—model parameters, optimizer buffers, scheduler state, and training history
- **Understand training systems architecture**: Master memory overhead (4-6× model size), gradient accumulation strategies, checkpoint management, and the difference between training and evaluation modes

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement CosineSchedule for learning rate scheduling, clip_grad_norm for gradient stability, and complete Trainer class with checkpointing
2. **Use**: Train neural networks end-to-end with real optimization dynamics, observe learning rate adaptation, and experiment with gradient accumulation
3. **Reflect**: Analyze training memory overhead (parameters + gradients + optimizer state), understand when to checkpoint, and compare training strategies across different scenarios

## Implementation Guide

### The Training Loop Cycle

Training orchestrates data, forward pass, loss, gradients, and updates in an iterative cycle:

```{mermaid}
graph LR
    A[Data Batch] --> B[Forward Pass<br/>Model]
    B --> C[Loss<br/>Compute]
    C --> D[Backward Pass<br/>Autograd]
    D --> E[Optimizer Step<br/>Update θ]
    E --> F[Next Batch]
    F --> A

    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffe0b2
    style E fill:#fce4ec
    style F fill:#f0fdf4
```

**Cycle**: Load batch → Forward through model → Compute loss → Backward gradients → Update parameters → Repeat

### CosineSchedule - Adaptive Learning Rate Management

Learning rate scheduling is like adjusting driving speed based on road conditions—start fast on the highway, slow down in neighborhoods for precision. Cosine annealing provides smooth transitions from aggressive learning to fine-tuning:

```python
class CosineSchedule:
    """
    Cosine annealing learning rate schedule.

    Starts at max_lr, decreases following cosine curve to min_lr.
    Formula: lr = min_lr + (max_lr - min_lr) * (1 + cos(π*epoch/T)) / 2
    """
    def __init__(self, max_lr=0.1, min_lr=0.01, total_epochs=100):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing: smooth decrease from max to min
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor

# Usage example
schedule = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=50)
print(schedule.get_lr(0))   # 0.1 - fast learning initially
print(schedule.get_lr(25))  # ~0.05 - gradual slowdown
print(schedule.get_lr(50))  # 0.001 - fine-tuning at end
```

### Gradient Clipping - Preventing Training Explosions

Gradient clipping is a speed governor that prevents dangerously large gradients from destroying training progress. Global norm clipping scales all gradients uniformly while preserving their relative magnitudes:

```python
def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    """
    Clip gradients by global norm to prevent exploding gradients.

    Computes total_norm = sqrt(sum of all gradient squares).
    If total_norm > max_norm, scales all gradients by max_norm/total_norm.
    """
    if not parameters:
        return 0.0

    # Compute global norm across all parameters
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
            total_norm += np.sum(grad_data ** 2)

    total_norm = np.sqrt(total_norm)

    # Clip if necessary - preserves gradient direction
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                if hasattr(param.grad, 'data'):
                    param.grad.data *= clip_coef
                else:
                    param.grad *= clip_coef

    return float(total_norm)

# Usage example
params = model.parameters()
original_norm = clip_grad_norm(params, max_norm=1.0)
print(f"Gradient norm: {original_norm:.2f} → clipped to 1.0")
```

### Trainer Class - Complete Training Orchestration

The Trainer class conducts the symphony of training—coordinating model, optimizer, loss function, and scheduler into cohesive learning loops with checkpointing and evaluation:

```python
class Trainer:
    """
    Complete training orchestrator for neural networks.

    Handles training loops, evaluation, scheduling, gradient clipping,
    checkpointing, and train/eval mode switching.
    """
    def __init__(self, model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None):
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

        Supports gradient accumulation for effective larger batch sizes.
        """
        self.model.training = True
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

            # Update every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_norm is not None:
                    clip_grad_norm(self.model.parameters(), self.grad_clip_norm)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                self.step += 1

        # Update learning rate
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            self.optimizer.lr = current_lr
            self.history['learning_rates'].append(current_lr)

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)
        self.epoch += 1

        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluate model without updating parameters.

        Sets model.training = False for proper evaluation behavior.
        """
        self.model.training = False
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            # Forward pass only - no gradients
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)
            total_loss += loss.data

            # Calculate accuracy for classification
            if len(outputs.data.shape) > 1:
                predictions = np.argmax(outputs.data, axis=1)
                if len(targets.data.shape) == 1:
                    correct += np.sum(predictions == targets.data)
                else:
                    correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                total += len(predictions)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history['eval_loss'].append(avg_loss)
        return avg_loss, accuracy

    def save_checkpoint(self, path: str):
        """Save complete training state for resumption."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state': {i: p.data.copy() for i, p in enumerate(self.model.parameters())},
            'optimizer_state': self._get_optimizer_state(),
            'scheduler_state': self._get_scheduler_state(),
            'history': self.history,
            'training_mode': self.training_mode
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str):
        """Load training state from checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']

        # Restore model parameters
        for i, param in enumerate(self.model.parameters()):
            if i in checkpoint['model_state']:
                param.data = checkpoint['model_state'][i].copy()
```

### Complete Training Example

Bringing all components together into production-ready training:

```python
from tinytorch.core.training import Trainer, CosineSchedule, clip_grad_norm
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss
from tinytorch.core.optimizers import SGD

# Build model
class SimpleNN:
    def __init__(self):
        self.layer1 = Linear(3, 5)
        self.layer2 = Linear(5, 2)
        self.training = True

    def forward(self, x):
        x = self.layer1.forward(x)
        x = Tensor(np.maximum(0, x.data))  # ReLU
        return self.layer2.forward(x)

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()

# Configure training
model = SimpleNN()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = MSELoss()
scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=10)

# Create trainer with gradient clipping
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    grad_clip_norm=1.0  # Prevent exploding gradients
)

# Train for multiple epochs
for epoch in range(10):
    train_loss = trainer.train_epoch(train_data)
    eval_loss, accuracy = trainer.evaluate(val_data)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
          f"eval_loss={eval_loss:.4f}, accuracy={accuracy:.4f}")

    # Save checkpoint periodically
    if epoch % 5 == 0:
        trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pkl')

# Restore from checkpoint
trainer.load_checkpoint('checkpoint_epoch_5.pkl')
print(f"Resumed training from epoch {trainer.epoch}")
```

## Getting Started

### Prerequisites

Ensure you have completed all Foundation tier modules:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify all prerequisites (Training is the Foundation capstone!)
tito test tensor      # Module 01: Tensor operations
tito test activations # Module 02: Activation functions
tito test layers      # Module 03: Neural network layers
tito test losses      # Module 04: Loss functions
tito test autograd    # Module 05: Automatic differentiation
tito test optimizers  # Module 06: Parameter update algorithms
```

### Development Workflow

1. **Open the development file**: `modules/07_training/training.py`
2. **Implement CosineSchedule**: Build learning rate scheduler with cosine annealing (smooth max_lr → min_lr transition)
3. **Create clip_grad_norm**: Implement global norm gradient clipping to prevent exploding gradients
4. **Build Trainer class**: Orchestrate complete training loop with train_epoch(), evaluate(), and checkpointing
5. **Add gradient accumulation**: Support effective larger batch sizes with limited memory
6. **Test end-to-end training**: Validate complete pipeline with real models and data
7. **Export and verify**: `tito module complete 07 && tito test training`

## Testing

### Comprehensive Test Suite

Run the full test suite to verify complete training infrastructure:

```bash
# TinyTorch CLI (recommended)
tito test training

# Direct pytest execution
python -m pytest tests/ -k training -v
```

### Test Coverage Areas

- **CosineSchedule Correctness**: Verify cosine annealing produces correct learning rates at start, middle, and end epochs
- **Gradient Clipping Stability**: Test global norm computation and uniform scaling when gradients exceed threshold
- **Trainer Orchestration**: Ensure proper coordination of forward pass, backward pass, optimization, and scheduling
- **Checkpointing Completeness**: Validate save/load preserves model state, optimizer buffers, scheduler state, and training history
- **Memory Analysis**: Measure training memory overhead (parameters + gradients + optimizer state = 4-6× model size)

### Inline Testing & Training Analysis

The module includes comprehensive validation of training dynamics:

```python
# CosineSchedule validation
schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
print(schedule.get_lr(0))    # 0.1 - aggressive learning initially
print(schedule.get_lr(50))   # ~0.055 - gradual slowdown
print(schedule.get_lr(100))  # 0.01 - fine-tuning at end

# Gradient clipping validation
param.grad = np.array([100.0, 200.0])  # Large gradients
original_norm = clip_grad_norm([param], max_norm=1.0)
# original_norm ≈ 223.6 → clipped to 1.0
assert np.linalg.norm(param.grad.data) ≈ 1.0

# Trainer integration validation
trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)
loss = trainer.train_epoch(train_data)
eval_loss, accuracy = trainer.evaluate(test_data)
trainer.save_checkpoint('checkpoint.pkl')
```

### Manual Testing Examples

```python
from training import Trainer, CosineSchedule, clip_grad_norm
from layers import Linear
from losses import MSELoss
from optimizers import SGD
from tensor import Tensor

# Test complete training pipeline
class SimpleModel:
    def __init__(self):
        self.layer = Linear(2, 1)
        self.training = True

    def forward(self, x):
        return self.layer.forward(x)

    def parameters(self):
        return self.layer.parameters()

# Create training system
model = SimpleModel()
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = MSELoss()
scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)

trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)

# Create simple dataset
train_data = [
    (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
    (Tensor([[0.5, 1.0]]), Tensor([[1.5]]))
]

# Train and monitor
for epoch in range(10):
    loss = trainer.train_epoch(train_data)
    lr = scheduler.get_lr(epoch)
    print(f"Epoch {epoch}: loss={loss:.4f}, lr={lr:.4f}")

# Test checkpointing
trainer.save_checkpoint('test_checkpoint.pkl')
trainer.load_checkpoint('test_checkpoint.pkl')
print(f"Restored from epoch {trainer.epoch}")
```

## Systems Thinking Questions

### Real-World Applications

- **Production Training Pipelines**: PyTorch Lightning, Hugging Face Transformers, TensorFlow Estimators all use similar Trainer architectures with checkpointing and scheduling
- **Large-Scale Model Training**: GPT, BERT, and vision models rely on gradient clipping and learning rate scheduling for stable convergence across billions of parameters
- **Research Experimentation**: Academic ML uses checkpointing for long experiments with periodic evaluation and model selection
- **Fault-Tolerant Training**: Cloud training systems use checkpoints to resume after infrastructure failures or spot instance interruptions

### Training System Architecture

- **Memory Breakdown**: Training requires parameters (1×) + gradients (1×) + optimizer state (2-3×) = 4-6× model memory footprint
- **Gradient Accumulation**: Enables effective batch size of accumulation_steps × actual_batch_size with fixed memory—trades time for memory efficiency
- **Train/Eval Modes**: Different layer behaviors during training (dropout active, batch norm updates) vs evaluation (dropout off, fixed batch norm)
- **Checkpoint Components**: Must save model parameters, optimizer buffers (momentum, Adam m/v), scheduler state, epoch counter, and training history for exact resumption

### Training Dynamics

- **Learning Rate Scheduling**: Cosine annealing starts fast (quick convergence when far from optimum) then slows (stable fine-tuning near solution)
- **Exploding Gradients**: Occur in deep networks and RNNs when gradient magnitudes grow exponentially through backpropagation—gradient clipping prevents training collapse
- **Gradient Accumulation Trade-offs**: Reduces memory by processing small batches but increases training time linearly with accumulation steps
- **Checkpointing Strategy**: Balance disk space (1GB+ per checkpoint) vs fault tolerance (more frequent = less lost work) and evaluation frequency (save best model)

### Performance Characteristics

- **Training Memory Scaling**: Adam optimizer uses 4× parameter memory (params + grads + m + v) vs SGD with momentum at 3× (params + grads + momentum)
- **Checkpoint Overhead**: Pickle serialization adds 10-30% overhead beyond raw parameter data—use compression for large models
- **Learning Rate Impact**: Too high causes instability/divergence, too low causes slow convergence—scheduling adapts automatically
- **Global Norm vs Individual Clipping**: Global norm preserves gradient direction while preventing explosion—individual clipping can distort optimization trajectory

## Ready to Build?

You're about to complete the Foundation tier by building the training infrastructure that brings neural networks to life! This is where all your work on tensors, activations, layers, losses, gradients, and optimizers comes together into a cohesive system that actually learns from data.

Training is the heart of machine learning—the process that transforms random initialization into intelligent models. You're implementing the same patterns used to train GPT, BERT, ResNet, and every production AI system. Understanding how scheduling, gradient clipping, checkpointing, and mode switching work together gives you mastery over the training process.

This module is the culmination of everything you've built. Take your time understanding how each piece fits into the bigger picture, and enjoy creating a complete ML training system from scratch!

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/07_training/training.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/07_training/training.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/07_training/training.py
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
<a class="left-prev" href="../modules/06_optimizers_ABOUT.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../modules/08_dataloader_ABOUT.html" title="next page">Next Module →</a>
</div>
