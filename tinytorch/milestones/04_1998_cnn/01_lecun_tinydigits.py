#!/usr/bin/env python3
"""
The CNN Revolution (1998) - LeNet Part 1: TinyDigits
====================================================

📚 HISTORICAL CONTEXT:
After backpropagation proved MLPs could learn (1986), researchers still struggled
with image recognition. MLPs treated pixels independently, requiring millions of
parameters and ignoring spatial structure.

Then in 1998, Yann LeCun's LeNet-5 revolutionized computer vision with
Convolutional Neural Networks (CNNs). By using:
- Shared weights (convolution) → parameters independent of image size
- Local connectivity → preserves spatial structure
- Pooling → translation invariance

LeNet achieved 99%+ accuracy on handwritten digits, launching the deep learning
revolution that led to modern computer vision.

🎯 MILESTONE 4 PART 1: CONVOLUTION VS. DENSE ON THE SAME DIGITS (Offline)
Using YOUR Tiny🔥Torch spatial modules, you'll build a CNN that OUTPERFORMS the
MLP from Milestone 03 on the SAME dataset. This proves spatial operations matter!

✅ REQUIRED MODULES (Run after Module 09):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autodiff
  Module 02 (Activations)   : YOUR ReLU activation
  Module 03 (Layers)        : YOUR Linear layer for classification
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (DataLoader)    : YOUR data batching system
  Module 06 (Autograd)      : YOUR automatic differentiation
  Module 07 (Optimizers)    : YOUR SGD optimizer
  Module 08 (Training)      : YOUR training loops
  Module 09 (Convolutions)  : YOUR Conv2d + MaxPool2d  <-- NEW!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Simple LeNet-style CNN):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Input Image │    │   Conv2d    │    │    ReLU     │    │  MaxPool2d  │    │   Flatten   │    │   Linear    │
    │   8×8×1     │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ YOUR Module │───▶│             │───▶│ YOUR Module │
    │  Grayscale  │    │     09      │    │     02      │    │     09      │    │   72 dims   │    │   72→10     │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         1→8 channels      Non-linear         2×2 pooling        Spatial→Dense     10 Classes
                         3×3 kernel        activation         Reduces size

    Feature Map Dimensions:
    Input: (batch, 1, 8, 8)
    After Conv: (batch, 8, 6, 6)   ← 3×3 kernel reduces 8→6
    After Pool: (batch, 8, 3, 3)   ← 2×2 pooling halves dimensions
    Flatten: (batch, 72)           ← 8 × 3 × 3 = 72 features
    Output: (batch, 10)            ← 10 class probabilities

🔍 WHAT CONVOLUTION CHANGES - The Key Insight:

    MLP (Milestone 03):                  CNN (This Milestone):

    ┌────────┐                           ┌────────┐
    │8×8 img │ → Flatten → 64 numbers    │8×8 img │ → Conv2d → 8 feature maps
    └────────┘                           └────────┘
        ↓                                    ↓
    Each pixel treated                   LOCAL patterns detected:
    INDEPENDENTLY                        • Horizontal edges
                                         • Vertical edges
    Shifting image 1 pixel               • Corners
    = COMPLETELY different               • Curves
    input to network!
                                         Shifting image 1 pixel
    ❌ No spatial awareness              = SAME features detected!

                                         ✅ Translation invariance!

📊 EXPECTED RESULTS (Comparison with Milestone 03):

    ┌──────────────────┬────────────────┬────────────────┐
    │ Architecture     │ Parameters     │ Expected Acc.  │
    ├──────────────────┼────────────────┼────────────────┤
    │ MLP (M03)        │ 2,378          │ 75-85%         │
    │ CNN (This)       │ ~800           │ 85-95%         │  ← BETTER with FEWER params!
    └──────────────────┴────────────────┴────────────────┘

    The CNN should achieve ~10% HIGHER accuracy with 3× FEWER parameters!
    This is the power of exploiting spatial structure.

📌 PART 2: After proving CNNs work, Part 2 (02_lecun_cifar10.py) scales to
   real 32×32 color images using YOUR DataLoader!
"""

import sys
import os
import time
import pickle
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box

# Add paths for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import TinyTorch components
from tinytorch import Tensor, SGD, CrossEntropyLoss
from tinytorch.core.spatial import Conv2d, MaxPool2d
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.core.dataloader import DataLoader, TensorDataset

console = Console()

# Success criterion. The docstring expects 85-95% test accuracy, but measured
# 2026-09-28 over three full 50-epoch runs of this script: 85.5%, 85.0%, 86.0%.
# A correct CNN lands at the bottom of that stated range, so gating at 85%
# would fail correct work on an unlucky shuffle. 75% fails a network that has
# not learned while leaving margin for run-to-run variation.
MIN_TEST_ACCURACY = 75.0

# Accuracy alone cannot prove the CONVOLUTION learned. 2026-09-29: with every
# Conv2d filter gradient forced to zero (filters frozen at their random init),
# this script still reached 80.5-81% test accuracy, because the Linear head can
# classify 8x8 digits from 72 random-filter features. So the milestone also
# checks the thing it claims to teach: YOUR Conv2d backward must deliver a
# non-zero gradient to every filter, and the filters must actually move.
#
# Relative change = ||W_final - W_init|| / ||W_init|| for each Conv2d weight.
# Measured 2026-09-29, full 50-epoch runs of this script:
#
#   run                                test acc     filters moved   max |grad|
#   correct (3 runs)                   86.0-86.5%   76.0-76.1%      0.25-0.34
#   filter gradients zeroed (frozen)   81.0%         0.0%           0
#   filter gradients scaled by 0.1     81.0%        14.5%           4.2e-2
#   filter gradients scaled by 0.01    81.0%         1.7%           4.5e-3
#   filter gradients scaled by 0.001   81.0%         0.2%           4.1e-4
#
# Accuracy separates correct from broken by 5 points (10 of 200 test images);
# filter movement separates them by 76% vs at most 14.5%. The 25% threshold
# sits 3x below the correct runs and fails a conv whose gradient is missing
# or 10x too small. That is why MIN_TEST_ACCURACY can stay at 75%: raising it
# to ~83% would split today's numbers too, but on a margin of a few test
# images, and it would still say nothing about whether the convolution learned.
MIN_CONV_RELATIVE_CHANGE = 0.25

# Note: Autograd is automatically enabled when tinytorch is imported

# =============================================================================
# 🎯 YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone showcases YOUR NEW spatial modules for the first time:
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01: Tensor   │ 4D tensors for images          │ (batch, channels, H, W)     │
# │                     │ (batch, 1, 8, 8) grayscale     │ format for spatial ops      │
# │                     │                                │                             │
# │ Module 02: ReLU     │ Non-linearity after convolution│ Same as MLP, but on         │
# │                     │ on 3D feature maps             │ spatial feature maps!       │
# │                     │                                │                             │
# │ Module 03: Linear   │ Classification head only       │ 72→10 (much smaller than    │
# │                     │ (after spatial features)       │ MLP's 64→32→10)             │
# │                     │                                │                             │
# │ Module 09: Conv2d   │ 3×3 kernel detects local       │ WEIGHT SHARING: same 3×3    │
# │ ★ NEW MODULE ★      │ patterns (edges, curves)       │ kernel used everywhere!     │
# │                     │                                │                             │
# │ Module 09: MaxPool2d│ 2×2 pooling reduces spatial    │ TRANSLATION INVARIANCE:     │
# │ ★ NEW MODULE ★      │ dimensions while keeping       │ small shifts don't matter   │
# │                     │ strongest activations          │                             │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================
# 🆕 WHAT'S NEW SINCE MILESTONE 03 (MLP)
# =============================================================================
#
# ┌──────────────────────┬─────────────────────────┬────────────────────────────┐
# │ MLP (Milestone 03)   │ CNN (This Milestone)    │ Why It's Better            │
# ├──────────────────────┼─────────────────────────┼────────────────────────────┤
# │ Flatten first        │ Conv2d first            │ Preserves spatial structure│
# │ 2,378 parameters     │ ~800 parameters         │ Weight sharing = efficient │
# │ No spatial awareness │ Local connectivity      │ Neighbors processed together│
# │ Global patterns only │ Hierarchical features   │ Edges → Shapes → Objects   │
# │ 75-85% accuracy      │ 85-95% accuracy         │ ~10% improvement!          │
# └──────────────────────┴─────────────────────────┴────────────────────────────┘
#
# =============================================================================


# ============================================================================
# 🎓 ZONE 1: STUDENT LEGO BRICKS (LeNet CNN Architecture)
# ============================================================================

class SimpleCNN:
    """
    Simple Convolutional Neural Network for digit classification.

    Architecture inspired by LeNet-5 (1998):
    - Conv2d: Detects local patterns (edges, curves)
    - ReLU: Nonlinearity
    - MaxPool: Spatial down-sampling + translation invariance
    - Linear: Final classification

    Input: (batch, 1, 8, 8)
    Conv1: 1 → 8 channels, 3×3 kernel → (batch, 8, 6, 6)
    Pool1: 2×2 max pooling → (batch, 8, 3, 3)
    Flatten: → (batch, 72)
    Linear: 72 → 10 classes
    """

    def __init__(self):
        # Convolutional layers
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        # After conv(3×3) and pool(2×2): 8×8 → 6×6 → 3×3
        # Flattened size: 8 channels × 3 × 3 = 72
        self.fc = Linear(in_features=72, out_features=10)

        self.params = [self.conv1.weight, self.conv1.bias, self.fc.weight, self.fc.bias]

    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)

    def forward(self, x):
        # Conv + ReLU + Pool
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        # Flatten: (batch, 8, 3, 3) → (batch, 72)
        batch_size = out.shape[0]
        # Preserve the graph so gradients reach the convolutional filters.
        out = out.reshape(batch_size, -1)

        # Final classification
        out = self.fc.forward(out)
        return out

    def parameters(self):
        return self.params


# ============================================================================
# 📊 ZONE 2: MILESTONE HARNESS & TRAINING UX
# ============================================================================

def load_digits_dataset():
    """
    Load the TinyDigits dataset (8×8 curated digits).

    Returns 150 training + 47 test grayscale images of handwritten digits (0-9).
    Each image is 8×8 pixels, perfect for quick CNN demonstrations.
    Ships with TinyTorch - no downloads needed!
    """
    # Load from TinyDigits dataset (shipped with TinyTorch)
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / "datasets" / "tinydigits" / "train.pkl"
    test_path = project_root / "datasets" / "tinydigits" / "test.pkl"

    if not train_path.exists() or not test_path.exists():
        console.print(f"[red]✗ TinyDigits dataset not found![/red]")
        console.print(f"[yellow]Expected location: {train_path.parent}[/yellow]")
        console.print("[yellow]Run: python3 datasets/tinydigits/create_tinydigits.py[/yellow]")
        sys.exit(1)

    # Load training data
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    train_images = train_data['images']  # (150, 8, 8)
    train_labels = train_data['labels']  # (150,)

    # Load test data
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    test_images = test_data['images']  # (47, 8, 8)
    test_labels = test_data['labels']  # (47,)

    # CNN expects (batch, channels, height, width)
    # Add channel dimension: (N, 8, 8) → (N, 1, 8, 8)
    train_images = train_images[:, np.newaxis, :, :]  # (150, 1, 8, 8)
    test_images = test_images[:, np.newaxis, :, :]    # (47, 1, 8, 8)

    return (
        Tensor(train_images.astype(np.float32)),
        Tensor(train_labels.astype(np.int64)),
        Tensor(test_images.astype(np.float32)),
        Tensor(test_labels.astype(np.int64))
    )


# ============================================================================
# 🎯 TRAINING & EVALUATION
# ============================================================================

def conv_layers(model):
    """Every Conv2d the model owns, in attribute order."""
    return [layer for layer in vars(model).values() if isinstance(layer, Conv2d)]


class ConvGradientMonitor:
    """Remember the largest filter gradient each Conv2d received in training.

    Called right after loss.backward() and before optimizer.step(), which is
    the only moment the gradient YOUR autograd computed is visible.
    """

    def __init__(self, layers):
        self.layers = list(layers)
        self.max_abs_grad = [0.0 for _ in self.layers]

    def __call__(self):
        for i, layer in enumerate(self.layers):
            grad = layer.weight.grad
            if grad is None:
                continue
            grad = np.asarray(getattr(grad, "data", grad), dtype=np.float64)
            if np.all(np.isfinite(grad)):
                self.max_abs_grad[i] = max(self.max_abs_grad[i], float(np.max(np.abs(grad))))


def relative_change(before, after):
    """||after - before|| / ||before||: how far a weight tensor moved."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    scale = np.linalg.norm(before)
    if scale == 0:
        scale = 1.0
    return float(np.linalg.norm(after - before) / scale)


def conv_learning_failures(max_abs_grads, relative_changes,
                           min_relative_change=MIN_CONV_RELATIVE_CHANGE):
    """Return a list of reasons the convolution did not learn (empty = learned).

    max_abs_grads: largest |gradient| each Conv2d weight saw during training.
    relative_changes: relative_change(init, final) for each Conv2d weight.
    """
    failures = []
    if not max_abs_grads:
        failures.append("the model has no Conv2d layer")
    for i, (g, rel) in enumerate(zip(max_abs_grads, relative_changes)):
        if not np.isfinite(g) or g == 0.0:
            failures.append(f"Conv2d #{i + 1} never received a non-zero filter gradient")
        if not np.isfinite(rel):
            failures.append(f"Conv2d #{i + 1} filters became NaN/inf")
        elif rel < min_relative_change:
            failures.append(
                f"Conv2d #{i + 1} filters moved only {rel:.1%} from their random init "
                f"(a learning conv moves at least {min_relative_change:.0%})")
    return failures


def train_epoch(model, dataloader, criterion, optimizer, on_backward=None):
    """Train for one epoch.

    on_backward, if given, is called after each loss.backward() and before
    optimizer.step(), so the gate can see the gradients YOUR autograd produced.
    """
    total_loss = 0.0
    n_samples = 0

    for batch_images, batch_labels in dataloader:
        # Forward pass
        logits = model(batch_images)
        loss = criterion.forward(logits, batch_labels)

        # Backward pass
        loss.backward()
        if on_backward is not None:
            on_backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        batch_size = batch_images.shape[0]
        total_loss += loss.data.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


# ============================================================================
# 🔎 LOSS FORWARD CHECK (is YOUR Module 04 loss value right?)
# ============================================================================
#
# 2026-09-29: with CrossEntropy's forward returning 0, this milestone still
# passed. Training only needs the gradient, and the gradient comes from the
# backward pass, so the network learned while every printed loss was wrong.
# One batch, checked against an independent NumPy computation before training
# starts, catches that without touching the training run.

def reference_cross_entropy(logits, targets):
    """Mean cross-entropy computed directly in NumPy (Module 04's definition).

    Uses the stable log-softmax: subtract each row's max before exponentiating.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z.reshape(-1, z.shape[-1])
    t = np.asarray(targets).reshape(-1).astype(int)
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(t)), t]))


def loss_forward_failure(loss_fn, outputs, targets, reference_fn, rtol=1e-3, atol=1e-4):
    """Compare YOUR loss value on one batch to the NumPy reference.

    Returns None when they agree, otherwise a message naming both numbers.
    """
    name = type(loss_fn).__name__
    expected = reference_fn(outputs.data, targets.data)
    value = np.asarray(getattr(loss_fn(outputs, targets), "data", None), dtype=np.float64)
    if value.size != 1:
        return (f"your {name} forward returns an array of shape {value.shape} for this batch, "
                f"but a loss is one number (the mean over the batch, here {expected:.4f}): "
                "check Module 04")
    got = float(value.reshape(()))
    if np.isfinite(got) and np.isclose(got, expected, rtol=rtol, atol=atol):
        return None
    return (f"your {name} forward returns {got:.4f} for this batch, "
            f"but the loss of these outputs is {expected:.4f}: check Module 04")


def report_loss_forward_failure(message):
    """Print the teaching panel for a wrong loss value."""
    console.print(Panel.fit(
        "[bold red]❌ YOUR loss value is wrong[/bold red]\n\n"
        f"{message}.\n\n"
        "Training could still work, because the gradient comes from the backward\n"
        "pass, but every loss this milestone prints would be wrong. Cross-entropy\n"
        "is -mean(log_softmax(logits)[i, target_i]), with the row max subtracted\n"
        "before exp so large logits cannot overflow.",
        title="Needs Work",
        border_style="red",
    ))


def evaluate_accuracy(model, images, labels):
    """Evaluate model accuracy on a dataset."""
    logits = model(images)
    predictions = np.argmax(logits.data, axis=1)
    accuracy = 100.0 * np.mean(predictions == labels.data)
    avg_loss = float(CrossEntropyLoss()(logits, labels).data)
    return accuracy, avg_loss

def press_enter_to_continue():
    if os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1" or os.environ.get("CI") == "true":
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError:
            pass
        console.print()

# ============================================================================
# 🎬 MAIN MILESTONE DEMONSTRATION
# ============================================================================

def train_cnn():
    """Main training loop following 5-Act structure."""

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 1: THE CHALLENGE 🎯
    # ═══════════════════════════════════════════════════════════════════════

    console.print(Panel.fit(
        "[bold cyan]1998: The Computer Vision Challenge[/bold cyan]\n\n"
        "[yellow]The Problem:[/yellow]\n"
        "MLPs flatten images → lose spatial structure\n"
        "Each pixel treated independently\n"
        "Millions of parameters needed for larger images\n\n"
        "[green]The Innovation:[/green]\n"
        "Convolutional Neural Networks (CNNs)\n"
        "  • Shared weights across space (convolution)\n"
        "  • Local connectivity (receptive fields)\n"
        "  • Pooling for translation invariance\n\n"
        "[bold]Can spatial operations outperform dense layers?[/bold]",
        title="🎯 ACT 1: THE CHALLENGE",
        border_style="cyan",
        box=box.DOUBLE
    ))

    press_enter_to_continue()

    # Load data
    console.print("[bold]📊 Loading Handwritten Digits Dataset...[/bold]")
    train_images, train_labels, test_images, test_labels = load_digits_dataset()

    console.print(f"  Training samples: [cyan]{len(train_images.data)}[/cyan]")
    console.print(f"  Test samples: [cyan]{len(test_images.data)}[/cyan]")
    console.print(f"  Image shape: [cyan]{train_images.data[0].shape}[/cyan] (1 channel, 8×8 pixels)")
    console.print(f"  Classes: [cyan]10[/cyan] (digits 0-9)")

    # Show training data structure
    console.print(f"\n  [dim]Sample digit values (first image, top-left 3×3):[/dim]")
    sample = train_images.data[0, 0, :3, :3]
    for row in sample:
        console.print(f"    {' '.join(f'{val:.2f}' for val in row)}")

    press_enter_to_continue()

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 2: THE SETUP 🏗️
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]🏗️  The Architecture:[/bold]")
    console.print("""
    ┌──────────┐    ┌──────────┐    ┌──────┐    ┌─────────┐    ┌─────────┐    ┌────────┐
    │  Input   │    │  Conv2d  │    │ ReLU │    │MaxPool2d│    │ Flatten │    │ Linear │
    │ 1×8×8    │───▶│  1→8     │───▶│      │───▶│  2×2    │───▶│ 8×3×3   │───▶│ 72→10  │
    │          │    │  3×3     │    │      │    │         │    │  =72    │    │        │
    └──────────┘    └──────────┘    └──────┘    └─────────┘    └─────────┘    └────────┘
                    ↑ Detects                   ↑ Spatial
                    local patterns              downsampling
    """)

    console.print("[bold]🔧 Components:[/bold]")
    console.print("  • Conv layer: Detects local patterns (edges, curves)")
    console.print("  • ReLU: Non-linear activation")
    console.print("  • MaxPool: Spatial downsampling + translation invariance")
    console.print("  • Linear: Final classification (72 → 10 classes)")
    console.print("  • [bold cyan]Key insight: Shared weights → parameter count set by filter size, not image size[/bold cyan]")

    # Create model
    console.print("\n🧠 Building Convolutional Neural Network...")
    model = SimpleCNN()

    # Count parameters
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    conv_params = np.prod(model.conv1.weight.shape) + np.prod(model.conv1.bias.shape)
    fc_params = np.prod(model.fc.weight.shape) + np.prod(model.fc.bias.shape)

    console.print(f"  ✓ Conv layer: [cyan]{conv_params}[/cyan] parameters")
    console.print(f"  ✓ FC layer: [cyan]{fc_params}[/cyan] parameters")
    console.print(f"  ✓ Total: [bold cyan]{total_params}[/bold cyan] parameters")

    # Hyperparameters
    console.print("\n[bold]⚙️  Training Configuration:[/bold]")
    epochs = 50
    batch_size = 32
    learning_rate = 0.01

    config_table = Table(show_header=False, box=None)
    config_table.add_row("Epochs:", f"[cyan]{epochs}[/cyan]")
    config_table.add_row("Batch size:", f"[cyan]{batch_size}[/cyan]")
    config_table.add_row("Learning rate:", f"[cyan]{learning_rate}[/cyan]")
    config_table.add_row("Optimizer:", "[cyan]SGD[/cyan]")
    config_table.add_row("Loss:", "[cyan]CrossEntropyLoss[/cyan]")
    console.print(config_table)

    # Create optimizer and loss
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # Create dataloader
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    press_enter_to_continue()

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 3: THE EXPERIMENT 🔬
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]🔬 Training CNN on Handwritten Digits...[/bold]\n")

    # Before training: snapshot the filters so we can prove they learned.
    convs = conv_layers(model)
    initial_filters = [layer.weight.data.copy() for layer in convs]
    grad_monitor = ConvGradientMonitor(convs)
    initial_acc, initial_loss = evaluate_accuracy(model, test_images, test_labels)
    console.print(f"[yellow]Before training:[/yellow] Accuracy = {initial_acc:.1f}%")

    # One training-sized batch, sliced directly so the DataLoader's shuffle
    # order is untouched.
    check_images = Tensor(train_images.data[:batch_size])
    check_labels = Tensor(train_labels.data[:batch_size])
    loss_failure = loss_forward_failure(criterion, model(check_images), check_labels,
                                        reference_cross_entropy)
    if loss_failure:
        report_loss_forward_failure(loss_failure)
        return 1
    console.print("[green]✓[/green] YOUR CrossEntropyLoss matches a NumPy check on one batch\n")

    # Training loop
    history = {
        "train_loss": [],
        "test_accuracy": [],
        "train_accuracy": []  # Track training accuracy to detect overfitting
    }
    start_time = time.time()

    # Use Live display with spinner for real-time feedback
    with Live(console=console, refresh_per_second=10) as live:
        for epoch in range(epochs):
            # Update spinner before training
            spinner_text = Text()
            spinner_text.append("⠋ ", style="cyan")
            spinner_text.append(f"Epoch {epoch+1:3d}/{epochs}  Training...")
            live.update(spinner_text)

            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                     on_backward=grad_monitor)

            # Evaluate on both train and test
            train_acc, _ = evaluate_accuracy(model, train_images, train_labels)
            test_acc, _ = evaluate_accuracy(model, test_images, test_labels)

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["test_accuracy"].append(test_acc)

            if (epoch + 1) % 5 == 0:  # Print every 5 epochs
                gap = train_acc - test_acc
                gap_indicator = "⚠️" if gap > 10 else "✓"
                live.console.print(
                    f"Epoch {epoch+1:3d}/{epochs}  "
                    f"Loss: {train_loss:.4f}  "
                    f"Train: {train_acc:.1f}%  "
                    f"Test: {test_acc:.1f}%  "
                    f"{gap_indicator} Gap: {gap:.1f}%"
                )

    training_time = time.time() - start_time

    press_enter_to_continue()

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 4: THE DIAGNOSIS 📊
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]📊 The Results:[/bold]\n")

    final_train_acc = history["train_accuracy"][-1]
    final_test_acc = history["test_accuracy"][-1]
    final_loss = history["train_loss"][-1]
    overfitting_gap = final_train_acc - final_test_acc

    table = Table(title="Training Outcome", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=20)
    table.add_column("Status", style="magenta", width=20)

    table.add_row(
        "Train Accuracy",
        f"{final_train_acc:.1f}%",
        f"↑ +{final_train_acc - initial_acc:.1f}%"
    )
    table.add_row(
        "Test Accuracy",
        f"{final_test_acc:.1f}%",
        f"↑ +{final_test_acc - initial_acc:.1f}%"
    )
    table.add_row(
        "Overfitting Gap",
        f"{overfitting_gap:.1f}%",
        "✓ Healthy" if overfitting_gap < 10 else "⚠️ Overfitting"
    )
    table.add_row(
        "Training Time",
        f"{training_time*1000:.0f}ms",
        "-"
    )

    console.print(table)

    press_enter_to_continue()

    # Sample predictions
    console.print("[bold]🔍 Sample Predictions:[/bold]")
    sample_images = Tensor(test_images.data[:10])  # First 10 test samples
    logits = model(sample_images)
    predictions = np.argmax(logits.data, axis=1)

    samples_table = Table(show_header=True, box=box.SIMPLE)
    samples_table.add_column("True", style="cyan", justify="center")
    samples_table.add_column("Pred", style="green", justify="center")
    samples_table.add_column("Result", justify="center")

    for i in range(10):
        true_label = int(test_labels.data[i])
        pred_label = int(predictions[i])
        result = "✓" if true_label == pred_label else "✗"
        style = "green" if true_label == pred_label else "red"
        samples_table.add_row(str(true_label), str(pred_label), f"[{style}]{result}[/{style}]")

    console.print(samples_table)

    # Key insights
    console.print("\n[bold]💡 Key Insights:[/bold]")
    console.print(f"  • CNNs preserve spatial structure")
    console.print(f"  • Conv layers detect local patterns (edges → digits)")
    console.print(f"  • Pooling provides translation invariance")
    console.print(f"  • {total_params} params vs 2,410 for the Milestone 03 MLP (about 3× fewer)")

    press_enter_to_continue()

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 5: THE REFLECTION 🌟
    # ═══════════════════════════════════════════════════════════════════════

    # 2026-09-28: this milestone once printed "Success!" and exited 0 at any accuracy.
    # 2026-09-29: accuracy alone also passed a frozen convolution (see
    # MIN_CONV_RELATIVE_CHANGE), so the filters must be shown to learn too.
    filter_changes = [relative_change(before, layer.weight.data)
                      for before, layer in zip(initial_filters, convs)]
    console.print("[bold]🔎 Did YOUR convolution learn?[/bold]")
    for i, (g, rel) in enumerate(zip(grad_monitor.max_abs_grad, filter_changes)):
        console.print(f"  Conv2d #{i + 1}: largest filter gradient {g:.2e}, "
                      f"filters moved {rel:.1%} from init "
                      f"(needs ≥ {MIN_CONV_RELATIVE_CHANGE:.0%})")
    conv_failures = conv_learning_failures(grad_monitor.max_abs_grad, filter_changes)

    if final_test_acc < MIN_TEST_ACCURACY:
        console.print(Panel.fit(
            f"[bold red]❌ MILESTONE 04 FAILED: test accuracy {final_test_acc:.1f}% "
            f"is below the {MIN_TEST_ACCURACY:.0f}% target[/bold red]\n\n"
            "A working CNN reaches about 85% here. Check YOUR Conv2d, MaxPool2d,\n"
            "and their backward passes.",
            title="Needs Work",
            border_style="red",
        ))
        return 1

    if conv_failures:
        console.print(Panel.fit(
            "[bold red]❌ MILESTONE 04 FAILED: YOUR convolution did not learn[/bold red]\n\n"
            + "\n".join(f"  • {reason}" for reason in conv_failures) + "\n\n"
            f"Test accuracy was {final_test_acc:.1f}%, but that came from the Linear\n"
            "head reading features from filters stuck at their random start.\n"
            "Random 3×3 filters already extract usable features from 8×8 digits;\n"
            "the point of a CNN is that the filters themselves are learned.\n\n"
            "Check YOUR Conv2dFunction.backward (Module 09): it must return a\n"
            "grad_weight of shape (out_channels, in_channels, kH, kW), built by\n"
            "correlating the input patches with grad_output, and Conv2d must pass\n"
            "its weight into Conv2dFunction.apply so autograd can reach it.",
            title="Needs Work",
            border_style="red",
        ))
        return 1

    generalization = (f"  ✓ Model generalizes well (gap: {overfitting_gap:.1f}%)\n"
                      if overfitting_gap < 10 else
                      f"  ⚠ Train/test gap is {overfitting_gap:.1f}%, a sign of overfitting\n")

    console.print(Panel.fit(
        "[bold green]🎉 Success! Your CNN Learned to Recognize Digits![/bold green]\n\n"

        f"Test accuracy: [bold]{final_test_acc:.1f}%[/bold] (Gap: {overfitting_gap:.1f}%)\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]💡 What YOU Just Accomplished:[/bold]\n"
        "  ✓ Built a Convolutional Neural Network from scratch\n"
        "  ✓ Used Conv2d for spatial feature extraction\n"
        "  ✓ Applied MaxPooling for translation invariance\n"
        f"  ✓ Achieved {final_test_acc:.1f}% test accuracy!\n"
        + generalization +
        "  ✓ Used about 3× fewer parameters than the Milestone 03 MLP\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]🎓 Why This Matters:[/bold]\n"
        "  LeNet-5 (1998) proved CNNs work for real-world vision.\n"
        "  This breakthrough led to:\n"
        "  • AlexNet (2012) - ImageNet revolution\n"
        "  • VGG, ResNet, modern computer vision\n"
        "  • Self-driving cars, medical imaging, face recognition\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]📌 The Key Breakthrough:[/bold]\n"
        "  [yellow]Spatial structure matters![/yellow]\n"
        "  MLPs: Every pixel connects to everything → explosion\n"
        "  CNNs: Local connectivity + shared weights → efficiency\n"
        "  \n"
        "  This is why CNNs dominate computer vision today!\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]🚀 What's Next:[/bold]\n"
        "[dim]You've now built the complete ML training pipeline:\n"
        "  Tensors → Layers → Optimizers → DataLoaders → CNNs\n"
        "  \n"
        "  Next modules will add modern techniques:\n"
        "  • Normalization, Dropout, Advanced architectures\n"
        "  • Attention mechanisms, Transformers\n"
        "  • Production systems, Optimization, Deployment![/dim]",

        title="🌟 1998 CNN Revolution Complete",
        border_style="green",
        box=box.DOUBLE
    ))

    press_enter_to_continue()
    return 0

if __name__ == "__main__":
    sys.exit(train_cnn())
