#!/usr/bin/env python3
"""
MLP on Digits (1986) - Rumelhart, Hinton, Williams
==================================================

📚 HISTORICAL CONTEXT:
In 1986, Rumelhart, Hinton, and Williams published "Learning representations by
back-propagating errors" in Nature. This paper proved that multi-layer networks
could learn useful internal representations - ending the AI Winter that began
after the 1969 XOR crisis.

🎯 MILESTONE 3: MULTI-LAYER PERCEPTRON ON TINYDIGITS
The 1986 backpropagation paper proved multi-layer networks could solve
real-world problems. Let's recreate that breakthrough using YOUR Tiny🔥Torch!

✅ REQUIRED MODULES (Run after Module 08):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autodiff
  Module 02 (Activations)   : YOUR ReLU activation
  Module 03 (Layers)        : YOUR Linear layer
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (DataLoader)    : YOUR data batching system
  Module 06 (Autograd)      : YOUR automatic differentiation
  Module 07 (Optimizers)    : YOUR SGD optimizer
  Module 08 (Training)      : YOUR training loops
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (The Classic MLP):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Input Image │    │   Flatten   │    │   Linear    │    │    ReLU     │    │   Linear    │
    │    8×8      │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ YOUR Module │
    │   Pixels    │    │     01      │    │     03      │    │     02      │    │     03      │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
        8×8=64             64 dims           64→32           Non-linear          32→10
        pixels                             Hidden Layer      activation       10 Classes

    Sample 8×8 Digit Images:              What the Hidden Layer Learns:

    ┌────────┐  ┌────────┐               • Edge detectors (horizontal, vertical)
    │░░██░░░░│  │░░░██░░░│               • Curve patterns (loops in 0, 6, 8, 9)
    │░██░░░░░│  │░░░██░░░│               • Stroke endings (1, 7 vs 8, 0)
    │░██░░░░░│  │░░░██░░░│               • Intersection points (4, 8)
    │░██░░░░░│  │░░░██░░░│
    │░██░░░░░│  │░░░██░░░│               32 hidden units = 32 feature detectors
    │░░██████│  │░░░██░░░│               that YOUR network learns automatically!
    └────────┘  └────────┘
       "1"         "7"

🔍 MLP LIMITATION (Why CNNs will be better):
    MLP treats each pixel INDEPENDENTLY - no spatial awareness!

    ┌────────────────────────────────────────────────────────────────────┐
    │ MLP sees: [p₁, p₂, p₃, ..., p₆₄]  (just 64 numbers)                │
    │                                                                    │
    │ MLP doesn't know that p₁ and p₂ are NEIGHBORS!                     │
    │ Shifting an image 1 pixel changes ALL feature values!              │
    │                                                                    │
    │ This is why Milestone 04 (CNNs) will show ~10% improvement:        │
    │ CNNs exploit SPATIAL STRUCTURE that MLPs ignore.                   │
    └────────────────────────────────────────────────────────────────────┘

📊 DATASET: TinyDigits (8×8 Handwritten Digits)
  - 150 training + 47 test samples (curated for fast learning)
  - 8×8 grayscale images (64 features)
  - 10 classes (digits 0-9)
  - Ships with TinyTorch (~310 KB, no download!)

📊 EXPECTED RESULTS:
- Training time: ~1-2 minutes
- Accuracy: 75-85% (decent for 8×8 images without spatial features!)
- Parameters: 2,378 (64×32 + 32 + 32×10 + 10)
- 📌 BASELINE: CNN (Milestone 04) will show improvement by using spatial structure!
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, ReLU, CrossEntropyLoss, SGD
from tinytorch.core.dataloader import TensorDataset, DataLoader

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()

# =============================================================================
# 🎯 YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone showcases YOUR complete ML system on a classification task:
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01: Tensor   │ Stores 8×8 images as 64-dim    │ Automatic gradient tracking │
# │                     │ vectors + all gradients        │ through entire network      │
# │                     │                                │                             │
# │ Module 02: ReLU     │ Non-linearity after hidden     │ Enables learning non-linear │
# │                     │ layer (64→32)                  │ digit patterns              │
# │                     │                                │                             │
# │ Module 03: Linear   │ Two layers: feature extraction │ 2,378 parameters total      │
# │                     │ (64→32) + classification (32→10)│ learned by YOUR autograd    │
# │                     │                                │                             │
# │ Module 04: Loss     │ CrossEntropyLoss for           │ Multi-class loss guides     │
# │                     │ 10-way classification          │ learning of all 10 digits   │
# │                     │                                │                             │
# │ Module 05: DataLoader│ Batches training images       │ Memory efficient iteration  │
# │                     │ (no need to load all at once)  │ over dataset                │
# │                     │                                │                             │
# │ Module 06: Autograd │ .backward() computes gradients │ Chain rule through 2 layers │
# │                     │ for 2,378 parameters           │ automatically               │
# │                     │                                │                             │
# │ Module 07: SGD      │ Updates all weights each step  │ Stochastic gradient descent │
# │                     │                                │ with learning rate          │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================
# 🆕 WHAT'S NEW SINCE MILESTONE 02 (XOR)
# =============================================================================
#
# ┌──────────────────────┬─────────────────────────┬────────────────────────────┐
# │ XOR (Milestone 02)   │ This Milestone          │ Why It Matters             │
# ├──────────────────────┼─────────────────────────┼────────────────────────────┤
# │ 2 inputs             │ 64 inputs (8×8 image)   │ Images are high-dim!       │
# │ 4 data points        │ 150+ data points        │ Need more data to learn    │
# │ Binary classification│ 10-way classification   │ Multi-class is harder      │
# │ 13 parameters        │ 2,378 parameters        │ More capacity = more power │
# │ No batching          │ DataLoader batching     │ Memory efficiency matters  │
# │ BinaryCrossEntropy   │ CrossEntropyLoss        │ Different loss for multi   │
# └──────────────────────┴─────────────────────────┴────────────────────────────┘
#
# =============================================================================


# ============================================================================
# 🎓 STUDENT CODE: Multi-Layer Perceptron
# ============================================================================

class DigitMLP:
    """
    Multi-Layer Perceptron for digit classification.

    Architecture:
      Input (64) → Linear(64→32) → ReLU → Linear(32→10) → Output
    """

    def __init__(self, input_size=64, hidden_size=32, num_classes=10):
        console.print("🧠 Building Multi-Layer Perceptron...")

        # Hidden layer
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()

        # Output layer
        self.fc2 = Linear(hidden_size, num_classes)

        console.print(f"  ✓ Hidden layer: {input_size} → {hidden_size} (with ReLU)")
        console.print(f"  ✓ Output layer: {hidden_size} → {num_classes}")

        total_params = (input_size * hidden_size + hidden_size) + \
                      (hidden_size * num_classes + num_classes)
        console.print(f"  ✓ Total parameters: {total_params:,}\n")

    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)

    def forward(self, x):
        """Forward pass through the network."""
        # Flatten if needed (8×8 → 64)
        if len(x.data.shape) > 2:
            batch_size = x.data.shape[0]
            x = Tensor(x.data.reshape(batch_size, -1))

        # Hidden layer
        x = self.fc1(x)
        x = self.relu(x)

        # Output layer
        x = self.fc2(x)
        return x

    def parameters(self):
        """Get all trainable parameters."""
        return [self.fc1.weight, self.fc1.bias,
                self.fc2.weight, self.fc2.bias]


def load_digit_dataset():
    """Load the TinyDigits dataset (8×8 curated digits)."""
    console.print(Panel.fit(
        "[bold]Loading TinyDigits Dataset[/bold]\n"
        "Curated 8×8 handwritten digits optimized for fast learning",
        title="📊 Dataset",
        border_style="cyan"
    ))

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
    train_images_np = train_data['images']  # (150, 8, 8)
    train_labels_np = train_data['labels']  # (150,)

    # Load test data
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    test_images_np = test_data['images']  # (47, 8, 8)
    test_labels_np = test_data['labels']  # (47,)

    console.print(f"✓ TinyDigits loaded ({train_images_np.shape[0] + test_images_np.shape[0]} total samples)")
    console.print(f"✓ Image shape: {train_images_np[0].shape}")
    console.print(f"✓ Classes: {np.unique(train_labels_np)}")

    # Convert to Tensors
    train_images = Tensor(train_images_np.astype(np.float32))
    train_labels = Tensor(train_labels_np.astype(np.int64))
    test_images = Tensor(test_images_np.astype(np.float32))
    test_labels = Tensor(test_labels_np.astype(np.int64))

    console.print(f"\n📊 Split:")
    console.print(f"  Training: {len(train_images.data)} samples")
    console.print(f"  Testing:  {len(test_images.data)} samples\n")

    return train_images, train_labels, test_images, test_labels


def evaluate_accuracy(model, images, labels):
    """Compute classification accuracy."""
    # Forward pass
    logits = model(images)

    # Get predictions (argmax)
    predictions = np.argmax(logits.data, axis=1)

    # Compare with labels
    correct = (predictions == labels.data).sum()
    total = len(labels.data)
    accuracy = 100.0 * correct / total

    return accuracy, predictions


def compare_batch_sizes(train_images, train_labels, test_images, test_labels):
    """
    Compare different batch sizes to show DataLoader's impact on training.

    This demonstrates a key systems trade-off in ML:
    - Larger batches: Faster throughput (fewer Python loops)
    - Smaller batches: More gradient updates per epoch
    """
    import time

    console.print(Panel.fit(
        "[bold cyan]🔬 Systems Experiment: Batch Size Impact[/bold cyan]\n\n"
        "[dim]Let's explore how batch size affects training speed and learning.\n"
        "This shows YOUR DataLoader in action![/dim]",
        title="⚙️ DataLoader Capabilities",
        border_style="yellow"
    ))

    batch_sizes = [16, 64, 256]
    epochs = 5  # Quick experiment
    results = []

    for batch_size in batch_sizes:
        console.print(f"\n[bold]Testing batch_size={batch_size}[/bold]")

        # Create DataLoader with this batch size
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        console.print(f"  Batches per epoch: {len(train_loader)}")

        # Create fresh model
        model = DigitMLP(input_size=64, hidden_size=32, num_classes=10)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()

        # Time the training
        start_time = time.time()

        for epoch in range(epochs):
            for batch_images, batch_labels in train_loader:
                logits = model(batch_images)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        elapsed = time.time() - start_time

        # Evaluate
        final_acc, _ = evaluate_accuracy(model, test_images, test_labels)

        # Calculate throughput
        total_samples = len(train_dataset) * epochs
        samples_per_sec = total_samples / elapsed
        updates = len(train_loader) * epochs

        results.append({
            'batch_size': batch_size,
            'time': elapsed,
            'accuracy': final_acc,
            'updates': updates,
            'throughput': samples_per_sec
        })

        console.print(f"  Time: {elapsed*1000:.0f}ms, Accuracy: {final_acc:.1f}%")

    # Show comparison table
    console.print("\n")
    table = Table(title="📊 Batch Size Comparison", box=box.ROUNDED)
    table.add_column("Batch Size", style="cyan", justify="center")
    table.add_column("Training Time", style="green")
    table.add_column("Gradient Updates", style="yellow", justify="center")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Throughput", style="blue")

    for r in results:
        table.add_row(
            str(r['batch_size']),
            f"{r['time']*1000:.0f}ms",
            str(r['updates']),
            f"{r['accuracy']:.1f}%",
            f"{r['throughput']:.0f} samples/s"
        )

    console.print(table)

    # Key insights
    console.print("\n")
    console.print(Panel.fit(
        "[bold]💡 Key Systems Insights:[/bold]\n\n"
        "[green]✓ Larger batches process data faster[/green] (fewer Python loops)\n"
        "[green]✓ Smaller batches give more gradient updates[/green] (more optimization steps)\n"
        "[green]✓ Throughput vs update frequency trade-off[/green]\n\n"
        "[bold]What This Shows:[/bold]\n"
        f"  • Batch 16:  Slowest but {results[0]['updates']} updates\n"
        f"  • Batch 64:  Balanced - {results[1]['updates']} updates\n"
        f"  • Batch 256: Fastest but only {results[2]['updates']} updates\n\n"
        "[bold]🚀 Production Tip:[/bold] In real systems, batch size is limited by:\n"
        "  • GPU memory (larger batches need more VRAM)\n"
        "  • Gradient noise (tiny batches → unstable training)\n"
        "  • Sweet spot: Usually 32-128 for most tasks\n\n"
        "[dim]YOUR DataLoader makes experimenting with this trivial -\n"
        "just change one number and the whole pipeline adapts![/dim]",
        title="⚙️ DataLoader Impact",
        border_style="cyan"
    ))


def train_mlp():
    """Train MLP on digit recognition task."""

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 1: THE CHALLENGE 🎯
    # ═══════════════════════════════════════════════════════════════════════

    console.print(Panel.fit(
        "[bold cyan]🎯 1986 - Deep Learning on Real Data[/bold cyan]\n\n"
        "[dim]Can multi-layer networks learn from real handwritten digits?[/dim]\n"
        "[dim]Rumelhart, Hinton & Williams prove backprop works on real tasks![/dim]",
        title="🔥 1986 Backpropagation Revolution",
        border_style="cyan",
        box=box.DOUBLE
    ))

    console.print("\n[bold]📊 The Data:[/bold]")
    train_images, train_labels, test_images, test_labels = load_digit_dataset()
    console.print("  • Dataset: 8×8 handwritten digits (UCI repository)")
    console.print(f"  • Training samples: {len(train_images.data)}")
    console.print(f"  • Test samples: {len(test_images.data)}")
    console.print("  • Classes: 10 digits (0-9)")
    console.print("  • Challenge: Recognize handwritten digits from pixels!")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 2: THE SETUP 🏗️
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]🏗️ The Architecture:[/bold]")
    console.print("""
    ┌─────────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Input Image │    │ Flatten │    │ Linear  │    │ Linear  │
    │    8×8      │───▶│   64    │───▶│ 64→32   │───▶│ 32→10   │
    │   Pixels    │    │         │    │  +ReLU  │    │         │
    └─────────────┘    └─────────┘    └─────────┘    └─────────┘
                                      Hidden Layer   10 Classes
    """)

    console.print("[bold]🔧 Components:[/bold]")
    model = DigitMLP(input_size=64, hidden_size=32, num_classes=10)
    console.print("  • Hidden layer: 64 → 32 (learns digit features)")
    console.print("  • ReLU activation: Non-linear transformations")
    console.print("  • Output layer: 32 → 10 (one per digit class)")
    console.print(f"  • Total parameters: ~{64*32 + 32 + 32*10 + 10:,}")

    console.print("\n[bold]⚙️ Hyperparameters:[/bold]")
    console.print("  • Batch size: 32 (using YOUR DataLoader!)")
    console.print("  • Learning rate: 0.01")
    console.print("  • Epochs: 20")
    console.print("  • Loss: CrossEntropyLoss (for multi-class)")
    console.print("  • Optimizer: SGD with backprop")

    # Create DataLoader
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    console.print(f"  • Batches per epoch: {len(train_loader)}")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 3: THE EXPERIMENT 🔬
    # ═══════════════════════════════════════════════════════════════════════

    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    initial_acc, _ = evaluate_accuracy(model, test_images, test_labels)

    console.print("[bold]📌 Before Training:[/bold]")
    console.print(f"  Initial accuracy: {initial_acc:.1f}% (random ~10%)")
    console.print("  Model has random weights - knows nothing about digits yet!")

    console.print("\n[bold]🔥 Training in Progress...[/bold]")
    console.print("[dim](Watch backpropagation optimize through hidden layers!)[/dim]\n")

    epochs = 20
    initial_loss = None
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }

    # Use Live display with spinner for real-time feedback
    with Live(console=console, refresh_per_second=10) as live:
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_images, batch_labels in train_loader:
                # Forward pass
                logits = model(batch_images)
                loss = loss_fn(logits, batch_labels)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.data
                batch_count += 1

                # Update spinner with current batch progress
                spinner_text = Text()
                spinner_text.append("⠋ ", style="cyan")
                spinner_text.append(f"Epoch {epoch+1:2d}/{epochs}  Batch {batch_count}/{len(train_loader)}")
                live.update(spinner_text)

            avg_loss = epoch_loss / batch_count

            # Evaluate on both train and test to detect overfitting
            train_acc, _ = evaluate_accuracy(model, train_images, train_labels)
            test_acc, _ = evaluate_accuracy(model, test_images, test_labels)

            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(train_acc)
            history["test_accuracy"].append(test_acc)

            if initial_loss is None:
                initial_loss = avg_loss

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                gap = train_acc - test_acc
                gap_indicator = "⚠️" if gap > 10 else "✓"
                live.console.print(
                    f"Epoch {epoch+1:2d}/{epochs}  "
                    f"Loss: {avg_loss:.4f}  "
                    f"Train: {train_acc:.1f}%  "
                    f"Test: {test_acc:.1f}%  "
                    f"{gap_indicator} Gap: {gap:.1f}%"
                )

    console.print("\n[green]✅ Training Complete![/green]")

    final_train_acc = history["train_accuracy"][-1]
    final_test_acc = history["test_accuracy"][-1]
    overfitting_gap = final_train_acc - final_test_acc

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 4: THE DIAGNOSIS 📊
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]📊 The Results:[/bold]\n")

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

    console.print(table)

    # Also get predictions for later use
    _, predictions = evaluate_accuracy(model, test_images, test_labels)

    console.print("\n[bold]🔍 Sample Predictions:[/bold]")
    console.print("[dim](First 10 test images)[/dim]\n")

    n_samples = 10
    for i in range(n_samples):
        true_label = test_labels.data[i]
        pred_label = predictions[i]
        status = "✓" if pred_label == true_label else "✗"
        color = "green" if pred_label == true_label else "red"
        console.print(f"  {status} True: {true_label}, Predicted: {pred_label}", style=color)

    console.print("\n[bold]💡 Key Insights:[/bold]")
    console.print("  • MLP learned to recognize handwritten digits from pixels")
    console.print("  • Hidden layer discovered useful digit features")
    console.print("  • DataLoader enabled efficient batch processing")
    console.print("  • Backprop through hidden layers works on image data!")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 5: THE REFLECTION 🌟
    # ═══════════════════════════════════════════════════════════════════════

    console.print("")
    console.print(Panel.fit(
        "[bold green]🎉 Success! Your MLP Learned to Recognize Digits![/bold green]\n\n"

        f"Test accuracy: [bold]{final_test_acc:.1f}%[/bold] (Gap: {overfitting_gap:.1f}%)\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]💡 What YOU Just Accomplished:[/bold]\n"
        "  ✓ Built multi-layer network with YOUR components\n"
        "  ✓ Trained on TinyDigits (synthetic handwritten digits)\n"
        "  ✓ Used YOUR DataLoader for efficient batching\n"
        f"  ✓ Model generalizes well (gap: {overfitting_gap:.1f}%)\n"
        "  ✓ Backprop through hidden layers works on image data!\n"
        f"  ✓ Achieved {final_test_acc:.1f}% test accuracy!\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]🎓 Why This Matters:[/bold]\n"
        "  This proved backprop works on practical tasks, not just XOR!\n"
        "  1986 paper by Rumelhart, Hinton & Williams launched\n"
        "  modern deep learning revolution.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]📌 The Key Insight:[/bold]\n"
        "  MLPs flatten images → lose spatial structure.\n"
        "  Each pixel treated independently with no neighborhood info.\n"
        "  \n"
        "  [yellow]Limitation:[/yellow] 8×8 images work, but larger images?\n"
        "  We need better architectures for spatial data...\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]🚀 What's Next:[/bold]\n"
        "[dim]Milestone 04 (CNN) will show how preserving spatial structure\n"
        "dramatically improves performance on images![/dim]",

        title="🌟 1986 MLP Breakthrough Complete",
        border_style="green",
        box=box.DOUBLE
    ))

    # Optional: Batch size experiment
    console.print("\n")
    run_experiment = input("\n🔬 Run batch size experiment? (y/n): ").lower().strip() == 'y'

    if run_experiment:
        compare_batch_sizes(train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    train_mlp()
