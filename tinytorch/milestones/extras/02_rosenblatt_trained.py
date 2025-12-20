#!/usr/bin/env python3
"""
The Perceptron (1957) - Frank Rosenblatt [WITH TRAINING]
=========================================================

📚 HISTORICAL CONTEXT:
In 1957, Frank Rosenblatt created the Perceptron - the first machine that could
LEARN from examples. Before this, all pattern recognition was hand-coded!
This breakthrough proved machines could improve themselves through training.

🎯 MILESTONE 1 PART 2: TRAINED PERCEPTRON (After Modules 01-08)

Now that you've completed training modules, let's see the SAME architecture
actually LEARN! Watch random weights → intelligent predictions through training.

✅ REQUIRED MODULES (Run after Module 08):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure
  Module 02 (Activations)   : YOUR sigmoid activation
  Module 03 (Layers)        : YOUR Linear layer
  Module 04 (Losses)        : YOUR loss function (BinaryCrossEntropyLoss)
  Module 06 (Autograd)      : YOUR automatic differentiation
  Module 07 (Optimizers)    : YOUR SGD optimizer
  Module 08 (Training)      : YOUR training loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Single-Layer Perceptron):
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
    │ Input       │     │ Linear      │     │ Sigmoid     │     │ Output  │
    │  (x₁, x₂)   │────▶│ w₁x₁+w₂x₂+b│────▶│  σ(z)       │────▶│  ŷ      │
    │ YOUR M1     │     │ YOUR M3     │     │ YOUR M2     │     │ [0,1]   │
    └─────────────┘     └─────────────┘     └─────────────┘     └─────────┘
           ↑                   ↑
           │                   │ Gradients flow backward!
           │                   │ (YOUR Module 06 Autograd)
           └───────────────────┘

# =============================================================================
# 📊 YOUR MODULES IN ACTION
# =============================================================================
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 06: Autograd │ Computes ∂Loss/∂w for each     │ Enables automatic learning! │
# │                     │ weight automatically           │ No manual derivatives       │
# │                     │                                │                             │
# │ Module 07: SGD      │ Updates weights: w = w - lr*∂L │ The learning algorithm!     │
# │                     │ after each batch               │                             │
# │                     │                                │                             │
# │ Module 08: Training │ Orchestrates forward → loss →  │ The complete learning       │
# │                     │ backward → update cycle        │ loop you control!           │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================

🔥 THE TRANSFORMATION:
  Before Training:  50% accuracy (random guessing) ❌
  After Training:   95%+ accuracy (intelligent!)   ✅

  SAME architecture. SAME data. Just add LEARNING.

💡 KEY INSIGHT:
The magic isn't in the architecture (it's just 3 parameters!)
The magic is in YOUR training loop that adjusts those parameters to fit data.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, Sigmoid, BinaryCrossEntropyLoss, SGD

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich import box

console = Console()

# ============================================================================
# 🎓 STUDENT CODE: Same Perceptron class from forward_pass.py
# ============================================================================

class Perceptron:
    """
    Simple perceptron: Linear + Sigmoid

    SAME as forward_pass.py - but now we'll TRAIN it!
    """

    def __init__(self, input_size=2, output_size=1):
        self.linear = Linear(input_size, output_size)
        self.activation = Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        """Return all trainable parameters."""
        return self.linear.parameters()


# ============================================================================
# 📊 DATA GENERATION: Linearly separable 2D data
# ============================================================================

def generate_data(n_samples=100, seed=None):
    """Generate linearly separable data."""
    if seed is not None:
        np.random.seed(seed)

    # Class 1: Top-right cluster (high x1, high x2)
    class1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([3, 3])
    labels1 = np.ones((n_samples // 2, 1))

    # Class 0: Bottom-left cluster (low x1, low x2)
    class0 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([1, 1])
    labels0 = np.zeros((n_samples // 2, 1))

    # Combine
    X = np.vstack([class1, class0])
    y = np.vstack([labels1, labels0])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return Tensor(X), Tensor(y)


# ============================================================================
# 🎯 TRAINING FUNCTION
# ============================================================================

def train_perceptron(model, X, y, epochs=100, lr=0.1):
    """Train the perceptron using SGD."""

    # Setup training components
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    console.print("\n[bold cyan]🔥 Starting Training...[/bold cyan]\n")

    history = {"loss": [], "accuracy": []}

    # Use Live display with spinner for real-time feedback
    with Live(console=console, refresh_per_second=10) as live:
        for epoch in range(epochs):
            # Forward pass
            predictions = model(X)
            loss = loss_fn(predictions, y)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Calculate accuracy
            pred_classes = (predictions.data > 0.5).astype(int)
            accuracy = (pred_classes == y.data).mean()

            history["loss"].append(loss.data.item())
            history["accuracy"].append(accuracy)

            # Update spinner with current progress
            spinner_text = Text()
            spinner_text.append("⠋ ", style="cyan")
            spinner_text.append(f"Epoch {epoch+1:3d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")
            live.update(spinner_text)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                live.console.print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")

    console.print("\n[bold green]✅ Training Complete![/bold green]\n")

    return history


# ============================================================================
# 📈 EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_model(model, X, y):
    """Evaluate trained model."""
    predictions = model(X)
    pred_classes = (predictions.data > 0.5).astype(int)
    accuracy = (pred_classes == y.data).mean()

    # Get model weights
    weights = model.linear.weight.data
    bias = model.linear.bias.data

    return accuracy, weights, bias, predictions


def main():
    """Main training pipeline."""

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 1: THE CHALLENGE 🎯
    # ═══════════════════════════════════════════════════════════════════════

    console.print(Panel.fit(
        "[bold cyan]🎯 1957 - The First Neural Network[/bold cyan]\n\n"
        "[dim]Can a machine learn from examples to classify data?[/dim]\n"
        "[dim]Frank Rosenblatt's perceptron attempts to answer this![/dim]",
        title="🔥 1957 Perceptron Revolution",
        border_style="cyan",
        box=box.DOUBLE
    ))

    console.print("\n[bold]📊 The Data:[/bold]")
    X, y = generate_data(n_samples=100, seed=42)
    console.print("  • Dataset: Linearly separable 2D points")
    console.print(f"  • Samples: {len(X.data)} (50 per class)")
    console.print("  • Challenge: Learn decision boundary from examples")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 2: THE SETUP 🏗️
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]🏗️ The Architecture:[/bold]")
    console.print("""
    ┌─────────────┐    ┌──────────────┐    ┌──────────┐
    │   Input     │    │   Weights    │    │  Output  │
    │   (x₁, x₂)  │───▶│ w₁·x₁ + w₂·x₂│───▶│    ŷ     │
    │  2 features │    │   + bias     │    │ binary   │
    └─────────────┘    └──────────────┘    └──────────┘
    """)

    console.print("[bold]🔧 Components:[/bold]")
    console.print("  • Single layer: Maps 2D input → 1D output")
    console.print("  • Linear transformation: Weighted sum")
    console.print("  • Total parameters: 3 (2 weights + 1 bias)")

    console.print("\n[bold]⚙️ Hyperparameters:[/bold]")
    console.print("  • Learning rate: 0.1")
    console.print("  • Epochs: 100")
    console.print("  • Optimizer: Gradient descent")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 3: THE EXPERIMENT 🔬
    # ═══════════════════════════════════════════════════════════════════════

    model = Perceptron(input_size=2, output_size=1)
    acc_before, w_before, b_before, _ = evaluate_model(model, X, y)

    console.print("[bold]📌 Before Training:[/bold]")
    console.print(f"  Initial accuracy: {acc_before:.1%} (random guessing)")
    console.print(f"  Weights: w₁={w_before.flatten()[0]:.3f}, w₂={w_before.flatten()[1]:.3f}, b={b_before.flatten()[0]:.3f}")
    console.print("  Model has random weights - no knowledge yet!")

    console.print("\n[bold]🔥 Training in Progress...[/bold]")
    console.print("[dim](Watch gradient descent optimize the weights!)[/dim]\n")

    history = train_perceptron(model, X, y, epochs=100, lr=0.1)

    console.print("\n[green]✅ Training Complete![/green]")

    acc_after, w_after, b_after, predictions = evaluate_model(model, X, y)

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 4: THE DIAGNOSIS 📊
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]📊 The Results:[/bold]\n")

    table = Table(title="Training Outcome", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Before Training", style="yellow", width=16)
    table.add_column("After Training", style="green", width=16)
    table.add_column("Improvement", style="magenta", width=14)

    table.add_row(
        "Accuracy",
        f"{acc_before:.1%}",
        f"{acc_after:.1%}",
        f"+{(acc_after - acc_before):.1%}" if acc_after > acc_before else "—"
    )

    table.add_row(
        "Weight w₁",
        f"{w_before.flatten()[0]:.4f}",
        f"{w_after.flatten()[0]:.4f}",
        "→ Learned" if abs(w_after.flatten()[0] - w_before.flatten()[0]) > 0.1 else "—"
    )

    table.add_row(
        "Weight w₂",
        f"{w_before.flatten()[1]:.4f}",
        f"{w_after.flatten()[1]:.4f}",
        "→ Learned" if abs(w_after.flatten()[1] - w_before.flatten()[1]) > 0.1 else "—"
    )

    table.add_row(
        "Bias b",
        f"{b_before.flatten()[0]:.4f}",
        f"{b_after.flatten()[0]:.4f}",
        "→ Learned" if abs(b_after.flatten()[0] - b_before.flatten()[0]) > 0.1 else "—"
    )

    console.print(table)

    console.print("\n[bold]🔍 Sample Predictions:[/bold]")
    console.print("[dim](First 10 samples - seeing is believing!)[/dim]\n")

    n_samples = min(10, len(y.data))
    for i in range(n_samples):
        true_val = int(y.data.flatten()[i])
        pred_val = int(predictions.data.flatten()[i])
        status = "✓" if pred_val == true_val else "✗"
        color = "green" if pred_val == true_val else "red"
        console.print(f"  {status} True: {true_val}, Predicted: {pred_val}", style=color)

    console.print("\n[bold]💡 Key Insights:[/bold]")
    console.print("  • The model LEARNED from data (not programmed!)")
    console.print(f"  • Weights changed: random → optimized values")
    console.print("  • Simple gradient descent found the solution")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 5: THE REFLECTION 🌟
    # ═══════════════════════════════════════════════════════════════════════

    console.print("")
    if acc_after >= 0.9:
        console.print(Panel.fit(
            "[bold green]🎉 Success! Your Perceptron Learned to Classify![/bold green]\n\n"

            f"Final accuracy: [bold]{acc_after:.1%}[/bold]\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            "[bold]💡 What YOU Just Accomplished:[/bold]\n"
            "  ✓ Built the FIRST neural network (1957 Rosenblatt)\n"
            "  ✓ Implemented forward pass with YOUR Tensor\n"
            "  ✓ Used gradient descent to optimize weights\n"
            "  ✓ Watched machine learning happen in real-time!\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            "[bold]🎓 Why This Matters:[/bold]\n"
            "  This is the FOUNDATION of all neural networks.\n"
            "  Every model from GPT-4 to AlphaGo uses this same core idea:\n"
            "  Adjust weights via gradients to minimize error.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            "[bold]📌 The Key Insight:[/bold]\n"
            "  The architecture is simple (~10 lines of code).\n"
            "  The MAGIC is the training loop:\n"
            "    Forward → Loss → Backward → Update\n"
            "  \n"
            "  [yellow]Limitation:[/yellow] Single layers can only solve\n"
            "  linearly separable problems.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            "[bold]🚀 What's Next:[/bold]\n"
            "[dim]Milestone 02 shows what happens when data ISN'T\n"
            "linearly separable... the 17-year AI Winter begins![/dim]",

            title="🌟 1957 Perceptron Complete",
            border_style="green",
            box=box.DOUBLE
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠️  Training in Progress[/bold yellow]\n\n"
            f"Current accuracy: {acc_after:.1%}\n\n"
            "Your perceptron is learning but needs more training.\n"
            "Try: More epochs (500+) or different learning rate (0.5).\n\n"
            "[dim]The gradient descent algorithm is working - just needs more steps![/dim]",
            title="🔄 Learning in Progress",
            border_style="yellow",
            box=box.DOUBLE
        ))


if __name__ == "__main__":
    main()
