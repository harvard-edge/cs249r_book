#!/usr/bin/env python3
"""
The XOR Crisis (1969) - Minsky & Papert
========================================

📚 HISTORICAL CONTEXT:
In 1969, Marvin Minsky and Seymour Papert published "Perceptrons," mathematically
proving that single-layer perceptrons CANNOT solve the XOR problem. This revelation
killed neural network research funding for over a decade - the "AI Winter."

🎯 MILESTONE 2 PART 1: THE CRISIS (After Modules 01-04)

This demonstrates WHY the crisis happened. Watch a perceptron fail to learn XOR,
no matter how much we train it. This is what convinced the world that neural
networks were a dead end.

✅ REQUIRED MODULES (Run after Module 04):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure
  Module 02 (Activations)   : YOUR sigmoid activation
  Module 03 (Layers)        : YOUR Linear layer (single layer only!)
  Module 04 (Losses)        : YOUR loss function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Single-Layer Perceptron - WILL FAIL):
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Input       │     │ Linear      │     │ Sigmoid     │
    │  (x₁, x₂)   │────▶│ w₁x₁+w₂x₂+b│────▶│  σ(z)       │
    │ YOUR M1     │     │ YOUR M3     │     │ YOUR M2     │
    └─────────────┘     └─────────────┘     └─────────────┘
                                                    │
                                                    ▼
                                          Output: 0 or 1?
                                          ❌ Cannot separate XOR!

# =============================================================================
# 📊 YOUR MODULES IN ACTION
# =============================================================================
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ What This Proves            │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01: Tensor   │ Stores XOR inputs and outputs  │ Data flows correctly        │
# │                     │                                │                             │
# │ Module 02: Sigmoid  │ Squashes linear output to [0,1]│ Activation works, but...    │
# │                     │                                │ can't fix linear limits     │
# │                     │                                │                             │
# │ Module 03: Linear   │ Computes w₁·x₁ + w₂·x₂ + b     │ Creates LINEAR boundary     │
# │                     │                                │ ❌ XOR needs NON-linear!    │
# │                     │                                │                             │
# │ Module 04: Loss     │ Measures prediction error      │ Loss won't decrease!        │
# │                     │                                │ (stuck at ~0.69)            │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================

🔍 THE XOR PROBLEM - Why It's Impossible for Single Layers:

XOR (Exclusive OR) outputs 1 when inputs DIFFER, 0 when they're the SAME:

    Visual Representation:        Truth Table:

    1 │ ○ (0,1)    ● (1,1)        │ x₁ │ x₂ │ XOR │
      │   [1]       [0]           ├────┼────┼─────┤
      │                           │ 0  │ 0  │  0  │ ← same
    0 │ ● (0,0)    ○ (1,0)        │ 0  │ 1  │  1  │ ← different
      │   [0]       [1]           │ 1  │ 0  │  1  │ ← different
      └─────────────────          │ 1  │ 1  │  0  │ ← same
        0          1              └────┴────┴─────┘

🚫 THE FUNDAMENTAL PROBLEM:

No single straight line can separate the points!

    Try drawing a line:           Any line fails:

    1 │ 1 ╱ ╱ ╱ 0                1 │ 1 ╲ ╲ ╲ 0
      │ ╱ ╱ ╱ ╱ ╱                  │ ╲ ╲ ╲ ╲ ╲
    0 │ 0 ╱ ╱ ╱ 1                0 │ 0 ╲ ╲ ╲ 1
      └────────────                └────────────
        Line can't separate!        Still wrong!

This is called "non-linear separability" - the problem that ended the first
neural network era.

⚠️ WHAT TO EXPECT:
- Training will complete (no errors)
- Loss will NOT decrease (stuck around 0.69)
- Accuracy will NOT improve (stuck at 50% - random guessing)
- The model CANNOT learn XOR - Minsky was right!

This failure launched the AI Winter. Part 2 (xor_solved.py) shows the solution!
"""

import sys
import os
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, Sigmoid, BinaryCrossEntropyLoss, SGD

console = Console()


# ============================================================================
# 🎲 DATA GENERATION
# ============================================================================

def generate_xor_data(n_samples=100):
    """
    Generate XOR dataset with slight noise.

    Returns clean XOR data to clearly demonstrate the failure.
    """
    console.print("\n[bold]Step 1:[/bold] Generating XOR dataset...")

    # Generate each XOR case with repetition
    samples_per_case = n_samples // 4

    # Case 1: (0,0) → 0
    x1 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([0.0, 0.0])
    y1 = np.zeros((samples_per_case, 1))

    # Case 2: (0,1) → 1
    x2 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([0.0, 1.0])
    y2 = np.ones((samples_per_case, 1))

    # Case 3: (1,0) → 1
    x3 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([1.0, 0.0])
    y3 = np.ones((samples_per_case, 1))

    # Case 4: (1,1) → 0
    x4 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([1.0, 1.0])
    y4 = np.zeros((samples_per_case, 1))

    # Combine and shuffle
    X = np.vstack([x1, x2, x3, x4])
    y = np.vstack([y1, y2, y3, y4])

    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    console.print(f"  ✓ Created [bold]{n_samples}[/bold] XOR samples")
    console.print(f"  ✓ Problem: [bold red]NOT linearly separable![/bold red]")

    return Tensor(X), Tensor(y)


# ============================================================================
# 🏗️ SINGLE-LAYER PERCEPTRON (The Architecture That FAILS)
# ============================================================================

class SingleLayerPerceptron:
    """
    Single-layer perceptron - the architecture that CANNOT solve XOR.

    This is the exact architecture Minsky proved insufficient in 1969.
    """

    def __init__(self):
        self.linear = Linear(2, 1)
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        """Forward pass: Input → Linear → Sigmoid → Output"""
        logits = self.linear(x)
        output = self.sigmoid(logits)
        return output

    def parameters(self):
        """Return trainable parameters."""
        return self.linear.parameters()


# ============================================================================
# 🔥 TRAINING FUNCTION (That Will FAIL on XOR)
# ============================================================================

def train_perceptron(model, X, y, epochs=100, lr=0.1):
    """
    Train single-layer perceptron on XOR.

    This will fail - the model CANNOT learn XOR.
    """
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    console.print("\n[bold cyan]🔥 Attempting to Train on XOR...[/bold cyan]")
    console.print("[dim](This will fail - Minsky proved it mathematically!)[/dim]\n")

    history = {"loss": [], "accuracy": []}

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

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            console.print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")

    console.print("\n[bold yellow]⚠️  Training Complete (But Failed to Learn!)[/bold yellow]\n")

    return history


# ============================================================================
# 📊 EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_and_explain(model, X, y, history):
    """Evaluate the failed model and explain WHY it failed."""

    predictions = model(X)
    pred_classes = (predictions.data > 0.5).astype(int)
    final_accuracy = (pred_classes == y.data).mean()

    # Get final metrics
    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    initial_acc = history["accuracy"][0]
    final_acc = history["accuracy"][-1]

    # Show results table
    table = Table(title="\n🎯 The XOR Crisis - Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Initial", style="white")
    table.add_column("Final", style="white")
    table.add_column("Change", style="bold")

    loss_change = "No improvement" if abs(final_loss - initial_loss) < 0.1 else f"{initial_loss - final_loss:+.4f}"
    acc_change = "No improvement" if abs(final_acc - initial_acc) < 0.05 else f"{final_acc - initial_acc:+.1%}"

    table.add_row("Loss", f"{initial_loss:.4f}", f"{final_loss:.4f}", loss_change)
    table.add_row("Accuracy", f"{initial_acc:.1%}", f"{final_acc:.1%}", acc_change)

    console.print(table)

    # Show the failure
    if final_accuracy < 0.6:
        console.print(Panel(
            "[bold red]❌ FAILURE: Cannot Learn XOR[/bold red]\n\n"
            f"Final accuracy: {final_accuracy:.1%} (essentially random guessing)\n"
            f"Loss stuck at: {final_loss:.4f} (not decreasing)\n\n"
            "[bold]This is the XOR Crisis![/bold]\n"
            "Single-layer perceptrons cannot solve non-linearly separable problems.",
            title="⚠️  The 1969 AI Winter Begins",
            border_style="red"
        ))
    else:
        console.print(Panel(
            "[yellow]⚠️  PARTIAL SUCCESS (Unexpected!)[/yellow]\n\n"
            f"Accuracy: {final_accuracy:.1%}\n"
            "This shouldn't happen with clean XOR data.\n"
            "The problem is fundamentally non-linearly separable.",
            border_style="yellow"
        ))

    # Show XOR truth table vs predictions
    console.print("\n[bold]XOR Truth Table vs Model Predictions:[/bold]")
    test_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    test_preds = model(Tensor(test_inputs))

    truth_table = Table(show_header=True)
    truth_table.add_column("x₁", style="cyan")
    truth_table.add_column("x₂", style="cyan")
    truth_table.add_column("XOR (True)", style="green")
    truth_table.add_column("Predicted", style="yellow")
    truth_table.add_column("Correct?", style="white")

    for i, (x1, x2) in enumerate(test_inputs):
        true_xor = int(x1 != x2)
        pred = int(test_preds.data[i, 0] > 0.5)
        correct = "✓" if pred == true_xor else "✗"
        truth_table.add_row(
            f"{int(x1)}",
            f"{int(x2)}",
            f"{true_xor}",
            f"{pred}",
            correct
        )

    console.print(truth_table)


# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

def main():
    """Demonstrate the XOR crisis - single-layer perceptron failure."""

    console.print(Panel.fit(
        "[bold]The XOR Crisis (1969) - Minsky & Papert[/bold]\n\n"
        "[dim]Watch a single-layer perceptron FAIL to learn XOR.[/dim]\n"
        "[dim]This failure convinced the world neural networks were useless.[/dim]",
        border_style="red"
    ))

    # Generate data
    X, y = generate_xor_data(n_samples=100)

    # Create single-layer perceptron
    console.print("\n[bold]Step 2:[/bold] Creating single-layer perceptron...")
    model = SingleLayerPerceptron()
    console.print("  ✓ Architecture: Input(2) → Linear(2→1) → Sigmoid → Output")
    console.print("  ⚠️  [bold red]No hidden layer - this is the problem![/bold red]")

    # Attempt to train (will fail)
    console.print("\n[bold]Step 3:[/bold] Training on XOR...")
    history = train_perceptron(model, X, y, epochs=100, lr=0.5)

    # Evaluate and explain the failure
    evaluate_and_explain(model, X, y, history)

    # Historical context
    console.print(Panel(
        "[bold]💡 Historical Significance[/bold]\n\n"
        "[bold cyan]1969:[/bold cyan] Minsky & Papert prove single-layer networks can't solve XOR\n"
        "[bold red]1970s:[/bold red] AI Winter begins - funding disappears\n"
        "[bold yellow]1986:[/bold yellow] Multi-layer networks + backprop solve it (see xor_solved.py!)\n"
        "[bold green]Today:[/bold green] Deep learning powers GPT, AlphaGo, etc.\n\n"
        "[dim]The solution? Hidden layers! See [bold]xor_solved.py[/bold] to witness the revival.[/dim]",
        title="🌨️  The AI Winter",
        border_style="blue"
    ))


if __name__ == "__main__":
    main()
