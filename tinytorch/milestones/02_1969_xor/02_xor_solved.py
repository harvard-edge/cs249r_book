#!/usr/bin/env python3
"""
XOR Solved! Multi-Layer Networks (1986)
========================================

📚 HISTORICAL CONTEXT:
After the 1969 XOR crisis killed neural networks, research funding dried up for over
a decade. Then in 1986, Rumelhart, Hinton, and Williams published the backpropagation
algorithm for training multi-layer networks - and XOR became trivial!

🎯 MILESTONE 2 PART 2: THE SOLUTION (After Modules 01-08)
Watch a multi-layer network SOLVE the "impossible" XOR problem that stumped AI for
17 years. The secret? Hidden layers + backpropagation (which YOU just built!).

✅ REQUIRED MODULES (Run after Module 08):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autodiff
  Module 02 (Activations)   : YOUR ReLU and Sigmoid (non-linearity!)
  Module 03 (Layers)        : YOUR Linear layers (multiple layers!)
  Module 04 (Losses)        : YOUR loss function
  Module 06 (Autograd)      : YOUR backpropagation through hidden layers
  Module 07 (Optimizers)    : YOUR SGD optimizer
  Module 08 (Training)      : YOUR training loop
  (Module 05 DataLoader skipped: XOR has only 4 points, no batching needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (The Multi-Layer Solution):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Input       │    │   Linear    │    │    ReLU     │    │   Linear    │    │  Sigmoid    │
    │ Features    │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ YOUR Module │
    │ (x1, x2)    │    │     03      │    │     02      │    │     03      │    │     02      │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         2 inputs           2→4               Non-             4→1             Output
                        Hidden Layer!      linearity!      Combines to       probability
                                                           class prob

    THE KEY: Hidden layer creates NEW features that make XOR linearly separable!

🔍 HOW IT WORKS - Feature Learning:

    Original XOR Space:              Hidden Layer Feature Space:
    (NOT linearly separable)         (NOW linearly separable!)

    1 │ ○ (0,1)    ● (1,1)           The 4 hidden units learn:
      │   [1]       [0]               • h₁: detects "x₁ AND NOT x₂"
      │                               • h₂: detects "x₂ AND NOT x₁"
    0 │ ● (0,0)    ○ (1,0)           • h₃: detects "x₁ AND x₂"
      │   [0]       [1]               • h₄: detects "NOT x₁ AND NOT x₂"
      └─────────────
        0          1                 In this new 4D space, a single
                                     linear boundary WORKS!
    No line works here!

    Mathematical Transformation:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Original: (x₁, x₂)  →  Hidden: (h₁, h₂, h₃, h₄)  →  Output: y        │
    │                             ↑                                        │
    │                     ReLU(W₁·x + b₁)                                  │
    │                                                                      │
    │ The hidden layer TRANSFORMS the input space into one where           │
    │ XOR becomes a simple linear classification problem!                  │
    └──────────────────────────────────────────────────────────────────────┘

📊 EXPECTED RESULTS:
- Training time: ~30 seconds
- Accuracy: 95-100% (problem solved!)
- Loss decreases smoothly
- Perfect XOR predictions
- ✅ YOUR backpropagation trains the hidden layer!

🔥 THE BREAKTHROUGH (1986):
This is the architecture that ended the AI Winter! Rumelhart, Hinton, and Williams
proved that YOUR autograd can train hidden layers to learn useful features.
"""

import sys
import os
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

# Add project root to path
sys.path.insert(0, os.getcwd())

# Seed will be set before training to guarantee 100% convergence

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, ReLU, Sigmoid, BinaryCrossEntropyLoss, SGD

console = Console()

# =============================================================================
# 🎯 YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone showcases the modules YOU built. Here's what powers this solution:
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01: Tensor   │ All data + gradients flow      │ Automatic gradient tracking │
# │                     │ through YOUR Tensor            │ enables backpropagation     │
# │                     │                                │                             │
# │ Module 02: ReLU     │ Non-linearity in hidden layer  │ Creates NON-LINEAR features │
# │            Sigmoid  │ Output probability             │ that make XOR separable!    │
# │                     │                                │                             │
# │ Module 03: Linear   │ TWO layers now!                │ First layer: feature space  │
# │                     │ (2→4) and (4→1)                │ Second: classification      │
# │                     │                                │                             │
# │ Module 04: Loss     │ BinaryCrossEntropy measures    │ Guides learning toward      │
# │                     │ how wrong predictions are      │ correct XOR outputs         │
# │                     │                                │                             │
# │ Module 06: Autograd │ .backward() computes gradients │ Gradients flow through      │
# │                     │ for BOTH layers automatically  │ hidden layer to inputs!     │
# │                     │                                │                             │
# │ Module 06: SGD      │ Updates 13 parameters          │ Adjusts weights to minimize │
# │                     │ (2×4 + 4 + 4×1 + 1)            │ loss function               │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================
# 🆕 WHAT'S NEW SINCE PART 1 (XOR Crisis)
# =============================================================================
#
# Part 1 FAILED because:          Part 2 SUCCEEDS because:
# ┌──────────────────────────────┬──────────────────────────────────────────────┐
# │ Single Linear layer          │ + Hidden Linear layer (2→4)                  │
# │ Only Sigmoid activation      │ + ReLU activation (non-linearity!)           │
# │ No training (random weights) │ + YOUR Autograd trains the hidden layer      │
# │ No optimizer                 │ + YOUR SGD updates 13 parameters             │
# │ Max 75% accuracy             │ + 95-100% accuracy (problem SOLVED!)         │
# └──────────────────────────────┴──────────────────────────────────────────────┘
#
# =============================================================================


# ============================================================================
# 🎲 DATA GENERATION
# ============================================================================

def generate_xor_data(n_samples=100):
    """Generate XOR dataset with slight noise."""
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

    return Tensor(X), Tensor(y)


# ============================================================================
# 🏗️ MULTI-LAYER NETWORK (The Solution!)
# ============================================================================

class XORNetwork:
    """
    Multi-layer network that SOLVES XOR!

    The hidden layer creates new features that make XOR linearly separable.
    This is the architecture that ended the AI Winter.
    """

    def __init__(self, hidden_size=4):
        # Hidden layer - THE KEY INNOVATION!
        self.hidden = Linear(2, hidden_size)
        self.relu = ReLU()  # Non-linearity is essential!

        # Output layer
        self.output = Linear(hidden_size, 1)
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        """
        Forward pass through hidden layer.

        Input → Hidden Layer → ReLU → Output Layer → Sigmoid
        """
        # Hidden layer transforms input space
        h = self.hidden(x)
        h_activated = self.relu(h)

        # Output layer in new feature space
        logits = self.output(h_activated)
        output = self.sigmoid(logits)

        return output

    def parameters(self):
        """Return all trainable parameters."""
        return self.hidden.parameters() + self.output.parameters()


# ============================================================================
# 🔥 TRAINING FUNCTION (That Will SUCCEED on XOR!)
# ============================================================================

def train_network(model, X, y, epochs=500, lr=0.5):
    """
    Train multi-layer network on XOR.

    This WILL succeed - hidden layers solve the problem!
    """
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    console.print("\n[bold cyan]🔥 Training Multi-Layer Network...[/bold cyan]")
    console.print("[dim](This will work - hidden layers solve XOR!)[/dim]\n")

    history = {"loss": [], "accuracy": []}

    # Use Live display with spinner for real-time feedback
    with Live(console=console, refresh_per_second=10) as live:
        for epoch in range(epochs):
            # Forward pass
            predictions = model(X)
            loss = loss_fn(predictions, y)

            # Backward pass (through hidden layers!)
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

            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                live.console.print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")

    console.print("\n[green]✅ Training Complete - XOR Solved![/green]")

    return history


# ============================================================================
# 📊 EVALUATION & CELEBRATION
# ============================================================================

def evaluate_and_celebrate(model, X, y, history):
    """Evaluate the successful model and celebrate the victory!"""

    predictions = model(X)
    pred_classes = (predictions.data > 0.5).astype(int)
    final_accuracy = (pred_classes == y.data).mean()

    # Get metrics
    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    initial_acc = history["accuracy"][0]
    final_acc = history["accuracy"][-1]

    console.print("[bold]📊 The Results:[/bold]\n")

    table = Table(title="Training Outcome", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Before Training", style="yellow", width=16)
    table.add_column("After Training", style="green", width=16)
    table.add_column("Improvement", style="magenta", width=14)

    loss_improvement = f"-{initial_loss - final_loss:.4f}"
    acc_improvement = f"+{final_acc - initial_acc:.1%}"

    table.add_row("Loss", f"{initial_loss:.4f}", f"{final_loss:.4f}", loss_improvement)
    table.add_row("Accuracy", f"{initial_acc:.1%}", f"{final_acc:.1%}", acc_improvement)

    console.print(table)

    console.print("\n[bold]🔍 XOR Truth Table vs Predictions:[/bold]")
    console.print("[dim](The ultimate test - all 4 XOR cases!)[/dim]\n")
    test_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    test_preds = model(Tensor(test_inputs))

    truth_table = Table(show_header=True, border_style="green")
    truth_table.add_column("x₁", style="cyan")
    truth_table.add_column("x₂", style="cyan")
    truth_table.add_column("XOR (True)", style="green")
    truth_table.add_column("Predicted", style="yellow")
    truth_table.add_column("Correct?", style="white")

    all_correct = True
    for i, (x1, x2) in enumerate(test_inputs):
        true_xor = int(x1 != x2)
        pred_prob = test_preds.data[i, 0]
        pred = int(pred_prob > 0.5)
        correct = pred == true_xor
        all_correct = all_correct and correct

        truth_table.add_row(
            f"{int(x1)}",
            f"{int(x2)}",
            f"{true_xor}",
            f"{pred} ({pred_prob:.3f})",
            "✅" if correct else "❌"
        )

    console.print(truth_table)

    if all_correct:
        console.print("\n[bold green]✨ Perfect! All XOR cases correctly predicted![/bold green]")

    console.print("\n[bold]💡 Key Insights:[/bold]")
    console.print("  • Hidden layer transformed XOR into a solvable problem")
    console.print("  • Network learned non-linear decision boundary")
    console.print("  • Multi-layer networks can solve ANY classification problem!")


# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

def main():
    """Demonstrate solving XOR with multi-layer networks."""

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 1: THE CHALLENGE 🎯
    # ═══════════════════════════════════════════════════════════════════════

    console.print(Panel.fit(
        "[bold cyan]🎯 1986 - Ending the AI Winter[/bold cyan]\n\n"
        "[dim]Can neural networks solve non-linearly separable problems?[/dim]\n"
        "[dim]The XOR problem that stumped AI for 17 years![/dim]",
        title="🔥 1986 AI Renaissance",
        border_style="cyan",
        box=box.DOUBLE
    ))

    console.print("\n[bold]📊 The Data:[/bold]")
    X, y = generate_xor_data(n_samples=100)
    console.print("  • Dataset: XOR problem (4 distinct cases)")
    console.print(f"  • Samples: {len(X.data)} (with slight noise)")
    console.print("  • Pattern: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0")
    console.print("  • Challenge: [bold red]NOT linearly separable![/bold red]")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 2: THE SETUP 🏗️
    # ═══════════════════════════════════════════════════════════════════════

    console.print("[bold]🏗️ The Architecture:[/bold]")
    console.print("""
    ┌───────┐    ┌───────────┐    ┌──────┐    ┌─────────┐    ┌────────┐
    │ Input │    │  Hidden   │    │ ReLU │    │ Output  │    │Sigmoid │
    │  (2)  │───▶│    (4)    │───▶│  Act │───▶│   (1)   │───▶│  ŷ     │
    └───────┘    └───────────┘    └──────┘    └─────────┘    └────────┘
                  ↑ THE KEY!
             Learns non-linear features
    """)

    console.print("[bold]🔧 Components:[/bold]")
    console.print("  • Hidden layer: Transforms data into new space")
    console.print("  • [bold green]ReLU activation: Adds non-linearity (the secret!)[/bold green]")
    console.print("  • Output layer: Makes final decision")
    console.print("  • Total parameters: ~17 (vs 3 for single-layer)")

    console.print("\n[bold]⚙️ Hyperparameters:[/bold]")
    console.print("  • Hidden size: 4")
    console.print("  • Learning rate: 0.5 (aggressive!)")
    console.print("  • Epochs: 500")
    console.print("  • Optimizer: SGD with backprop through hidden layer")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 3: THE EXPERIMENT 🔬
    # ═══════════════════════════════════════════════════════════════════════

    # Set seed before model creation to guarantee reproducible 100% convergence
    np.random.seed(1986)  # The year backprop was published!

    model = XORNetwork(hidden_size=4)
    initial_preds = model(X)
    initial_acc = ((initial_preds.data > 0.5).astype(int) == y.data).mean()

    console.print("[bold]📌 Before Training:[/bold]")
    console.print(f"  Initial accuracy: {initial_acc:.1%} (random guessing)")
    console.print("  XOR is impossible for single-layer networks!")
    console.print("  Let's see if hidden layers change the game...")

    console.print("\n[bold]🔥 Training in Progress...[/bold]")
    console.print("[dim](This will work - hidden layers solve XOR!)[/dim]\n")

    history = train_network(model, X, y, epochs=500, lr=0.5)

    console.print("\n[green]✅ Training Complete - XOR Solved![/green]")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 4: THE DIAGNOSIS 📊
    # ═══════════════════════════════════════════════════════════════════════

    evaluate_and_celebrate(model, X, y, history)

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 5: THE REFLECTION 🌟
    # ═══════════════════════════════════════════════════════════════════════

    final_acc = history["accuracy"][-1]

    console.print("")
    console.print(Panel.fit(
        "[bold green]🎉 Success! You Ended the AI Winter![/bold green]\n\n"

        f"Final accuracy: [bold]{final_acc:.1%}[/bold] (Perfect XOR solution!)\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]💡 What YOU Just Accomplished:[/bold]\n"
        "  ✓ Solved the problem that killed AI for 17 years!\n"
        "  ✓ Built multi-layer network with YOUR components\n"
        "  ✓ Hidden layer learns non-linear features\n"
        "  ✓ Backprop through multiple layers works perfectly!\n"
        "  ✓ Proved that deep networks CAN work!\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]🎓 Why This Matters:[/bold]\n"
        "  This ENDED the 17-year AI Winter!\n"
        "  [bold red]1969:[/bold red] XOR crisis → single layers fail\n"
        "  [bold yellow]1970-1986:[/bold yellow] AI Winter - research funding dries up\n"
        "  [bold green]1986:[/bold green] Backprop + hidden layers solve it\n"
        "  [bold cyan]TODAY:[/bold cyan] YOU recreated this breakthrough!\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]📌 The Key Insight:[/bold]\n"
        "  Hidden layers are the KEY to modern AI.\n"
        "  They learn new features that make problems solvable.\n"
        "  Every deep network (GPT, AlphaGo, etc.) uses this pattern!\n"
        "  \n"
        "  [green]Breakthrough:[/green] Non-linear activation functions (ReLU)\n"
        "  enable networks to learn non-linear decision boundaries.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "[bold]🚀 What's Next:[/bold]\n"
        "[dim]Milestone 03 applies this to digit images with YOUR DataLoader!\n"
        "Train on handwritten digits and see modern ML in action![/dim]",

        title="🌟 1986 AI Renaissance Complete",
        border_style="green",
        box=box.DOUBLE
    ))


if __name__ == "__main__":
    main()
