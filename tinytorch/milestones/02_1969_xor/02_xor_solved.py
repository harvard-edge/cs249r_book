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
- Training time: about a second
- Accuracy: 100% on the four XOR cases (problem solved!)
- Pass condition: all 4 truth-table rows correct after training, AND
  YOUR backprop moved the hidden-layer weights (not just the output layer)
- Loss decreases smoothly
- Perfect XOR predictions
- ✅ YOUR backpropagation trains the hidden layer!

🔥 THE BREAKTHROUGH (1986):
This is the architecture that ended the AI Winter! Rumelhart, Hinton, and Williams
proved that YOUR autograd can train hidden layers to learn useful features.
"""

import argparse
import sys
from pathlib import Path
import os
import numpy as np
rng = np.random.default_rng(7)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
# │ Module 07: SGD      │ Updates 17 parameters          │ Adjusts weights to minimize │
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
# │ No optimizer                 │ + YOUR SGD updates 17 parameters             │
# │ Max 75% accuracy             │ + 100%: all 4 XOR cases (problem SOLVED!)    │
# └──────────────────────────────┴──────────────────────────────────────────────┘
#
# =============================================================================


# =============================================================================
# 🎓 ZONE 1: STUDENT CORE LEGO BRICKS (Model Architecture)
# =============================================================================

class XORNetwork:
    """
    Multi-layer network that SOLVES XOR!

    The hidden layer creates new features that make XOR linearly separable.
    This is the architecture that ended the AI Winter.
    """

    def __init__(self, hidden_size=4):
        # Hidden layer: the key innovation!
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


# =============================================================================
# 📊 ZONE 2: MILESTONE HARNESS & VALIDATION UX
# =============================================================================

def generate_xor_data(n_samples=100):
    """Generate balanced XOR cases with slight noise."""
    if n_samples < 4 or n_samples % 4:
        raise ValueError("n_samples must be a positive multiple of four")
    # Generate each XOR case with repetition
    samples_per_case = n_samples // 4

    # Case 1: (0,0) → 0
    x1 = rng.standard_normal((samples_per_case, 2)) * 0.1 + np.array([0.0, 0.0])
    y1 = np.zeros((samples_per_case, 1))

    # Case 2: (0,1) → 1
    x2 = rng.standard_normal((samples_per_case, 2)) * 0.1 + np.array([0.0, 1.0])
    y2 = np.ones((samples_per_case, 1))

    # Case 3: (1,0) → 1
    x3 = rng.standard_normal((samples_per_case, 2)) * 0.1 + np.array([1.0, 0.0])
    y3 = np.ones((samples_per_case, 1))

    # Case 4: (1,1) → 0
    x4 = rng.standard_normal((samples_per_case, 2)) * 0.1 + np.array([1.0, 1.0])
    y4 = np.zeros((samples_per_case, 1))

    # Combine and shuffle
    X = np.vstack([x1, x2, x3, x4])
    y = np.vstack([y1, y2, y3, y4])

    indices = rng.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return Tensor(X), Tensor(y)


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

    history = {"loss": [], "accuracy": [], "hidden_grad_max": 0.0}

    # Use Live display with spinner for real-time feedback
    with Live(console=console, refresh_per_second=10) as live:
        for epoch in range(epochs):
            # Forward pass
            predictions = model(X)
            loss = loss_fn(predictions, y)

            # Backward pass (through hidden layers!)
            loss.backward()

            # Record how much gradient reached the HIDDEN layer. If backprop
            # stops at the output layer, this stays at zero.
            g = model.hidden.weight.grad
            if g is not None:
                g = np.asarray(getattr(g, "data", g), dtype=np.float64)
                history["hidden_grad_max"] = max(history["hidden_grad_max"],
                                                 float(np.abs(g).max()))

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

    # The per-epoch accuracy above is measured BEFORE each update, so it lags
    # the model by one step. Report the trained model itself; the milestone's
    # pass/fail verdict comes later, from the XOR truth table.
    final_preds = model(X)
    final_accuracy = ((final_preds.data > 0.5).astype(int) == y.data).mean()
    console.print(f"\n[bold]Training complete.[/bold] Accuracy of the trained model: {final_accuracy:.1%}")

    return history

# ============================================================================
# 🔎 LOSS FORWARD CHECK (is YOUR Module 04 loss value right?)
# ============================================================================
#
# 2026-09-29: with BinaryCrossEntropy's forward returning 0, this milestone
# still passed. Training only needs the gradient, and the gradient comes from
# the backward pass, so the network learned while every printed loss was
# wrong. One batch, checked against an independent NumPy computation before
# training starts, catches that without touching the training run.

BCE_EPSILON = 1e-7  # the clip Module 04 applies before taking logs


def reference_bce(predictions, targets):
    """Mean binary cross-entropy computed directly in NumPy (Module 04's definition).

    The clip runs in the predictions' own dtype, as Module 04's does: in
    float32, 1 - 1e-7 rounds to 1 - 1.19e-7, which changes log(1 - p) at p = 1.
    """
    p = np.clip(np.asarray(predictions), BCE_EPSILON, 1 - BCE_EPSILON).astype(np.float64)
    t = np.asarray(targets, dtype=np.float64)
    return float(np.mean(-(t * np.log(p) + (1 - t) * np.log(1 - p))))


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


def press_enter_to_continue():
    """Pause in interactive sessions; skip in CI or non-interactive runs."""
    if os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1" or os.environ.get("CI") == "true":
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError:
            pass
        console.print()

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
    final_loss = float(BinaryCrossEntropyLoss()(predictions, y).data)
    initial_acc = history["accuracy"][0]
    final_acc = final_accuracy

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

    press_enter_to_continue()

    console.print("[bold]🔍 XOR Truth Table vs Predictions:[/bold]")
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

    return all_correct, final_acc


# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

DEFAULT_SEED = 11  # why 11: see the seed comment in main()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="XOR solved with a hidden layer (1986)", allow_abbrev=False,
        epilog="Try --seed 5 to watch a correct network stall at 75% in a dead-ReLU saddle point.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"weight-init seed (default {DEFAULT_SEED}, which converges with correct code)")
    args, _unknown = parser.parse_known_args(argv)
    return args


def main():
    """Demonstrate solving XOR with multi-layer networks."""
    args = parse_args()

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

    press_enter_to_continue()

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

    press_enter_to_continue()

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 3: THE EXPERIMENT 🔬
    # ═══════════════════════════════════════════════════════════════════════

    # Seed the layers' weight-init RNG so the run is reproducible.
    #
    # Seed choice (measured 2026-09-29 over seeds 0-59 and 1986, reference
    # implementation, 500 epochs, lr=0.5, 4 hidden units):
    #   * 51 of 61 seeds solve XOR with correct code; 10 land in the 75%
    #     dead-ReLU saddle point.
    #   * The old seed, 1986, draws hidden features that ALREADY separate XOR,
    #     so training only the output layer solved it and a broken backprop
    #     passed this milestone.
    #   * Seed 11's initial hidden features are NOT linearly separable on the
    #     four XOR inputs, so no output-layer-only training can solve it: the
    #     hidden layer has to learn. With correct code it converges with every
    #     truth-table probability within 0.005 of its target, and still
    #     converges at lr 0.3 or 0.8 and at 300 epochs.
    import tinytorch.core.layers as _layers
    _layers.rng = np.random.default_rng(args.seed)

    model = XORNetwork(hidden_size=4)
    hidden_w_before = np.array(model.hidden.weight.data, dtype=np.float64, copy=True)
    initial_preds = model(X)
    initial_acc = ((initial_preds.data > 0.5).astype(int) == y.data).mean()

    console.print("[bold]📌 Before Training:[/bold]")
    console.print(f"  Initial accuracy: {initial_acc:.1%} (random guessing)")
    console.print("  XOR is impossible for single-layer networks!")
    console.print("  Let's see if hidden layers change the game...")

    loss_failure = loss_forward_failure(BinaryCrossEntropyLoss(), initial_preds, y, reference_bce)
    if loss_failure:
        console.print(Panel.fit(
            "[bold red]❌ YOUR loss value is wrong[/bold red]\n\n"
            f"{loss_failure}.\n\n"
            "Training could still work, because the gradient comes from the backward\n"
            "pass, but every loss this milestone prints would be wrong. BCE is\n"
            "mean(-(y·log(p) + (1-y)·log(1-p))) with p clipped to [1e-7, 1-1e-7].",
            title="❌ Milestone FAILED",
            border_style="red",
            box=box.DOUBLE
        ))
        return 1
    console.print("  [green]✓[/green] YOUR BinaryCrossEntropyLoss matches a NumPy check on this batch")

    press_enter_to_continue()

    console.print("[bold]🔥 Training in Progress...[/bold]")
    console.print("[dim](This will work - hidden layers solve XOR!)[/dim]\n")

    history = train_network(model, X, y, epochs=500, lr=0.5)
    hidden_w_after = np.asarray(model.hidden.weight.data, dtype=np.float64)
    hidden_rel_change = float(np.linalg.norm(hidden_w_after - hidden_w_before)
                              / max(np.linalg.norm(hidden_w_before), 1e-12))

    #console.print("\n[green]✅ Training Complete - XOR Solved![/green]")

    console.print("\n" + "─" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 4: THE DIAGNOSIS 📊
    # ═══════════════════════════════════════════════════════════════════════

    truth_table_correct, final_acc = evaluate_and_celebrate(model, X, y, history)

    press_enter_to_continue()

    # ═══════════════════════════════════════════════════════════════════════
    # ACT 5: THE REFLECTION 🌟
    # ═══════════════════════════════════════════════════════════════════════

    # The milestone passes only if all three hold for the TRAINED model:
    #   1. All four XOR truth-table rows are predicted correctly.
    #   2. Backprop delivered a non-zero gradient to the hidden layer.
    #   3. The hidden weights moved by at least HIDDEN_MIN_REL_CHANGE of their
    #      initial size (||W_after - W_before|| / ||W_before||).
    # Threshold measured 2026-09-29: correct code moves the hidden weights by
    # 1.43x-13.7x their initial norm on the 51 converging seeds of 0-59 (and
    # 0.92x-2.4x on the 10 that stall); with backprop stopped at the output
    # layer, or an optimizer step that does nothing, the change is exactly 0.
    # 0.25 sits far from both.
    HIDDEN_MIN_REL_CHANGE = 0.25
    hidden_grad_max = history.get("hidden_grad_max", 0.0)
    hidden_grad_ok = hidden_grad_max > 0.0
    hidden_moved_ok = hidden_rel_change >= HIDDEN_MIN_REL_CHANGE
    passed = truth_table_correct and hidden_grad_ok and hidden_moved_ok

    console.print("[bold]🔎 Milestone checks:[/bold]")
    for ok, label in [
        (truth_table_correct, "All 4 XOR truth-table rows correct after training"),
        (hidden_grad_ok, f"Hidden layer received gradients (max |grad| = {hidden_grad_max:.3g})"),
        (hidden_moved_ok, f"Hidden weights changed by {hidden_rel_change:.2f}x their initial size "
                          f"(need ≥ {HIDDEN_MIN_REL_CHANGE})"),
    ]:
        console.print(f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} {label}")
    console.print()

    if passed:
        console.print(Panel.fit(
            "[bold green]🎉 Success! You Ended the AI Winter![/bold green]\n\n"

            f"Final accuracy: [bold]{final_acc:.1%}[/bold], all 4 XOR cases correct!\n\n"

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
    else:
        # Say which check failed and what it points at, instead of falsely
        # advertising XOR as solved.
        causes = []
        if not hidden_grad_ok:
            causes.append(
                "  • [bold]No gradient reached the hidden layer.[/bold] Backprop stops at\n"
                "    the output layer, so the hidden features never learn. Check that\n"
                "    Module 06's matmul backward returns the gradient for its INPUT\n"
                "    (grad @ W.T), not only for its weights, and that ReLU's\n"
                "    backward passes gradient through where the input was positive.\n")
        if hidden_grad_ok and not hidden_moved_ok:
            causes.append(
                "  • [bold]The hidden weights barely moved[/bold] even though gradients\n"
                "    reached them. Check Module 07's SGD.step: it must update\n"
                "    param.data in place (param.data -= lr * grad) for every parameter.\n")
        if not truth_table_correct:
            causes.append(
                "  • [bold]The trained model gets at least one XOR row wrong.[/bold]\n"
                "    Check Module 04's BinaryCrossEntropy gradient and Module 07's\n"
                "    SGD.step (does the loss in the log above go down?).\n")
        if args.seed == DEFAULT_SEED:
            luck_note = (
                f"[bold]🎲 Is it bad luck?[/bold] Not with the default seed ({DEFAULT_SEED}).\n"
                "  The run is seeded, so re-running repeats it exactly, and this\n"
                "  seed converges with correct code. Fix the code first.\n\n")
        else:
            luck_note = (
                f"[bold]🎲 Is it bad luck?[/bold] Possibly: you chose --seed {args.seed}.\n"
                "  About 1 seed in 6 leaves one XOR case stuck at p≈0.5 (a 75%\n"
                "  dead-ReLU saddle point) even with correct code. Run without\n"
                f"  --seed (default {DEFAULT_SEED}) to test your code.\n\n")
        console.print(Panel.fit(
            "[bold red]❌ XOR Not Solved by a Learning Hidden Layer[/bold red]\n\n"

            f"Accuracy of the trained model: [bold]{final_acc:.1%}[/bold]\n\n"

            "[bold]🔍 What the checks point at:[/bold]\n"
            + "".join(causes) +
            "\n"
            + luck_note +

            "[dim]Do not move on to Milestone 03 (TinyDigits) until XOR\n"
            "actually converges - otherwise you are debugging on top of a\n"
            "broken foundation.[/dim]",

            title="❌ Milestone FAILED",
            border_style="red",
            box=box.DOUBLE
        ))

    press_enter_to_continue()
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
