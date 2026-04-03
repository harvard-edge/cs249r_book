#!/usr/bin/env python3
"""
The Perceptron (1958) - Frank Rosenblatt [FORWARD PASS ONLY]
=============================================================

📚 HISTORICAL CONTEXT:
Frank Rosenblatt's Perceptron was the first trainable artificial neural network that
could learn from examples. It sparked the first AI boom and demonstrated that machines
could actually learn to recognize patterns, launching the neural network revolution.

🎯 MILESTONE 1: FORWARD PASS (BEFORE TRAINING)
Using YOUR Tiny🔥Torch implementations, you'll build a perceptron with RANDOM weights.
This milestone shows you WHY training is essential - the model won't work without it!

⚠️ IMPORTANT: This is NOT the trained version!
- You've completed Modules 01-04 (Tensor, Activations, Layers, Losses)
- You HAVEN'T learned training yet (Modules 05-07: Autograd, Optimizers, Training)
- This milestone demonstrates the PROBLEM that training will solve

✅ REQUIRED MODULES (Run after Module 04):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure (gradients dormant for now)
  Module 02 (Activations)   : YOUR sigmoid activation function
  Module 03 (Layers)        : YOUR Linear layer with RANDOM weights
  Module 04 (Losses)        : YOUR loss functions (for measuring failure)
  Data Generation           : Directly generated within this script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Original 1957 Design):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Input       │    │   Linear    │    │  Sigmoid    │    │ Binary      │
    │ Features    │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ Output      │
    │ (x1, x2)    │    │     03      │    │     02      │    │ (0 or 1)    │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

🔍 WHAT YOU'LL SEE - EXPECTATION vs REALITY:

    WHAT YOU MIGHT EXPECT:           WHAT YOU'LL ACTUALLY GET:
    "I built it, so it works!"       "Wait... it's just guessing!"

    4 │ • • • • •                    4 │ • ○ • ○ •
      │ • • • • • ╱  Perfect!          │ ○ • • ○ • ╲  Random!
    2 │ • • • • ╱ •                  2 │ • ○ • • ○ •
      │ ○ ○ ○ ╱ ○ ○                    │ ○ • ○ ○ • ○
    0 │ ○ ○ ╱ ○ ○ ○                  0 │ • ○ • ○ ○ •
      └────────────                    └────────────
        0   2   4                        0   2   4

    ❌ Accuracy: ~50%                ❌ Accuracy: ~50%
       (What you hoped for)             (What random weights give you)

    WHY IS IT SO BAD?
    The weights are RANDOM! Without training:
    - w₁, w₂, b are random numbers from initialization
    - The decision boundary is in a random position
    - Predictions are essentially coin flips

    Mathematical Reality:
    y = sigmoid(w₁·x₁ + w₂·x₂ + b)  ← These are RANDOM values!

    Where YOUR modules compute:
    - Linear: z = w₁·x₁ + w₂·x₂ + b  (random w₁, w₂, b!)
    - Sigmoid: y = 1/(1 + e⁻ᶻ)       (squash to [0,1])
    - Decision: class = 1 if y > 0.5 else 0  (random decision boundary!)

🔍 KEY INSIGHTS (This Milestone):
- ✅ Architecture works: Forward pass executes correctly
- ❌ But it's useless: Random weights = random predictions (~50% accuracy)
- 💡 The lesson: Building the model is easy; making it LEARN is the hard part
- 🎯 Motivation: You NEED training (coming in Modules 05-07!)

📊 WHAT TO EXPECT (This Milestone):
- Dataset: 10 linearly separable synthetic points (just for testing)
- No training: Just forward pass with random weights
- Expected accuracy: ~40-60% (essentially random guessing)
- Key takeaway: "My model doesn't work... yet!"

🚀 WHAT COMES NEXT (After Module 07):
- Same architecture, but WITH training
- Expected accuracy: 95%+ on same problem
- Training time: ~30 seconds
- You'll see the SAME perceptron transform from useless → intelligent
"""

import sys
import os
import numpy as np
rng = np.random.default_rng(7)
import argparse

# Add project root to path for correct tinytorch imports
# This allows the script to be run from the root of the project
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor        # Module 01: YOU built this!
from tinytorch.core.layers import Linear        # Module 03: YOU built this!
from tinytorch.core.activations import Sigmoid  # Module 02: YOU built this!

# Import Rich for beautiful CLI output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

console = Console()

# ============================================================================
# 🎓 STUDENT CODE: This is what YOU built with Modules 01-03!
# ============================================================================

class Perceptron:
    """
    Simple perceptron: Linear + Sigmoid

    This uses components YOU built in:
      - Module 01: Tensor (data structure)
      - Module 02: Sigmoid (activation function)
      - Module 03: Linear (layer with weights)

    The entire model is just ~10 lines of code!
    """

    def __init__(self, input_size=2, output_size=1):
        # Module 03: Linear layer (w1*x1 + w2*x2 + b)
        self.linear = Linear(input_size, output_size)

        # Module 02: Sigmoid activation (squashes to [0,1])
        self.activation = Sigmoid()

    def forward(self, x):
        # Step 1: Linear transformation (Module 03)
        x = self.linear(x)

        # Step 2: Activation function (Module 02)
        x = self.activation(x)

        return x

    def __call__(self, x):
        """PyTorch-style: model(x) calls forward(x)"""
        return self.forward(x)

# ============================================================================
# 📊 VISUALIZATION CODE: Rich CLI formatting (you can ignore this!)
# ============================================================================

def draw_network_architecture():
    """Draw the perceptron architecture using ASCII art."""
    network = """
    Input Layer        Linear Layer              Activation         Output

    ┌─────────┐       ┌──────────────┐         ┌──────────┐      ┌─────────┐
    │         │       │              │         │          │      │         │
    │   x₁    │───────┤              │         │          │      │         │
    │         │  w₁   │              │    z    │          │  y   │  class  │
    └─────────┘       │    Linear    │─────────│  Sigmoid │──────│ (0 or 1)│
                      │   (Wx + b)   │         │  σ(z)    │      │         │
    ┌─────────┐       │              │         │          │      │         │
    │         │  w₂   │              │         │          │      │         │
    │   x₂    │───────┤              │         │          │      └─────────┘
    │         │       │              │         │          │
    └─────────┘       └──────────────┘         └──────────┘
                            ↑
                            b (bias)

    Computation Flow:
      1. Linear:  z = w₁·x₁ + w₂·x₂ + b
      2. Sigmoid: y = 1 / (1 + e⁻ᶻ)
      3. Decision: class = 1 if y > 0.5 else 0
    """
    return network.strip()

def visualize_data_points(X, y, predictions=None, weights=None):
    """Create ASCII visualization of data points with decision boundary."""
    # Create a simple scatter plot
    grid_size = 20
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

    # Find bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Draw decision boundary if weights provided
    # Decision boundary: w1*x1 + w2*x2 + b = 0 → x2 = -(w1*x1 + b)/w2
    if weights is not None:
        w1, w2, b = weights
        if abs(w2) > 0.001:  # Avoid division by zero
            # Determine slope for choosing line character
            slope = -w1 / w2
            line_char = '/' if slope > 0 else '\\'

            for gx in range(grid_size):
                # Map grid x to real x
                px = x_min + (gx / (grid_size - 1)) * (x_max - x_min)
                # Calculate decision boundary y
                py = -(w1 * px + b) / w2
                # Map to grid y
                gy = int((py - y_min) / (y_max - y_min) * (grid_size - 1))
                gy = grid_size - 1 - gy  # Flip y-axis

                if 0 <= gy < grid_size and grid[gy][gx] == ' ':
                    grid[gy][gx] = line_char  # Decision boundary line

    # Plot points (after boundary so they overlap)
    for i, (px, py) in enumerate(X):
        # Map to grid
        gx = int((px - x_min) / (x_max - x_min) * (grid_size - 1))
        gy = int((py - y_min) / (y_max - y_min) * (grid_size - 1))
        gy = grid_size - 1 - gy  # Flip y-axis

        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            true_label = int(y[i])
            if predictions is not None:
                pred_label = int(predictions[i])
                # Show correct (green) vs incorrect (red) predictions
                if true_label == pred_label:
                    grid[gy][gx] = '●' if true_label == 1 else '○'
                else:
                    grid[gy][gx] = '✗'  # Wrong prediction
            else:
                grid[gy][gx] = '●' if true_label == 1 else '○'

    # Build the plot
    lines = []
    lines.append("   " + "─" * grid_size)
    for row in grid:
        lines.append("  │" + "".join(row) + "│")
    lines.append("   " + "─" * grid_size)
    lines.append("   ● = Class 1 (should cluster top-right)")
    lines.append("   ○ = Class 0 (should cluster bottom-left)")
    if weights is not None:
        lines.append("   / or \\ = Decision boundary (where z = 0)")
    if predictions is not None:
        lines.append("   ✗ = Incorrect prediction")

    return "\n".join(lines)

def press_enter_to_continue() :
    if sys.stdin.isatty() and sys.stdout.isatty() :
        try :
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError :
            pass
        console.print()

def main():
    """Demonstrate Rosenblatt's Perceptron using YOUR Tiny🔥Torch system!"""

    # Header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🎯 MILESTONE 1: The Perceptron (1958)[/bold cyan]\n"
        "[yellow]⚠️  FORWARD PASS ONLY - Random Weights[/yellow]\n\n"
        "[dim]Components: YOUR Tensor + YOUR Linear + YOUR Sigmoid[/dim]",
        border_style="cyan"
    ))
    press_enter_to_continue()

    # Introduction - What to expect
    intro = (
        "[bold]What You're Demonstrating:[/bold]\n\n"
        "You've completed Modules 01-03 and built these components:\n"
        "  • [cyan]Module 01:[/cyan] Tensor (data structure)\n"
        "  • [cyan]Module 02:[/cyan] Sigmoid (activation function)\n"
        "  • [cyan]Module 03:[/cyan] Linear (layer with weights)\n\n"
        "[bold yellow]What to Expect:[/bold yellow]\n"
        "  • The architecture [green]WORKS[/green] - forward pass succeeds ✓\n"
        "  • Accuracy is [red]POOR[/red] - random weights = random predictions ✗\n"
        "  • Decision boundary (/) is in a [yellow]RANDOM[/yellow] position\n"
        "  • Each run gives [yellow]DIFFERENT[/yellow] results (no seed!)\n\n"
        "[bold cyan]The Key Lesson:[/bold cyan]\n"
        "  Building the model is easy. Making it [bold]LEARN[/bold] is hard.\n"
        "  That's why you need Modules 04-07: Losses, Autograd, Optimizers, Training!"
    )
    console.print(Panel(intro, title="[bold cyan]📖 Introduction[/bold cyan]", border_style="cyan"))
    press_enter_to_continue()

    # Step 1: Prepare synthetic data
    console.print("[bold]📊 Step 1: Preparing Data[/bold]")
    console.print("   Creating linearly separable clusters...")
    console.print("   [dim]This is a SIMPLE problem - a trained model achieves 95%+ easily[/dim]")
    console.print("   [yellow]⚠️  No random seed - each run will be different![/yellow]")

    cluster1 = rng.normal([2, 2], 0.5, (5, 2))   # Class 1: top-right
    cluster2 = rng.normal([-2, -2], 0.5, (5, 2)) # Class 0: bottom-left
    X = np.vstack([cluster1, cluster2]).astype(np.float32)
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)  # True labels

    # Show data visualization
    console.print()
    data_viz = visualize_data_points(X, y)
    console.print(Panel(data_viz, title="[cyan]Training Data[/cyan]", border_style="cyan"))
    console.print(f"   [green]✓[/green] Created {X.shape[0]} points in 2 clearly separated clusters")
    press_enter_to_continue()

    # Step 2: Create the Perceptron model with YOUR components
    console.print("[bold]🧠 Step 2: Building Model[/bold]")
    console.print("   [yellow]⚠️  No training yet - you haven't learned Modules 05-07![/yellow]")
    console.print("   🧠 Assembling perceptron with YOUR Tiny🔥Torch modules...")

    model = Perceptron(input_size=2, output_size=1)

    console.print(f"      [green]✓[/green] Linear layer: 2 → 1 [dim](YOUR Module 03!)[/dim]")
    console.print(f"      [green]✓[/green] Activation: Sigmoid [dim](YOUR Module 02!)[/dim]")
    console.print("   [yellow]⚠️  Model assembled - but weights are RANDOM![/yellow]\n")

    # Show network architecture
    network_diagram = draw_network_architecture()
    console.print(Panel(network_diagram, title="[cyan]🏗️  Network Architecture (1957 Design)[/cyan]", border_style="cyan"))
    press_enter_to_continue()

    # Step 3: Test with random weights
    console.print("[bold]🔬 Step 3: Testing with Random Weights[/bold]")
    console.print("   Running forward pass...\n")

    input_tensor = Tensor(X)
    predictions = model(input_tensor)

    # Convert to binary predictions
    pred_classes = (predictions.data > 0.5).astype(int).flatten()
    accuracy = (pred_classes == y).mean()

    # Format arrays nicely for display
    true_str = ' '.join([f"{int(val)}" for val in y])
    pred_str = ' '.join([f"{val}" for val in pred_classes])
    match_str = ' '.join(['[green]✓[/green]' if m else '[red]✗[/red]' for m in (pred_classes == y)])

    # Create results table
    results_table = Table(title="📊 Prediction Results", box=box.ROUNDED, border_style="cyan")
    results_table.add_column("Metric", style="cyan", no_wrap=True)
    results_table.add_column("Value", style="white")

    results_table.add_row("True Labels", f"[{true_str}]")
    results_table.add_row("Predictions", f"[{pred_str}]")
    results_table.add_row("Matches", match_str)

    # Determine status
    if accuracy < 0.6:
        accuracy_display = f"[red]{accuracy:.1%} ❌ Random Guessing![/red]"
        status = "FAILED"
        status_color = "red"
    else:
        accuracy_display = f"[yellow]{accuracy:.1%} 🎲 Got Lucky![/yellow]"
        status = "LUCKY"
        status_color = "yellow"

    results_table.add_row("Accuracy", accuracy_display)
    console.print(results_table)
    press_enter_to_continue()

    # Extract weights for visualization and display
    w1 = model.linear.weight.data[0,0]
    w2 = model.linear.weight.data[1,0]
    b = model.linear.bias.data[0]

    # Calculate z values (linear output before sigmoid)
    z_values = X @ np.array([[w1], [w2]]) + b

    # Show visualization with predictions AND decision boundary
    pred_viz = visualize_data_points(X, y, pred_classes, weights=(w1, w2, b))
    console.print(Panel(pred_viz, title="[cyan]Predictions with Decision Boundary[/cyan]", border_style=status_color))
    press_enter_to_continue()
    
    # Show weights AND equation
    decision_eq = f"z = {w1:.4f}·x₁ + {w2:.4f}·x₂ + {b:.4f}"
    boundary_eq = f"Decision boundary (z=0): x₂ = {-w1/w2:.4f}·x₁ + {-b/w2:.4f}" if abs(w2) > 0.001 else "Decision boundary: vertical line"

    weights_content = (
        f"[bold]Random Weights:[/bold]\n"
        f"  w₁ = [yellow]{w1:7.4f}[/yellow]\n"
        f"  w₂ = [yellow]{w2:7.4f}[/yellow]\n"
        f"  b  = [yellow]{b:7.4f}[/yellow]\n\n"
        f"[bold]Linear Function:[/bold]\n"
        f"  {decision_eq}\n\n"
        f"[bold]Decision Line:[/bold]\n"
        f"  {boundary_eq}\n"
        f"  [dim](Everything above line → Class 1, below → Class 0)[/dim]"
    )
    console.print(Panel(weights_content, title="[yellow]🔧 Model Parameters[/yellow]", border_style="yellow"))
    press_enter_to_continue()

    # Diagnosis
    if status == "FAILED":
        diagnosis = (
            "[red]❌ The model is essentially guessing randomly[/red]\n"
            "[red]❌ Random initialization = random decision boundary[/red]\n\n"
            "[bold cyan]💡 KEY INSIGHT:[/bold cyan] Building the architecture is easy.\n"
            "   Making it [bold]LEARN[/bold] is the hard part!"
        )
    else:
        diagnosis = (
            "[yellow]🎲 You got lucky with this random initialization![/yellow]\n"
            "[yellow]🎲 But this is NOT learning - just chance[/yellow]\n\n"
            "[bold cyan]💡 KEY INSIGHT:[/bold cyan] Even when it works, random weights\n"
            "   won't generalize. You need [bold]TRAINING[/bold]!"
        )

    console.print(Panel(diagnosis, title=f"[{status_color}]🔍 Diagnosis: {status}[/{status_color}]", border_style=status_color))
    press_enter_to_continue()

    # Tip for multiple runs
    tip = (
        "💡 [bold yellow]Run this script multiple times![/bold yellow]\n\n"
        "Each run uses different random weights and data.\n"
        "You'll see varying results:\n"
        "  • Sometimes: High accuracy (got lucky!) 🎲\n"
        "  • Usually: Low accuracy (random guessing) ❌\n\n"
        "[dim]This demonstrates why training is essential - it must work EVERY time![/dim]"
    )
    console.print(Panel(tip, title="[bold yellow]💡 Experiment[/bold yellow]", border_style="yellow"))
    press_enter_to_continue()

    # Next steps
    next_steps = (
        "[bold]Complete Modules 05-07 to unlock TRAINING:[/bold]\n\n"
        "  [cyan]•[/cyan] Module 06 (Autograd):   Calculate gradients automatically\n"
        "  [cyan]•[/cyan] Module 07 (Optimizers): Update weights intelligently\n"
        "  [cyan]•[/cyan] Module 08 (Training):   Put it all together\n\n"
        "[dim]Then return to this SAME perceptron and watch it achieve 95%+!\n"
        "You'll see random → intelligent through the power of learning![/dim]"
    )
    console.print(Panel(next_steps, title="[bold green]🚀 Next Steps[/bold green]", border_style="green"))
    press_enter_to_continue()

if __name__ == "__main__":
    main()
