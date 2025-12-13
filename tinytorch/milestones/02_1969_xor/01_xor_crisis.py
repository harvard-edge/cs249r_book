#!/usr/bin/env python3
"""
The XOR Crisis (1969) - Minsky & Papert
========================================

📚 HISTORICAL CONTEXT:
In 1969, Marvin Minsky and Seymour Papert published "Perceptrons," mathematically
proving that single-layer perceptrons CANNOT solve the XOR problem. This revelation
killed neural network research funding for over a decade - the "AI Winter."

🎯 MILESTONE 2 PART 1: THE CRISIS (After Modules 01-03)
This demonstrates WHY the crisis happened. We'll show that NO MATTER what weights
you choose, a single-layer perceptron cannot correctly classify all XOR patterns.
This is a mathematical impossibility - not a training failure.

✅ REQUIRED MODULES (Run after Module 03):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure
  Module 02 (Activations)   : YOUR sigmoid activation
  Module 03 (Layers)        : YOUR Linear layer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (The Failing Single-Layer Perceptron):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Input       │    │   Linear    │    │  Sigmoid    │    │ Binary      │
    │ Features    │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ Output      │
    │ (x1, x2)    │    │     03      │    │     02      │    │ (0 or 1)    │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                    │
                           │   y = sigmoid(w1·x1 + w2·x2 + b)
                           │   Decision boundary: w1·x1 + w2·x2 + b = 0
                           │
                           ↓
                    Always a STRAIGHT LINE - cannot solve XOR!

🔍 THE XOR PROBLEM - Why It's Impossible for Single Layers:

XOR (Exclusive OR) outputs 1 when inputs DIFFER, 0 when they're the SAME:

    Visual Representation:        Truth Table:

    1 │ ○ (0,1)    ● (1,1)        ┌────┬────┬─────┐
      │   [1]       [0]           │ x1 │ x2 │ XOR │
      │                           ├────┼────┼─────┤
    0 │ ● (0,0)    ○ (1,0)        │ 0  │ 0  │  0  │ same
      │   [0]       [1]           │ 0  │ 1  │  1  │ different
      └─────────────              │ 1  │ 0  │  1  │ different
        0          1              │ 1  │ 1  │  0  │ same
                                  └────┴────┴─────┘

    THE FUNDAMENTAL PROBLEM - No single line can separate the points:

    Try 1: Vertical line?      Try 2: Horizontal line?    Try 3: Diagonal line?
    1 │ ○   │ ●                1 │ ○     ●                1 │ ○     ●
      │     │                    │───────────               │  ╲
    0 │ ●   │ ○                0 │ ●     ○                0 │ ●   ╲ ○
      └─────┼──                  └───────────               └────────╲
        0   │ 1                    0       1                  0       1
    ❌ Wrong: (0,1) with (0,0) ❌ Wrong: (0,1) with (1,1) ❌ Wrong: (0,0) with (1,0)

    THERE IS NO SOLUTION! This is called "non-linear separability"

📊 WHAT TO EXPECT:
- We'll try MANY different weight configurations
- NONE of them will achieve 100% accuracy
- Best possible: 75% (3 out of 4 correct)
- This proves Minsky was right!
- ❌ Expected accuracy: 75% max (mathematically impossible to do better!)

🚀 WHAT COMES NEXT:
Part 2 (02_xor_solved.py) shows how hidden layers solve this!
The secret? Multi-layer networks can learn NON-LINEAR decision boundaries.
"""

import sys
import os
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
# Only needs Modules 01-03 (no training required)
from tinytorch import Tensor, Linear, Sigmoid

console = Console()

# =============================================================================
# 🎯 YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone uses YOUR modules to demonstrate a fundamental limitation:
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ What It Proves              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01: Tensor   │ Stores XOR inputs and outputs  │ Data flows correctly        │
# │                     │                                │                             │
# │ Module 02: Sigmoid  │ Squashes linear output to [0,1]│ Activation works, but...    │
# │                     │                                │ can't fix linear limits     │
# │                     │                                │                             │
# │ Module 03: Linear   │ Computes w1·x1 + w2·x2 + b     │ Creates LINEAR boundary     │
# │                     │                                │ ❌ XOR needs NON-linear!     │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# KEY INSIGHT: YOUR modules work perfectly! The failure is ARCHITECTURAL.
# A single Linear layer can only create straight-line decision boundaries.
# XOR requires a curved boundary - that's why hidden layers are needed!
#
# =============================================================================


# ============================================================================
# XOR DATA
# ============================================================================

XOR_INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

XOR_TARGETS = np.array([
    [0.0],  # 0 XOR 0 = 0
    [1.0],  # 0 XOR 1 = 1
    [1.0],  # 1 XOR 0 = 1
    [0.0],  # 1 XOR 1 = 0
])


# ============================================================================
# SINGLE-LAYER PERCEPTRON
# ============================================================================

class SingleLayerPerceptron:
    """
    Single-layer perceptron - the architecture that CANNOT solve XOR.

    This is the exact architecture Minsky proved insufficient in 1969.
    Decision boundary: w1*x1 + w2*x2 + b = 0 (always a straight line)
    """

    def __init__(self):
        self.linear = Linear(2, 1)
        self.sigmoid = Sigmoid()

    def set_weights(self, w1, w2, b):
        """Manually set weights to test different configurations."""
        # Weight shape is (in_features, out_features) = (2, 1)
        self.linear.weight.data = np.array([[w1], [w2]])
        self.linear.bias.data = np.array([b])

    def __call__(self, x):
        """Forward pass: Input -> Linear -> Sigmoid -> Output"""
        return self.sigmoid(self.linear(x))

    def get_weights(self):
        """Return current weights."""
        # Weight shape is (2, 1), flatten to get [w1, w2]
        w = self.linear.weight.data.flatten()
        b = float(self.linear.bias.data.item() if hasattr(self.linear.bias.data, 'item') else self.linear.bias.data[0])
        return w[0], w[1], b


def evaluate_on_xor(model):
    """Evaluate model on XOR and return accuracy + predictions."""
    X = Tensor(XOR_INPUTS)
    predictions = model(X)
    pred_classes = (predictions.data > 0.5).astype(int)
    accuracy = (pred_classes == XOR_TARGETS).mean()
    return accuracy, predictions.data.flatten()


def describe_decision_boundary(w1, w2, b):
    """Describe what the decision boundary looks like."""
    if abs(w2) < 0.001:
        if abs(w1) < 0.001:
            return "Horizontal line (degenerate)"
        return f"Vertical line at x1 = {-b/w1:.2f}"

    slope = -w1/w2
    intercept = -b/w2
    return f"Line: x2 = {slope:.2f}*x1 + {intercept:.2f}"


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_crisis():
    """Show that NO weight configuration can solve XOR."""

    console.print(Panel.fit(
        "[bold red]The XOR Crisis (1969)[/bold red]\n\n"
        "[dim]Minsky & Papert proved single-layer perceptrons cannot solve XOR.[/dim]\n"
        "[dim]Let's verify this by trying MANY different weight configurations.[/dim]",
        border_style="red"
    ))

    # Show the XOR problem
    console.print("\n[bold]The XOR Truth Table:[/bold]")
    table = Table(show_header=True, box=box.ROUNDED)
    table.add_column("x1", style="cyan", justify="center")
    table.add_column("x2", style="cyan", justify="center")
    table.add_column("XOR", style="green", justify="center")
    table.add_column("Visual", style="yellow", justify="center")

    table.add_row("0", "0", "0", "Bottom-left")
    table.add_row("0", "1", "1", "Top-left")
    table.add_row("1", "0", "1", "Bottom-right")
    table.add_row("1", "1", "0", "Top-right")
    console.print(table)

    console.print("\n[bold yellow]The Challenge:[/bold yellow]")
    console.print("  Separate (0,0) and (1,1) from (0,1) and (1,0)")
    console.print("  Using only a SINGLE STRAIGHT LINE")
    console.print("  [red]Spoiler: It's impossible![/red]\n")

    # Try various weight configurations
    console.print("[bold cyan]Testing Different Weight Configurations...[/bold cyan]\n")

    model = SingleLayerPerceptron()

    # Different weight configurations to try
    configurations = [
        (1.0, 1.0, -0.5, "Diagonal line (positive slope)"),
        (1.0, 1.0, -1.5, "Diagonal line (shifted)"),
        (-1.0, 1.0, 0.0, "Diagonal line (negative slope)"),
        (1.0, -1.0, 0.0, "Diagonal line (other direction)"),
        (2.0, 2.0, -1.0, "Steep diagonal"),
        (0.0, 1.0, -0.5, "Horizontal split"),
        (1.0, 0.0, -0.5, "Vertical split"),
        (1.0, 1.0, 0.0, "Through origin"),
        (5.0, 5.0, -2.5, "Sharp sigmoid"),
        (-1.0, -1.0, 1.5, "Inverted diagonal"),
    ]

    results_table = Table(title="Weight Configuration Results", box=box.ROUNDED)
    results_table.add_column("Config", style="dim")
    results_table.add_column("w1", justify="right")
    results_table.add_column("w2", justify="right")
    results_table.add_column("b", justify="right")
    results_table.add_column("Accuracy", justify="center")
    results_table.add_column("Predictions", style="dim")

    best_accuracy = 0
    best_config = None

    for i, (w1, w2, b, description) in enumerate(configurations):
        model.set_weights(w1, w2, b)
        accuracy, preds = evaluate_on_xor(model)

        # Format predictions
        pred_str = " ".join([f"{int(p)}" for p in (preds > 0.5).astype(int)])

        # Color accuracy based on value
        if accuracy == 1.0:
            acc_style = "[bold green]100%[/bold green]"
        elif accuracy >= 0.75:
            acc_style = "[yellow]75%[/yellow]"
        else:
            acc_style = "[red]{:.0%}[/red]".format(accuracy)

        results_table.add_row(
            f"#{i+1}",
            f"{w1:.1f}",
            f"{w2:.1f}",
            f"{b:.1f}",
            acc_style.format(accuracy) if "{" in acc_style else acc_style,
            pred_str
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = (w1, w2, b, description)

    console.print(results_table)

    # Also try random configurations
    console.print("\n[bold cyan]Trying 100 Random Configurations...[/bold cyan]")

    random_best = 0
    for _ in range(100):
        w1 = np.random.randn() * 5
        w2 = np.random.randn() * 5
        b = np.random.randn() * 3
        model.set_weights(w1, w2, b)
        accuracy, _ = evaluate_on_xor(model)
        if accuracy > random_best:
            random_best = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = (w1, w2, b, "Random")

    console.print(f"  Best from random search: [yellow]{random_best:.0%}[/yellow]")

    # Show the conclusion
    console.print("\n")
    if best_accuracy < 1.0:
        console.print(Panel(
            f"[bold red]CONFIRMED: XOR is UNSOLVABLE![/bold red]\n\n"
            f"Best accuracy achieved: [yellow]{best_accuracy:.0%}[/yellow] (need 100%)\n"
            f"We tried {len(configurations) + 100} different configurations.\n\n"
            "[bold]Why does this happen?[/bold]\n"
            "A single-layer perceptron can only draw ONE straight line.\n"
            "XOR requires separating diagonal corners - impossible with one line!\n\n"
            "[dim]This mathematical impossibility ended neural network research for 17 years.[/dim]",
            title="The 1969 AI Winter Begins",
            border_style="red"
        ))
    else:
        console.print(Panel(
            "[yellow]Unexpected: Found a solution![/yellow]\n"
            "This shouldn't happen with standard XOR.",
            border_style="yellow"
        ))

    # Visual explanation
    console.print("\n[bold]Why No Line Works:[/bold]")
    console.print("""
    The XOR pattern:          Any line fails:

      1 | O       *             1 | O   /   *
        | (1)     (0)             |   /
        |                         | /
      0 | *       O             0 |*      O
        | (0)     (1)             |
        +----------               +----------
          0       1                 0       1

    O = should output 1          The line separates (0,0) from the rest,
    * = should output 0          but (1,1) is on the WRONG side!

    No matter how you draw a single line, at least one point is wrong.
    """)

    # Historical context
    console.print(Panel(
        "[bold]Historical Significance[/bold]\n\n"
        "[bold cyan]1969:[/bold cyan] Minsky & Papert publish 'Perceptrons'\n"
        "       Mathematically prove XOR is unsolvable\n\n"
        "[bold red]1970s:[/bold red] AI Winter begins\n"
        "       Neural network research funding disappears\n\n"
        "[bold yellow]1986:[/bold yellow] Rumelhart, Hinton & Williams\n"
        "       Multi-layer networks + backprop SOLVE XOR!\n\n"
        "[dim]Run Part 2 to see how hidden layers break through this barrier![/dim]",
        title="The Path Forward",
        border_style="blue"
    ))

    return 0


def main():
    """Demonstrate the XOR crisis."""
    return demonstrate_crisis()


if __name__ == "__main__":
    sys.exit(main())
