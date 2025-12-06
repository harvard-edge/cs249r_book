#!/usr/bin/env python3
"""
[MILESTONE NAME] ([YEAR]) - [HISTORICAL FIGURE]
===============================================

📚 HISTORICAL CONTEXT:
[2-3 sentences about why this breakthrough mattered historically]

🎯 WHAT YOU'RE BUILDING:
[1-2 sentences about what students will demonstrate with their implementation]

✅ REQUIRED MODULES (Run after Module X):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module XX (Component)  : YOUR [description]
  Module YY (Component)  : YOUR [description]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE:
    [ASCII diagram showing the model architecture]
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Input   │───▶│ Layer 1 │───▶│ Output  │
    └─────────┘    └─────────┘    └─────────┘

📊 EXPECTED PERFORMANCE:
- Dataset: [Dataset name and size]
- Training time: ~X minutes
- Expected accuracy: Y%
- Parameters: Z
"""

import sys
import os
import numpy as np
import time

# Add project paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'milestones'))

# Import TinyTorch components YOU BUILT
from tinytorch import (
    Tensor,
    Linear,
    ReLU,
    SGD,
    CrossEntropyLoss,
    # ... other components
)

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MilestoneModel:
    """
    [Brief description of what this model does]

    Architecture:
      [Simple text description of layers]

    This demonstrates YOUR [specific TinyTorch modules] working together!
    """

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the model with YOUR TinyTorch components."""
        self.layer1 = Linear(input_size, hidden_size)
        self.activation = ReLU()
        self.layer2 = Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)

    def parameters(self):
        """Return all trainable parameters."""
        return [
            self.layer1.weight, self.layer1.bias,
            self.layer2.weight, self.layer2.bias
        ]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_data():
    """
    Load and prepare the dataset.

    Returns:
        train_data: Training dataset
        test_data: Test dataset
    """
    # Example data generation
    train_X = np.random.randn(1000, 10)
    train_y = np.random.randint(0, 2, (1000, 1))

    test_X = np.random.randn(200, 10)
    test_y = np.random.randint(0, 2, (200, 1))

    return (Tensor(train_X), Tensor(train_y)), (Tensor(test_X), Tensor(test_y))


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_data, epochs=100, lr=0.01):
    """
    Train the model and display progress.

    Args:
        model: The model to train
        train_data: Training dataset
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        dict: Final metrics (accuracy, loss, etc.)
    """
    optimizer = SGD(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    train_X, train_y = train_data

    console.print("\n[bold cyan]🔥 Starting Training...[/bold cyan]\n")

    for epoch in range(epochs):
        # Forward pass
        predictions = model(train_X)
        loss = loss_fn(predictions, train_y)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        # Calculate accuracy
        pred_classes = (predictions.data > 0.5).astype(int)
        accuracy = (pred_classes == train_y.data).mean() * 100

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            console.print(
                f"Epoch {epoch+1:3d}/{epochs}  "
                f"Loss: {loss.data.item():.4f}  "
                f"Accuracy: {accuracy:.1f}%"
            )

    console.print("\n[bold green]✅ Training Complete![/bold green]\n")

    return {
        "accuracy": accuracy,
        "loss": loss.data.item()
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete milestone demonstration."""

    # Header
    console.print(Panel.fit(
        "[bold cyan]🎯 [MILESTONE NAME] ([YEAR])[/bold cyan]\n\n"
        "[dim]Historical Context: [Brief context here][/dim]\n"
        "[dim]What You're Proving: YOUR modules work on [task]![/dim]",
        title="🔥 [Historical Figure] Milestone",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Load data
    console.print("\n[bold]📊 Loading Data...[/bold]")
    train_data, test_data = load_data()
    console.print("  • Training samples: 1,000")
    console.print("  • Test samples: 200")

    # Create model
    console.print("\n[bold]🏗️ Building Model...[/bold]")
    model = MilestoneModel(input_size=10, hidden_size=20, output_size=2)
    console.print("  • Architecture: Linear(10→20) + ReLU + Linear(20→2)")
    console.print("  • Parameters: ~262 total")

    console.print("\n" + "─" * 70 + "\n")

    # Train
    start_time = time.time()
    final_metrics = train_model(model, train_data, epochs=100, lr=0.01)
    training_time = time.time() - start_time

    console.print("─" * 70 + "\n")

    # Results table
    table = Table(title="Training Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Final Accuracy", f"{final_metrics['accuracy']:.1f}%")
    table.add_row("Final Loss", f"{final_metrics['loss']:.4f}")
    table.add_row("Training Time", f"{training_time:.1f}s")
    table.add_row("Epochs", "100")

    console.print(table)

    # Success message
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]🎉 Success! Your Implementation Works![/bold green]\n\n"
        f"Final accuracy: [bold]{final_metrics['accuracy']:.1f}%[/bold]\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[bold]💡 What YOU Just Accomplished:[/bold]\n"
        "  ✓ Built [architecture description]\n"
        "  ✓ Trained using YOUR autograd and optimizer\n"
        "  ✓ Achieved [expected accuracy] on [task]!\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[bold]🎓 Why This Matters:[/bold]\n"
        "  [Educational significance here]\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[bold]🚀 What's Next:[/bold]\n"
        "  [Next milestone or concept to explore]",
        title="🌟 Milestone Complete",
        border_style="green",
        box=box.DOUBLE
    ))


if __name__ == "__main__":
    main()







