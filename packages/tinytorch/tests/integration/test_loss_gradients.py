"""
Comprehensive test for loss function gradients.

Tests which losses have proper autograd integration and work for training.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tinytorch import Tensor, Linear, MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss, SGD
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def test_mse_loss_gradients():
    """Test MSELoss with autograd - SHOULD WORK"""
    console.print("\n[bold cyan]Test 1: MSELoss with Gradients[/bold cyan]")

    # Simple regression problem
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD([model.weight, model.bias], lr=0.01)

    # Training data
    X = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = Tensor(np.array([[3.0], [7.0]], dtype=np.float32))  # y = x1 + 2*x2

    # Record initial loss
    pred = model(X)
    initial_loss = loss_fn(pred, y)
    console.print(f"  Initial loss: {initial_loss.data:.4f}")

    # Train for 10 steps
    for _ in range(10):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Check if loss decreased
    pred = model(X)
    final_loss = loss_fn(pred, y)
    console.print(f"  Final loss: {final_loss.data:.4f}")

    improvement = initial_loss.data - final_loss.data
    console.print(f"  Improvement: {improvement:.4f}")

    assert improvement > 0, (
        f"MSELoss did not learn: loss went from {initial_loss.data:.4f} to "
        f"{final_loss.data:.4f} (improvement {improvement:.4f}, expected > 0)"
    )
    console.print("  [green]✅ MSELoss works - gradients flow correctly![/green]")


def test_bce_loss_gradients():
    """Test BinaryCrossEntropyLoss with autograd - SHOULD WORK"""
    console.print("\n[bold cyan]Test 2: BinaryCrossEntropyLoss with Gradients[/bold cyan]")

    # Simple binary classification
    model = Linear(2, 1)

    from tinytorch.core.activations import Sigmoid
    activation = Sigmoid()

    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD([model.weight, model.bias], lr=0.1)

    # Training data (XOR-like)
    X = Tensor(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    y = Tensor(np.array([[0.0], [0.0]], dtype=np.float32))

    # Record initial loss
    logits = model(X)
    pred = activation(logits)
    initial_loss = loss_fn(pred, y)
    console.print(f"  Initial loss: {initial_loss.data:.4f}")

    # Train for 10 steps
    for _ in range(10):
        logits = model(X)
        pred = activation(logits)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Check if loss decreased
    logits = model(X)
    pred = activation(logits)
    final_loss = loss_fn(pred, y)
    console.print(f"  Final loss: {final_loss.data:.4f}")

    improvement = initial_loss.data - final_loss.data
    console.print(f"  Improvement: {improvement:.4f}")

    assert improvement > 0, (
        f"BinaryCrossEntropyLoss did not learn: loss went from "
        f"{initial_loss.data:.4f} to {final_loss.data:.4f} "
        f"(improvement {improvement:.4f}, expected > 0)"
    )
    console.print("  [green]✅ BinaryCrossEntropyLoss works - gradients flow correctly![/green]")


def test_crossentropy_loss_gradients():
    """Test CrossEntropyLoss with autograd."""
    console.print("\n[bold cyan]Test 3: CrossEntropyLoss with Gradients[/bold cyan]")

    # Simple multi-class classification
    model = Linear(2, 3)  # 2 features → 3 classes
    loss_fn = CrossEntropyLoss()
    optimizer = SGD([model.weight, model.bias], lr=0.01)

    # Training data
    X = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = Tensor(np.array([0, 2], dtype=np.int64))  # Class labels

    # Record initial loss
    logits = model(X)
    initial_loss = loss_fn(logits, y)
    console.print(f"  Initial loss: {initial_loss.data:.4f}")

    # Check if gradients exist
    has_grad_fn = hasattr(initial_loss, '_grad_fn') and initial_loss._grad_fn is not None
    console.print(f"  Has gradient function: {has_grad_fn}")

    assert has_grad_fn, (
        "CrossEntropyLoss output carries no _grad_fn, so backward() would "
        "silently do nothing"
    )

    # Train for 10 steps. Any exception here is a real failure, not something
    # to swallow -- letting it propagate is what makes this test meaningful.
    for _ in range(10):
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    logits = model(X)
    final_loss = loss_fn(logits, y)
    console.print(f"  Final loss: {final_loss.data:.4f}")

    improvement = initial_loss.data - final_loss.data
    console.print(f"  Improvement: {improvement:.4f}")

    assert improvement > 0, (
        f"CrossEntropyLoss did not learn: loss went from {initial_loss.data:.4f} "
        f"to {final_loss.data:.4f} (improvement {improvement:.4f}, expected > 0)"
    )
    console.print("  [green]✅ CrossEntropyLoss works - gradients flow correctly![/green]")


def main():
    """Run all loss gradient tests."""
    console.print(Panel.fit(
        "[bold]TinyTorch Loss Function Gradient Tests[/bold]\n\n"
        "Testing which losses have proper autograd integration\n"
        "and can be used for training neural networks.",
        title="🧪 Loss Gradient Tests",
        border_style="cyan"
    ))

    results = {}
    results['MSELoss'] = test_mse_loss_gradients()
    results['BinaryCrossEntropyLoss'] = test_bce_loss_gradients()
    results['CrossEntropyLoss'] = test_crossentropy_loss_gradients()

    # Summary table
    console.print("\n")
    table = Table(title="📊 Loss Function Status", show_header=True)
    table.add_column("Loss Function", style="cyan")
    table.add_column("Autograd Integration", style="yellow")
    table.add_column("Gradient Flow", style="magenta")
    table.add_column("Status", style="white")

    for loss_name, passed in results.items():
        if passed is None:
            table.add_row(loss_name, "Unknown", "Unknown", "[yellow]⚠️ Skipped[/yellow]")
        elif passed:
            table.add_row(loss_name, "✅ Yes", "✅ Working", "[green]✅ Ready for training[/green]")
        else:
            table.add_row(loss_name, "❌ No", "❌ Broken", "[red]❌ Cannot train[/red]")

    console.print(table)

    # Recommendations
    console.print("\n")
    console.print(Panel.fit(
        "[bold]💡 Recommendations:[/bold]\n\n"
        "[green]✅ Use MSELoss for:[/green]\n"
        "  • Regression tasks\n"
        "  • Simple multi-class with one-hot encoding\n\n"
        "[green]✅ Use BinaryCrossEntropyLoss for:[/green]\n"
        "  • Binary classification (2 classes)\n"
        "  • Requires Sigmoid activation output\n\n"
        "[green]✅ Use CrossEntropyLoss for:[/green]\n"
        "  • Multi-class classification (preferred!)\n"
        "  • Works with raw class labels (no one-hot needed)\n"
        "  • Numerically stable via log-softmax\n\n"
        "[bold]For Milestone 03 (MLP on digits):[/bold]\n"
        "Use CrossEntropyLoss with raw labels (0-9).",
        title="🎯 Usage Guide",
        border_style="yellow"
    ))


if __name__ == "__main__":
    main()
