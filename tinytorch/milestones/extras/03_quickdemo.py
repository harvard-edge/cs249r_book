#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🚀 MILESTONE 05.3: TinyTalks Quick Demo (2-Minute Training)        ║
║           Watch Your Transformer Learn to Answer Questions Live!             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 HISTORICAL CONTEXT:
This demo shows the magic of transformer learning in real-time. Watch as
random noise becomes coherent answers - the same progression that happens
(at much larger scale) when training GPT, Claude, and other LLMs.

🎯 WHAT YOU'RE BUILDING:
A live training dashboard using YOUR Tiny🔥Torch implementations!
See model responses evolve from gibberish to coherent answers in ~2 minutes.

Features:
- Smaller model (~50K params) for fast training
- Live dashboard showing training progress
- Rotating prompts to show diverse capabilities
- Learning progression display (gibberish → coherent)

✅ REQUIRED MODULES (Run after Module 13):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)         : YOUR data structure for all computations
  Module 04 (Losses)         : YOUR CrossEntropyLoss for training
  Module 07 (Optimizers)     : YOUR Adam optimizer for fast convergence
  Module 10 (Tokenization)   : YOUR CharTokenizer for text → tokens
  Module 13 (Transformer)    : YOUR GPT model for generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Live Training Pipeline):
    ┌──────────────────────────────────────────────────────────────┐
    │                    LIVE TRAINING DASHBOARD                    │
    ├──────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────────────────────────────┐   │
    │  │ Progress    │  │ Learning Progression                 │   │
    │  │ Epoch: 3/8  │  │ Q: What is 2+2?                      │   │
    │  │ Loss: 1.234 │  │   Ep1: sdfj3kj... (gibberish)        │   │
    │  │ ████░░ 40%  │  │   Ep2: four is th... (learning)      │   │
    │  └─────────────┘  │   Ep3: 4 (correct!)                  │   │
    │  ┌─────────────┐  └─────────────────────────────────────┘   │
    │  │ Systems     │                                             │
    │  │ Tokens/s    │  YOUR GPT model processes batches and       │
    │  │ Memory      │  updates weights using YOUR optimizer       │
    │  └─────────────┘                                             │
    └──────────────────────────────────────────────────────────────┘

# =============================================================================
# 📊 YOUR MODULES IN ACTION
# =============================================================================
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 04: Loss     │ Computes cross-entropy loss    │ Guides learning direction   │
# │                     │ to measure prediction error    │ (gradient information)      │
# │                     │                                │                             │
# │ Module 07: Adam     │ Updates weights with adaptive  │ Fast convergence in 2 min   │
# │                     │ learning rates                 │ (vs hours with vanilla SGD) │
# │                     │                                │                             │
# │ Module 10: Tokenize │ Converts Q&A text to tokens    │ Character-level enables     │
# │                     │ for model processing           │ learning any vocabulary     │
# │                     │                                │                             │
# │ Module 13: GPT      │ Learns to predict next token   │ Watch answers evolve from   │
# │                     │ given question context         │ noise to coherent text!     │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================

💡 KEY INSIGHT:
This demo shows the same learning progression as GPT training:
  Epoch 1: "sdf3kj2l" (random weights = random output)
  Epoch 4: "fouris" (learning patterns, merging words)
  Epoch 8: "4" or "four" (correct answers!)

📊 EXPECTED RESULTS:
  Training time: ~2 minutes
  Final loss: ~1.0-1.5 (down from ~4.0)
  Visible progression: gibberish → partial → coherent answers
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Rich for live dashboard
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

# TinyTorch imports
from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import Adam
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.transformer import GPT
from tinytorch.core.tokenization import CharTokenizer

console = Console()

# =============================================================================
# Configuration - Optimized for ~2 minute training
# =============================================================================

CONFIG = {
    # Model (smaller for speed)
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "max_seq_len": 32,  # Shorter sequences for speed

    # Training (optimized for ~2 min on pure Python)
    "epochs": 8,
    "batches_per_epoch": 30,
    "batch_size": 8,
    "learning_rate": 0.003,  # Balanced LR for stable convergence

    # Display
    "update_interval": 5,  # Update dashboard every N batches
}

# Test prompts to show model learning (3 prompts for better progression display)
TEST_PROMPTS = [
    "Q: What is 2+2?\nA:",
    "Q: What color is the sky?\nA:",
    "Q: Say hello\nA:",
]


# =============================================================================
# Dataset
# =============================================================================

class TinyTalksDataset:
    """Simple character-level dataset from TinyTalks."""

    def __init__(self, data_path: Path, seq_len: int):
        self.seq_len = seq_len

        # Load text
        with open(data_path, 'r') as f:
            self.text = f.read()

        # Create tokenizer and build vocabulary
        self.tokenizer = CharTokenizer()
        self.tokenizer.build_vocab([self.text])

        # Tokenize entire text
        self.tokens = self.tokenizer.encode(self.text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def get_batch(self, batch_size: int):
        """Get random batch of sequences."""
        indices = np.random.randint(0, len(self) - 1, size=batch_size)

        inputs = []
        targets = []

        for idx in indices:
            seq = self.tokens[idx:idx + self.seq_len + 1]
            inputs.append(seq[:-1])
            targets.append(seq[1:])

        return (
            Tensor(np.array(inputs)),
            Tensor(np.array(targets))
        )


# =============================================================================
# Text Generation
# =============================================================================

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 30) -> str:
    """Generate text from prompt."""
    # Encode prompt
    tokens = tokenizer.encode(prompt)

    for _ in range(max_tokens):
        # Prepare input
        context = tokens[-CONFIG["max_seq_len"]:]
        x = Tensor(np.array([context]))

        # Forward pass
        logits = model.forward(x)

        # Get next token probabilities
        last_logits = logits.data[0, -1, :]

        # Temperature sampling
        temperature = 0.8
        last_logits = last_logits / temperature
        exp_logits = np.exp(last_logits - np.max(last_logits))
        probs = exp_logits / np.sum(exp_logits)

        # Sample
        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

        # Stop at newline (end of answer)
        if tokenizer.decode([next_token]) == '\n':
            break

    # Decode and extract answer
    full_text = tokenizer.decode(tokens)

    # Get just the answer part
    if "A:" in full_text:
        answer = full_text.split("A:")[-1].strip()
        # Clean up - take first line
        answer = answer.split('\n')[0].strip()
        return answer if answer else "(empty)"

    return full_text[len(prompt):].strip() or "(empty)"


# =============================================================================
# Dashboard Layout
# =============================================================================

def make_layout() -> Layout:
    """Create the dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="outputs", ratio=2),
    )

    layout["left"].split_column(
        Layout(name="progress", ratio=2),
        Layout(name="stats", ratio=1),
    )

    return layout


def make_header() -> Panel:
    """Create header panel."""
    return Panel(
        Text("TinyTalks Quick Demo - Watch Your Transformer Learn!",
             style="bold cyan", justify="center"),
        box=box.ROUNDED,
        style="cyan",
    )


def make_progress_panel(epoch: int, total_epochs: int, batch: int,
                        total_batches: int, loss: float, elapsed: float) -> Panel:
    """Create training progress panel."""
    # Calculate overall progress
    total_steps = total_epochs * total_batches
    current_step = (epoch - 1) * total_batches + batch
    progress_pct = (current_step / total_steps) * 100

    # Progress bar
    bar_width = 20
    filled = int(bar_width * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Estimate time remaining
    if current_step > 0:
        time_per_step = elapsed / current_step
        remaining_steps = total_steps - current_step
        eta = remaining_steps * time_per_step
        eta_str = f"{eta:.0f}s"
    else:
        eta_str = "..."

    content = Text()
    content.append(f"Epoch: {epoch}/{total_epochs}\n", style="bold")
    content.append(f"Batch: {batch}/{total_batches}\n")
    content.append(f"Loss: {loss:.3f}\n\n", style="yellow")
    content.append(f"{bar} {progress_pct:.0f}%\n\n", style="green")
    content.append(f"Elapsed: {elapsed:.0f}s\n")
    content.append(f"ETA: {eta_str}")

    return Panel(
        content,
        title="[bold]Training Progress[/bold]",
        border_style="green",
        box=box.ROUNDED,
    )


def make_outputs_panel(responses: dict, epoch: int) -> Panel:
    """Create model outputs panel showing all epoch responses as a log."""
    content = Text()

    # Show all 3 prompts with full epoch history
    for i, prompt in enumerate(TEST_PROMPTS):
        q = prompt.split('\n')[0]
        content.append(f"{q}\n", style="cyan bold")

        # Show all epochs completed so far
        for ep in range(1, epoch + 1):
            key = f"epoch_{ep}_{i}"
            response = responses.get(key, "...")
            # Most recent epoch is highlighted
            style = "white" if ep == epoch else "dim"
            content.append(f"  Ep{ep}: ", style="yellow")
            # Truncate long responses to fit
            display_response = response[:25] + "..." if len(response) > 25 else response
            content.append(f"{display_response}\n", style=style)

        content.append("\n")

    return Panel(
        content,
        title=f"[bold]Learning Progression (Epoch {epoch})[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    )


def make_stats_panel(stats: dict) -> Panel:
    """Create systems stats panel."""
    content = Text()

    content.append("Performance Metrics\n", style="bold")
    content.append(f"  Tokens/sec: {stats.get('tokens_per_sec', 0):.1f}\n")
    content.append(f"  Batch time: {stats.get('batch_time_ms', 0):.0f}ms\n")
    content.append(f"  Memory: {stats.get('memory_mb', 0):.1f}MB\n\n")

    content.append("Model Stats\n", style="bold")
    content.append(f"  Parameters: {stats.get('params', 0):,}\n")
    content.append(f"  Vocab size: {stats.get('vocab_size', 0)}\n")

    return Panel(
        content,
        title="[bold]Systems[/bold]",
        border_style="magenta",
        box=box.ROUNDED,
    )


def make_footer(message: str = "") -> Panel:
    """Create footer panel."""
    if not message:
        message = "Training in progress... Watch the model learn to answer questions!"

    return Panel(
        Text(message, style="dim", justify="center"),
        box=box.ROUNDED,
        style="dim",
    )


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    """Main training function with live dashboard."""

    # Welcome
    console.print()
    console.print(Panel.fit(
        "[bold cyan]TinyTalks Quick Demo[/bold cyan]\n\n"
        "Watch a transformer learn to answer questions in real-time!\n"
        "The model starts with random weights (gibberish output)\n"
        "and learns to produce coherent answers.\n\n"
        "[dim]Training time: ~2 minutes[/dim]",
        title="Welcome",
        border_style="cyan",
    ))
    console.print()

    # Load dataset
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "datasets" / "tinytalks" / "splits" / "train.txt"

    if not data_path.exists():
        console.print(f"[red]Error: Dataset not found at {data_path}[/red]")
        console.print("[yellow]Please ensure TinyTalks dataset is available.[/yellow]")
        return

    console.print(f"[dim]Loading dataset from {data_path}...[/dim]")
    dataset = TinyTalksDataset(data_path, CONFIG["max_seq_len"])
    console.print(f"[green]✓[/green] Loaded {len(dataset.text):,} characters, vocab size: {dataset.tokenizer.vocab_size}")

    # Create model
    console.print("[dim]Creating model...[/dim]")
    model = GPT(
        vocab_size=dataset.tokenizer.vocab_size,
        embed_dim=CONFIG["n_embd"],
        num_heads=CONFIG["n_head"],
        num_layers=CONFIG["n_layer"],
        max_seq_len=CONFIG["max_seq_len"],
    )

    # Count parameters
    param_count = sum(p.data.size for p in model.parameters())
    console.print(f"[green]✓[/green] Model created: {param_count:,} parameters")
    console.print(f"[dim]  {CONFIG['n_layer']} layers, {CONFIG['n_head']} heads, {CONFIG['n_embd']} embed dim[/dim]")

    # Setup training
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = CrossEntropyLoss()

    console.print()
    console.print("[bold green]Starting training with live dashboard...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop early[/dim]")
    console.print()
    time.sleep(1)

    # Storage for responses and stats
    responses = {}
    stats = {
        "params": param_count,
        "vocab_size": dataset.tokenizer.vocab_size,
        "tokens_per_sec": 0,
        "batch_time_ms": 0,
        "memory_mb": param_count * 4 / (1024 * 1024),  # Rough estimate
    }

    # Create layout
    layout = make_layout()

    # Training loop with live display
    start_time = time.time()
    current_loss = 0.0
    total_tokens = 0

    try:
        with Live(layout, console=console, refresh_per_second=4) as live:
            for epoch in range(1, CONFIG["epochs"] + 1):
                epoch_loss = 0.0

                for batch_idx in range(1, CONFIG["batches_per_epoch"] + 1):
                    batch_start = time.time()

                    # Get batch
                    inputs, targets = dataset.get_batch(CONFIG["batch_size"])

                    # Forward pass
                    logits = model.forward(inputs)

                    # Reshape for loss
                    batch_size, seq_len, vocab_size = logits.shape
                    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
                    targets_flat = targets.reshape(-1)

                    # Compute loss
                    loss = criterion(logits_flat, targets_flat)

                    # Backward pass
                    loss.backward()

                    # Update
                    optimizer.step()
                    optimizer.zero_grad()

                    # Track loss and stats
                    batch_loss = float(loss.data)
                    epoch_loss += batch_loss
                    current_loss = epoch_loss / batch_idx

                    # Update systems stats
                    batch_time = time.time() - batch_start
                    tokens_in_batch = batch_size * seq_len
                    total_tokens += tokens_in_batch
                    elapsed = time.time() - start_time

                    stats["batch_time_ms"] = batch_time * 1000
                    stats["tokens_per_sec"] = total_tokens / elapsed if elapsed > 0 else 0

                    # Update dashboard
                    layout["header"].update(make_header())
                    layout["progress"].update(make_progress_panel(
                        epoch, CONFIG["epochs"],
                        batch_idx, CONFIG["batches_per_epoch"],
                        current_loss, elapsed
                    ))
                    layout["stats"].update(make_stats_panel(stats))
                    layout["outputs"].update(make_outputs_panel(responses, epoch))
                    layout["footer"].update(make_footer())

                # End of epoch - generate sample responses
                for i, prompt in enumerate(TEST_PROMPTS):
                    response = generate_response(model, dataset.tokenizer, prompt)
                    responses[f"epoch_{epoch}_{i}"] = response

                # Update display with new responses
                layout["outputs"].update(make_outputs_panel(responses, epoch))

                # Show epoch completion message
                layout["footer"].update(make_footer(
                    f"Epoch {epoch} complete! Loss: {current_loss:.3f}"
                ))

        # Training complete
        total_time = time.time() - start_time

        console.print()
        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"Total time: {total_time:.1f} seconds\n"
            f"Final loss: {current_loss:.3f}\n"
            f"Epochs: {CONFIG['epochs']}\n\n"
            "[cyan]Watch how your transformer learned to talk![/cyan]",
            title="Success",
            border_style="green",
        ))

        # Show learning progression for all prompts
        console.print()
        console.print("[bold]Full Learning Progression:[/bold]")
        console.print()

        for i, prompt in enumerate(TEST_PROMPTS):
            q = prompt.split('\n')[0]
            table = Table(box=box.ROUNDED, title=q)
            table.add_column("Epoch", style="cyan")
            table.add_column("Response", style="white")

            for epoch in range(1, CONFIG["epochs"] + 1):
                key = f"epoch_{epoch}_{i}"
                resp = responses.get(key, "...")
                table.add_row(str(epoch), resp)

            console.print(table)
            console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Training stopped by user[/yellow]")


if __name__ == "__main__":
    main()
