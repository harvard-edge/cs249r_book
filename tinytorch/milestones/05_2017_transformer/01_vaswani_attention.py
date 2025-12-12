#!/usr/bin/env python3
"""
Attention is All You Need (2017) - The Transformer Challenge
=============================================================

MILESTONE 05: PROVE YOUR ATTENTION MECHANISM WORKS

Before GPT changed everything, Vaswani et al. proved transformers work using
simple sequence tasks. Now YOU must prove YOUR attention implementation works
by passing THREE increasingly difficult challenges.

REQUIRED MODULES:
  Module 01 (Tensor)        : YOUR data structure with autograd
  Module 02 (Activations)   : YOUR ReLU activation
  Module 03 (Layers)        : YOUR Linear layers
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (Autograd)      : YOUR automatic differentiation
  Module 06 (Optimizers)    : YOUR Adam optimizer
  Module 11 (Embeddings)    : YOUR token & positional embeddings
  Module 12 (Attention)     : YOUR multi-head self-attention
  Module 13 (Transformer)   : YOUR transformer blocks

THE THREE CHALLENGES:

  Challenge 1: SEQUENCE REVERSAL (Warm-up)
  ----------------------------------------
  Input:  PYTHON  ->  Output: NOHTYP

  This is IMPOSSIBLE without attention working correctly.
  Each output position must attend to the OPPOSITE input position.


  Challenge 2: SEQUENCE COPYING (Verification)
  --------------------------------------------
  Input:  TENSOR  ->  Output: TENSOR

  Sounds easy? The model must learn DIFFERENT attention patterns:
  - Reversal: anti-diagonal attention
  - Copying: diagonal attention

  Same model, two opposite tasks = real understanding.


  Challenge 3: MIXED TASK INFERENCE (The Real Test)
  -------------------------------------------------
  Given a PREFIX token, the model must:
  - [R] PYTHON  ->  NOHTYP  (reverse)
  - [C] PYTHON  ->  PYTHON  (copy)

  This proves your attention can dynamically route information
  based on context - the foundation of all modern LLMs.

SUCCESS CRITERIA:
  - Challenge 1: 95%+ accuracy on reversal
  - Challenge 2: 95%+ accuracy on copying
  - Challenge 3: 90%+ accuracy on mixed tasks

  Pass all three = Your attention is production-ready!

WHAT THIS PROVES:
  Query-Key-Value computation works
  Attention weights are computed correctly
  Multi-head attention aggregates properly
  Positional encoding preserves position information
  Your architecture can dynamically route information
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, ReLU, CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.text.embeddings import Embedding, PositionalEncoding
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.core.transformer import LayerNorm

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn
from rich import box

console = Console()


class AttentionTransformer:
    """
    Transformer for proving attention works across multiple tasks.

    Architecture:
      Embedding -> Positional -> Attention -> FFN -> Output
    """

    def __init__(self, vocab_size, embed_dim=64, num_heads=4, seq_len=10, num_layers=2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Embedding layers
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(seq_len, embed_dim)

        # Transformer layers
        self.attention_layers = []
        self.ln1_layers = []
        self.ln2_layers = []
        self.fc1_layers = []
        self.fc2_layers = []

        for _ in range(num_layers):
            self.attention_layers.append(MultiHeadAttention(embed_dim, num_heads))
            self.ln1_layers.append(LayerNorm(embed_dim))
            self.ln2_layers.append(LayerNorm(embed_dim))
            self.fc1_layers.append(Linear(embed_dim, embed_dim * 4))
            self.fc2_layers.append(Linear(embed_dim * 4, embed_dim))

        self.relu = ReLU()

        # Output projection
        self.output_proj = Linear(embed_dim, vocab_size)

        # Collect parameters
        self._params = [self.embedding.weight]
        for i in range(num_layers):
            self._params.extend(self.attention_layers[i].parameters())
            self._params.extend(self.ln1_layers[i].parameters())
            self._params.extend(self.ln2_layers[i].parameters())
            self._params.extend([self.fc1_layers[i].weight, self.fc1_layers[i].bias])
            self._params.extend([self.fc2_layers[i].weight, self.fc2_layers[i].bias])
        self._params.extend([self.output_proj.weight, self.output_proj.bias])

        self.total_params = sum(np.prod(p.shape) for p in self._params)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # Embed tokens and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Transformer layers
        for i in range(self.num_layers):
            # Self-attention with residual
            attn_out = self.attention_layers[i].forward(x, mask=None)
            x = self.ln1_layers[i](x + attn_out)

            # FFN with residual
            ffn_out = self.fc2_layers[i](self.relu(self.fc1_layers[i](x)))
            x = self.ln2_layers[i](x + ffn_out)

        # Project to vocabulary
        batch, seq, embed = x.shape
        x_2d = x.reshape(batch * seq, embed)
        logits_2d = self.output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, self.vocab_size)

        return logits

    def parameters(self):
        return self._params


# Token mappings
PADDING = 0
REVERSE_TOKEN = 27  # [R] prefix
COPY_TOKEN = 28     # [C] prefix


def tokens_to_letters(tokens, skip_special=True):
    """Convert token indices to readable letters."""
    result = []
    for t in tokens:
        if t == 0:
            if skip_special:
                continue
            result.append('_')
        elif t == REVERSE_TOKEN:
            result.append('[R]')
        elif t == COPY_TOKEN:
            result.append('[C]')
        elif 1 <= t <= 26:
            result.append(chr(ord('A') + t - 1))
        else:
            result.append('?')
    return ''.join(result)


def letters_to_tokens(s):
    """Convert letters to token indices."""
    return [ord(c) - ord('A') + 1 for c in s.upper() if c.isalpha()]


def generate_reversal_data(num_samples, seq_len=6):
    """Generate sequence reversal dataset."""
    dataset = []
    for _ in range(num_samples):
        seq = np.random.randint(1, 27, size=seq_len)
        reversed_seq = seq[::-1].copy()
        dataset.append((seq, reversed_seq))
    return dataset


def generate_copy_data(num_samples, seq_len=6):
    """Generate sequence copying dataset."""
    dataset = []
    for _ in range(num_samples):
        seq = np.random.randint(1, 27, size=seq_len)
        dataset.append((seq, seq.copy()))
    return dataset


def generate_mixed_data(num_samples, seq_len=6):
    """Generate mixed task dataset with prefix tokens."""
    dataset = []
    for _ in range(num_samples):
        seq = np.random.randint(1, 27, size=seq_len)

        if np.random.random() < 0.5:
            # Reverse task
            input_seq = np.concatenate([[REVERSE_TOKEN], seq])
            target_seq = np.concatenate([[REVERSE_TOKEN], seq[::-1]])
        else:
            # Copy task
            input_seq = np.concatenate([[COPY_TOKEN], seq])
            target_seq = np.concatenate([[COPY_TOKEN], seq])

        dataset.append((input_seq, target_seq))
    return dataset


def train_epoch(model, dataset, optimizer, loss_fn):
    """Train for one epoch."""
    total_loss = 0.0
    correct_sequences = 0

    np.random.shuffle(dataset)

    for input_seq, target_seq in dataset:
        input_tensor = Tensor(input_seq.reshape(1, -1))
        target_tensor = Tensor(target_seq.reshape(1, -1))

        logits = model(input_tensor)

        logits_2d = logits.reshape(-1, model.vocab_size)
        target_1d = target_tensor.reshape(-1)
        loss = loss_fn(logits_2d, target_1d)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.data

        pred = np.argmax(logits.data, axis=-1).flatten()
        if np.array_equal(pred, target_seq):
            correct_sequences += 1

    return total_loss / len(dataset), (correct_sequences / len(dataset)) * 100


def evaluate(model, dataset):
    """Evaluate model on dataset."""
    correct = 0
    predictions = []

    for input_seq, target_seq in dataset:
        input_tensor = Tensor(input_seq.reshape(1, -1))
        logits = model(input_tensor)
        pred = np.argmax(logits.data, axis=-1).flatten()

        predictions.append((input_seq, target_seq, pred))
        if np.array_equal(pred, target_seq):
            correct += 1

    return (correct / len(dataset)) * 100, predictions


def run_challenge(name, model, train_data, test_data, optimizer, loss_fn, epochs, target_acc):
    """Run a single challenge."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]{name}[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    # Show examples
    console.print("[dim]Examples:[/dim]")
    for inp, tgt in train_data[:3]:
        console.print(f"  {tokens_to_letters(inp)} -> {tokens_to_letters(tgt)}")
    console.print()

    # Training
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Training...", total=epochs)

        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, list(train_data), optimizer, loss_fn)
            test_acc, _ = evaluate(model, test_data)
            best_acc = max(best_acc, test_acc)

            progress.update(task, advance=1)

            if (epoch + 1) % 10 == 0:
                console.print(
                    f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                    f"Train={train_acc:.1f}%, Test={test_acc:.1f}%"
                )

            # Early stopping if we hit target
            if test_acc >= target_acc:
                progress.update(task, completed=epochs)
                break

    # Final evaluation
    final_acc, predictions = evaluate(model, test_data)

    # Results
    console.print()
    passed = final_acc >= target_acc

    if passed:
        console.print(f"[bold green]PASSED![/bold green] Accuracy: {final_acc:.1f}% (target: {target_acc}%)")
    else:
        console.print(f"[bold red]FAILED[/bold red] Accuracy: {final_acc:.1f}% (target: {target_acc}%)")

    # Show sample predictions
    console.print("\n[dim]Sample predictions:[/dim]")
    for inp, tgt, pred in predictions[:5]:
        match = "" if np.array_equal(pred, tgt) else ""
        style = "green" if np.array_equal(pred, tgt) else "red"
        console.print(f"  [{style}]{match}[/{style}] {tokens_to_letters(inp)} -> {tokens_to_letters(pred)}")

    return passed, final_acc


def main():
    """Main training loop with three challenges."""

    # Banner
    console.print()
    console.print(Panel.fit(
        "[bold cyan]MILESTONE 05: ATTENTION IS ALL YOU NEED[/bold cyan]\n\n"
        "[yellow]Prove your attention mechanism works by passing THREE challenges.[/yellow]\n\n"
        "Challenge 1: Sequence Reversal (PYTHON -> NOHTYP)\n"
        "Challenge 2: Sequence Copying  (TENSOR -> TENSOR)\n"
        "Challenge 3: Mixed Tasks       ([R]ABC -> CBA, [C]ABC -> ABC)",
        border_style="cyan",
        title="The Transformer Challenge"
    ))
    console.print()

    # Configuration
    vocab_size = 29  # 0=pad, 1-26=A-Z, 27=[R], 28=[C]
    seq_len = 6
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    lr = 0.001

    console.print(Panel(
        f"[bold]Model Configuration[/bold]\n"
        f"  Vocabulary:  {vocab_size} tokens (A-Z + special)\n"
        f"  Sequence:    {seq_len} letters\n"
        f"  Embedding:   {embed_dim} dimensions\n"
        f"  Attention:   {num_heads} heads\n"
        f"  Layers:      {num_layers} transformer blocks\n"
        f"  Learning:    {lr}",
        title="Configuration",
        border_style="blue"
    ))

    # Build model
    console.print("\n[bold]Building Transformer...[/bold]")
    model = AttentionTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len + 1,  # +1 for task prefix in challenge 3
        num_layers=num_layers
    )
    console.print(f"  Total parameters: {model.total_params:,}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    results = {}

    # Challenge 1: Sequence Reversal
    train_rev = generate_reversal_data(600, seq_len)
    test_rev = generate_reversal_data(200, seq_len)
    passed1, acc1 = run_challenge(
        "CHALLENGE 1: SEQUENCE REVERSAL",
        model, train_rev, test_rev, optimizer, loss_fn,
        epochs=50, target_acc=95
    )
    results['reversal'] = (passed1, acc1)

    # Challenge 2: Sequence Copying (same model, different task)
    train_copy = generate_copy_data(600, seq_len)
    test_copy = generate_copy_data(200, seq_len)
    passed2, acc2 = run_challenge(
        "CHALLENGE 2: SEQUENCE COPYING",
        model, train_copy, test_copy, optimizer, loss_fn,
        epochs=50, target_acc=95
    )
    results['copying'] = (passed2, acc2)

    # Challenge 3: Mixed Tasks (the real test)
    train_mixed = generate_mixed_data(800, seq_len)
    test_mixed = generate_mixed_data(300, seq_len)
    passed3, acc3 = run_challenge(
        "CHALLENGE 3: MIXED TASK INFERENCE",
        model, train_mixed, test_mixed, optimizer, loss_fn,
        epochs=60, target_acc=90
    )
    results['mixed'] = (passed3, acc3)

    # Final Summary
    console.print("\n" + "="*60)
    console.print(Panel.fit("[bold]FINAL RESULTS[/bold]", border_style="cyan"))

    table = Table(box=box.ROUNDED)
    table.add_column("Challenge", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    table.add_row(
        "1. Reversal",
        f"{acc1:.1f}%",
        "95%",
        "[green]PASSED[/green]" if passed1 else "[red]FAILED[/red]"
    )
    table.add_row(
        "2. Copying",
        f"{acc2:.1f}%",
        "95%",
        "[green]PASSED[/green]" if passed2 else "[red]FAILED[/red]"
    )
    table.add_row(
        "3. Mixed Tasks",
        f"{acc3:.1f}%",
        "90%",
        "[green]PASSED[/green]" if passed3 else "[red]FAILED[/red]"
    )

    console.print(table)
    console.print()

    all_passed = passed1 and passed2 and passed3

    if all_passed:
        console.print(Panel.fit(
            "[bold green]MILESTONE 05 COMPLETE![/bold green]\n\n"
            "Your attention mechanism has proven it can:\n"
            "  Query-Key-Value computation works\n"
            "  Learn different attention patterns (diagonal vs anti-diagonal)\n"
            "  Dynamically route information based on context\n"
            "  Handle multiple tasks with a single model\n\n"
            "[bold]This is the foundation of GPT, BERT, and all modern LLMs![/bold]",
            border_style="green",
            title="ATTENTION IS ALL YOU NEED"
        ))
        return 0
    else:
        failed = []
        if not passed1:
            failed.append("Reversal")
        if not passed2:
            failed.append("Copying")
        if not passed3:
            failed.append("Mixed Tasks")

        console.print(Panel.fit(
            f"[bold yellow]CHALLENGES FAILED: {', '.join(failed)}[/bold yellow]\n\n"
            "Check your implementation:\n"
            "  MultiHeadAttention: Q, K, V projections\n"
            "  Positional encoding is being added\n"
            "  Attention scores use softmax correctly\n"
            "  Gradients flow through all layers",
            border_style="yellow",
            title="Keep Working"
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
