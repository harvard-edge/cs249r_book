#!/usr/bin/env python3
"""
Attention is All You Need (2017) - The Transformer Challenge
=============================================================

📚 HISTORICAL CONTEXT:
In 2017, Vaswani et al. published "Attention is All You Need," introducing the
Transformer architecture that would power GPT, BERT, and all modern LLMs. The
key innovation: SELF-ATTENTION allows each position to attend to all others,
enabling parallel training and capturing long-range dependencies.

🎯 MILESTONE 05: PROVE YOUR ATTENTION MECHANISM WORKS
Before GPT changed everything, Vaswani et al. proved transformers work using
simple sequence tasks. Now YOU must prove YOUR attention implementation works
by passing THREE increasingly difficult challenges.

✅ REQUIRED MODULES (Run after Module 13):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autograd
  Module 02 (Activations)   : YOUR ReLU activation
  Module 03 (Layers)        : YOUR Linear layers
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (DataLoader)    : YOUR Dataset/DataLoader for batching
  Module 06 (Autograd)      : YOUR automatic differentiation
  Module 07 (Optimizers)    : YOUR Adam optimizer
  Module 11 (Embeddings)    : YOUR token & positional embeddings  <-- NEW!
  Module 12 (Attention)     : YOUR multi-head self-attention      <-- NEW!
  Module 13 (Transformer)   : YOUR transformer blocks             <-- NEW!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (The Original Transformer):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Tokens    │    │  Embedding  │    │ Positional  │    │ Multi-Head  │    │    FFN      │
    │  "PYTHON"   │───▶│ YOUR Module │───▶│   Encoding  │───▶│  Attention  │───▶│ YOUR Modules│
    │   Input     │    │     11      │    │   YOUR M11  │    │   YOUR M12  │    │   02 + 03   │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         Token→Vector      Position info       Self-attention     Feed-forward
                                           injected            Q·K·V computed     network

    🔍 ATTENTION MECHANISM - The Key Innovation:

    Query (Q): "What am I looking for?"     ┌──────────────────────────────────────┐
    Key (K):   "What do I contain?"         │ Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V│
    Value (V): "What information to pass?"  └──────────────────────────────────────┘

    For sequence reversal "PYTHON" → "NOHTYP":

    Output position 0 (N) must attend to input position 5 (N)
    Output position 1 (O) must attend to input position 4 (O)
    Output position 2 (H) must attend to input position 3 (H)
    ...

    This is IMPOSSIBLE without correct attention!

🎮 THE THREE CHALLENGES:

    Challenge 1: SEQUENCE REVERSAL (Warm-up)
    ────────────────────────────────────────
    Input:  P Y T H O N  →  Output: N O H T Y P

    ┌─┬─┬─┬─┬─┬─┐              ┌─┬─┬─┬─┬─┬─┐
    │P│Y│T│H│O│N│     →        │N│O│H│T│Y│P│
    └─┴─┴─┴─┴─┴─┘              └─┴─┴─┴─┴─┴─┘
     0 1 2 3 4 5                5 4 3 2 1 0  ← Attention pattern

    Success requires: Anti-diagonal attention weights!


    Challenge 2: SEQUENCE COPYING (Verification)
    ────────────────────────────────────────────
    Input:  T E N S O R  →  Output: T E N S O R

    Same model must learn DIFFERENT attention pattern:
    - Reversal: anti-diagonal
    - Copying: diagonal (identity)

    This proves YOUR attention can learn task-specific patterns!


    Challenge 3: MIXED TASK INFERENCE (The Real Test)
    ─────────────────────────────────────────────────
    [R] PYTHON  →  NOHTYP  (reverse prefix)
    [C] PYTHON  →  PYTHON  (copy prefix)

    The PREFIX token controls the behavior!
    This is how GPT-style models work: context determines output.

📊 SUCCESS CRITERIA:
  ✅ Challenge 1: 95%+ accuracy on reversal
  ✅ Challenge 2: 95%+ accuracy on copying
  ✅ Challenge 3: 90%+ accuracy on mixed tasks

  Pass all three = Your attention is production-ready!

🔥 WHAT THIS PROVES ABOUT YOUR IMPLEMENTATION:
  • Query-Key-Value computation works correctly
  • Attention weights are computed properly (softmax(Q·K/√d))
  • Multi-head attention aggregates information from multiple perspectives
  • Positional encoding preserves sequence position information
  • YOUR architecture can dynamically route information based on context
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
from tinytorch.core.dataloader import Dataset, DataLoader  # Module 05: YOUR DataLoader!
from tinytorch.core.embeddings import Embedding, PositionalEncoding
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.core.transformers import LayerNorm

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn
from rich import box

console = Console()

# =============================================================================
# 🎯 YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone showcases YOUR attention and transformer modules:
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01: Tensor   │ 3D tensors for sequences       │ (batch, seq_len, embed_dim) │
# │                     │ + attention weight matrices    │ format for transformer ops  │
# │                     │                                │                             │
# │ Module 02: ReLU     │ Non-linearity in FFN blocks    │ Feed-forward network after  │
# │                     │ after attention                │ each attention layer        │
# │                     │                                │                             │
# │ Module 03: Linear   │ Q, K, V projections + FFN      │ 4× width expansion in FFN   │
# │                     │ + output projection            │ (standard transformer)      │
# │                     │                                │                             │
# │ Module 05: DataLoader│ Batches sequence data         │ Efficient training with     │
# │                     │ for transformer training       │ shuffled mini-batches       │
# │                     │                                │                             │
# │ Module 11: Embedding│ Token → Dense vector           │ Learned representations     │
# │ ★ NEW MODULE ★      │ + Positional encoding          │ with position information   │
# │                     │                                │                             │
# │ Module 12: Attention│ Multi-head self-attention      │ Q·K·V computation with      │
# │ ★ NEW MODULE ★      │ computes attention weights     │ softmax and scaling         │
# │                     │                                │                             │
# │ Module 13: LayerNorm│ Normalizes activations         │ Stabilizes deep transformer │
# │ ★ NEW MODULE ★      │ before/after attention         │ training                    │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================
# 🆕 WHAT'S NEW SINCE MILESTONE 04 (CNN)
# =============================================================================
#
# ┌──────────────────────┬─────────────────────────┬────────────────────────────┐
# │ CNN (Milestone 04)   │ Transformer (This)      │ Why It's Different         │
# ├──────────────────────┼─────────────────────────┼────────────────────────────┤
# │ Local connectivity   │ Global attention        │ Every position sees all!   │
# │ Spatial features     │ Sequence features       │ Text/audio/time series     │
# │ Fixed receptive field│ Dynamic attention       │ Model decides what to see  │
# │ Conv2d kernels       │ Q·K·V projections       │ Learned attention patterns │
# │ Translation invariance│ Position encoding      │ Explicit position info     │
# │ Images only          │ ANY sequence data       │ Language, music, code...   │
# └──────────────────────┴─────────────────────────┴────────────────────────────┘
#
# =============================================================================


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


# =============================================================================
# 📦 SEQUENCE DATASET - Using YOUR DataLoader from Module 05
# =============================================================================

class SequenceDataset(Dataset):
    """
    Dataset for sequence-to-sequence tasks using YOUR Dataset interface.
    
    This shows how YOUR DataLoader handles sequence data for transformers,
    enabling efficient batched training with shuffling.
    """
    
    def __init__(self, data):
        """
        Args:
            data: List of (input_seq, target_seq) tuples
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return Tensor(input_seq), Tensor(target_seq)


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


def train_epoch(model, dataloader, optimizer, loss_fn):
    """Train for one epoch using YOUR DataLoader."""
    total_loss = 0.0
    correct_sequences = 0
    total_samples = 0

    # Use YOUR DataLoader for batched training!
    for input_batch, target_batch in dataloader:
        batch_size = input_batch.shape[0]

        logits = model(input_batch)

        # Reshape for loss computation
        logits_2d = logits.reshape(-1, model.vocab_size)
        target_1d = target_batch.reshape(-1)
        loss = loss_fn(logits_2d, target_1d)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.data * batch_size

        # Check sequence accuracy per sample in batch
        pred = np.argmax(logits.data, axis=-1)
        for i in range(batch_size):
            if np.array_equal(pred[i], target_batch.data[i]):
                correct_sequences += 1
        total_samples += batch_size

    return total_loss / total_samples, (correct_sequences / total_samples) * 100

def press_enter_to_continue() :
    if sys.stdin.isatty() and sys.stdout.isatty() :
        try :
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError :
            pass
        console.print()

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


def run_challenge(name, model, train_data, test_data, optimizer, loss_fn, epochs, target_acc, batch_size=16):
    """Run a single challenge using YOUR DataLoader."""
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]{name}[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    # Show examples
    console.print("[dim]Examples:[/dim]")
    for inp, tgt in train_data[:3]:
        console.print(f"  {tokens_to_letters(inp)} -> {tokens_to_letters(tgt)}")
    press_enter_to_continue()

    # Create DataLoader for training (YOUR Module 05!)
    train_dataset = SequenceDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    console.print(f"[dim]DataLoader: {len(train_dataset)} samples, batch_size={batch_size}, {len(train_loader)} batches[/dim]\n")

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
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn)
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

    press_enter_to_continue()

    # Show sample predictions
    console.print("[dim]Sample predictions:[/dim]")
    for inp, tgt, pred in predictions[:5]:
        match = "" if np.array_equal(pred, tgt) else ""
        style = "green" if np.array_equal(pred, tgt) else "red"
        console.print(f"  [{style}]{match}[/{style}] {tokens_to_letters(inp)} -> {tokens_to_letters(pred)}")

    press_enter_to_continue()

    return passed, final_acc


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model hyperparameters (shared across all challenges)
CONFIG = {
    'vocab_size': 29,      # 0=pad, 1-26=A-Z, 27=[R], 28=[C]
    'seq_len': 6,          # Sequence length for tasks
    'embed_dim': 64,       # Embedding dimensions
    'num_heads': 4,        # Attention heads
    'num_layers': 2,       # Transformer blocks
    'lr': 0.001,           # Learning rate
}


def build_model(config=CONFIG):
    """
    Build a fresh transformer model with optimizer.

    Returns:
        model: AttentionTransformer instance
        optimizer: Adam optimizer
        loss_fn: CrossEntropyLoss
    """
    model = AttentionTransformer(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        seq_len=config['seq_len'] + 1,  # +1 for task prefix in challenge 3
        num_layers=config['num_layers']
    )

    for param in model.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters(), lr=config['lr'])
    loss_fn = CrossEntropyLoss()

    return model, optimizer, loss_fn


# =============================================================================
# CHALLENGE 1: SEQUENCE REVERSAL
# =============================================================================

def challenge_1_reversal(model, optimizer, loss_fn, config=CONFIG):
    """
    Challenge 1: Learn to reverse sequences (PYTHON -> NOHTYP).

    Task: Input sequence reversed in output
    ──────────────────────────────────────
        Input:   P  Y  T  H  O  N
        Output:  N  O  H  T  Y  P

    Required Attention Pattern (anti-diagonal):
    ───────────────────────────────────────────
        Output position 0 (N) attends to input position 5 (N)
        Output position 1 (O) attends to input position 4 (O)
        Output position 2 (H) attends to input position 3 (H)
        ...

        Attention weights matrix (ideal):
                    Input positions
                    0  1  2  3  4  5
        Output  0 [ .  .  .  .  .  █ ]  → attends to pos 5
        pos     1 [ .  .  .  .  █  . ]  → attends to pos 4
                2 [ .  .  .  █  .  . ]  → attends to pos 3
                3 [ .  .  █  .  .  . ]  → attends to pos 2
                4 [ .  █  .  .  .  . ]  → attends to pos 1
                5 [ █  .  .  .  .  . ]  → attends to pos 0

    Returns:
        passed: bool - whether target accuracy was achieved
        accuracy: float - final test accuracy
    """
    seq_len = config['seq_len']

    train_data = generate_reversal_data(600, seq_len)
    test_data = generate_reversal_data(200, seq_len)

    return run_challenge(
        "CHALLENGE 1: SEQUENCE REVERSAL",
        model, train_data, test_data, optimizer, loss_fn,
        epochs=50, target_acc=95
    )


# =============================================================================
# CHALLENGE 2: SEQUENCE COPYING
# =============================================================================

def challenge_2_copying(model, optimizer, loss_fn, config=CONFIG):
    """
    Challenge 2: Learn to copy sequences (TENSOR -> TENSOR).

    Task: Input sequence copied to output (identity)
    ────────────────────────────────────────────────
        Input:   T  E  N  S  O  R
        Output:  T  E  N  S  O  R

    Required Attention Pattern (diagonal):
    ──────────────────────────────────────
        Output position 0 (T) attends to input position 0 (T)
        Output position 1 (E) attends to input position 1 (E)
        Output position 2 (N) attends to input position 2 (N)
        ...

        Attention weights matrix (ideal):
                    Input positions
                    0  1  2  3  4  5
        Output  0 [ █  .  .  .  .  . ]  → attends to pos 0
        pos     1 [ .  █  .  .  .  . ]  → attends to pos 1
                2 [ .  .  █  .  .  . ]  → attends to pos 2
                3 [ .  .  .  █  .  . ]  → attends to pos 3
                4 [ .  .  .  .  █  . ]  → attends to pos 4
                5 [ .  .  .  .  .  █ ]  → attends to pos 5

    Why same model?
    ───────────────
    Uses the SAME model from Challenge 1, demonstrating that
    transformers can adapt their attention patterns to new tasks
    without architectural changes.

    Returns:
        passed: bool - whether target accuracy was achieved
        accuracy: float - final test accuracy
    """
    seq_len = config['seq_len']

    train_data = generate_copy_data(600, seq_len)
    test_data = generate_copy_data(200, seq_len)

    return run_challenge(
        "CHALLENGE 2: SEQUENCE COPYING",
        model, train_data, test_data, optimizer, loss_fn,
        epochs=50, target_acc=95
    )


# =============================================================================
# CHALLENGE 3: MIXED TASK INFERENCE
# =============================================================================

def challenge_3_mixed(config=CONFIG):
    """
    Challenge 3: Learn BOTH tasks with prefix conditioning.

    Task: Prefix token controls behavior
    ────────────────────────────────────
        [R] A B C D E F  →  F E D C B A   (reverse)
        [C] A B C D E F  →  A B C D E F   (copy)

        The prefix token [R] or [C] tells the model what to do!
        This is how GPT-style instruction following works.

    Required Attention Pattern (context-dependent):
    ───────────────────────────────────────────────
        When prefix = [R]:
                      [R] A  B  C  D  E  F
            Output A [ .  .  .  .  .  .  █ ]  → attends to F
            Output B [ .  .  .  .  .  █  . ]  → attends to E
            ...

        When prefix = [C]:
                      [C] A  B  C  D  E  F
            Output A [ .  █  .  .  .  .  . ]  → attends to A
            Output B [ .  .  █  .  .  .  . ]  → attends to B
            ...

        The model learns to READ the prefix and ROUTE accordingly!

    Why fresh model?
    ────────────────
    After challenges 1 & 2, the model has been trained sequentially:
      1. Learn reversal → weights encode anti-diagonal pattern
      2. Learn copying  → weights OVERWRITE to diagonal pattern

    By challenge 3, the model is "stuck" in copy mode and can't
    learn conditional behavior. Starting fresh allows it to learn
    BOTH patterns simultaneously, conditioned on the prefix token.

    This is the key insight: transformers can learn to dynamically
    route information based on context (the prefix), which is the
    foundation of instruction-following in modern LLMs.

    Returns:
        passed: bool - whether target accuracy was achieved
        accuracy: float - final test accuracy
    """
    seq_len = config['seq_len']

    console.print("[dim]Building fresh model for mixed task learning...[/dim]")
    model, optimizer, loss_fn = build_model(config)
    console.print(f"[dim]  Total parameters: {model.total_params:,}[/dim]\n")

    train_data = generate_mixed_data(800, seq_len)
    test_data = generate_mixed_data(300, seq_len)

    return run_challenge(
        "CHALLENGE 3: MIXED TASK INFERENCE",
        model, train_data, test_data, optimizer, loss_fn,
        epochs=60, target_acc=90
    )


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_final_results(results):
    """
    Print the final results table and success/failure message.

    Args:
        results: dict with keys 'reversal', 'copying', 'mixed'
                 each containing (passed: bool, accuracy: float)

    Returns:
        0 if all passed, 1 otherwise
    """
    passed1, acc1 = results['reversal']
    passed2, acc2 = results['copying']
    passed3, acc3 = results['mixed']

    console.print("=" * 60)
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
    press_enter_to_continue()

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
        press_enter_to_continue()
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
        press_enter_to_continue()
        return 1


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point: Run all three transformer challenges.

    Challenge Structure:
    ────────────────────
    1. REVERSAL  - Can attention learn anti-diagonal patterns?
    2. COPYING   - Can attention learn diagonal patterns? (same model)
    3. MIXED     - Can attention learn conditional behavior? (fresh model)

    The first two challenges use the same model to show adaptability.
    The third uses a fresh model to properly test prefix conditioning.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # BANNER
    # ─────────────────────────────────────────────────────────────────────────
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
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Model Configuration[/bold]\n"
        f"  Vocabulary:  {CONFIG['vocab_size']} tokens (A-Z + special)\n"
        f"  Sequence:    {CONFIG['seq_len']} letters\n"
        f"  Embedding:   {CONFIG['embed_dim']} dimensions\n"
        f"  Attention:   {CONFIG['num_heads']} heads\n"
        f"  Layers:      {CONFIG['num_layers']} transformer blocks\n"
        f"  Learning:    {CONFIG['lr']}",
        title="Configuration",
        border_style="blue"
    ))
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD MODEL (shared for challenges 1 & 2)
    # ─────────────────────────────────────────────────────────────────────────
    console.print("[bold]Building Transformer...[/bold]")
    model, optimizer, loss_fn = build_model()
    console.print(f"  Total parameters: {model.total_params:,}")
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # RUN CHALLENGES
    # ─────────────────────────────────────────────────────────────────────────
    results = {}

    # Challenge 1: Sequence Reversal
    results['reversal'] = challenge_1_reversal(model, optimizer, loss_fn)

    # Challenge 2: Sequence Copying (same model, different task)
    results['copying'] = challenge_2_copying(model, optimizer, loss_fn)

    # Challenge 3: Mixed Tasks (fresh model - see docstring for why)
    results['mixed'] = challenge_3_mixed()

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    return print_final_results(results)


if __name__ == "__main__":
    sys.exit(main())
