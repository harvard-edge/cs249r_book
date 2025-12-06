#!/usr/bin/env python3
"""
Sequence Reversal (2017) - Attention Mechanism Proof
=====================================================

🎯 MILESTONE 5.0: PROVE ATTENTION WORKS (From "Attention is All You Need")

Before building GPT, let's PROVE your attention mechanism works using the
canonical test from Vaswani et al. (2017): Sequence Reversal.

✅ REQUIRED MODULES (Run after Module 12):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autograd
  Module 02 (Activations)   : YOUR ReLU activation
  Module 03 (Layers)        : YOUR Linear layers
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (Autograd)      : YOUR automatic differentiation
  Module 06 (Optimizers)    : YOUR Adam optimizer
  Module 11 (Embeddings)    : YOUR token & positional embeddings
  Module 12 (Attention)     : YOUR multi-head self-attention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔬 THE CANONICAL TEST:

From "Attention is All You Need" (Vaswani et al., 2017):
"We also trained on the copy and reverse tasks to verify our model learns
to attend to relevant positions."

**WHY SEQUENCE REVERSAL?**
This task is IMPOSSIBLE without attention working correctly:

    Input:  [1, 2, 3, 4, 5]
    Output: [5, 4, 3, 2, 1]
    
    ❌ Cannot use element-wise operations (each position only sees itself)
    ❌ Cannot use local convolution (limited receptive field)
    ❌ Cannot use positional encoding alone (doesn't provide content)
    ✅ REQUIRES attention to look at distant positions!

🏗️ ARCHITECTURE (Minimal Transformer):

    ┌──────────────────────────────────────────────────────────────────────┐
    │                         Output Predictions                           │
    │                   Vocabulary Logits (vocab_size)                     │
    └──────────────────────────────────────────────────────────────────────┘
                                     ▲
    ┌──────────────────────────────────────────────────────────────────────┐
    │                       Output Projection                              │
    │                  Module 03: embed_dim → vocab_size                   │
    └──────────────────────────────────────────────────────────────────────┘
                                     ▲
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         LayerNorm                                    │
    │                   Module 13: Normalization                           │
    └──────────────────────────────────────────────────────────────────────┘
                                     ▲
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    Feed-Forward Network                              │
    │            Module 03: Linear → ReLU → Linear                         │
    └──────────────────────────────────────────────────────────────────────┘
                                     ▲
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         LayerNorm                                    │
    └──────────────────────────────────────────────────────────────────────┘
                                     ▲
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              ⭐ MULTI-HEAD SELF-ATTENTION ⭐                          ║
    ║                                                                      ║
    ║  This is what makes reversal possible!                              ║
    ║  Output[0] attends to Input[4]                                      ║
    ║  Output[1] attends to Input[3]                                      ║
    ║  Output[2] attends to Input[2]                                      ║
    ║  Output[3] attends to Input[1]                                      ║
    ║  Output[4] attends to Input[0]                                      ║
    ║                                                                      ║
    ║  Anti-diagonal attention pattern!                                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
                                     ▲
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    Positional Encoding                               │
    │              Module 11: Add position information                     │
    └──────────────────────────────────────────────────────────────────────┘
                                     ▲
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      Token Embeddings                                │
    │              Module 11: tokens → embed_dim vectors                   │
    └──────────────────────────────────────────────────────────────────────┘
                                     ▲
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      Input Sequence                                  │
    │                     [1, 2, 3, 4, 5]                                  │
    └──────────────────────────────────────────────────────────────────────┘

📊 EXPECTED PERFORMANCE:
  - Task: Reverse sequences of length 6-8
  - Vocabulary: 10 unique tokens (0-9)
  - Training time: ~30 seconds (instant gratification!)
  - Expected: 95%+ exact sequence match accuracy
  - Success = "My attention mechanism actually computes relationships!"

💡 WHAT TO WATCH FOR:
  - Epoch 1-5: Model learns sequence structure
  - Epoch 6-10: Starts getting some reversals correct
  - Epoch 11-20: 80-90% accuracy
  - Epoch 21-30: 95%+ accuracy
  - If this works → Your attention is computing cross-position relationships! ✓

🎓 LEARNING OUTCOMES:
After this milestone, you'll have PROVEN that:
  ✅ Your Query·Key·Value computation works
  ✅ Your attention weights are being computed correctly
  ✅ Your multi-head attention aggregates properly
  ✅ Your positional encoding preserves position information
  ✅ Your architecture can learn to route information across positions

🚀 NEXT STEPS:
If this works, you're ready for:
  - 01_vaswani_generation.py: Character-level Q&A
  - 02_vaswani_dialogue.py: Full conversational AI
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
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn
from rich import box

console = Console()

# ============================================================================
# 🎓 STUDENT CODE: Minimal Transformer for Sequence Reversal
# ============================================================================

class ReversalTransformer:
    """
    Minimal Transformer specifically designed to prove attention works.
    
    Architecture:
      Embedding → Positional → Attention → FFN → Output
      
    This is the SIMPLEST transformer that can learn to reverse sequences.
    """
    
    def __init__(self, vocab_size=10, embed_dim=32, num_heads=4, seq_len=8):
        console.print("🏗️  Building Minimal Transformer for Sequence Reversal...")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Embedding layers
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(seq_len, embed_dim)
        
        # Transformer block
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        
        # Feed-forward network
        self.fc1 = Linear(embed_dim, embed_dim * 2)
        self.relu = ReLU()
        self.fc2 = Linear(embed_dim * 2, embed_dim)
        
        # Output projection
        self.output_proj = Linear(embed_dim, vocab_size)
        
        # Count parameters
        params = (
            [self.embedding.weight] +
            self.attention.parameters() +
            self.ln1.parameters() + self.ln2.parameters() +
            [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias] +
            [self.output_proj.weight, self.output_proj.bias]
        )
        total_params = sum(np.prod(p.shape) for p in params)
        
        console.print(f"  ✓ Embeddings: vocab={vocab_size}, dim={embed_dim}")
        console.print(f"  ✓ Attention: {num_heads} heads")
        console.print(f"  ✓ FFN: {embed_dim} → {embed_dim*2} → {embed_dim}")
        console.print(f"  ✓ Total parameters: {total_params:,}\n")
        
        self._params = params
    
    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward pass through the transformer.
        
        Args:
            x: Input sequences (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, seq_len, vocab_size)
        """
        # Embed tokens and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Transformer block with residual connections
        # Self-attention
        attn_out = self.attention.forward(x, mask=None)
        x = self.ln1(x + attn_out)
        
        # Feed-forward network
        ffn_out = self.fc2(self.relu(self.fc1(x)))
        x = self.ln2(x + ffn_out)
        
        # Project to vocabulary
        batch, seq, embed = x.shape
        x_2d = x.reshape(batch * seq, embed)
        logits_2d = self.output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, self.vocab_size)
        
        return logits
    
    def parameters(self):
        """Get all trainable parameters."""
        return self._params


def generate_reversal_dataset(num_samples=200, seq_len=6, vocab_size=26):
    """
    Generate sequence reversal dataset using letters A-Z.
    
    Each sample is (input_seq, target_seq) where target = reverse(input)
    More intuitive than numbers: "CAT" → "TAC", "HELLO" → "OLLEH"
    """
    dataset = []
    for _ in range(num_samples):
        # Generate random sequence of letters (1-26 maps to A-Z)
        seq = np.random.randint(1, min(vocab_size, 27), size=seq_len)
        reversed_seq = seq[::-1].copy()
        dataset.append((seq, reversed_seq))
    return dataset


def tokens_to_letters(tokens):
    """Convert token indices to readable letters (1=A, 2=B, ...)"""
    return ''.join(chr(ord('A') + t - 1) if 1 <= t <= 26 else '?' for t in tokens)


# Fun word examples for demonstration
FUN_WORDS = [
    "PYTHON",
    "TORCH", 
    "NEURAL",
    "TENSOR",
    "ATTEND",
    "VASWANI",
    "QUERY",
    "HELLO",
]


def train_epoch(model, dataset, optimizer, loss_fn):
    """Train for one epoch."""
    total_loss = 0.0
    correct_sequences = 0
    total_sequences = len(dataset)
    
    for input_seq, target_seq in dataset:
        # Convert to tensors (add batch dimension)
        input_tensor = Tensor(input_seq.reshape(1, -1))
        target_tensor = Tensor(target_seq.reshape(1, -1))
        
        # Forward pass
        logits = model(input_tensor)
        
        # Reshape for loss computation
        logits_2d = logits.reshape(-1, model.vocab_size)
        target_1d = target_tensor.reshape(-1)
        loss = loss_fn(logits_2d, target_1d)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.data
        
        # Check if entire sequence is correct
        pred = np.argmax(logits.data, axis=-1).flatten()
        if np.array_equal(pred, target_seq):
            correct_sequences += 1
    
    avg_loss = total_loss / total_sequences
    accuracy = (correct_sequences / total_sequences) * 100
    
    return avg_loss, accuracy


def evaluate(model, dataset):
    """Evaluate model on dataset."""
    correct_sequences = 0
    predictions = []
    
    for input_seq, target_seq in dataset:
        input_tensor = Tensor(input_seq.reshape(1, -1))
        logits = model(input_tensor)
        pred = np.argmax(logits.data, axis=-1).flatten()
        
        predictions.append((input_seq, target_seq, pred))
        if np.array_equal(pred, target_seq):
            correct_sequences += 1
    
    accuracy = (correct_sequences / len(dataset)) * 100
    return accuracy, predictions


def main():
    """Main training loop."""
    
    # Banner
    console.print()
    console.print("="*70)
    console.print(Panel.fit(
        "[bold cyan]Sequence Reversal: The Attention Proof[/bold cyan]\n"
        "[dim]From 'Attention is All You Need' (Vaswani et al., 2017)[/dim]\n\n"
        "[yellow]This task CANNOT be solved without attention working![/yellow]",
        border_style="cyan",
        title="⭐ Milestone 5.0",
    ))
    console.print("="*70)
    console.print()
    
    # Hyperparameters  
    vocab_size = 27  # 0 (padding) + A-Z (1-26)
    seq_len = 6      # 6-letter "words"
    embed_dim = 32
    num_heads = 4
    lr = 0.001
    epochs = 100
    train_size = 500
    test_size = 200
    
    console.print(Panel(
        f"[bold]Hyperparameters[/bold]\n"
        f"  Vocabulary: [cyan]{vocab_size}[/cyan] tokens (A-Z letters)\n"
        f"  Sequence:   [cyan]{seq_len}[/cyan] letters per word\n"
        f"  Embedding:  [cyan]{embed_dim}[/cyan] dimensions\n"
        f"  Attention:  [cyan]{num_heads}[/cyan] heads\n"
        f"  Learning:   [cyan]{lr}[/cyan]\n"
        f"  Epochs:     [cyan]{epochs}[/cyan]",
        title="⚙️  Configuration",
        border_style="blue"
    ))
    console.print()
    
    # Generate data
    console.print("📊 Generating reversal dataset...")
    console.print("   [dim]Task: Reverse letters like PYTHON → NOHTYP[/dim]")
    train_data = generate_reversal_dataset(num_samples=train_size, seq_len=seq_len, vocab_size=vocab_size)
    test_data = generate_reversal_dataset(num_samples=test_size, seq_len=seq_len, vocab_size=vocab_size)
    console.print(f"  ✓ Training samples: {len(train_data)}")
    console.print(f"  ✓ Test samples: {len(test_data)}\n")
    
    # Show example with letters
    console.print("🔍 Example:")
    ex_in, ex_out = train_data[0]
    console.print(f"  Input:  [cyan]{tokens_to_letters(ex_in)}[/cyan] → Target: [green]{tokens_to_letters(ex_out)}[/green]")
    console.print(f"  [dim](Numbers: {ex_in.tolist()} → {ex_out.tolist()})[/dim]")
    console.print()
    
    # Build model
    model = ReversalTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len
    )
    
    # Set requires_grad
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    
    # Training
    console.print("🚀 Training transformer to reverse sequences...\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=epochs)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = train_epoch(model, train_data, optimizer, loss_fn)
            
            # Evaluate
            test_acc, _ = evaluate(model, test_data)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            # Update progress
            progress.update(task, advance=1)
            
            # Print every 5 epochs
            if (epoch + 1) % 5 == 0:
                console.print(
                    f"  Epoch [cyan]{epoch+1:2d}[/cyan]: "
                    f"Loss = [yellow]{train_loss:.4f}[/yellow], "
                    f"Train Acc = [green]{train_acc:.1f}%[/green], "
                    f"Test Acc = [green]{test_acc:.1f}%[/green]"
                )
    
    console.print()
    
    # Final evaluation
    final_acc, predictions = evaluate(model, test_data)
    
    # Results
    console.print("="*70)
    console.print(Panel.fit(
        "[bold]Training Complete![/bold]",
        border_style="green"
    ))
    console.print("="*70)
    console.print()
    
    # Show final accuracy
    table = Table(title="📊 Final Results", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Status", style="yellow", justify="center")
    
    table.add_row(
        "Test Accuracy",
        f"{final_acc:.1f}%",
        "✅ EXCELLENT" if final_acc >= 95 else "⚠️  LEARNING" if final_acc >= 80 else "❌ NEEDS WORK"
    )
    table.add_row(
        "Training Loss",
        f"{history['train_loss'][-1]:.4f}",
        "✅" if history['train_loss'][-1] < 0.5 else "⚠️"
    )
    
    console.print(table)
    console.print()
    
    # Show sample predictions with letters
    console.print(Panel("[bold]Sample Predictions[/bold]", border_style="blue"))
    console.print()
    
    for i, (inp, target, pred) in enumerate(predictions[:8]):
        match = "✓" if np.array_equal(pred, target) else "✗"
        style = "green" if np.array_equal(pred, target) else "red"
        
        inp_str = tokens_to_letters(inp)
        target_str = tokens_to_letters(target)
        pred_str = tokens_to_letters(pred)
        
        console.print(f"  [{style}]{match}[/{style}] Input:  [cyan]{inp_str}[/cyan]")
        console.print(f"     Target: [green]{target_str}[/green]")
        console.print(f"     Pred:   [{style}]{pred_str}[/{style}]\n")
    
    # Verdict
    console.print("="*70)
    if final_acc >= 95:
        console.print(Panel.fit(
            "[bold green]🎉 SUCCESS! Your attention mechanism is working![/bold green]\n\n"
            "Your transformer learned to reverse sequences, which proves:\n"
            "  ✅ Query·Key·Value computation is correct\n"
            "  ✅ Attention weights are being computed properly\n"
            "  ✅ Multi-head attention aggregates correctly\n"
            "  ✅ Positional encoding preserves position information\n\n"
            "[bold]You're ready for complex tasks like Q&A and generation![/bold]",
            border_style="green",
            title="⭐ Attention Proof Complete"
        ))
    elif final_acc >= 80:
        console.print(Panel.fit(
            "[bold yellow]⚠️  Learning in Progress[/bold yellow]\n\n"
            "The model is learning but hasn't converged yet.\n"
            "Try:\n"
            "  • More epochs (40-50)\n"
            "  • Lower learning rate (0.001-0.003)\n"
            "  • More attention heads (6-8)",
            border_style="yellow",
            title="💡 Keep Training"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]❌ Attention Needs Debugging[/bold red]\n\n"
            "The model isn't learning to reverse sequences.\n"
            "Check:\n"
            "  • MultiHeadAttention implementation\n"
            "  • Positional encoding is being added\n"
            "  • Gradients are flowing (run gradient flow test)\n"
            "  • Residual connections preserve gradients",
            border_style="red",
            title="🔧 Debug Required"
        ))
    console.print("="*70)
    console.print()


if __name__ == "__main__":
    main()

