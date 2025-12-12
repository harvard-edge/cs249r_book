#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🗣️ TINYTALKS: Your First Language Model                   ║
║              Watch YOUR Transformer Complete Simple Phrases                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

After proving attention works (sequence reversal), let's see YOUR transformer 
complete phrases - just like a tiny GPT!

🎯 THE TASK: Next Character Prediction
    Given: "hel"  → Predict: "l" (to form "hell")
    Given: "hell" → Predict: "o" (to form "hello")

This is exactly how GPT works - predict the next token!

✅ REQUIRED MODULES:
  Module 01-03: Tensor, Activations, Layers
  Module 06: Optimizers (Adam)
  Module 11: Embeddings
  Module 12: Attention  
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.getcwd())

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

console = Console()


def main():
    # ========================================================================
    # WELCOME
    # ========================================================================
    
    console.print(Panel(
        "[bold magenta]╔═══════════════════════════════╗[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]🗣️ TINYTALKS                  [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]   Phrase Completion Demo     [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta]                               [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] YOUR Transformer predicts    [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] the next character!          [bold magenta]║[/bold magenta]\n"
        "[bold magenta]╚═══════════════════════════════╝[/bold magenta]",
        border_style="bright_magenta"
    ))
    
    # ========================================================================
    # IMPORT YOUR IMPLEMENTATIONS
    # ========================================================================
    
    console.print("\n[bold cyan]📦 Loading YOUR TinyTorch...[/bold cyan]\n")
    
    try:
        from tinytorch import Tensor, Linear, ReLU, CrossEntropyLoss
        from tinytorch import LayerNorm, create_causal_mask
        from tinytorch.core.optimizers import Adam
        from tinytorch.text.embeddings import Embedding, PositionalEncoding
        from tinytorch.core.attention import MultiHeadAttention
        
        console.print("  [green]✓[/green] All YOUR implementations loaded!")
        
    except ImportError as e:
        console.print(f"[red]Import Error: {e}[/red]")
        return 1
    
    # ========================================================================
    # TRAINING DATA
    # ========================================================================
    
    console.print(Panel(
        "[bold cyan]📚 Training Data: Simple Words[/bold cyan]\n\n"
        "Teaching the model to complete:\n"
        "  [cyan]'ca'[/cyan]  → [green]'cat'[/green]\n"
        "  [cyan]'do'[/cyan]  → [green]'dog'[/green]\n"
        "  [cyan]'su'[/cyan]  → [green]'sun'[/green]\n"
        "  [cyan]'sta'[/cyan] → [green]'star'[/green]",
        border_style="cyan"
    ))
    
    # Training words - distinct patterns to avoid confusion
    words = ["cat", "dog", "red", "blue", "sun", "moon", "star"]
    
    # Build vocabulary
    all_chars = set()
    for word in words:
        all_chars.update(word)
    all_chars.add('_')  # Padding
    
    chars = sorted(list(all_chars))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    pad_idx = char_to_idx['_']
    
    console.print(f"  [green]✓[/green] Vocabulary: {vocab_size} characters\n")
    
    # ========================================================================
    # BUILD MODEL
    # ========================================================================
    
    console.print(Panel(
        "[bold cyan]🏗️ Building Model[/bold cyan]\n\n"
        "Using YOUR implementations:\n"
        "  • Embedding (Module 11)\n"
        "  • MultiHeadAttention (Module 12)\n"
        "  • Linear, LayerNorm (Modules 03, 13)",
        border_style="cyan"
    ))
    
    # Small but capable config
    embed_dim = 32
    num_heads = 2
    max_len = 12
    
    # Build components
    embedding = Embedding(vocab_size, embed_dim)
    pos_encoding = PositionalEncoding(max_len, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    all_params = (embedding.parameters() + attention.parameters() + 
                  ln.parameters() + output_proj.parameters())
    
    param_count = sum(p.data.size for p in all_params)
    console.print(f"  [green]✓[/green] Model: {param_count:,} parameters\n")
    
    # Using create_causal_mask from tinytorch.core.transformer (Module 13)
    
    def forward(tokens):
        """Forward pass with causal masking for autoregressive generation."""
        batch, seq_len = tokens.shape[0], tokens.data.shape[1]
        
        x = embedding(tokens)
        x = pos_encoding(x)
        
        # Create causal mask - each position can only see past + current
        mask = create_causal_mask(seq_len)
        attn_out = attention(x, mask)
        
        x = ln(x + attn_out)  # Residual connection
        
        # Reshape for output projection
        batch, seq, embed = x.shape
        x_2d = x.reshape(batch * seq, embed)
        logits_2d = output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, vocab_size)
        return logits
    
    # ========================================================================
    # PREPARE TRAINING DATA
    # ========================================================================
    
    def encode(text):
        """Convert text to indices."""
        return [char_to_idx.get(c, pad_idx) for c in text]
    
    def pad(seq, length):
        """Pad sequence to length."""
        return seq + [pad_idx] * (length - len(seq))
    
    # Create training examples: for each word, train to predict next char
    # Input: "hel__" Target at each position: "ello_"
    train_inputs = []
    train_targets = []
    
    for word in words:
        # Pad word
        word_padded = word + '_' * (max_len - len(word))
        
        # Input is word, target is shifted by 1
        inp = encode(word_padded[:max_len])
        tgt = encode(word_padded[1:max_len] + '_')
        
        train_inputs.append(inp)
        train_targets.append(tgt)
    
    X = Tensor(np.array(train_inputs))
    y = Tensor(np.array(train_targets))
    
    console.print(f"  [dim]Training examples: {len(words)} words[/dim]\n")
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    console.print(Panel(
        "[bold yellow]🏋️ Training: Next Character Prediction[/bold yellow]\n\n"
        "For 'star': s→t, t→a, a→r, r→_",
        border_style="yellow"
    ))
    
    optimizer = Adam(all_params, lr=0.03)
    loss_fn = CrossEntropyLoss()
    
    num_epochs = 300  # More training for better completion
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        transient=True
    ) as progress:
        task = progress.add_task("Training...", total=num_epochs)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(len(words)):
                # Get batch
                inp = Tensor(X.data[i:i+1])
                tgt = Tensor(y.data[i:i+1])
                
                # Forward
                logits = forward(inp)
                
                # Reshape for loss (batch*seq, vocab)
                batch, seq, vocab = logits.shape
                logits_2d = logits.reshape(batch * seq, vocab)
                target_1d = tgt.reshape(-1)
                
                # Compute loss over all positions
                loss = loss_fn(logits_2d, target_1d)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += float(loss.data)
            
            progress.advance(task)
    
    console.print(f"  [green]✓[/green] Training complete! (Loss: {total_loss/len(words):.4f})\n")
    
    # ========================================================================
    # GENERATION DEMO
    # ========================================================================
    
    console.print(Panel(
        "[bold green]🎉 PHRASE COMPLETION DEMO[/bold green]\n\n"
        "Watch YOUR transformer complete words!",
        border_style="green"
    ))
    
    def complete(prefix, max_chars=10):
        """Complete a word character by character."""
        text = prefix
        
        console.print(f"\n  [bold cyan]Start:[/bold cyan] [yellow]{prefix}[/yellow]", end="")
        
        for _ in range(max_chars):
            # Encode and pad
            inp = pad(encode(text), max_len)
            tokens = Tensor(np.array([inp]))
            
            # Forward
            logits = forward(tokens)
            
            # Get prediction for next position
            pos = len(text) - 1
            if pos >= max_len - 1:
                break
                
            next_logits = logits.data[0, pos, :]
            
            # Softmax + sample
            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / probs.sum()
            next_idx = np.argmax(probs)
            next_char = idx_to_char[next_idx]
            
            if next_char == '_':
                break
                
            console.print(f"[green]{next_char}[/green]", end="")
            text += next_char
            time.sleep(0.1)
        
        console.print()
        return text
    
    # Test completions
    test_prefixes = ["ca", "do", "re", "blu", "su", "sta"]
    
    for prefix in test_prefixes:
        complete(prefix)
        time.sleep(0.2)
    
    # ========================================================================
    # SUCCESS
    # ========================================================================
    
    console.print(Panel(
        "[bold green]🏆 TINYTALKS COMPLETE![/bold green]\n\n"
        "[green]YOUR transformer completed words![/green]\n\n"
        "[bold]How it works:[/bold]\n"
        "  1. [cyan]Embedding[/cyan]: Characters → Vectors\n"
        "  2. [cyan]Attention[/cyan]: Look at previous chars\n"
        "  3. [cyan]Predict[/cyan]: What comes next?\n"
        "  4. [cyan]Repeat[/cyan]: Generate char by char\n\n"
        "[dim]This is exactly how GPT works![/dim]\n\n"
        "[bold]🎓 You've built a language model![/bold]",
        title="🗣️ TinyTalks",
        border_style="bright_green",
        box=box.DOUBLE,
        padding=(1, 2)
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
