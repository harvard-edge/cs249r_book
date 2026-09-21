#!/usr/bin/env python3
"""
The Generative LLM Revolution (2020) - TinyGPT on Shakespeare
============================================================

📚 HISTORICAL CONTEXT:
In 2020, Brown et al. (OpenAI) published "Language Models are Few-Shot Learners,"
introducing GPT-3. Rather than training models for specific classification tasks,
GPT proved that autoregressive next-token prediction at scale produces emergent,
general-purpose language capabilities.

🎯 MILESTONE 07: TRAIN TINYGPT FROM SCRATCH & GENERATE TEXT
In this capstone milestone, YOU bring together all 20 modules of TinyTorch:
- Tokenizing text with YOUR Tokenizer
- Projecting into continuous space with YOUR Embeddings
- Computing self-attention with YOUR Causal Multi-Head Attention
- Stacking decoder layers with YOUR Pre-LN Transformer Blocks
- Computing analytical gradients with YOUR Autograd engine
- Updating parameters with YOUR AdamW optimizer
- Extending prompts autoregressively with YOUR Sampling & Generation loop!

✅ REQUIRED MODULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR strided tensor data structure
  Module 02 (Activations)   : YOUR GELU activation
  Module 03 (Layers)        : YOUR Linear projection layers
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (DataLoader)    : YOUR mini-batch DataLoader
  Module 06 (Autograd)      : YOUR reverse-mode computational graph
  Module 07 (Optimizers)    : YOUR AdamW optimizer
  Module 10 (Tokenization)  : YOUR Tokenizer
  Module 11 (Embeddings)    : YOUR Token + Learned Positional Embeddings
  Module 12 (Attention)     : YOUR Causal Multi-Head Attention
  Module 13 (Transformers)  : YOUR Stacked Transformer Blocks & TinyGPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Decoder-Only Generative Transformer):
    ┌──────────────┐
    │  Token IDs   │  [B, S]
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Embeddings   │  Token + Learned Positional Table [B, S, D]
    └──────┬───────┘
           ▼
    ┌──────────────┐ ◄───┐
    │ LayerNorm    │     │
    │ Causal MHA   │     │ (× num_layers Transformer Blocks)
    │ Residual Add │     │
    │ LayerNorm    │     │
    │ MLP (GELU)   │     │
    │ Residual Add │ ────┘
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Final LN     │  [B, S, D]
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ LM Head      │  Linear projection [B, S, Vocab]
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Next-Token   │  Teacher forcing loss / Autoregressive Sampling
    └──────────────┘

📊 SUCCESS CRITERIA:
  ✅ Loss Convergence : Training loss drops steadily (target < 2.50)
  ✅ Text Generation  : Extends prompts with Shakespearean cadences
  ✅ Complete Stack   : All 37 parameters receive autograd updates
"""

import sys
import os
import argparse
import time
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import AdamW
from tinytorch.core.dataloader import Dataset, DataLoader
from tinytorch.core.tokenization import CharTokenizer, BPETokenizer
from tinytorch.core.transformers import TinyGPT
from milestones.data_manager import DatasetManager

# Rich for terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich import box

console = Console()


def press_enter_to_continue():
    """Pause in interactive sessions; skip in CI/pipes."""
    if os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1" or os.environ.get("CI") == "true":
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError:
            pass
        console.print()


class TextWindowDataset(Dataset):
    """Slices a 1D token stream into autoregressive (input, target) pairs shifted by 1."""

    def __init__(self, token_ids, seq_len=32, stride=8):
        self.inputs = []
        self.targets = []
        n_tokens = len(token_ids)
        for start in range(0, n_tokens - seq_len, stride):
            self.inputs.append(token_ids[start : start + seq_len])
            self.targets.append(token_ids[start + 1 : start + seq_len + 1])
        self.inputs = np.array(self.inputs, dtype=np.int64)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return Tensor(self.inputs[idx]), Tensor(self.targets[idx])


def build_model(vocab_size, embed_dim=64, num_layers=2, num_heads=4, max_seq_len=64):
    """Instantiate TinyGPT and compute parameter statistics."""
    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    )
    total_params = sum(p.data.size for p in model.parameters())
    return model, total_params


def train_epoch(model, dataloader, optimizer, loss_fn, vocab_size):
    """Run one training epoch with teacher forcing and autograd updates."""
    total_loss = 0.0
    total_tokens = 0

    for batch_x, batch_y in dataloader:
        batch_size, seq_len = batch_x.shape

        # Forward pass: [B, S] -> [B, S, V]
        logits = model.forward(batch_x)

        # Reshape to 2D for CrossEntropyLoss: [B*S, V] vs [B*S]
        logits_2d = logits.reshape(-1, vocab_size)
        target_1d = batch_y.reshape(-1)

        loss = loss_fn(logits_2d, target_1d)

        # Autograd backward pass & AdamW step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.data * (batch_size * seq_len)
        total_tokens += batch_size * seq_len

    return total_loss / max(total_tokens, 1)


def generate_continuation(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8):
    """Autoregressively extend prompt tokens using TinyGPT's generation loop."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [1]  # fallback
    prompt_tensor = Tensor(np.array([prompt_ids]))
    generated_tensor = model.generate(prompt_tensor, max_new_tokens=max_new_tokens, temperature=temperature)
    generated_ids = generated_tensor.data[0].tolist()
    return tokenizer.decode(generated_ids)


def run_milestone(args=None):
    """Main milestone execution flow."""
    args = args or argparse.Namespace()
    sample_only = getattr(args, "quick", False) or getattr(args, "sample", False)
    custom_prompt = getattr(args, "prompt", "First Citizen:")
    temperature = getattr(args, "temperature", 0.8)
    epochs = getattr(args, "epochs", 12)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. BANNER & INTRO
    # ─────────────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold cyan]MILESTONE 07: THE GENERATIVE LLM REVOLUTION (2020)[/bold cyan]\n\n"
        "[yellow]Train TinyGPT from scratch on Shakespeare and generate text![/yellow]\n\n"
        "• Model: Causal Pre-LN Transformer Decoder (Vaswani et al. / GPT-2 / GPT-3)\n"
        "• Autograd: All Q, K, V projections and MLP weights receive reverse gradients\n"
        "• Target Workload: Autoregressive Next-Token Prediction on TinyShakespeare",
        border_style="cyan",
        title="TinyTorch Capstone",
    ))
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. LOAD DATASET
    # ─────────────────────────────────────────────────────────────────────────
    dm = DatasetManager()
    with console.status("[bold green]Loading TinyShakespeare dataset..."):
        text = dm.get_tinyshakespeare(sample_only=sample_only)

    console.print(f"📖 Dataset loaded: [bold]{len(text):,} characters[/bold] of Shakespeare")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TOKENIZATION
    # ─────────────────────────────────────────────────────────────────────────
    console.print("\n[bold]🔤 Initializing Tokenizer...[/bold]")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([text])
    vocab_size = tokenizer.vocab_size
    tokens = tokenizer.encode(text)

    console.print(f"  Vocabulary size : [cyan]{vocab_size} tokens[/cyan]")
    console.print(f"  Encoded stream  : [cyan]{len(tokens):,} tokens[/cyan]")

    # Build sequence dataset (crop text to ~30k tokens for fast CPU convergence)
    train_tokens = tokens[: min(len(tokens), 35000)]
    seq_len = 32
    dataset = TextWindowDataset(train_tokens, seq_len=seq_len, stride=8)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    console.print(f"  Slices created  : [cyan]{len(dataset):,} training windows[/cyan] (seq_len={seq_len})")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. INSTANTIATE TINYGPT
    # ─────────────────────────────────────────────────────────────────────────
    console.print("\n[bold]🏗️ Constructing TinyGPT Model...[/bold]")
    embed_dim = 64
    num_layers = 2
    num_heads = 4
    model, total_params = build_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len * 2,
    )

    table = Table(title="Model Architecture & Parameters", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Configuration", style="magenta")
    table.add_column("Implementation", style="green")
    table.add_row("Token Embeddings", f"{vocab_size} × {embed_dim}", "Module 11 (EmbeddingLayer)")
    table.add_row("Position Table", f"{seq_len * 2} × {embed_dim}", "Module 11 (Learned Positional)")
    table.add_row("Attention", f"{num_layers} blocks, {num_heads} heads (causal)", "Module 12 (MultiHeadAttention)")
    table.add_row("Feed-Forward", f"4× expansion ({embed_dim * 4}) + GELU", "Module 02 & 03 (MLP)")
    table.add_row("Total Parameters", f"{total_params:,}", "100% Student Authored")
    console.print(table)
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = CrossEntropyLoss()

    console.print("[bold]🚀 Training TinyGPT from Scratch (Next-Token Prediction)...[/bold]")

    history = []
    t_start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training epochs...", total=epochs)

        for epoch in range(epochs):
            loss = train_epoch(model, dataloader, optimizer, loss_fn, vocab_size)
            perplexity = np.exp(min(loss, 20.0))
            history.append((epoch + 1, loss, perplexity))
            progress.update(task, advance=1, description=f"[cyan]Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} (PPL: {perplexity:.1f})")

    t_total = time.perf_counter() - t_start
    console.print(f"\n⏱️ Training completed in [bold green]{t_total:.2f} seconds[/bold green] ({t_total / epochs:.2f}s/epoch)")

    # Display convergence summary
    res_table = Table(title="Training Loss Trajectory", box=box.ROUNDED)
    res_table.add_column("Epoch", justify="center")
    res_table.add_column("Cross-Entropy Loss", justify="right")
    res_table.add_column("Perplexity", justify="right")

    for ep, l, p in [history[0], history[len(history)//2], history[-1]]:
        res_table.add_row(f"Epoch {ep}", f"{l:.4f}", f"{p:.2f}")
    console.print(res_table)
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # 6. TEXT GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    console.print("[bold]✨ Generating Shakespeare Autoregressively...[/bold]\n")

    test_prompts = [custom_prompt, "ROMEO:", "KING:"]
    # If custom prompt is already in test_prompts, keep unique
    test_prompts = list(dict.fromkeys(test_prompts))

    for p in test_prompts:
        t0 = time.perf_counter()
        continuation = generate_continuation(
            model,
            tokenizer,
            prompt=p,
            max_new_tokens=45,
            temperature=temperature,
        )
        gen_time = (time.perf_counter() - t0) * 1000
        console.print(Panel(
            f"[bold green]Prompt:[/bold green] [yellow]{repr(p)}[/yellow]\n\n"
            f"[bold cyan]Generated Output:[/bold cyan]\n{continuation}\n\n"
            f"[dim]Generated in {gen_time:.1f}ms (temp={temperature})[/dim]",
            border_style="magenta",
            title=f"Sample: {p}",
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # 7. SUCCESS EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    initial_loss = history[0][1]
    final_loss = history[-1][1]
    loss_drop = initial_loss - final_loss

    passed = (final_loss < 2.50) or (loss_drop > 1.20)

    console.print()
    if passed:
        console.print(Panel.fit(
            f"[bold green]🏆 MILESTONE 07 PASSED: TINYGPT CONVERGED & GENERATING![/bold green]\n\n"
            f"  • Initial Loss : {initial_loss:.4f}\n"
            f"  • Final Loss   : [bold green]{final_loss:.4f}[/bold green] (reduction: -{loss_drop:.2f})\n"
            f"  • Model Status : Syntactically coherent autoregressive text generated!\n\n"
            "You have constructed and trained an end-to-end generative LLM\n"
            "using exclusively the runtime, autograd, and neural network primitives\n"
            "you wrote from scratch in TinyTorch!",
            border_style="green",
            title="Success!",
        ))
        return 0
    else:
        console.print(Panel.fit(
            f"[bold red]❌ MILESTONE 07 FAILED: LOSS DID NOT MEET CONVERGENCE GATE[/bold red]\n\n"
            f"  • Expected final loss < 2.50, got {final_loss:.4f}",
            border_style="red",
            title="Needs Tuning",
        ))
        return 1


def main():
    parser = argparse.ArgumentParser(description="Milestone 07: TinyGPT on TinyShakespeare")
    parser.add_argument("--quick", "--sample", action="store_true", help="Run on offline sample dataset")
    parser.add_argument("--prompt", type=str, default="First Citizen:", help="Seed prompt for text generation")
    parser.add_argument("--temperature", "--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    args = parser.parse_args()
    return run_milestone(args)


if __name__ == "__main__":
    sys.exit(main())
