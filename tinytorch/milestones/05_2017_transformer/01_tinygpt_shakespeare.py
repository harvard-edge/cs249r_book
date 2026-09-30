#!/usr/bin/env python3
"""
The Transformer Era (2017-2022) - TinyGPT & The Foundation of ChatGPT
=============================================================================

📚 HISTORICAL CONTEXT:
In 2017, Vaswani et al. published "Attention Is All You Need," introducing the
Transformer. In 2020, Brown et al. (OpenAI) published "Language Models are Few-Shot Learners,"
introducing GPT-3 and proving that autoregressive next-token prediction at scale
produces emergent, general-purpose language reasoning. In late 2022, OpenAI launched
ChatGPT, bringing this exact generative pretraining foundation into everyday life.

Behind ChatGPT sits this exact mathematical and systems engine:
1. Autoregressive Next-Token Prediction: Teacher forcing with CrossEntropyLoss.
2. Causal Self-Attention: Masked attention preventing future token leakage.
3. Pre-LayerNorm Residual Highway: Clean gradient propagation through deep blocks.
4. Systems Serving Efficiency: The KV-cache (Module 18) and quantization (Module 15)
   that make generative sampling fast and interactive in production.

🎯 MILESTONE 05: TRAIN TINYGPT FROM SCRATCH & GENERATE TEXT
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
  Module 08 (Training)      : YOUR Trainer training loop
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
  ✅ Loss Convergence : Final training loss < 1.20 (a model without working
                        attention stalls near 2.37, the bigram level)
  ✅ Generalization   : Best loss on held-out Shakespeare sits at least 0.35
                        below a counted bigram model of the training text
  ✅ Causality        : Changing future tokens never changes past predictions
  ✅ Honest Loss      : YOUR cross-entropy matches a NumPy reference, and every
                        gated loss is measured in NumPy from the model's logits
  ✅ Positions        : A repeated token gives different logits at different
                        positions (YOUR positional encoding is doing its job)
  ✅ Text Generation  : Samples continuations from the trained model (printed
                        for you to read; text quality is not scored)
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
# Shared Milestone 05 checks live next to this script (transformer_gates.py)
milestone_dir = str(Path(__file__).resolve().parent)
if milestone_dir not in sys.path:
    sys.path.insert(0, milestone_dir)

from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import AdamW
from tinytorch.core.dataloader import TensorDataset, DataLoader
from tinytorch.core.training import Trainer
from tinytorch.core.tokenization import CharTokenizer, BPETokenizer
from tinytorch.core.embeddings import EmbeddingLayer
from tinytorch.core.layers import Linear
from tinytorch.core.transformers import (
    LayerNorm,
    TransformerBlock,
    create_causal_mask,
    generate,
)
from milestones.data_manager import DatasetManager
from transformer_gates import (
    attention_content_failure_message,
    attention_content_probe,
    attention_content_report,
    bigram_cross_entropy,
    causality_failure_message,
    causality_probe,
    causality_report,
    cross_entropy_check,
    cross_entropy_failure_message,
    measured_loss,
    position_failure_message,
    position_probe,
)

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


# =============================================================================
# 🎓 ZONE 1: STUDENT LEGO BRICKS (TinyGPT Architecture)
# =============================================================================

class TinyGPT:
    """
    Complete Decoder-Only Generative Pretrained Transformer (TinyGPT).

    Assembled entirely from TinyTorch LEGO bricks:
      1. Token + Learned Positional Embeddings: Module 11 (EmbeddingLayer)
      2. Causal Multi-Head Self-Attention: Module 12 (MultiHeadAttention inside TransformerBlock)
      3. Feed-Forward Expansion (4x) with GELU: Module 02 & 03 (MLP inside TransformerBlock)
      4. Deep Pre-LayerNorm Residual Highway: Module 13 (TransformerBlock)
      5. Final Layer Normalization: Module 13 (LayerNorm)
      6. Un-embedding Vocabulary Projection: Module 03 (Linear)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 64,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token + Positional Embeddings (Module 11)
        self.embedding_layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)

        # Stack of Pre-LN Transformer Blocks (Module 13)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ]

        # Final Layer Normalization (Module 13)
        self.ln_f = LayerNorm(embed_dim)

        # LM Head (Module 03): projects hidden state back to vocabulary logits
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        """Forward pass: [B, S] -> [B, S, V]."""
        batch_size, seq_len = tokens.shape

        # Token + positional embeddings
        x = self.embedding_layer.forward(tokens, start_pos)

        # Causal autoregressive mask
        mask = create_causal_mask(seq_len)

        # Stacked transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final LayerNorm & LM Head projection
        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)

        return logits

    def __call__(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        return self.forward(tokens, start_pos)

    def parameters(self) -> list:
        """Return all learnable parameters across all subsystems."""
        params = []
        params.extend(self.embedding_layer.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())
        return params


# =============================================================================
# 📊 ZONE 2: MILESTONE HARNESS & TRAINING UX
# =============================================================================

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


def train_epoch(model, dataloader, criterion, optimizer, vocab_size=None):
    """Train TinyGPT for one epoch of next-token prediction with flattened sequence loss."""
    total_loss = 0.0
    total_tokens = 0
    v_size = vocab_size if vocab_size is not None else getattr(model, "vocab_size", None)

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        logits = model(inputs)
        cur_vocab = v_size if v_size is not None else logits.shape[-1]
        logits_flat = logits.reshape(-1, cur_vocab)
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        batch_tokens = targets.data.size
        total_loss += float(loss.data) * batch_tokens
        total_tokens += batch_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def generate_continuation(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8):
    """Autoregressively extend prompt tokens using TinyGPT's generation loop."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [1]  # fallback
    prompt_tensor = Tensor(np.array([prompt_ids]))
    generated_tensor = generate(model, prompt_tensor, max_new_tokens=max_new_tokens, temperature=temperature)
    generated_ids = generated_tensor.data[0].tolist()
    return tokenizer.decode(generated_ids)


def convergence_report(initial_loss: float, final_loss: float, vocab_size: int,
                       heldout_loss: float = None, bigram_loss: float = None,
                       heldout_epoch: int = None) -> list:
    """
    Success-panel lines built only from measured values.

    The gate checks loss, not text quality, so the report states the loss,
    the perplexity, and what uniform guessing would score, and leaves the
    judgment of the samples to the reader.
    """
    # 2026-09-28: the panel once said "Syntactically coherent autoregressive
    # text generated!" on every pass while the samples were gibberish.
    # 2026-09-29: "Initial Loss" was the epoch-1 average, taken while the model
    # was already learning; it is now measured before the first update.
    perplexity = float(np.exp(min(final_loss, 20.0)))
    lines = [
        f"  • Loss before training : {initial_loss:.4f}",
        f"  • Final training loss  : [bold green]{final_loss:.4f}[/bold green] "
        f"(reduction: -{initial_loss - final_loss:.2f})",
        f"  • Perplexity   : {perplexity:.2f} on the training text (uniform guessing "
        f"over {vocab_size} characters scores {vocab_size})",
    ]
    if heldout_loss is not None and bigram_loss is not None:
        where = f" (epoch {heldout_epoch})" if heldout_epoch is not None else ""
        lines.append(
            f"  • Held-out loss: {heldout_loss:.4f}{where} on text never trained on\n"
            f"                   (a counted bigram model scores {bigram_loss:.4f})"
        )
    lines.append(
        "  • Samples      : printed above. Low loss does not guarantee readable\n"
        "                   text. Nothing here scores coherence, so judge the\n"
        "                   samples yourself."
    )
    return lines


# Gate calibration, 2026-09-29 (default 12 epochs; "full" = 1.1M-char
# TinyShakespeare, "sample" = the 21k-char offline file used when nothing is
# downloaded). Numbers are final training loss / best held-out loss / margin
# below the counted bigram (2.48 full, 2.49 sample):
#   correct code     full 0.923-0.936 / 1.947-1.958 / 0.52-0.53   (3 runs)
#                    sample 0.645-0.659 / 1.978-2.009 / 0.48-0.51 (2 runs)
#   uniform attention (scores all zero, no content routing)
#                    full 1.467-1.477 / margin 0.30-0.31; sample 1.307-1.312 / 0.27-0.28
#   attention zeroed full 2.373 / margin 0.04; sample 2.356 / margin 0.05
#   no causal mask   0.06 training loss, caught by the causality probe
#   no sqrt(d_k) scale  sample 0.827 / margin 0.46 (passes: a mild bug)
# The old gate (loss < 2.50 or drop > 1.20) passed all of these. A loss
# target of 1.50 would still pass uniform attention, so it is 1.20.
TRAIN_LOSS_TARGET = 1.20
HELDOUT_MARGIN = 0.35   # nats the best held-out loss must sit below the bigram
TRAIN_TOKEN_LIMIT = 35000
HELDOUT_TOKENS = 4000


def run_milestone(args=None):
    """Main milestone execution flow."""
    args = args or argparse.Namespace()
    sample_only = getattr(args, "quick", False) or getattr(args, "sample", False)
    custom_prompt = getattr(args, "prompt", "First Citizen:")
    temperature = getattr(args, "temperature", 0.6)
    epochs = getattr(args, "epochs", 12)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. BANNER & INTRO
    # ─────────────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold cyan]MILESTONE 05: THE TRANSFORMER ERA (2017-2022)[/bold cyan]\n\n"
        "[yellow]Train TinyGPT from scratch on Shakespeare: the foundation of ChatGPT![/yellow]\n\n"
        "• Model: Causal Pre-LN Transformer Decoder (Vaswani et al. / GPT-2 / GPT-3)\n"
        "• Autograd: All Q, K, V projections and MLP weights receive reverse gradients\n"
        "• Systems: Parallel teacher-forcing training; foundation for KV-cache serving\n"
        "• Target Workload: Autoregressive Next-Token Prediction on TinyShakespeare",
        border_style="cyan",
        title="TinyTorch Architecture Tier",
    ))
    press_enter_to_continue()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. LOAD DATASET
    # ─────────────────────────────────────────────────────────────────────────
    data_path = getattr(args, "data_path", None)
    if data_path:
        custom_file = Path(data_path)
        if not custom_file.exists():
            raise FileNotFoundError(f"Custom text file not found at {data_path}")
        with open(custom_file, "r", encoding="utf-8") as f:
            text = f.read()
        dataset_name = custom_file.name
    else:
        dm = DatasetManager()
        with console.status("[bold green]Loading TinyShakespeare dataset..."):
            text = dm.get_tinyshakespeare(sample_only=sample_only)
        dataset_name = "Shakespeare"

    console.print(f"📖 Dataset loaded: [bold]{len(text):,} characters[/bold] of {dataset_name}")

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

    # Build sequence dataset (crop text to ~35k tokens for fast CPU convergence).
    # The next HELDOUT_TOKENS characters are never trained on; they measure
    # whether the model learned Shakespeare or memorized its crop.
    heldout_size = min(HELDOUT_TOKENS, len(tokens) // 10)
    train_limit = min(TRAIN_TOKEN_LIMIT, len(tokens) - heldout_size)
    train_tokens = tokens[:train_limit]
    heldout_tokens = tokens[train_limit:train_limit + heldout_size]
    seq_len = 32
    stride = 4

    inputs = []
    targets = []
    for start in range(0, len(train_tokens) - seq_len, stride):
        inputs.append(train_tokens[start : start + seq_len])
        targets.append(train_tokens[start + 1 : start + seq_len + 1])
    x_tensor = Tensor(np.array(inputs, dtype=np.int32))
    y_tensor = Tensor(np.array(targets, dtype=np.int32))
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    console.print(f"  Slices created  : [cyan]{len(dataset):,} training windows[/cyan] (seq_len={seq_len})")

    # Held-out windows do not overlap, so every held-out character is scored once.
    heldout_x = np.array([heldout_tokens[s:s + seq_len]
                          for s in range(0, len(heldout_tokens) - seq_len, seq_len)], dtype=np.int32)
    heldout_y = np.array([heldout_tokens[s + 1:s + seq_len + 1]
                          for s in range(0, len(heldout_tokens) - seq_len, seq_len)], dtype=np.int32)
    console.print(f"  Held-out text   : [cyan]{len(heldout_tokens):,} tokens[/cyan] "
                  "(never trained on)")

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
    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Two checks that need no training, so a broken module fails in seconds:
    # YOUR cross-entropy forward must report the true loss of the logits (the
    # gates below measure loss in NumPy and would otherwise never notice), and
    # YOUR positional encoding must make positions distinguishable.
    train_x = np.array(inputs, dtype=np.int32)
    train_y = np.array(targets, dtype=np.int32)
    ce_check = cross_entropy_check(model, loss_fn, train_x[:32], train_y[:32], vocab_size)
    if not ce_check.passed:
        console.print(Panel.fit(cross_entropy_failure_message(ce_check),
                                border_style="red", title="Loss Check Failed"))
        return 1
    positions = position_probe(model, int(train_x[0, 0]), seq_len)
    if not positions.passed:
        console.print(Panel.fit(position_failure_message(positions),
                                border_style="red", title="Position Check Failed"))
        return 1

    # Loss of the untrained model, measured before the first update. Uniform
    # guessing over the vocabulary scores ln(vocab_size).
    loss_before = measured_loss(model, heldout_x, heldout_y)
    console.print(f"  Loss before training: [cyan]{loss_before:.4f}[/cyan] on held-out text "
                  f"(uniform guessing: {np.log(vocab_size):.4f})\n")

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
            loss = trainer.train_epoch(dataloader)
            perplexity = np.exp(min(loss, 20.0))
            heldout = measured_loss(model, heldout_x, heldout_y)
            history.append((epoch + 1, loss, perplexity, heldout))
            progress.update(task, advance=1, description=(
                f"[cyan]Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} (PPL: {perplexity:.1f}) "
                f"| Held-out: {heldout:.4f}"))

    t_total = time.perf_counter() - t_start
    console.print(f"\n⏱️ Training completed in [bold green]{t_total:.2f} seconds[/bold green] ({t_total / epochs:.2f}s/epoch)")

    # Display convergence summary
    res_table = Table(title="Training Loss Trajectory", box=box.ROUNDED)
    res_table.add_column("Epoch", justify="center")
    res_table.add_column("Epoch Loss (your Trainer)", justify="right")
    res_table.add_column("Perplexity", justify="right")
    res_table.add_column("Held-out Loss", justify="right")

    best_row = min(history, key=lambda row: row[3])
    shown = sorted({0, len(history) // 2, len(history) - 1, history.index(best_row)})
    for ep, l, p, h in (history[i] for i in shown):
        note = " (lowest)" if ep == best_row[0] else ""
        res_table.add_row(f"Epoch {ep}", f"{l:.4f}", f"{p:.2f}", f"{h:.4f}{note}")
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
    # Training loss is re-measured on every training window after training,
    # in NumPy (2026-09-29): the epoch average from Trainer.train_epoch and the
    # value of YOUR cross-entropy forward are student code, and gating on them
    # passed a loss that returned 0. Measured this way (sample text, 3 runs):
    # correct code 0.565-0.593 (its last epoch average was ~0.66, since weights
    # improve during an epoch); uniform attention 1.258, which also fails the
    # held-out margin. Full-text values were not re-measured.
    final_loss = measured_loss(model, train_x, train_y)
    # Best held-out loss over the epochs (the early-stopping point). By the last
    # epoch the model has memorized its 35k-character crop and held-out loss
    # climbs back up (2.70 on 2026-09-29), so the final epoch is the wrong test
    # of whether attention learned anything that transfers.
    best_epoch, heldout_loss = best_row[0], best_row[3]
    bigram_loss = bigram_cross_entropy(train_tokens, heldout_tokens, vocab_size)
    causality = causality_probe(model, heldout_x[0], vocab_size)
    attention = attention_content_probe(model, heldout_x[0])

    loss_ok = final_loss < TRAIN_LOSS_TARGET
    heldout_ok = heldout_loss < bigram_loss - HELDOUT_MARGIN
    passed = loss_ok and heldout_ok and causality.passed and attention.passed

    console.print()
    if not causality.passed:
        # A future-peeking model scores a tiny loss, so say why before anything else.
        console.print(Panel.fit(causality_failure_message(causality),
                                border_style="red", title="Causality Check Failed"))
        return 1
    if not attention.passed:
        # Uniform attention still beats a bigram model, so name it directly.
        console.print(Panel.fit(attention_content_failure_message(attention),
                                border_style="red", title="Attention Check Failed"))
        return 1

    if passed:
        console.print(Panel.fit(
            "[bold green]🏆 MILESTONE ACHIEVED: TINYGPT LEARNED FROM CONTEXT[/bold green]\n\n"
            + "\n".join(convergence_report(loss_before, final_loss, vocab_size,
                                           heldout_loss=heldout_loss,
                                           bigram_loss=bigram_loss,
                                           heldout_epoch=best_epoch))
            + "\n  • " + causality_report(causality)
            + "\n  • " + attention_content_report(attention) + "\n\n"
            "You have constructed and trained the core generative engine of ChatGPT\n"
            "using exclusively the runtime, autograd, and neural network primitives\n"
            "you wrote from scratch in TinyTorch!\n\n"
            "[dim]In Modules 14-19, you'll optimize this model with KV caching (Module 18),\n"
            "INT8 quantization (Module 15), and hardware acceleration (Module 17).[/dim]",
            border_style="green",
            title="Success!",
        ))
        return 0
    else:
        mark = lambda ok: "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(Panel.fit(
            f"[bold red]❌ MILESTONE 05 FAILED: TINYGPT DID NOT LEARN FROM CONTEXT[/bold red]\n\n"
            f"  {mark(loss_ok)} Training loss {final_loss:.4f}, measured after training "
            f"(needed < {TRAIN_LOSS_TARGET:.2f})\n"
            f"  {mark(heldout_ok)} Best held-out loss {heldout_loss:.4f} (epoch {best_epoch}) "
            f"vs counted bigram {bigram_loss:.4f} (needed at least "
            f"{HELDOUT_MARGIN:.2f} below it)\n\n"
            "  A model that only sees the current character plateaus at the bigram\n"
            "  loss (about 2.4 on this text), and attention that averages every\n"
            "  earlier character equally stalls near 1.4. Getting well below both\n"
            "  takes attention that picks WHICH earlier characters matter: check the\n"
            "  Q @ K^T scores, the softmax over keys, the Q/K/V projections, and that\n"
            "  the optimizer updates every parameter.",
            border_style="red",
            title="Needs Tuning",
        ))
        return 1


def main():
    parser = argparse.ArgumentParser(description="Milestone 05: TinyGPT on TinyShakespeare")
    parser.add_argument("--quick", "--sample", action="store_true", help="Run on offline sample dataset")
    parser.add_argument("--data-path", type=str, default=None, help="Path to custom text file to train on")
    parser.add_argument("--prompt", type=str, default="First Citizen:", help="Seed prompt for text generation")
    parser.add_argument("--temperature", "--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    args = parser.parse_args()
    return run_milestone(args)


if __name__ == "__main__":
    sys.exit(main())
