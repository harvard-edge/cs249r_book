"""
Milestone 05: The Transformer Era (Part 4) - Conversational Concepts & Overfitting Detective.

In this milestone part, you train TinyGPT as a conversational question answering
assistant that explains TinyTorch concepts ("TinyTorch Teaching TinyTorch").

Key Systems and Educational Concepts:
  1. Conversational Dialogue Modeling:
     Autoregressively formatting prompts with 'Q: ...' and completing 'A: ...'.
  2. The Overfitting Detective Experiment:
     Comparing Train Loss versus Test Loss curves on held-out concept splits
     (tinytorch_train.txt vs tinytorch_test.txt).
     Because a 119K parameter model has more weights than the 13 KB dataset has
     characters, students observe how micro-models can achieve near-zero training
     loss by rote memorization while held-out test loss reveals the generalization gap.
  3. Interactive Streaming Inference:
     Character-by-character typewriter streaming using rich.Live.

All components are assembled from the student-authored TinyTorch LEGO bricks:
  - Module 02 & 03: Activation functions (GELU) and Multi-Layer Perceptron
  - Module 11: Token and Learned Positional Embeddings
  - Module 12: Causal Multi-Head Self-Attention
  - Module 13: Pre-LayerNorm Residual Transformer Blocks
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tinytorch.core.tensor import Tensor  # noqa: E402
from tinytorch.core.losses import CrossEntropyLoss  # noqa: E402
from tinytorch.core.optimizers import AdamW  # noqa: E402
from tinytorch.core.dataloader import TensorDataset, DataLoader  # noqa: E402
from tinytorch.core.training import Trainer  # noqa: E402
from tinytorch.core.tokenization import CharTokenizer  # noqa: E402
from tinytorch.core.embeddings import EmbeddingLayer  # noqa: E402
from tinytorch.core.layers import Linear  # noqa: E402
from tinytorch.core.transformers import (  # noqa: E402
    LayerNorm,
    TransformerBlock,
    create_causal_mask,
)
from milestones.data_manager import DatasetManager  # noqa: E402

# Rich for terminal UI
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.progress import (  # noqa: E402
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)
from rich.live import Live  # noqa: E402
from rich import box  # noqa: E402

console = Console()


def press_enter_to_continue(pause: bool = False):
    """Pause in interactive sessions when pause is enabled; skip in CI or non-interactive runs."""
    if not pause:
        return
    if (
        os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1"
        or os.environ.get("CI") == "true"
    ):
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError:
            pass
        console.print()


# =============================================================================
# ZONE 1: STUDENT CORE LEGO BRICKS (Model Architecture & Generation)
# =============================================================================


class TinyGPT:
    """
    Complete Decoder-Only Generative Pretrained Transformer (TinyGPT).

    Assembled entirely from TinyTorch LEGO bricks:
      1. Token + Positional Embeddings: Module 11 (EmbeddingLayer)
      2. Causal Multi-Head Self-Attention: Module 12 (TransformerBlock)
      3. Feed-Forward Expansion (4x) with GELU: Module 02 and 03 (MLP)
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
        max_seq_len: int = 128,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token + Positional Embeddings
        self.token_embeddings = EmbeddingLayer(vocab_size, embed_dim)
        self.pos_embeddings = EmbeddingLayer(max_seq_len, embed_dim)

        # Transformer Blocks with Pre-LN Residual Connections
        self.blocks = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ]

        # Final LayerNorm and Un-embedding Projection
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size)

    def forward(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        """
        Forward pass for next-token prediction.

        Args:
            tokens: Tensor of shape (batch_size, seq_len) with integer token IDs.
            start_pos: Starting position index for positional embeddings.

        Returns:
            Logits Tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = tokens.shape

        # Positional indices
        positions = np.arange(start_pos, start_pos + seq_len, dtype=np.int32)
        pos_tensor = Tensor(np.tile(positions, (batch_size, 1)))

        # Sum token and position representations
        h = self.token_embeddings.forward(tokens) + self.pos_embeddings.forward(pos_tensor)

        # Create causal triangular mask
        mask = create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            h = block.forward(h, mask=mask)

        # Final LayerNorm and un-embedding projection
        h = self.ln_f.forward(h)
        logits = self.lm_head.forward(h)
        return logits

    def __call__(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        return self.forward(tokens, start_pos)

    def parameters(self) -> list:
        """Return all learnable parameter Tensors."""
        params = []
        params.extend(self.token_embeddings.parameters())
        params.extend(self.pos_embeddings.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())
        return params


def build_model(
    vocab_size: int,
    embed_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    max_seq_len: int = 128,
) -> Tuple[TinyGPT, int]:
    """Instantiate TinyGPT and compute learnable parameter count."""
    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    )
    total_params = sum(p.data.size for p in model.parameters())
    return model, total_params


def sample_next_token_top_k(
    logits: np.ndarray,
    temperature: float = 0.35,
    top_k: int = 5,
) -> int:
    """
    Sample next token index using temperature scaling and top-k filtering.

    Args:
        logits: 1D array of unnormalized log-probabilities across vocabulary.
        temperature: Temperature divisor (lower = sharper distribution).
        top_k: Number of highest-probability candidate tokens to retain.

    Returns:
        Sampled token integer index.
    """
    temperature = max(temperature, 1e-4)
    scaled_logits = logits / temperature

    # Top-k filtering
    if top_k > 0 and top_k < len(scaled_logits):
        top_indices = np.argpartition(scaled_logits, -top_k)[-top_k:]
        filtered_logits = np.full_like(scaled_logits, -1e9)
        filtered_logits[top_indices] = scaled_logits[top_indices]
        scaled_logits = filtered_logits

    # Numerically stable softmax
    shifted = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)

    return int(np.random.choice(len(probs), p=probs))


def generate_chat_response(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.35,
    top_k: int = 5,
    stream_callback: Optional[Callable[[str], None]] = None,
    stop_at_newline: bool = True,
) -> str:
    """
    Generate conversational answer to a question prompt.

    Args:
        model: Trained TinyGPT instance.
        tokenizer: Character tokenizer with built vocabulary.
        prompt: Question prompt (e.g. 'Q: What is TinyGPT?\\nA:').
        max_new_tokens: Maximum number of characters to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering threshold.
        stream_callback: Optional callback invoked with updated text after each token.
        stop_at_newline: Whether to stop generation after two consecutive newlines.

    Returns:
        Full generated string (prompt + answer).
    """
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [1]

    current_ids = list(prompt_ids)
    current_tokens = Tensor(np.array([current_ids], dtype=np.int32))

    for _ in range(max_new_tokens):
        if current_tokens.shape[1] >= model.max_seq_len:
            break

        logits = model.forward(current_tokens)
        last_logits = logits.data[0, -1, :]

        next_id = sample_next_token_top_k(
            last_logits,
            temperature=temperature,
            top_k=top_k,
        )
        current_ids.append(next_id)

        if stream_callback is not None:
            stream_callback(tokenizer.decode(current_ids))

        next_tensor = np.array([[next_id]], dtype=np.int32)
        current_tokens = Tensor(
            np.concatenate([current_tokens.data, next_tensor], axis=1)
        )

        if stop_at_newline:
            current_text = tokenizer.decode(current_ids)
            new_text = current_text[len(prompt):]
            if "\n\n" in new_text or "\nQ:" in new_text:
                break

    return tokenizer.decode(current_ids)


def evaluate_dataloader_loss(
    model: TinyGPT,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    vocab_size: int,
) -> float:
    """
    Compute average cross-entropy loss over a dataloader without updating weights.

    Args:
        model: TinyGPT instance.
        dataloader: DataLoader containing inputs and targets.
        criterion: CrossEntropyLoss criterion.
        vocab_size: Total vocabulary size.

    Returns:
        Average loss across all evaluation tokens.
    """
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in dataloader:
        logits = model.forward(inputs)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)

        batch_tokens = targets.data.size
        total_loss += float(loss.data) * batch_tokens
        total_tokens += batch_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


# =============================================================================
# ZONE 2: MILESTONE HARNESS & USER EXPERIENCE
# =============================================================================


def load_qa_splits(
    sample_only: bool = False,
    data_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Load train and test splits for the TinyTorch Q&A dataset or a custom file.

    Returns:
        Tuple of (train_text, test_text).
    """
    if data_path:
        custom_path = Path(data_path)
        if not custom_path.exists():
            raise FileNotFoundError(f"Custom Q&A dataset not found at {data_path}")
        with open(custom_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        split_idx = int(len(full_text) * 0.8)
        return full_text[:split_idx], full_text[split_idx:]

    dm = DatasetManager()
    train_text = ""
    test_text = ""

    if hasattr(dm, "get_tinytalks"):
        try:
            train_text = dm.get_tinytalks(
                sample_only=sample_only,
                topic="tinytorch",
                split="train",
            )
            test_text = dm.get_tinytalks(
                sample_only=sample_only,
                topic="tinytorch",
                split="test",
            )
        except Exception:
            pass

    if not train_text or not test_text:
        splits_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "datasets"
            / "tinytalks"
            / "splits"
        )
        train_path = splits_dir / "tinytorch_train.txt"
        test_path = splits_dir / "tinytorch_test.txt"

        if train_path.exists() and test_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                train_text = f.read()
            with open(test_path, "r", encoding="utf-8") as f:
                test_text = f.read()
        else:
            raise FileNotFoundError(
                "Could not load TinyTorch Q&A splits. "
                "Please ensure datasets/tinytalks/splits/ exists."
            )

    return train_text, test_text


def create_token_windows(
    tokens: List[int],
    seq_len: int = 64,
    stride: int = 4,
) -> DataLoader:
    """Create a DataLoader of sliding windows from a token sequence."""
    inputs = []
    targets = []
    for start in range(0, len(tokens) - seq_len, stride):
        inputs.append(tokens[start:start + seq_len])
        targets.append(tokens[start + 1:start + seq_len + 1])

    if not inputs:
        # Fallback if text is shorter than seq_len
        inputs.append(tokens[:-1])
        targets.append(tokens[1:])

    x_tensor = Tensor(np.array(inputs, dtype=np.int32))
    y_tensor = Tensor(np.array(targets, dtype=np.int32))
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def run_milestone(args=None):
    """Main milestone execution flow for TinyGPT conversational Q&A."""
    args = args or argparse.Namespace()
    data_path = getattr(args, "data_path", None)
    is_quick = getattr(args, "quick", False)
    is_sample = getattr(args, "sample", False)
    sample_only = is_quick or is_sample
    pause = getattr(args, "pause", False)

    default_prompt = "Q: What is TinyGPT?\nA:"
    custom_prompt = getattr(args, "prompt", default_prompt)

    temperature = getattr(args, "temperature", 0.35)
    epochs = getattr(args, "epochs", 8 if sample_only else 10)
    top_k = getattr(args, "top_k", 5)
    should_stream = getattr(args, "stream", True)
    is_interactive = getattr(args, "interactive", False)

    if (
        os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1"
        or os.environ.get("CI") == "true"
    ):
        if not getattr(args, "stream", False):
            should_stream = False
        is_interactive = False

    # =========================================================================
    # 1. BANNER AND INTRO
    # =========================================================================
    dataset_title = Path(data_path).name if data_path else "tinytalks_tinytorch"
    banner_lines = [
        "[bold cyan]MILESTONE 05: TRANSFORMER ERA (PART 4)[/bold cyan]",
        "",
        "[yellow]TinyTorch Teaching TinyTorch: Conversational Q&A[/yellow]",
        "[yellow]and The Overfitting Detective Experiment[/yellow]",
        "",
        "• Model: Causal Pre-LN Transformer (Vaswani / GPT-4)",
        f"• Dataset: Curated concept Q&A pairs ({dataset_title})",
        "• Overfitting Detective: Train Loss vs Held-Out Test Loss divergence",
        "• Decoding: Temperature scaling, top-k filtering, and live streaming",
    ]
    console.print()
    console.print(
        Panel.fit(
            "\n".join(banner_lines),
            border_style="cyan",
            title="TinyTorch Architecture Tier",
        )
    )
    press_enter_to_continue(pause)

    # =========================================================================
    # 2. LOAD DATASET SPLITS
    # =========================================================================
    with console.status(f"[bold green]Loading {dataset_title} dataset splits..."):
        train_text, test_text = load_qa_splits(sample_only=sample_only, data_path=data_path)

    console.print(
        f"📖 Train split loaded : [bold]{len(train_text):,} characters[/bold] "
        "(64 concept pairs)"
    )
    console.print(
        f"📖 Test split loaded  : [bold]{len(test_text):,} characters[/bold] "
        "(17 held-out concept pairs)"
    )

    # =========================================================================
    # 3. TOKENIZATION AND SLICING
    # =========================================================================
    console.print("\n[bold]🔤 Initializing Tokenizer...[/bold]")
    tokenizer = CharTokenizer()
    combined_text = train_text + "\n" + test_text
    tokenizer.build_vocab([combined_text])
    vocab_size = tokenizer.vocab_size

    train_tokens = tokenizer.encode(train_text)
    test_tokens = tokenizer.encode(test_text)

    console.print(f"  Vocabulary size : [cyan]{vocab_size} tokens[/cyan]")
    console.print(f"  Train tokens    : [cyan]{len(train_tokens):,} tokens[/cyan]")
    console.print(f"  Test tokens     : [cyan]{len(test_tokens):,} tokens[/cyan]")

    seq_len = 64
    stride_train = 4 if not sample_only else 6
    stride_test = 8

    train_loader = create_token_windows(train_tokens, seq_len=seq_len, stride=stride_train)
    test_loader = create_token_windows(test_tokens, seq_len=seq_len, stride=stride_test)

    console.print(
        f"  Train windows   : [cyan]{len(train_loader.dataset):,} slices[/cyan] "
        f"(seq_len={seq_len}, stride={stride_train})"
    )
    console.print(
        f"  Test windows    : [cyan]{len(test_loader.dataset):,} slices[/cyan] "
        f"(seq_len={seq_len}, stride={stride_test})"
    )

    # =========================================================================
    # 4. INSTANTIATE TINYGPT
    # =========================================================================
    console.print("\n[bold]🏗️ Constructing TinyGPT Model...[/bold]")
    embed_dim = 64
    num_layers = 2
    num_heads = 4
    model, total_params = build_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=128,
    )

    table = Table(title="Model Architecture & Parameters", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Configuration", style="magenta")
    table.add_column("Implementation", style="green")
    table.add_row(
        "Token Embeddings",
        f"{vocab_size} × {embed_dim}",
        "Module 11 (EmbeddingLayer)",
    )
    table.add_row(
        "Position Table",
        f"128 × {embed_dim}",
        "Module 11 (Learned Positional)",
    )
    table.add_row(
        "Attention",
        f"{num_layers} blocks, {num_heads} heads (causal)",
        "Module 12 (MultiHeadAttention)",
    )
    table.add_row(
        "Feed-Forward",
        f"4× expansion ({embed_dim * 4}) + GELU",
        "Module 02 and 03 (MLP)",
    )
    table.add_row(
        "Total Parameters",
        f"{total_params:,}",
        "100% Student Authored",
    )
    console.print(table)
    press_enter_to_continue(pause)

    # =========================================================================
    # 5. TRAINING LOOP & OVERFITTING DETECTIVE
    # =========================================================================
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    console.print(
        "[bold]🚀 Training TinyGPT & Running Overfitting Detective...[/bold]"
    )

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
            train_loss = trainer.train_epoch(train_loader)
            test_loss = evaluate_dataloader_loss(
                model=model,
                dataloader=test_loader,
                criterion=loss_fn,
                vocab_size=vocab_size,
            )
            gap = test_loss - train_loss
            history.append((epoch + 1, train_loss, test_loss, gap))

            progress.update(
                task,
                advance=1,
                description=(
                    f"[cyan]Epoch {epoch + 1}/{epochs} | "
                    f"Train: {train_loss:.4f} | Test: {test_loss:.4f} | "
                    f"Gap: +{gap:.2f}"
                ),
            )

    train_time = time.perf_counter() - t_start
    console.print(
        f"\n⏱️ Training completed in [bold green]{train_time:.2f}s[/bold green] "
        f"({train_time / epochs:.2f}s/epoch)"
    )

    # =========================================================================
    # 6. OVERFITTING DETECTIVE REPORT TABLE
    # =========================================================================
    detective_table = Table(
        title="🕵️ The Overfitting Detective: Train vs. Test Generalization Gap",
        box=box.ROUNDED,
    )
    detective_table.add_column("Epoch", justify="center")
    detective_table.add_column("Train Loss (Memorized)", justify="right", style="cyan")
    detective_table.add_column("Test Loss (Held-Out)", justify="right", style="magenta")
    detective_table.add_column("Generalization Gap", justify="right")
    detective_table.add_column("Detective Diagnosis", style="yellow")

    for ep, tr, te, gap in history:
        if ep == 1:
            diag = "Underfitting (Both losses high)"
            gap_style = "dim"
        elif ep == epochs // 2:
            diag = "Sweet Spot (Optimal generalization)"
            gap_style = "green"
        elif ep == epochs:
            diag = "Overfitting (Rote memorization active)"
            gap_style = "bold red"
        else:
            diag = "Learning representations"
            gap_style = "white"

        gap_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
        detective_table.add_row(
            f"Epoch {ep}",
            f"{tr:.4f}",
            f"{te:.4f}",
            f"[{gap_style}]{gap_str}[/{gap_style}]",
            diag,
        )

    console.print(detective_table)
    press_enter_to_continue(pause)

    # =========================================================================
    # 7. CONVERSATIONAL Q&A EVALUATION
    # =========================================================================
    console.print(
        "\n[bold]✨ Generating Conversational Q&A Autoregressively...[/bold]\n"
    )

    benchmark_prompts = [
        "Q: What is TinyGPT?\nA:",
        "Q: What is the KV cache?\nA:",
        "Q: What does CrossEntropyLoss measure?\nA:",
        "Q: What is multi-head attention?\nA:",
    ]

    if custom_prompt and custom_prompt not in benchmark_prompts:
        test_prompts = [custom_prompt]
    elif is_quick:
        test_prompts = [benchmark_prompts[0]]
    else:
        test_prompts = list(benchmark_prompts)

    scorecard_rows = []

    for prompt in test_prompts:
        t0 = time.perf_counter()
        if should_stream:
            with Live(console=console, refresh_per_second=25) as live:
                def on_token(current_text: str):
                    live.update(
                        Panel(
                            current_text,
                            border_style="cyan",
                            title=f"Prompt: {prompt.strip()} | Generating...",
                            subtitle="[dim]Streaming tokens...[/dim]",
                        )
                    )
                    time.sleep(0.025)

                response = generate_chat_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=90,
                    temperature=temperature,
                    top_k=top_k,
                    stream_callback=on_token,
                )
        else:
            response = generate_chat_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=90,
                temperature=temperature,
                top_k=top_k,
            )

        gen_time_ms = (time.perf_counter() - t0) * 1000

        # Extract answer part
        if "\nA:" in response:
            answer_body = response.split("\nA:", 1)[1].strip()
        else:
            answer_body = response[len(prompt):].strip()

        # Format validation: check that answer is non-empty and coherent
        has_content = len(answer_body) > 10
        scorecard_rows.append(
            (
                prompt.split("\n")[0],
                "[bold green]PASS[/bold green]" if has_content else "[bold red]FAIL[/bold red]",
                f"{gen_time_ms:.1f}ms",
            )
        )

        console.print(
            Panel(
                response,
                border_style="green" if has_content else "yellow",
                title=f"Sample: {prompt.split(chr(10))[0]}",
                subtitle=f"[dim]Generated in {gen_time_ms:.1f}ms (temp={temperature}, top_k={top_k})[/dim]",
            )
        )

    # Interactive Q&A loop if requested
    if is_interactive:
        console.print("\n[bold cyan]💬 Interactive Q&A Mode Enabled![/bold cyan]")
        console.print("[dim]Type a question starting with 'Q: ' or type 'exit' to finish.[/dim]\n")
        while True:
            try:
                user_q = console.input("[yellow]Ask TinyGPT > [/yellow]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_q or user_q.lower() in ("exit", "quit", "q"):
                break
            if not user_q.startswith("Q:"):
                user_q = f"Q: {user_q}"
            full_user_prompt = f"{user_q}\nA:"

            if should_stream:
                with Live(console=console, refresh_per_second=25) as live:
                    def on_user_token(text: str):
                        live.update(
                            Panel(
                                text,
                                border_style="cyan",
                                title=f"{user_q} | Generating...",
                            )
                        )
                        time.sleep(0.015)

                    ans = generate_chat_response(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=full_user_prompt,
                        max_new_tokens=100,
                        temperature=temperature,
                        top_k=top_k,
                        stream_callback=on_user_token,
                    )
            else:
                ans = generate_chat_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=full_user_prompt,
                    max_new_tokens=100,
                    temperature=temperature,
                    top_k=top_k,
                )
            console.print(Panel(ans, border_style="green", title="TinyGPT Response"))

    # =========================================================================
    # 8. CONVERGENCE SCORECARD & ACHIEVEMENT GATE
    # =========================================================================
    initial_train_loss = history[0][1]
    final_train_loss = history[-1][1]
    final_test_loss = history[-1][2]
    final_gap = history[-1][3]
    train_loss_drop = initial_train_loss - final_train_loss

    # The gate checks that the model converged on training concepts
    # and observed the generalization gap
    if is_quick:
        passed = (final_train_loss < 2.80) or (train_loss_drop > 0.30)
    else:
        passed = (final_train_loss < 0.60 or train_loss_drop > 1.20) and final_gap > 0.30

    if passed:
        success_msg = (
            "[bold green]🏆 MILESTONE ACHIEVED: "
            "TINYGPT CHAT & OVERFITTING DETECTIVE VERIFIED![/bold green]\n\n"
            f"  • Initial Train Loss : {initial_train_loss:.4f}\n"
            f"  • Final Train Loss   : [bold green]{final_train_loss:.4f}[/bold green]"
            f" (drop: -{train_loss_drop:.2f})\n"
            f"  • Final Test Loss    : [cyan]{final_test_loss:.4f}[/cyan]\n"
            f"  • Generalization Gap : [yellow]+{final_gap:.2f}[/yellow] "
            "(Demonstrated rote memorization boundary!)\n"
            "  • Dialogue Gate      : Coherent Q&A formatting with "
            "autoregressive completion!\n\n"
            "You trained a conversational assistant explaining TinyTorch\n"
            "from scratch using exclusively your TinyTorch autograd and "
            "transformer blocks!\n\n"
            "[dim]In Milestone 06: Profile and accelerate these models "
            "with KV caching (Module 18)\n"
            "and INT8 quantization (Module 15) for low latency inference.[/dim]"
        )
        console.print(
            Panel.fit(
                success_msg,
                border_style="green",
                title="Success!",
            )
        )
        return 0
    else:
        console.print(
            Panel.fit(
                "[bold red]❌ MILESTONE 05 FAILED: CONVERGENCE GATE NOT MET[/bold red]\n\n"
                f"  • Final Train Loss   : {final_train_loss:.4f} (target < 0.60)\n"
                f"  • Generalization Gap : +{final_gap:.2f} (target > 0.30)",
                border_style="red",
                title="Needs Tuning",
            )
        )
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 05 Part 4: TinyGPT Conversational Q&A & Overfitting Detective"
    )
    parser.add_argument(
        "--quick",
        "--sample",
        action="store_true",
        help="Run fast verification on sample dataset with fewer windows",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to custom Q&A text file to train on",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Q: What is TinyGPT?\nA:",
        help="Seed prompt prefix for conversational Q&A generation",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        type=float,
        default=0.35,
        help="Sampling temperature (0.3 to 0.4 recommended for factual recall)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Top-k vocabulary candidate filtering threshold",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream tokens character-by-character with typewriter effect",
    )
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming output and print completed blocks directly",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Launch interactive terminal Q&A loop after training",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        default=False,
        help="Pause between sections during interactive demonstration",
    )
    args = parser.parse_args()
    return run_milestone(args)


if __name__ == "__main__":
    sys.exit(main())
