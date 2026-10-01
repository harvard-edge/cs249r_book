#!/usr/bin/env python3
"""
The Transformer Era (2017-2022): TinyCopilot Python Code Generation
=============================================================================

📚 HISTORICAL CONTEXT:
In 2017, Vaswani et al. published "Attention Is All You Need," introducing the
Transformer architecture. In 2021, Chen et al. (OpenAI) released Codex,
showing that autoregressive transformers trained on source code develop deep
understandings of formal syntax, type signatures, and algorithmic logic.
GitHub Copilot brought this capability directly into developer workflows,
turning next-token prediction into the standard for modern code assistance.

Unlike natural language (where grammatical slips and typos are easily
forgiven), programming languages demand strict syntactic precision:
1. Balanced Delimiters: Every parenthesis, bracket, and quote must balance.
2. Indentation Scoping: Python block structure depends entirely on whitespace.
3. Keyword Semantics: Tokens like def, return, if, and class require valid
   downstream syntax to pass interpreter verification.
4. Abstract Syntax Tree (AST): The code must compile into a formal syntax tree.

🎯 MILESTONE 05 CAPSTONE: TRAIN TINYCOPILOT ON PYTHON CODE
In this capstone milestone, YOU bring together your full TinyTorch stack:
- Tokenizing source code characters with YOUR Tokenizer (Module 10)
- Projecting into continuous latent space with YOUR Embeddings (Module 11)
- Computing self-attention with YOUR Causal Multi-Head Attention (Module 12)
- Stacking decoder layers with YOUR Pre-LN Transformer Blocks (Module 13)
- Computing exact analytical gradients with YOUR Autograd engine (Module 06)
- Optimizing weights with YOUR AdamW optimizer (Module 07)
- Autoregressively completing code prompts with top-k sampling (Module 13)
- Quantitatively verifying generated code with Python AST parsing!

✅ REQUIRED MODULES (Run after Module 13):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR strided tensor data structure
  Module 02 (Activations)   : YOUR GELU activation function
  Module 03 (Layers)        : YOUR Linear projection layers
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (DataLoader)    : YOUR mini-batch DataLoader
  Module 06 (Autograd)      : YOUR reverse-mode computational graph
  Module 07 (Optimizers)    : YOUR AdamW optimizer
  Module 08 (Training)      : YOUR Trainer training loop
  Module 10 (Tokenization)  : YOUR Character Tokenizer
  Module 11 (Embeddings)    : YOUR Token + Learned Positional Embeddings
  Module 12 (Attention)     : YOUR Causal Multi-Head Attention
  Module 13 (Transformers)  : YOUR Pre-LN Transformer Blocks and TinyGPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Decoder-Only Generative Transformer):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Code Tokens │───▶│ Embeddings  │───▶│   Causal    │───▶│     MLP     │
    │ "def add("  │    │  YOUR M11   │    │  Attention  │    │  YOUR M03   │
    └─────────────┘    └─────────────┘    │  YOUR M12   │    └─────────────┘
                                          └─────────────┘           │
                                                 │                  ▼
                                                 └─────────▶ ┌─────────────┐
                                                             │ Next Token  │
                                                             │ AST Parsed! │
                                                             └─────────────┘

📊 SUCCESS CRITERIA:
  Loss Convergence  : Final training loss < 1.0 (a model without working
                      attention stalls near 2.1)
  Syntactic Validity: Greedy completions of function names absent from the
                      training corpus parse via ast.parse exactly as generated
                      (at least 1 of the 7 held-out prompts)
  Causality         : Changing future tokens never changes past predictions
  Honest Loss       : YOUR cross-entropy matches a NumPy reference, and the
                      gated loss is measured in NumPy from the model's logits
  Positions         : A repeated token gives different logits at different
                      positions (YOUR positional encoding is doing its job)
"""

import argparse
import ast
import os
from pathlib import Path
import sys
import time
from typing import Callable, Optional, Tuple

import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# Shared Milestone 05 checks live next to this script (transformer_gates.py)
milestone_dir = str(Path(__file__).resolve().parent)
if milestone_dir not in sys.path:
    sys.path.insert(0, milestone_dir)

# =============================================================================
# 🎓 ZONE 1: STUDENT CORE LEGO BRICKS (YOUR Modules 01-13)
# =============================================================================

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
from milestones.try_it import try_it  # noqa: E402
from transformer_gates import (  # noqa: E402
    attention_content_failure_message,
    attention_content_probe,
    attention_content_report,
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
from rich.syntax import Syntax  # noqa: E402
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
# 🎓 ZONE 1: STUDENT CORE LEGO BRICKS (Model Architecture & Generation)
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

        # Token + Positional Embeddings (Module 11)
        self.embedding_layer = EmbeddingLayer(
            vocab_size, embed_dim, max_seq_len
        )

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

        # Final LayerNorm and LM Head projection
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


def build_model(
    vocab_size: int,
    embed_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    max_seq_len: int = 128,
) -> Tuple[TinyGPT, int]:
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


def sample_next_token_top_k(
    logits: np.ndarray,
    temperature: float = 0.4,
    top_k: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample one token using temperature scaling and top-k filtering."""
    flat_logits = np.asarray(logits, dtype=np.float64).reshape(-1)

    # Top-k filtering: restrict sampling to top k highest scoring candidates
    if top_k is not None and 0 < top_k < len(flat_logits):
        k_indices = np.argpartition(flat_logits, -top_k)[-top_k:]
        min_k_val = np.min(flat_logits[k_indices])
        flat_logits = np.where(flat_logits >= min_k_val, flat_logits, -1e9)

    # Greedy decode for near-zero temperature
    if temperature <= 1e-6:
        return int(np.argmax(flat_logits))

    # Center logits for numerical stability before exponentiating
    centered = flat_logits - np.max(flat_logits)
    with np.errstate(over="ignore", under="ignore"):
        exp_logits = np.exp(centered / temperature)

    denom = np.sum(exp_logits)
    if np.isnan(denom) or denom <= 0:
        return int(np.argmax(flat_logits))

    probs = exp_logits / denom
    gen = rng if rng is not None else np.random.default_rng()
    return int(gen.choice(len(probs), p=probs))


def generate_code_completion(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.4,
    top_k: int = 5,
    stop_at_block: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Autoregressively extend code prompt using temperature and top-k sampling.

    Args:
        model: Trained TinyGPT instance.
        tokenizer: Character tokenizer.
        prompt: Initial Python or C code prefix.
        max_new_tokens: Maximum number of new characters to generate.
        temperature: Sampling temperature (0.3 to 0.5 recommended for syntax).
        top_k: Top-k vocabulary filtering threshold.
        stop_at_block: Whether to stop when encountering double newline.
        stream_callback: Optional callable invoked with decoded text on each token.

    Returns:
        Full generated code string (prompt + completion).
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

        if stop_at_block:
            current_text = tokenizer.decode(current_ids)
            new_text = current_text[len(prompt):]
            if "\n\n" in new_text:
                break

    return tokenizer.decode(current_ids)


# Scored prompts are function names that never appear in the training corpus,
# so a pass means the model learned Python syntax rather than recalled a
# snippet it trained on. tests/cli/test_release_regressions.py enforces that
# none of these leak into datasets/tinypy/tinypy_sample.txt.
HELDOUT_PROMPTS = [
    "def square(x):",
    "def mean(values):",
    "def clamp(x, lo, hi):",
    "def is_even(n):",
    "class Counter:",
    "def sigmoid(x):",
    "def dot(a, b):",
]

# Gate calibration, 2026-09-29, default full run (35,000 tokens, 10 epochs).
# The milestone trains ONE network per run and gates on it. Each held-out
# prompt is scored with a greedy completion (temperature 0), so the score is a
# property of the trained network, not of a random draw; the sampled
# completion is printed for display only. (Scoring the sampled output at 40%
# failed correct code in about 1 run in 4.)
#
#   Greedy prompts parsed out of 7, correct code, 20 separate trainings:
#     4,4,2,2,3,5,3,3,3,6 (release audit), 3,5,3,5,2,3,5 (calibration) and
#     1,4,5 (full script runs after the change)
#   Broken or untrained code (no attention, untrained): 0/7 in 6 of 6 runs.
#   Final training loss: correct 0.225-0.236 (20 runs); attention zeroed 2.130-2.131.
#
# A threshold of 2 of 7 failed correct code in the first verification run
# (1/7), so the syntax check asks for at least ONE parseable completion. The
# minimum observed on correct code (1 of 7, once in 20 runs) EQUALS that
# threshold, so this check carries no margin and cannot separate correct from
# broken code by itself. The loss gate (correct 0.23 vs 2.13 without
# attention) and the causality probe are the primary catchers of broken
# code; the syntax check confirms the network writes some parseable Python
# for a name it never saw (broken code scored 0/7 every time).
HELDOUT_MIN_VALID = 1
HELDOUT_VALIDITY_TARGET = 100.0 * HELDOUT_MIN_VALID / len(HELDOUT_PROMPTS)  # 14.3%
SCORED_MAX_NEW_TOKENS = 60
COPILOT_TRAIN_LOSS_TARGET = 1.0


def check_ast_validity(full_code: str, prompt: str) -> Tuple[bool, str]:
    """
    Check whether the model's completion parses into a valid Python AST.

    The code is judged exactly as the model wrote it. Two rules keep the gate
    honest: nothing is ever added to the code (padding an open block with
    ``pass`` makes any header valid), and at most ONE line may be dropped,
    only when generation stopped on the token budget mid-line. Trimming lines
    until something parses would accept a single good line followed by garbage.

    Args:
        full_code: Prompt plus generated completion.
        prompt: The prompt the completion extends.

    Returns:
        tuple: (is_valid: bool, status_message: str)
    """
    completion = full_code[len(prompt):]
    if not completion.strip():
        return False, "No completion generated"

    # generate_code_completion stops at a blank line when the block is done;
    # otherwise the token budget cut it off and the last line may be partial.
    block_finished = "\n\n" in completion
    code = prompt + completion.split("\n\n", 1)[0] if block_finished else full_code

    try:
        ast.parse(code)
        return True, "Valid AST (exact parse)"
    except SyntaxError as e:
        direct_err = f"SyntaxError: {e.msg} at line {e.lineno}"

    if not block_finished:
        lines = code.rstrip("\n").split("\n")
        prompt_lines = prompt.rstrip("\n").count("\n") + 1
        # Keep the prompt plus at least one generated line after the drop
        if len(lines) - 1 > prompt_lines:
            try:
                ast.parse("\n".join(lines[:-1]) + "\n")
                return True, "Valid AST (dropped final line cut off by token budget)"
            except SyntaxError:
                pass

    return False, direct_err


# =============================================================================
# --quick trains on fewer tokens of the same bundled corpus; epochs are unchanged.
# 2026-09-28: --quick was documented as "sample dataset" but the bundled sample
# was always loaded, so the only real difference is this token budget.
QUICK_TOKEN_LIMIT = 20000
FULL_TOKEN_LIMIT = 35000


# 📊 ZONE 2: MILESTONE HARNESS & VALIDATION UX
# =============================================================================

def load_tinypy_corpus(data_path: Optional[str] = None) -> str:
    """
    Load the bundled TinyPy corpus, or a custom text file.

    TinyPy ships as one offline file (datasets/tinypy/tinypy_sample.txt);
    there is no larger download, so quick and full runs read the same corpus
    and differ only in how many tokens they train on (see QUICK_TOKEN_LIMIT).
    """
    if data_path:
        custom_path = Path(data_path)
        if not custom_path.exists():
            raise FileNotFoundError(f"Custom dataset file not found at {data_path}")
        with open(custom_path, "r", encoding="utf-8") as f:
            return f.read()

    # Direct offline search path first
    sample_path = (
        Path(__file__).resolve().parent.parent.parent
        / "datasets"
        / "tinypy"
        / "tinypy_sample.txt"
    )
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return f.read()

    dm = DatasetManager()
    if hasattr(dm, "get_tinypy"):
        try:
            text = dm.get_tinypy()
            if text and len(text) > 0:
                return text
        except Exception:
            pass

    raise FileNotFoundError(
        "Could not load TinyPy dataset. "
        "Please ensure datasets/tinypy/tinypy_sample.txt exists."
    )


def run_milestone(args=None):
    """Main milestone execution flow for TinyGPT code completion."""
    args = args or argparse.Namespace()
    data_path = getattr(args, "data_path", None)
    is_quick = getattr(args, "quick", False) or getattr(args, "sample", False)
    pause = getattr(args, "pause", False)

    dataset_title = Path(data_path).name if data_path else "TinyPy"
    default_prompt = "def add(a, b):"
    custom_prompt = getattr(args, "prompt", default_prompt)

    temperature = getattr(args, "temperature", 0.4)
    epochs = getattr(args, "epochs", 10)
    top_k = getattr(args, "top_k", 5)
    should_stream = getattr(args, "stream", True)
    if (
        os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1"
        or os.environ.get("CI") == "true"
    ):
        should_stream = False

    # =========================================================================
    # 1. BANNER AND INTRO
    # =========================================================================
    banner_lines = [
        "[bold cyan]MILESTONE 05: TRANSFORMER ERA (2017-2022)[/bold cyan]",
        "",
        "[yellow]Train TinyGPT on Python Code: "
        "Foundation of GitHub Copilot & Codex![/yellow]",
        "",
        "• Model: Causal Pre-LN Transformer (Vaswani / Codex / GPT-4)",
        f"• Workload: Autoregressive Next-Char Code Completion on {dataset_title}",
        "• Syntax Gate: Rigorous syntax parsing to verify formal correctness",
        "• Systems: Parallel training with temperature and top-k decode",
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
    # 2. LOAD DATASET
    # =========================================================================
    with console.status(f"[bold green]Loading {dataset_title} dataset..."):
        text = load_tinypy_corpus(data_path=data_path)

    console.print(
        f"📖 Dataset loaded: [bold]{len(text):,} characters[/bold] "
        "of Python source code"
    )

    # =========================================================================
    # 3. TOKENIZATION AND SLICING
    # =========================================================================
    console.print("\n[bold]🔤 Initializing Tokenizer...[/bold]")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([text])
    vocab_size = tokenizer.vocab_size
    tokens = tokenizer.encode(text)

    console.print(f"  Vocabulary size : [cyan]{vocab_size} tokens[/cyan]")
    console.print(f"  Encoded stream  : [cyan]{len(tokens):,} tokens[/cyan]")

    # Crop the stream so training stays fast; --quick uses a smaller crop.
    token_limit = QUICK_TOKEN_LIMIT if is_quick else FULL_TOKEN_LIMIT
    train_tokens = tokens[: min(len(tokens), token_limit)]
    console.print(
        f"  Training crop   : [cyan]first {len(train_tokens):,} tokens[/cyan] "
        f"({'quick' if is_quick else 'full'} run)"
    )
    seq_len = 64
    stride = 4

    inputs = []
    targets = []
    for start in range(0, len(train_tokens) - seq_len, stride):
        inputs.append(train_tokens[start:start + seq_len])
        targets.append(train_tokens[start + 1:start + seq_len + 1])

    x_tensor = Tensor(np.array(inputs, dtype=np.int32))
    y_tensor = Tensor(np.array(targets, dtype=np.int32))
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    console.print(
        f"  Slices created  : [cyan]{len(dataset):,} training windows[/cyan] "
        f"(seq_len={seq_len})"
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
    # 5. TRAINING LOOP
    # =========================================================================
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Two checks that need no training, so a broken module fails in seconds:
    # YOUR cross-entropy forward must report the true loss of the logits, and
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

    # Loss of the untrained model on a fixed spread of training windows,
    # measured before the first update (uniform guessing scores ln(vocab)).
    probe_windows = np.linspace(0, len(inputs) - 1, num=min(256, len(inputs))).astype(int)
    probe_x = train_x[probe_windows]
    probe_y = train_y[probe_windows]
    loss_before = measured_loss(model, probe_x, probe_y)
    console.print(
        f"  Loss before training: [cyan]{loss_before:.4f}[/cyan] "
        f"(uniform guessing: {np.log(vocab_size):.4f})\n"
    )

    console.print(
        "[bold]🚀 Training TinyGPT from Scratch on Python Code...[/bold]"
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
            loss = trainer.train_epoch(dataloader)
            perplexity = np.exp(min(loss, 20.0))
            history.append((epoch + 1, loss, perplexity))
            progress.update(
                task,
                advance=1,
                description=(
                    f"[cyan]Epoch {epoch+1}/{epochs} | "
                    f"Loss: {loss:.4f} (PPL: {perplexity:.1f})"
                ),
            )

    t_total = time.perf_counter() - t_start
    t_sec = f"{t_total:.2f}s"
    t_rate = f"{t_total / epochs:.2f}s/epoch"
    console.print(
        f"\n⏱️ Training took [bold green]{t_sec}[/bold green] ({t_rate})"
    )

    # Display convergence summary
    res_table = Table(title="Training Loss Trajectory", box=box.ROUNDED)
    res_table.add_column("Epoch", justify="center")
    res_table.add_column("Epoch Loss (your Trainer)", justify="right")
    res_table.add_column("Perplexity", justify="right")

    summary_epochs = [history[0], history[len(history) // 2], history[-1]]
    for ep, l_val, p_val in summary_epochs:
        res_table.add_row(f"Epoch {ep}", f"{l_val:.4f}", f"{p_val:.2f}")
    console.print(res_table)
    press_enter_to_continue()

    # =========================================================================
    # 6. CODE COMPLETION GENERATION & AST VALIDATION
    # =========================================================================
    syntax_lang = "python"
    console.print(
        "[bold]✨ Generating Python Code Autoregressively...[/bold]\n"
    )

    # A prompt that appears in the training text only tests recall, so it is
    # shown as a demo but never counted toward the gate.
    scored_prompts = [p for p in HELDOUT_PROMPTS if p not in text]
    test_prompts = list(scored_prompts)
    if custom_prompt and custom_prompt not in test_prompts:
        test_prompts.insert(0, custom_prompt)

    scorecard_rows = []
    valid_count = 0

    for prompt in test_prompts:
        is_scored = prompt in scored_prompts

        # (a) Sampled completion (temperature + top-k): shown, never scored.
        #     Sampling is a random draw, so scoring it made a correct model
        #     pass only about 3 runs in 4.
        if should_stream:
            with Live(console=console, refresh_per_second=25) as live:
                def on_token(current_text: str):
                    syntax = Syntax(
                        current_text,
                        syntax_lang,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                    )
                    live.update(
                        Panel(
                            syntax,
                            border_style="cyan",
                            title=f"Prompt: {prompt} | Generating...",
                            subtitle="[dim]Streaming tokens...[/dim]",
                        )
                    )
                    if not (os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1" or os.environ.get("CI") == "true"):
                        time.sleep(0.015)

                sampled_code = generate_code_completion(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=60,
                    temperature=temperature,
                    top_k=top_k,
                    stream_callback=on_token,
                )
        else:
            sampled_code = generate_code_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=60,
                temperature=temperature,
                top_k=top_k,
            )
        console.print(
            Panel(
                Syntax(sampled_code, syntax_lang, theme="monokai",
                       line_numbers=True, word_wrap=True),
                border_style="dim",
                title=f"Prompt: {prompt} | sampled (temp={temperature}, top_k={top_k})",
                subtitle="[dim]display only, not scored[/dim]",
            )
        )

        # (b) Greedy completion (always the most likely next character): the
        #     model's single best answer, and the one that is scored.
        t0 = time.perf_counter()
        full_code = generate_code_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=SCORED_MAX_NEW_TOKENS,
            temperature=0.0,
            top_k=top_k,
        )
        gen_time_ms = (time.perf_counter() - t0) * 1000
        is_valid, diagnostic = check_ast_validity(full_code, prompt)

        if is_valid and is_scored:
            valid_count += 1
        if is_valid:
            status_badge = "[bold green]PASS (AST Valid)[/bold green]"
        else:
            status_badge = "[bold red]FAIL (Syntax Error)[/bold red]"
        if not is_scored:
            status_badge += " [dim](seen in training, not scored)[/dim]"

        scorecard_rows.append(
            (prompt, status_badge, diagnostic, f"{gen_time_ms:.1f}ms")
        )

        console.print(
            Panel(
                Syntax(full_code, syntax_lang, theme="monokai",
                       line_numbers=True, word_wrap=True),
                border_style="green" if is_valid else "yellow",
                title=f"Prompt: {prompt} | greedy | {status_badge}",
                subtitle=f"[dim]{diagnostic} in {gen_time_ms:.1f}ms (greedy decode)[/dim]",
            )
        )

    # =========================================================================
    # 7. CONVERGENCE SCORECARD & ACHIEVEMENT GATE
    # =========================================================================
    n_scored = len(scored_prompts)
    ast_validity_rate = (valid_count / n_scored) * 100.0 if n_scored else 0.0

    score_table = Table(
        title="Syntactic Validity on Unseen Function Names (AST Parsing Gate)",
        box=box.ROUNDED,
    )
    score_table.add_column("Prompt Prefix", style="cyan")
    score_table.add_column("AST Status (greedy)", justify="center")
    score_table.add_column("Compiler Diagnostic", style="magenta")
    score_table.add_column("Decode Latency", justify="right", style="green")

    for p_name, st, diag, lat in scorecard_rows:
        score_table.add_row(p_name, st, diag, lat)

    console.print(score_table)

    # Training loss is re-measured on every training window after training,
    # in NumPy (2026-09-29): gating on the Trainer's epoch average, which is
    # YOUR cross-entropy's value, passed a loss forward that returned 0.
    final_loss = measured_loss(model, train_x, train_y)
    loss_drop = loss_before - final_loss

    causality = causality_probe(model, inputs[0], vocab_size)
    if not causality.passed:
        console.print()
        console.print(Panel.fit(causality_failure_message(causality),
                                border_style="red", title="Causality Check Failed"))
        return 1

    # Uniform attention (every earlier character weighted equally) still
    # reaches a low loss and parses some prompts, so it is checked directly.
    attention = attention_content_probe(model, inputs[0])
    if not attention.passed:
        console.print()
        console.print(Panel.fit(attention_content_failure_message(attention),
                                border_style="red", title="Attention Check Failed"))
        return 1

    loss_converged = final_loss < COPILOT_TRAIN_LOSS_TARGET
    syntax_converged = n_scored > 0 and valid_count >= min(HELDOUT_MIN_VALID, n_scored)
    passed = loss_converged and syntax_converged

    console.print()
    if passed:
        pass_ratio = f"{valid_count}/{n_scored}"
        success_msg = (
            "[bold green]🏆 MILESTONE ACHIEVED: "
            "TINYCOPILOT CODE GENERATION VERIFIED![/bold green]\n\n"
            f"  • Loss Before      : {loss_before:.4f} (untrained model)\n"
            f"  • Final Loss       : [bold green]{final_loss:.4f}[/bold green]"
            f" (measured after training; drop: -{loss_drop:.2f}, needed < {COPILOT_TRAIN_LOSS_TARGET:.1f})\n"
            f"  • AST Validity Rate: [bold green]{ast_validity_rate:.1f}%"
            f"[/bold green] ({pass_ratio} unseen prompts parsed, greedy decode)\n"
            f"  • {causality_report(causality)}\n"
            f"  • {attention_content_report(attention)}\n"
            "  • Syntactic Gate   : Valid Python for function names "
            "the model never saw in training\n\n"
            "You trained the foundation of OpenAI Codex and GitHub Copilot\n"
            "from scratch using exclusively your TinyTorch autograd and "
            "transformer blocks!\n\n"
            "[dim]Next in Milestone 06: Profile and accelerate this model "
            "with KV caching (Module 18)\n"
            "and INT8 quantization (Module 15) for serving latency.[/dim]"
        )
        console.print(
            Panel.fit(
                success_msg,
                border_style="green",
                title="Success!",
            )
        )
        try_completions(model, tokenizer, should_stream)
        return 0
    else:
        mark = lambda ok: "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(
            Panel.fit(
                "[bold red]❌ MILESTONE 05 FAILED: "
                "SYNTAX OR LOSS GATE NOT MET[/bold red]\n\n"
                f"  {mark(loss_converged)} Final Loss       : {final_loss:.4f} "
                f"(measured after training, target < {COPILOT_TRAIN_LOSS_TARGET:.1f})\n"
                f"  {mark(syntax_converged)} AST Validity Rate: {ast_validity_rate:.1f}% "
                f"(target: at least {HELDOUT_MIN_VALID} of {n_scored} unseen prompts, greedy)\n"
                f"    Passed Prompts   : {valid_count}/{n_scored} unseen\n\n"
                "  Without working attention the model only learns which character\n"
                "  follows which: loss stalls near 2.1 and no completion parses.\n"
                "  Check MultiHeadAttention.forward and that every parameter updates.",
                border_style="red",
                title="Needs Tuning",
            )
        )
        return 1


TRY_IT_MAX_PROMPT_CHARS = 80


def try_completions(model, tokenizer, should_stream, read=None):
    """After a pass, let the student type their own code prefixes to complete.

    Completions are greedy (always the most likely next character), the same
    decode the syntax gate scores, so "parses" here means what it means there.
    """

    def complete(prompt: str) -> None:
        if len(prompt) > TRY_IT_MAX_PROMPT_CHARS:
            console.print(f"[yellow]Keep it under {TRY_IT_MAX_PROMPT_CHARS} characters; "
                          "TinyCopilot sees 128 at a time.[/yellow]")
            return
        if tokenizer.decode(tokenizer.encode(prompt)) != prompt:
            console.print("[yellow]That prompt uses characters TinyCopilot never saw "
                          "in training. Stick to ordinary Python.[/yellow]")
            return
        decode = dict(model=model, tokenizer=tokenizer, prompt=prompt,
                      max_new_tokens=SCORED_MAX_NEW_TOKENS, temperature=0.0)
        if should_stream:
            with Live(console=console, refresh_per_second=25, transient=True) as live:
                def on_token(current_text: str):
                    live.update(Panel(Syntax(current_text, "python", theme="monokai",
                                             line_numbers=True, word_wrap=True),
                                      border_style="cyan", title="Generating..."))
                code = generate_code_completion(stream_callback=on_token, **decode)
        else:
            code = generate_code_completion(**decode)
        is_valid, diagnostic = check_ast_validity(code, prompt)
        verdict = "[green]parses as Python[/green]" if is_valid else "[yellow]does not parse[/yellow]"
        console.print(Panel(
            Syntax(code, "python", theme="monokai", line_numbers=True, word_wrap=True),
            border_style="green" if is_valid else "yellow",
            title=f"YOUR TinyCopilot | {verdict}",
            subtitle=f"[dim]{diagnostic}[/dim]",
        ))

    return try_it(
        console,
        "Type the start of some Python and YOUR TinyCopilot finishes it.\n"
        "Try a function it never saw, like [cyan]def triple(x):[/cyan] or "
        "[cyan]def is_positive(n):[/cyan]\n"
        "It learned from a few thousand lines of code, so expect the shape of "
        "Python more than correct logic.",
        "[yellow]Complete > [/yellow]",
        complete,
        read=read,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 05: TinyCopilot Python Code Generation"
    )
    parser.add_argument(
        "--quick",
        "--sample",
        action="store_true",
        help=("Train on the first 20,000 tokens of the bundled corpus instead of "
              "35,000 (same corpus and epochs, fewer training windows)"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="def add(a, b):",
        help="Seed prompt prefix for Python code generation",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        type=float,
        default=0.4,
        help="Sampling temperature (0.3 to 0.5 recommended for syntax)",
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
        "--data-path",
        type=str,
        default=None,
        help="Path to custom source code or text file to train on",
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
        "--pause",
        action="store_true",
        default=False,
        help="Pause between sections during interactive demonstration",
    )
    args = parser.parse_args()
    return run_milestone(args)


if __name__ == "__main__":
    sys.exit(main())
