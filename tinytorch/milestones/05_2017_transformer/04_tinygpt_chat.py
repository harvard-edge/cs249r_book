#!/usr/bin/env python3
"""
The Transformer Era (2017-2022): TinyGPT Conversational Chat & Overfitting Detective
=============================================================================

📚 HISTORICAL CONTEXT:
In late 2022, OpenAI released ChatGPT, showing that fine-tuned autoregressive
transformers could serve as fluent conversational partners. Conversational AI
relies on dialogue formatting, next-token prediction, and live streaming decode.

In this milestone part, you train TinyGPT as an interactive question-answering
assistant that explains TinyTorch concepts ("TinyTorch Teaching TinyTorch").

🎯 MILESTONE 05 PART 4: CONVERSATIONAL CHAT & THE OVERFITTING DETECTIVE
In this interactive milestone, YOU bring together your student-authored modules:
1. Conversational Dialogue Modeling:
   Formatting prompts with "Q: ..." and completing answers with "A: ...".
2. The Overfitting Detective Experiment:
   Comparing Train Loss versus Test Loss on held-out concept splits.
   Because a 218K parameter model has more weights than the dataset has
   characters, you will observe how micro-models can achieve near-zero training
   loss through rote memorization, exposing the generalization gap.
3. Interactive Typewriter Streaming:
   Token-by-token terminal typewriter streaming via live UI rendering.

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
  Module 11 (Embeddings)    : YOUR Token + Learned Positional Embeddings
  Module 12 (Attention)     : YOUR Causal Multi-Head Attention
  Module 13 (Transformers)  : YOUR Pre-LN Transformer Blocks & TinyGPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Tokenizer: this part uses a word-level ConceptTokenizer defined in this
  script, not your Module 10 tokenizer. 64 Q&A pairs are too few for a
  character model to learn whole answers in a few epochs, and your
  BPETokenizer splits on whitespace, which drops the newlines that separate
  "Q:" from "A:". Your CharTokenizer runs in Parts 1 and 3.

📊 SUCCESS CRITERIA:
  Learned        : Training loss < 0.60, measured in NumPy after training
                   (a model without working attention stalls near 1.0)
  Generalized    : The lowest held-out loss sits well below the untrained
                   model's held-out loss (see HELDOUT_DROP_MARGIN)
  Overfitting    : The final held-out loss sits above the training loss
                   (the Overfitting Detective's finding, not a score)
  Causality      : Changing future tokens never changes past predictions
  Honest Loss    : YOUR cross-entropy matches a NumPy reference
  Positions      : A repeated token gives different logits at different
                   positions (YOUR positional encoding is doing its job)

🏗️ ARCHITECTURE (Decoder-Only Conversational Transformer):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Dialogue   │───▶│ Embeddings  │───▶│   Causal    │───▶│     MLP     │
    │  "Q: What"  │    │  YOUR M11   │    │  Attention  │    │  YOUR M03   │
    └─────────────┘    └─────────────┘    │  YOUR M12   │    └─────────────┘
                                          └─────────────┘           │
                                                 │                  ▼
                                                 └─────────▶ ┌─────────────┐
                                                             │ Next Token  │
                                                             │ Live Stream │
                                                             └─────────────┘
"""

import argparse
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, List, Optional, Tuple

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
from tinytorch.core.embeddings import EmbeddingLayer  # noqa: E402
from tinytorch.core.layers import Linear  # noqa: E402
from tinytorch.core.transformers import (  # noqa: E402
    LayerNorm,
    TransformerBlock,
    create_causal_mask,
)
from milestones.data_manager import DatasetManager  # noqa: E402
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


class ConceptTokenizer:
    """
    Word and punctuation tokenizer preserving exact Q&A formatting.
    Maps words, punctuation, and structural newlines to discrete tokens,
    enabling micro-transformers to learn syntax and grammar rapidly.
    """

    def __init__(self):
        self.unk_token = "<unk>"
        self.token_to_id = {self.unk_token: 0}
        self.id_to_token = {0: self.unk_token}
        self.vocab_size = 1

    def tokenize_raw(self, text: str) -> List[str]:
        return re.findall(r"(?:\n|Q:|A:|[\w]+|[^\s\w])", text)

    def build_vocab(self, texts: List[str]) -> None:
        vocab_set = set()
        for text in texts:
            vocab_set.update(self.tokenize_raw(text))
        for token in sorted(vocab_set):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        self.vocab_size = len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_raw(text)
        return [self.token_to_id.get(t, 0) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        out: List[str] = []
        for i, t in enumerate(tokens):
            if t == "\n":
                out.append("\n")
            elif i == 0 or (out and out[-1] == "\n"):
                out.append(t)
            elif t in {".", ",", "?", "!", ":", ";", ")", "]"} or (out and out[-1] in {"(", "["}):
                out.append(t)
            elif (out and out[-1] == "-") or t == "-":
                out.append(t)
            else:
                out.append(" " + t)
        return "".join(out)


def normalize_prompt(user_prompt: str, available_questions: List[str]) -> str:
    """Normalize user input question to standard Q&A prompt formatting."""
    cleaned = user_prompt.strip()
    if cleaned.lower().startswith("user:"):
        cleaned = cleaned[5:].strip()
    elif cleaned.lower().startswith("q:"):
        cleaned = cleaned[2:].strip()

    if cleaned.endswith("A:"):
        cleaned = cleaned[:-2].strip()

    clean_lower = cleaned.lower()
    best_match = None
    for q in available_questions:
        q_text = q[2:].strip() if q.startswith("Q:") else q
        if clean_lower in q_text.lower() or q_text.lower() in clean_lower:
            best_match = q
            break
        q_words = set(re.findall(r"\w+", q_text.lower()))
        prompt_words = set(re.findall(r"\w+", clean_lower))
        if len(q_words & prompt_words) >= 3:
            best_match = q

    if best_match:
        return f"{best_match}\nA:"

    if not user_prompt.startswith("Q:"):
        user_prompt = f"Q: {user_prompt}"
    if not user_prompt.endswith("\nA:"):
        if user_prompt.endswith("A:"):
            user_prompt = user_prompt
        else:
            user_prompt = f"{user_prompt}\nA:"
    return user_prompt


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

        # Token + learned positional embeddings (Module 11). 2026-09-29: this
        # used to add a second EmbeddingLayer indexed by position ids, so the
        # model knew positions even when YOUR PositionalEncoding returned its
        # input unchanged; Parts 1 and 3 use the same single layer.
        self.embedding_layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)

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

        # Token embeddings plus the position table rows start_pos..start_pos+seq_len
        h = self.embedding_layer.forward(tokens, start_pos)

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
    tokenizer: ConceptTokenizer,
    prompt: str,
    max_new_tokens: int = 45,
    temperature: float = 0.2,
    top_k: int = 1,
    stream_callback: Optional[Callable[[str], None]] = None,
    stop_at_newline: bool = True,
) -> str:
    """
    Generate conversational answer to a question prompt.

    Args:
        model: Trained TinyGPT instance.
        tokenizer: Concept tokenizer with built vocabulary.
        prompt: Question prompt (e.g. 'Q: What is TinyGPT?\\nA:').
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering threshold.
        stream_callback: Optional callback invoked with updated text after each token.
        stop_at_newline: Whether to stop generation after newline.

    Returns:
        Full generated string (prompt + answer).
    """
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    current_ids = list(prompt_ids)
    current_tokens = Tensor(np.array([current_ids], dtype=np.int32))

    for _ in range(max_new_tokens):
        if current_tokens.shape[1] >= model.max_seq_len:
            break

        logits = model.forward(current_tokens)
        last_logits = logits.data[0, -1, :]

        if top_k <= 1 or temperature <= 0.1:
            next_id = int(np.argmax(last_logits))
        else:
            next_id = sample_next_token_top_k(
                last_logits,
                temperature=temperature,
                top_k=top_k,
            )
        current_ids.append(next_id)

        if stream_callback is not None:
            stream_callback(tokenizer.decode(current_ids))

        token_str = tokenizer.id_to_token.get(next_id, "")
        if stop_at_newline and token_str in ("\n", "<unk>", "<pad>"):
            break

        next_tensor = np.array([[next_id]], dtype=np.int32)
        current_tokens = Tensor(
            np.concatenate([current_tokens.data, next_tensor], axis=1)
        )

        if stop_at_newline:
            current_text = tokenizer.decode(current_ids)
            new_text = current_text[len(prompt):]
            if "\n" in new_text or "\nQ:" in new_text:
                break

    return tokenizer.decode(current_ids)


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
    seq_len: int = 36,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding (input, next-token target) windows over a token sequence."""
    inputs = []
    targets = []
    for start in range(0, len(tokens) - seq_len, stride):
        inputs.append(tokens[start:start + seq_len])
        targets.append(tokens[start + 1:start + seq_len + 1])

    if not inputs:
        # Fallback if text is shorter than seq_len
        inputs.append(tokens[:-1])
        targets.append(tokens[1:])

    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)


def diagnose_epochs(history):
    """Label each epoch from the measured held-out curve.

    The sweet spot is the epoch with the lowest held-out loss; overfitting
    means held-out loss has risen past it. Labels were once assigned by epoch
    number, so a steadily rising test loss could still be called "Sweet Spot".

    Args:
        history: list of (epoch, train_loss, test_loss, gap) tuples.

    Returns:
        list of (diagnosis, rich_style) tuples, one per epoch.
    """
    best_ep, _, best_te, _ = min(history, key=lambda row: row[2])
    labels = []
    for ep, _, te, _ in history:
        if ep == best_ep:
            labels.append(("Sweet Spot (lowest held-out loss)", "green"))
        elif ep < best_ep:
            labels.append(("Learning (held-out loss still falling)", "white"))
        elif te > best_te:
            labels.append(
                (f"Overfitting (held-out loss up {te - best_te:.2f} since epoch {best_ep})", "bold red")
            )
        else:
            labels.append(("Plateau (held-out loss matches best)", "dim"))
    return labels


# Gate calibration, 2026-09-29 (final training loss; --quick trains 8 epochs
# on the same splits instead of 10):
#   correct code     0.087-0.093 (quick 0.098); gap about +11
#   attention zeroed 1.009-1.013 (quick 1.002); gap about +7.5
# --quick used to pass on (loss < 2.80 or drop > 0.30), which any model met.
CHAT_TRAIN_LOSS_TARGET = 0.60
OVERFIT_GAP_TARGET = 0.30
# The lowest held-out loss must sit this far below the untrained model's
# held-out loss. Added 2026-09-29: with Trainer.train_epoch a no-op, the old
# gate read the Trainer's "0.0000" and passed a model whose test loss (7.28)
# matched the untrained one (7.27). Measured 2026-09-29, default run, in
# NumPy after the single-EmbeddingLayer change (untrained held-out 7.20):
#   correct code        lowest held-out 5.86-5.94 (epoch 1), drop 1.26-1.34 (3 runs)
#                       final train 0.068-0.074, gap +10.6 to +11.0
#   attention zeroed    lowest 6.22, drop 1.01; final train 0.99 (fails the loss)
#   no-op train_epoch   lowest 7.20, drop 0.00; final train 7.22 (fails all three)
#   uniform attention   lowest 6.49, drop 0.70; final train 0.079 (PASSES: with
#                       positions available, averaging the prefix still lets
#                       this model memorize; Part 1 is the check that catches it)
HELDOUT_DROP_MARGIN = 0.50


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
    tokenizer = ConceptTokenizer()
    combined_text = train_text + "\n" + test_text
    tokenizer.build_vocab([combined_text])
    vocab_size = tokenizer.vocab_size

    train_tokens = tokenizer.encode(train_text)
    test_tokens = tokenizer.encode(test_text)

    console.print(f"  Vocabulary size : [cyan]{vocab_size} tokens[/cyan]")
    console.print(f"  Train tokens    : [cyan]{len(train_tokens):,} tokens[/cyan]")
    console.print(f"  Test tokens     : [cyan]{len(test_tokens):,} tokens[/cyan]")

    seq_len = 36
    stride_train = 2
    stride_test = 4

    train_x, train_y = create_token_windows(train_tokens, seq_len=seq_len, stride=stride_train)
    test_x, test_y = create_token_windows(test_tokens, seq_len=seq_len, stride=stride_test)
    train_loader = DataLoader(TensorDataset(Tensor(train_x), Tensor(train_y)),
                              batch_size=32, shuffle=True)

    console.print(
        f"  Train windows   : [cyan]{len(train_x):,} slices[/cyan] "
        f"(seq_len={seq_len}, stride={stride_train})"
    )
    console.print(
        f"  Test windows    : [cyan]{len(test_x):,} slices[/cyan] "
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
    optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=0.01)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Two checks that need no training, so a broken module fails in seconds:
    # YOUR cross-entropy forward must report the true loss of the logits, and
    # YOUR positional encoding must make positions distinguishable.
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

    # Losses of the untrained model, measured before the first update. Every
    # loss the gate reads is computed in NumPy from the model's logits.
    train_loss_before = measured_loss(model, train_x, train_y)
    test_loss_before = measured_loss(model, test_x, test_y)
    console.print(
        f"  Loss before training: train [cyan]{train_loss_before:.4f}[/cyan], "
        f"test [cyan]{test_loss_before:.4f}[/cyan] "
        f"(uniform guessing: {np.log(vocab_size):.4f})\n"
    )

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
            # The training loss is re-measured after the epoch rather than
            # taken from YOUR Trainer.train_epoch: a train_epoch that did
            # nothing once reported 0.0 and passed an untrained model.
            trainer.train_epoch(train_loader)
            train_loss = measured_loss(model, train_x, train_y)
            test_loss = measured_loss(model, test_x, test_y)
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

    best_ep, _, best_te, _ = min(history, key=lambda row: row[2])
    for (ep, tr, te, gap), (diag, gap_style) in zip(history, diagnose_epochs(history)):

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

    all_questions = [
        line.strip() for line in train_text.splitlines() if line.startswith("Q:")
    ]
    benchmark_prompts = [
        "Q: What is autograd in TinyTorch?\nA:",
        "Q: What is TinyGPT?\nA:",
        "Q: What does CrossEntropyLoss measure?\nA:",
        "Q: What is multi-head attention?\nA:",
    ]

    if custom_prompt:
        normalized_prompt = normalize_prompt(custom_prompt, all_questions)
        test_prompts = [normalized_prompt]
    elif is_quick:
        test_prompts = [benchmark_prompts[0]]
    else:
        test_prompts = list(benchmark_prompts)

    scorecard_rows = []
    answered_count = 0

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
                    time.sleep(0.04)

                response = generate_chat_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=45,
                    temperature=temperature,
                    top_k=top_k,
                    stream_callback=on_token,
                )
        else:
            response = generate_chat_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=45,
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
        answered_count += int(has_content)
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
    final_train_loss = history[-1][1]
    final_test_loss = history[-1][2]
    final_gap = history[-1][3]
    train_loss_drop = train_loss_before - final_train_loss

    causality = causality_probe(model, test_tokens[:seq_len], vocab_size)
    if not causality.passed:
        console.print()
        console.print(Panel.fit(causality_failure_message(causality),
                                border_style="red", title="Causality Check Failed"))
        return 1

    # Uniform attention (every earlier word weighted equally) still memorizes
    # these 64 Q&A pairs, so the loss gates alone cannot reject it.
    attention = attention_content_probe(model, test_tokens[:seq_len])
    if not attention.passed:
        console.print()
        console.print(Panel.fit(attention_content_failure_message(attention),
                                border_style="red", title="Attention Check Failed"))
        return 1

    # Every loss below was measured in NumPy from the model's logits.
    # Three separate checks (2026-09-29):
    #   1. Learned: training loss below CHAT_TRAIN_LOSS_TARGET. A model with
    #      attention zeroed plateaus at 1.01 here (it can only count which word
    #      follows which); correct code reaches 0.09.
    #   2. Generalized at some point: the lowest held-out loss must sit
    #      HELDOUT_DROP_MARGIN below the untrained model's. An untrained model
    #      (or one whose training loop does nothing) cannot.
    #   3. Overfitting demonstrated: the held-out gap must open, because the
    #      Overfitting Detective is the lesson of this part. The gap is a
    #      symptom of memorization, not a sign of a good model, so it is
    #      reported as the experiment's result, never as success.
    # The old gate, (loss < 0.60 or drop > 1.20) and gap > 0.30, passed the
    # no-attention model through the drop clause.
    loss_target = CHAT_TRAIN_LOSS_TARGET
    learned = final_train_loss < loss_target
    heldout_target = test_loss_before - HELDOUT_DROP_MARGIN
    generalized = best_te < heldout_target
    gap_opened = final_gap > OVERFIT_GAP_TARGET
    passed = learned and generalized and gap_opened

    gap_note = (
        " (train/test gap opened: memorization is visible)"
        if gap_opened else " (gap still small at this training length)"
    )

    if passed:
        success_msg = (
            "[bold green]🏆 MILESTONE ACHIEVED: "
            "TINYGPT CHAT TRAINED, OVERFITTING REPRODUCED[/bold green]\n\n"
            f"  • Train Loss Before  : {train_loss_before:.4f} (untrained model)\n"
            f"  • Final Train Loss   : [bold green]{final_train_loss:.4f}[/bold green]"
            f" (drop: -{train_loss_drop:.2f}, needed < {loss_target:.2f})\n"
            f"  • Test Loss Before   : {test_loss_before:.4f} (untrained model)\n"
            f"  • Final Test Loss    : [cyan]{final_test_loss:.4f}[/cyan]\n"
            f"  • Generalization Gap : [yellow]{final_gap:+.2f}[/yellow]"
            f"{gap_note}\n"
            f"  • Lowest Test Loss   : {best_te:.4f} at epoch {best_ep} of {epochs}"
            f" (needed < {heldout_target:.2f})\n"
            f"  • {causality_report(causality)}\n"
            f"  • {attention_content_report(attention)}\n"
            f"  • Answers Generated  : {answered_count}/{len(test_prompts)} "
            "prompts produced a non-empty answer (length check only)\n\n"
            "The gap is the Overfitting Detective's finding, not a score: on 64\n"
            "Q&A pairs the model memorizes its training answers and does worse\n"
            "on questions it never saw. Your attention and training work; the\n"
            "dataset is simply too small to generalize from.\n\n"
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
        mark = lambda ok: "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(
            Panel.fit(
                "[bold red]❌ MILESTONE 05 FAILED: CONVERGENCE GATE NOT MET[/bold red]\n\n"
                f"  {mark(learned)} Final Train Loss   : {final_train_loss:.4f} "
                f"(target < {loss_target:.2f}, measured after training)\n"
                f"  {mark(generalized)} Lowest Test Loss   : {best_te:.4f} at epoch {best_ep} "
                f"(target < {heldout_target:.2f}, {HELDOUT_DROP_MARGIN:.2f} below the "
                f"untrained {test_loss_before:.2f})\n"
                f"  {mark(gap_opened)} Generalization Gap : {final_gap:+.2f} "
                f"(the overfitting demonstration needs > {OVERFIT_GAP_TARGET:.2f})\n\n"
                "  A model whose attention does nothing plateaus near 1.0 here: it\n"
                "  learns which word follows which, but cannot recall a whole answer.\n"
                "  A held-out loss that never drops below the untrained model's means\n"
                "  the weights did not move: check that Trainer.train_epoch runs\n"
                "  forward, backward and optimizer.step on every batch.\n"
                "  Check MultiHeadAttention.forward and that every parameter updates.",
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
