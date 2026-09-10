#!/usr/bin/env python3
"""Lab 3. Compare dense and sparse language-model training systems costs.

NanoGPT and Nano-MoE train on the same fixed batches. The lab reports measured
loss, throughput, and parameter footprints without claiming that a short run
establishes model quality. ``--smoke`` uses deterministic synthetic tokens on
CPU, performs one real optimizer step per model, and never accesses a network.

This classroom comparison is separate from canonical benchmark artifacts. Run
the registered workloads with ``mlperf run`` when producing review evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Expert(torch.nn.Module):
    """Two-layer feed-forward expert used by the standalone lab."""

    def __init__(self, model_width: int):
        super().__init__()
        self.input = torch.nn.Linear(model_width, model_width * 4, bias=False)
        self.output = torch.nn.Linear(model_width * 4, model_width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.input(inputs)))


class SparseMoERouter(torch.nn.Module):
    """Route each token through the selected feed-forward experts."""

    def __init__(self, model_width: int, *, experts: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = torch.nn.Linear(model_width, experts, bias=False)
        self.experts = torch.nn.ModuleList(
            [Expert(model_width) for _ in range(experts)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = inputs.shape
        flattened = inputs.reshape(-1, width)
        weights = F.softmax(self.gate(flattened), dim=1)
        weights, selected = torch.topk(weights, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flattened)
        for index, expert in enumerate(self.experts):
            rows, selected_rank = torch.where(selected == index)
            if rows.numel():
                output[rows] += (
                    expert(flattened[rows]) * weights[rows, selected_rank, None]
                )
        return output.reshape(batch, tokens, width)


class SparseLanguageModel(torch.nn.Module):
    """Small decoder-style model with sparse feed-forward layers."""

    def __init__(
        self,
        *,
        vocab_size: int,
        model_width: int,
        heads: int,
        layers: int,
        max_sequence_length: int,
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, model_width)
        self.position_embedding = torch.nn.Embedding(max_sequence_length, model_width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attention_norm": torch.nn.LayerNorm(model_width),
                        "attention": torch.nn.MultiheadAttention(
                            model_width, heads, batch_first=True
                        ),
                        "moe_norm": torch.nn.LayerNorm(model_width),
                        "moe": SparseMoERouter(model_width),
                    }
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = torch.nn.LayerNorm(model_width)
        self.output = torch.nn.Linear(model_width, vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            tokens.shape[1], device=tokens.device
        )
        for layer in self.layers:
            normalized = layer["attention_norm"](hidden)
            attention, _ = layer["attention"](
                normalized,
                normalized,
                normalized,
                attn_mask=causal_mask,
                need_weights=False,
            )
            hidden = hidden + attention
            hidden = hidden + layer["moe"](layer["moe_norm"](hidden))
        logits = self.output(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train dense NanoGPT and sparse Nano-MoE on identical batches."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a deterministic, CPU-only, network-free functional smoke.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON result path."
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "max_batches",
        "batch_size",
        "sequence_length",
        "embedding_dim",
        "heads",
        "layers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be greater than zero")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be greater than zero")
    if args.embedding_dim % args.heads:
        raise ValueError("embedding dimension must be divisible by the head count")
    if args.sequence_length > 256:
        raise ValueError("Nano-MoE supports sequence lengths up to 256 tokens")


def choose_device(requested: str, *, smoke: bool) -> torch.device:
    if smoke or requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"unsupported device: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synthetic_batches(
    *, batch_size: int, sequence_length: int, count: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        tokens = torch.randint(
            0,
            128,
            (batch_size, sequence_length + 1),
            generator=generator,
        )
        batches.append((tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()))
    return batches


def real_batches(
    *, batch_size: int, sequence_length: int, count: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from mlperf.assets import ensure_tinyshakespeare, tinyshakespeare_paths

    ensure_tinyshakespeare(download=True)
    data = tinyshakespeare_paths()["train"].read_bytes()
    tokens = torch.tensor(list(data), dtype=torch.long)
    window = sequence_length + 1
    required = count * batch_size * window
    if tokens.numel() < required:
        raise RuntimeError(
            f"Tiny Shakespeare has {tokens.numel()} bytes but the lab needs {required}"
        )
    chunks = tokens[:required].reshape(count, batch_size, window)
    return [(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()) for chunk in chunks]


def train_model(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    total_tokens = 0
    total_duration = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss = 0.0
        epoch_tokens = 0
        start = time.perf_counter()
        for inputs, targets in batches:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets=targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("training produced a missing or non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = targets.numel()
            weighted_loss += float(loss.item()) * tokens
            epoch_tokens += tokens
        duration = time.perf_counter() - start
        if not epoch_tokens:
            raise RuntimeError("training consumed no tokens")
        history.append(
            {
                "epoch": epoch,
                "loss": weighted_loss / epoch_tokens,
                "tokens": epoch_tokens,
                "duration_seconds": duration,
                "tokens_per_second": epoch_tokens / duration,
            }
        )
        total_tokens += epoch_tokens
        total_duration += duration
    return {
        "history": history,
        "final_loss": history[-1]["loss"],
        "tokens": total_tokens,
        "duration_seconds": total_duration,
        "tokens_per_second": total_tokens / total_duration,
    }


def parameter_summary(
    dense: torch.nn.Module, sparse: torch.nn.Module, *, top_k: int
) -> dict[str, Any]:
    dense_total = sum(parameter.numel() for parameter in dense.parameters())
    sparse_total = sum(parameter.numel() for parameter in sparse.parameters())
    all_expert_parameters = 0
    active_expert_parameters = 0
    for layer in sparse.layers:
        experts = layer["moe"].experts
        expert_sizes = [
            sum(parameter.numel() for parameter in expert.parameters())
            for expert in experts
        ]
        all_expert_parameters += sum(expert_sizes)
        active_expert_parameters += sum(expert_sizes[:top_k])
    shared_parameters = sparse_total - all_expert_parameters
    active_sparse_parameters = shared_parameters + active_expert_parameters
    return {
        "dense_total": dense_total,
        "sparse_total": sparse_total,
        "sparse_shared": shared_parameters,
        "sparse_all_experts": all_expert_parameters,
        "sparse_active_per_token": active_sparse_parameters,
        "sparse_active_fraction": active_sparse_parameters / sparse_total,
        "fp32_parameter_bytes": {
            "dense": dense_total * 4,
            "sparse": sparse_total * 4,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    if args.smoke:
        args.epochs = 1
        args.max_batches = 1
        args.batch_size = 2
        args.sequence_length = 8
        args.embedding_dim = 32
        args.heads = 4
        args.layers = 1
        args.device = "cpu"
        torch.set_num_threads(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device, smoke=args.smoke)
    batches = (
        synthetic_batches(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            count=args.max_batches,
            seed=args.seed,
        )
        if args.smoke
        else real_batches(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            count=args.max_batches,
        )
    )

    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    dense = NanoGPTWhiteBox(
        vocab_size=128,
        n_embd=args.embedding_dim,
        n_head=args.heads,
        n_layer=args.layers,
        max_seq_len=args.sequence_length,
    ).to(device)
    sparse = SparseLanguageModel(
        vocab_size=128,
        model_width=args.embedding_dim,
        heads=args.heads,
        layers=args.layers,
        max_sequence_length=args.sequence_length,
    ).to(device)
    for layer in sparse.layers:
        layer["moe"].top_k = args.top_k

    parameters = parameter_summary(dense, sparse, top_k=args.top_k)
    dense_metrics = train_model(
        dense,
        batches,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    sparse_metrics = train_model(
        sparse,
        batches,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    finite = all(
        math.isfinite(value)
        for value in (dense_metrics["final_loss"], sparse_metrics["final_loss"])
    )
    if not finite:
        raise RuntimeError("functional check failed because a final loss is not finite")

    result: dict[str, Any] = {
        "schema": "mlperf-edu-lab-result/0.1",
        "lab": "lab3-dense-sparse-comparison",
        "status": "passed",
        "result_scope": "functional-smoke" if args.smoke else "classroom-experiment",
        "canonical_result": False,
        "seed": args.seed,
        "device": str(device),
        "data_mode": "synthetic-deterministic"
        if args.smoke
        else "tinyshakespeare-local",
        "config": {
            "epochs": args.epochs,
            "batches_per_epoch": len(batches),
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "embedding_dim": args.embedding_dim,
            "heads": args.heads,
            "layers": args.layers,
            "experts": 8,
            "top_k": args.top_k,
            "learning_rate": args.learning_rate,
        },
        "parameters": parameters,
        "dense": dense_metrics,
        "sparse": sparse_metrics,
        "functional_check": {
            "same_training_batches": True,
            "finite_losses": True,
            "passed": True,
        },
        "interpretation_note": (
            "This bounded run measures systems behavior. It does not establish "
            "convergence or a quality ranking between architectures."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"LAB 3 FAIL: {exc}", file=sys.stderr)
        return 1

    print("MLPerf EDU Lab 3")
    print(f"  scope: {result['result_scope']}")
    print(f"  device: {result['device']}")
    print(
        f"  dense: loss={result['dense']['final_loss']:.4f}, "
        f"throughput={result['dense']['tokens_per_second']:.2f} tokens/s"
    )
    print(
        f"  sparse: loss={result['sparse']['final_loss']:.4f}, "
        f"throughput={result['sparse']['tokens_per_second']:.2f} tokens/s"
    )
    print(
        "  sparse active parameter fraction: "
        f"{result['parameters']['sparse_active_fraction']:.2%}"
    )
    print("LAB 3 SMOKE PASS" if args.smoke else "LAB 3 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
