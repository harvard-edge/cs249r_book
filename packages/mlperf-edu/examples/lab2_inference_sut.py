#!/usr/bin/env python3
"""Lab 2. Compare naïve and KV-cache autoregressive decoding.

The lab implements the current :class:`mlperf.sut.SUT_Interface` protocol and
drives it locally. The product CLI does not yet accept arbitrary ``--sut``
plugins. Run the canonical built-in inference workload separately with:

    mlperf run --workload causal-language-modeling --mode inference --phase decode --profile min

Lab results measure a local, lab-scale model. They are not publishable MLPerf
EDU results. Token-for-token parity guards the optimization against changing
the model output. ``--smoke`` is deterministic, CPU-only, and network-free.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlperf.loadgen import QuerySample  # noqa: E402
from mlperf.sut import SUT_Interface  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure NanoGPT decode with and without a KV cache."
    )
    parser.add_argument(
        "--mode", choices=("baseline", "kv-cache", "compare"), default="compare"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a deterministic, CPU-only, network-free functional smoke.",
    )
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--generated-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON result path."
    )
    return parser


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


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "queries",
        "prompt_length",
        "generated_tokens",
        "repeats",
        "embedding_dim",
        "heads",
        "layers",
        "vocab_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be greater than zero")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    if args.embedding_dim % args.heads:
        raise ValueError("embedding dimension must be divisible by the head count")
    if args.prompt_length + args.generated_tokens > 4096:
        raise ValueError("prompt plus generated tokens exceeds the supported context")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise ValueError(f"checkpoint not found: {args.checkpoint}")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile without observations")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class StudentNanoGPTSUT(SUT_Interface):
    """A complete lab SUT with interchangeable decode implementations."""

    def __init__(self, config: dict[str, Any]):
        from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

        self.device = torch.device(config["device"])
        self.seed = int(config["seed"])
        self.prompt_length = int(config["prompt_length"])
        self.generated_tokens = int(config["generated_tokens"])
        self.vocab_size = int(config["vocab_size"])
        self.mode = str(config["mode"])
        max_seq_len = self.prompt_length + self.generated_tokens
        self.model = NanoGPTWhiteBox(
            vocab_size=self.vocab_size,
            n_embd=int(config["embedding_dim"]),
            n_head=int(config["heads"]),
            n_layer=int(config["layers"]),
            max_seq_len=max_seq_len,
        ).to(self.device)
        checkpoint = config.get("checkpoint")
        if checkpoint is not None:
            payload = torch.load(
                checkpoint, map_location=self.device, weights_only=True
            )
            if isinstance(payload, dict):
                for key in ("model_state_dict", "state_dict", "model"):
                    if key in payload and isinstance(payload[key], dict):
                        payload = payload[key]
                        break
            self.model.load_state_dict(payload)
        self.model.eval()

    def prompt_for(self, sample: QuerySample) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + int(sample.index))
        prompt = torch.randint(
            0,
            self.vocab_size,
            (1, self.prompt_length),
            generator=generator,
            dtype=torch.long,
        )
        return prompt.to(self.device)

    def decode_baseline(self, prompt: torch.Tensor) -> torch.Tensor:
        output = prompt
        for _ in range(self.generated_tokens):
            logits, _ = self.model(output)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            output = torch.cat((output, next_token), dim=1)
        return output

    def decode_with_kv_cache(self, prompt: torch.Tensor) -> torch.Tensor:
        output = prompt
        logits, past_key_values = self.model(prompt, use_kv_cache=True)
        for step in range(self.generated_tokens):
            next_token = logits[:, -1:, :].argmax(dim=-1)
            output = torch.cat((output, next_token), dim=1)
            if step + 1 < self.generated_tokens:
                logits, past_key_values = self.model(
                    next_token,
                    use_kv_cache=True,
                    past_key_values=past_key_values,
                )
        return output

    async def process_queries(self, samples: list[QuerySample]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with torch.inference_mode():
            for sample in samples:
                prompt = self.prompt_for(sample)
                start = time.perf_counter()
                if self.mode == "baseline":
                    output = self.decode_baseline(prompt)
                elif self.mode == "kv-cache":
                    output = self.decode_with_kv_cache(prompt)
                else:
                    raise ValueError(f"unsupported SUT mode: {self.mode}")
                latency_ms = (time.perf_counter() - start) * 1000.0
                token_ids = output[0, self.prompt_length :].cpu().tolist()
                digest = hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                results.append(
                    {
                        "query_id": int(sample.id),
                        "tokens_generated": len(token_ids),
                        "sequence_length": int(output.size(1)),
                        "generated_token_ids": token_ids,
                        "generated_token_sha256": digest,
                        "latency_ms": latency_ms,
                    }
                )
        return results


async def measure_mode(
    sut: StudentNanoGPTSUT,
    samples: list[QuerySample],
    *,
    mode: str,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sut.mode = mode
    for _ in range(warmup):
        await sut.process_queries(samples)

    observations: list[dict[str, Any]] = []
    start = time.perf_counter()
    first_results: list[dict[str, Any]] = []
    for repeat in range(repeats):
        current = await sut.process_queries(samples)
        if repeat == 0:
            first_results = current
        observations.extend(current)
    duration = time.perf_counter() - start
    latencies = [float(item["latency_ms"]) for item in observations]
    generated = sum(int(item["tokens_generated"]) for item in observations)
    metrics = {
        "mode": mode,
        "queries": len(observations),
        "generated_tokens": generated,
        "duration_seconds": duration,
        "tokens_per_second": generated / duration,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
        },
    }
    return metrics, first_results


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    if args.smoke:
        args.queries = 1
        args.prompt_length = 8
        args.generated_tokens = 2
        args.repeats = 1
        args.warmup = 0
        args.embedding_dim = 32
        args.heads = 4
        args.layers = 1
        args.vocab_size = 64
        args.device = "cpu"
        torch.set_num_threads(1)

    torch.manual_seed(args.seed)
    device = choose_device(args.device, smoke=args.smoke)
    config = {
        "device": str(device),
        "seed": args.seed,
        "prompt_length": args.prompt_length,
        "generated_tokens": args.generated_tokens,
        "vocab_size": args.vocab_size,
        "embedding_dim": args.embedding_dim,
        "heads": args.heads,
        "layers": args.layers,
        "mode": "baseline",
        "checkpoint": args.checkpoint,
    }
    sut = StudentNanoGPTSUT(config)
    modes = ("baseline", "kv-cache") if args.mode == "compare" else (args.mode,)
    metrics: dict[str, Any] = {}
    outputs: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        metrics[mode], outputs[mode] = await measure_mode(
            sut,
            samples=[
                QuerySample(id=index, index=index, arrival_time=0.0)
                for index in range(args.queries)
            ],
            mode=mode,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    expected_sequence_length = args.prompt_length + args.generated_tokens
    valid_shapes = all(
        len(mode_outputs) == args.queries
        and all(
            item["tokens_generated"] == args.generated_tokens
            and item["sequence_length"] == expected_sequence_length
            for item in mode_outputs
        )
        for mode_outputs in outputs.values()
    )
    if not valid_shapes:
        raise RuntimeError(
            "decode output failed the token-count or sequence-length check"
        )

    parity: bool | None = None
    speedup: float | None = None
    if args.mode == "compare":
        baseline_tokens = [item["generated_token_ids"] for item in outputs["baseline"]]
        cached_tokens = [item["generated_token_ids"] for item in outputs["kv-cache"]]
        parity = baseline_tokens == cached_tokens
        if not parity:
            raise RuntimeError("KV-cache output differs from baseline output")
        speedup = (
            metrics["baseline"]["latency_ms"]["p50"]
            / metrics["kv-cache"]["latency_ms"]["p50"]
        )

    result: dict[str, Any] = {
        "schema": "mlperf-edu-lab-result/0.1",
        "lab": "lab2-kv-cache-inference",
        "status": "passed",
        "result_scope": "functional-smoke" if args.smoke else "classroom-experiment",
        "canonical_result": False,
        "seed": args.seed,
        "device": str(device),
        "model": {
            "vocab_size": args.vocab_size,
            "embedding_dim": args.embedding_dim,
            "heads": args.heads,
            "layers": args.layers,
            "parameters": sum(
                parameter.numel() for parameter in sut.model.parameters()
            ),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        },
        "measurement": {
            "queries_per_repeat": args.queries,
            "prompt_length": args.prompt_length,
            "generated_tokens_per_query": args.generated_tokens,
            "warmup_repeats": args.warmup,
            "measured_repeats": args.repeats,
        },
        "metrics": metrics,
        "functional_check": {
            "expected_sequence_length": expected_sequence_length,
            "valid_output_shapes": True,
            "token_parity": parity,
            "passed": True,
        },
        "kv_cache_p50_speedup": speedup,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(run(args))
    except (RuntimeError, ValueError) as exc:
        print(f"LAB 2 FAIL: {exc}", file=sys.stderr)
        return 1

    print("MLPerf EDU Lab 2")
    print(f"  scope: {result['result_scope']}")
    print(f"  device: {result['device']}")
    for mode, metrics in result["metrics"].items():
        print(
            f"  {mode}: p50={metrics['latency_ms']['p50']:.3f} ms, "
            f"p90={metrics['latency_ms']['p90']:.3f} ms, "
            f"throughput={metrics['tokens_per_second']:.2f} tokens/s"
        )
    if result["functional_check"]["token_parity"] is not None:
        print(f"  token parity: {result['functional_check']['token_parity']}")
        print(f"  KV-cache p50 speedup: {result['kv_cache_p50_speedup']:.2f}x")
    print("LAB 2 SMOKE PASS" if args.smoke else "LAB 2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
