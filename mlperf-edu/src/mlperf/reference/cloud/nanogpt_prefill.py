"""
MLPerf EDU causal language modeling, prefill phase

Single forward pass over a long context while materializing the KV cache. Exercises
the compute-bound regime: every weight matrix is reused across `ctx_len`
tokens, giving high arithmetic intensity. Should sit on the compute side
of the roofline.

The prefill and decode phases share one quality-approved NanoGPT checkpoint.
"""

import hashlib
import json
import statistics
import time
import torch

from .nanogpt_train import NanoGPTWhiteBox


FIXED_PROMPT_SEED = 314159


def fixed_token_prompt(
    *, batch_size: int, context_len: int, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, str]:
    """Create the canonical inference prompt independently of the run seed."""
    generator = torch.Generator(device="cpu").manual_seed(FIXED_PROMPT_SEED)
    prompt = torch.randint(
        0,
        vocab_size,
        (batch_size, context_len),
        dtype=torch.long,
        generator=generator,
    )
    canonical = json.dumps(
        {"shape": list(prompt.shape), "token_ids": prompt.reshape(-1).tolist()},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return prompt.to(device), hashlib.sha256(canonical).hexdigest()


def _sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def estimate_activation_bytes(model: NanoGPTWhiteBox, ctx_len: int, batch: int) -> int:
    """Static estimate of peak activation memory during prefill.

    Counts the dominant tensors per layer: Q/K/V (3 x B x ctx x d_model),
    attention scores (B x n_head x ctx x ctx), and FFN hidden (B x ctx x 4*d).
    Per-layer peak is held simultaneously during backward; for forward-only
    inference, only one layer's worth is live at a time.
    """
    cfg = model.config
    d = cfg["n_embd"]
    nh = cfg["n_head"]
    bytes_per = 4  # fp32
    qkv = 3 * batch * ctx_len * d
    attn = batch * nh * ctx_len * ctx_len
    ffn = batch * ctx_len * 4 * d
    return (qkv + attn + ffn) * bytes_per


class NanoGPTPrefill:
    """Times one forward pass over `ctx_len` tokens.

    Reports prefill latency, throughput, KV-cache bytes, and a static
    peak-activation estimate. Each timed forward materializes a fresh cache,
    matching the prompt-processing phase that precedes cached decode.
    """

    def __init__(
        self, model: NanoGPTWhiteBox, context_len: int = 1792, batch_size: int = 1
    ):
        if context_len > model.config["max_seq_len"]:
            raise ValueError(
                f"context_len={context_len} exceeds model max_seq_len="
                f"{model.config['max_seq_len']}; bump NanoGPTWhiteBox(max_seq_len=) first."
            )
        self.model = model.eval()
        self.ctx_len = context_len
        self.batch = batch_size
        self.vocab = model.config["vocab_size"]

    def run(
        self, n_warmup: int = 3, n_iter: int = 10, emit_sidecar: bool = True
    ) -> dict:
        device = next(self.model.parameters()).device
        ids, prompt_sha256 = fixed_token_prompt(
            batch_size=self.batch,
            context_len=self.ctx_len,
            vocab_size=self.vocab,
            device=device,
        )
        act_bytes = estimate_activation_bytes(self.model, self.ctx_len, self.batch)

        with torch.no_grad():
            for _ in range(n_warmup):
                _, warmup_cache = self.model(ids, use_kv_cache=True)
                _validate_kv_cache(warmup_cache, expected_context=self.ctx_len)
                del warmup_cache
            _sync(device)

            latencies = []
            kv_cache = None
            for _ in range(n_iter):
                kv_cache = None
                _sync(device)
                t0 = time.perf_counter()
                _, kv_cache = self.model(ids, use_kv_cache=True)
                _sync(device)
                latencies.append(time.perf_counter() - t0)
                _validate_kv_cache(kv_cache, expected_context=self.ctx_len)

        if kv_cache is None:
            raise ValueError("prefill measurement requires at least one measured run")

        ordered = sorted(latencies)
        latency = statistics.median(ordered)
        p90 = ordered[max(0, int(len(ordered) * 0.90 + 0.999999) - 1)]
        p99 = ordered[max(0, int(len(ordered) * 0.99 + 0.999999) - 1)]
        return {
            "phase": "prefill",
            "context_length": self.ctx_len,
            "batch_size": self.batch,
            "prompt_seed": FIXED_PROMPT_SEED,
            "prompt_sha256": prompt_sha256,
            "kv_cache_materialized": True,
            "kv_cache_bytes": _kv_cache_bytes(kv_cache),
            "prefill_latency_s": latency,
            "prefill_latency_median_s": latency,
            "prefill_latency_p90_s": p90,
            "prefill_latency_p99_s": p99,
            "prefill_latency_samples_s": latencies,
            "prefill_tokens_per_sec": self.ctx_len * self.batch / latency,
            "peak_activation_bytes": act_bytes,
        }


def _validate_kv_cache(kv_cache: object, *, expected_context: int) -> None:
    if not isinstance(kv_cache, (list, tuple)) or not kv_cache:
        raise ValueError("prefill model did not return a KV cache")
    for layer in kv_cache:
        if not isinstance(layer, (list, tuple)) or len(layer) != 2:
            raise ValueError("prefill model returned an invalid KV-cache layer")
        key, value = layer
        if not torch.is_tensor(key) or not torch.is_tensor(value):
            raise ValueError("prefill KV-cache entries must be tensors")
        if key.shape[-2] != expected_context or value.shape[-2] != expected_context:
            raise ValueError("prefill KV cache does not cover the complete prompt")


def _kv_cache_bytes(kv_cache: object) -> int:
    if not isinstance(kv_cache, (list, tuple)) or not kv_cache:
        raise ValueError("prefill model did not return a KV cache")
    total = 0
    for layer in kv_cache:
        if not isinstance(layer, (list, tuple)) or len(layer) != 2:
            raise ValueError("prefill model returned an invalid KV-cache layer")
        for tensor in layer:
            if not torch.is_tensor(tensor):
                raise ValueError("prefill KV-cache entries must be tensors")
            total += tensor.numel() * tensor.element_size()
    return int(total)


def run_benchmark(
    checkpoint_path: str = None,
    scenario: str = "Offline",
    context_len: int = 1792,
    batch_size: int = 1,
) -> dict:
    """Entry point used by the CLI / smoke test."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = NanoGPTWhiteBox().to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return NanoGPTPrefill(model, context_len=context_len, batch_size=batch_size).run()
