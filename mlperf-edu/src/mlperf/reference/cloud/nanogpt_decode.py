"""
MLPerf EDU causal language modeling, decode phase

Autoregressive decode with a real KV cache. Each step appends one
token's K and V, and attention re-reads the entire cached K, V from
DRAM -- the canonical bandwidth-bound regime that dominates LLM
serving cost in production.

The prefill and decode phases share one quality-approved NanoGPT checkpoint.

This reference path is a sequential single-stream microbenchmark. It does
not model concurrent requests, an arrival process, queueing, or a server SLO.
"""

import statistics
import time
import torch

from mlperf.harness import percentile

from .nanogpt_prefill import FIXED_PROMPT_SEED, fixed_token_prompt
from .nanogpt_train import NanoGPTWhiteBox


def _sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def kv_cache_bytes(past_key_values, dtype_bytes: int = 4) -> int:
    """Total bytes held in the KV cache across all layers."""
    total = 0
    for k, v in past_key_values:
        total += k.numel() * dtype_bytes + v.numel() * dtype_bytes
    return total


class NanoGPTDecode:
    """Prefill once, emit the first token, then time cached-token steps.

    Request TTFT spans prompt processing through selection of the first output
    token from the prefill logits. Each inter-token latency (ITL) sample then
    measures one cache-reusing forward pass that emits the next token. The first
    such ITL is also retained as ``first_decode_latency_s`` for diagnostics.
    """

    def __init__(
        self,
        model: NanoGPTWhiteBox,
        prefill_ctx: int = 1792,
        decode_steps: int = 64,
        batch_size: int = 1,
    ):
        if prefill_ctx < 1:
            raise ValueError("prefill_ctx must be at least one token")
        if decode_steps < 2:
            raise ValueError(
                "decode_steps must be at least two so inter-token latency is defined"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        max_ctx = prefill_ctx + decode_steps
        if max_ctx > model.config["max_seq_len"]:
            raise ValueError(
                f"prefill_ctx + decode_steps = {max_ctx} exceeds model "
                f"max_seq_len={model.config['max_seq_len']}; bump it."
            )
        self.model = model.eval()
        self.prefill_ctx = prefill_ctx
        self.decode_steps = decode_steps
        self.batch = batch_size
        self.vocab = model.config["vocab_size"]

    def _sample(self, logits):
        # Argmax keeps the test deterministic; replace with multinomial
        # if students need temperature/top-p exploration.
        return logits.argmax(dim=-1, keepdim=True)

    def run(self, emit_sidecar: bool = True) -> dict:
        device = next(self.model.parameters()).device
        prompt, prompt_sha256 = fixed_token_prompt(
            batch_size=self.batch,
            context_len=self.prefill_ctx,
            vocab_size=self.vocab,
            device=device,
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        cfg = self.model.config
        head_dim = cfg["n_embd"] // cfg["n_head"]
        # Per-step bytes during decode: full weight reread + full KV stream.
        kv_bytes_per_step = (
            2 * cfg["n_layer"] * cfg["n_head"] * head_dim * self.prefill_ctx * 4
        )
        bytes_per_step = n_params * 4 + kv_bytes_per_step
        # Per-step FLOPs: one new token through all weights + attention over ctx.
        flops_per_step = (
            2 * n_params
            + 4 * cfg["n_layer"] * cfg["n_head"] * head_dim * self.prefill_ctx
        )

        with torch.no_grad():
            # Prefill the cache. Request timing starts before prompt processing.
            _sync(device)
            request_start = time.perf_counter()
            logits, kv = self.model(prompt, use_kv_cache=True)
            _sync(device)
            prefill_time = time.perf_counter() - request_start

            # A causal-LM prefill already produces the logits for the first
            # output token. TTFT ends when that token has been selected; an
            # additional cached forward pass would actually emit token two.
            output_token = self._sample(logits[:, -1, :])
            _sync(device)
            request_ttft = time.perf_counter() - request_start
            generated_tokens = [output_token]

            per_step = []
            n_loop = self.decode_steps - 1
            if emit_sidecar and n_loop > 0:
                from mlperf.roofline import measure_roofline

                with measure_roofline(
                    "causal-language-modeling",
                    analytic_flops=lambda: flops_per_step * n_loop,
                    analytic_bytes=lambda: bytes_per_step * n_loop,
                    n_iter=n_loop,
                ) as roofline_context:
                    roofline_context.update(
                        {"mode": "inference", "phase": "decode", "model": "NanoGPT"}
                    )
                    for _ in range(n_loop):
                        _sync(device)
                        t = time.perf_counter()
                        logits, kv = self.model(
                            output_token, use_kv_cache=True, past_key_values=kv
                        )
                        output_token = self._sample(logits[:, -1, :])
                        _sync(device)
                        per_step.append(time.perf_counter() - t)
                        generated_tokens.append(output_token)
            else:
                for _ in range(n_loop):
                    _sync(device)
                    t = time.perf_counter()
                    logits, kv = self.model(
                        output_token, use_kv_cache=True, past_key_values=kv
                    )
                    output_token = self._sample(logits[:, -1, :])
                    _sync(device)
                    per_step.append(time.perf_counter() - t)
                    generated_tokens.append(output_token)

            request_end_to_end_latency = time.perf_counter() - request_start

        kv_bytes = kv_cache_bytes(kv)
        median_itl = statistics.median(per_step) if per_step else float("nan")
        first_decode_latency = per_step[0] if per_step else float("nan")
        p90_itl = (
            sorted(per_step)[max(0, int(len(per_step) * 0.90 + 0.999999) - 1)]
            if per_step
            else float("nan")
        )
        p99_itl = percentile(per_step, 99) if per_step else float("nan")
        # Achieved bandwidth: each decode step re-reads the full KV cache
        # (the model also re-reads weights, but those usually live in LLC
        # after warmup). KV stream is the *additive* per-step cost.
        achieved_bw_gbps = kv_bytes / median_itl / 1e9 if per_step else 0.0

        return {
            "phase": "decode",
            "prefill_ctx": self.prefill_ctx,
            "decode_steps": self.decode_steps,
            "batch_size": self.batch,
            "prompt_seed": FIXED_PROMPT_SEED,
            "prompt_sha256": prompt_sha256,
            "prefill_warm_s": prefill_time,
            "prefill_latency_s": prefill_time,
            "first_decode_latency_s": first_decode_latency,
            "request_ttft_s": request_ttft,
            "ttft_s": request_ttft,
            "request_end_to_end_latency_s": request_end_to_end_latency,
            "itl_median_s": median_itl,
            "itl_p90_s": p90_itl,
            "itl_p99_s": p99_itl,
            "itl_samples_s": per_step,
            "generated_token_ids": torch.cat(generated_tokens, dim=-1)
            .detach()
            .cpu()
            .tolist(),
            "kv_cache_bytes": kv_bytes,
            "achieved_bw_gbps": achieved_bw_gbps,
            "output_tokens_per_sec": self.batch / median_itl if per_step else 0.0,
        }


def run_benchmark(
    checkpoint_path: str = None,
    scenario: str = "SingleStream",
    prefill_ctx: int = 1792,
    decode_steps: int = 64,
    batch_size: int = 1,
) -> dict:
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
    return NanoGPTDecode(
        model, prefill_ctx=prefill_ctx, decode_steps=decode_steps, batch_size=batch_size
    ).run()
