"""
MLPerf EDU: NanoGPT-Decode workload (Cloud Division)

Autoregressive decode with a real KV cache. Each step appends one
token's K and V, and attention re-reads the entire cached K, V from
DRAM -- the canonical bandwidth-bound regime that dominates LLM
serving cost in production.

Pair with nanogpt-prefill (same checkpoint) to observe the
prefill-vs-decode bottleneck split.
"""
import statistics
import time
import torch

from .nanogpt_train import NanoGPTWhiteBox


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def kv_cache_bytes(past_key_values, dtype_bytes: int = 4) -> int:
    """Total bytes held in the KV cache across all layers."""
    total = 0
    for k, v in past_key_values:
        total += k.numel() * dtype_bytes + v.numel() * dtype_bytes
    return total


class NanoGPTDecode:
    """Warms the KV cache to `prefill_ctx`, then times `decode_steps` single-token steps.

    Reports time-to-first-token (TTFT), median + p99 inter-token latency
    (ITL), final KV-cache bytes, and an achieved-bandwidth estimate
    derived from streaming the cached K,V each step.
    """

    def __init__(self, model: NanoGPTWhiteBox,
                 prefill_ctx: int = 1792, decode_steps: int = 64, batch_size: int = 1):
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

    def run(self) -> dict:
        device = next(self.model.parameters()).device
        prompt = torch.randint(0, self.vocab, (self.batch, self.prefill_ctx), device=device)

        with torch.no_grad():
            # Warm the cache and get the last-step logits.
            _sync()
            t_prefill_start = time.perf_counter()
            logits, kv = self.model(prompt, use_kv_cache=True)
            _sync()
            prefill_time = time.perf_counter() - t_prefill_start

            # First decode step (TTFT measured here; the prefill is the
            # "prompt processing" phase, not part of TTFT in serving SLOs).
            _sync()
            t0 = time.perf_counter()
            next_tok = self._sample(logits[:, -1, :])
            logits, kv = self.model(next_tok, use_kv_cache=True, past_key_values=kv)
            _sync()
            ttft = time.perf_counter() - t0

            per_step = []
            for _ in range(self.decode_steps - 1):
                next_tok = self._sample(logits[:, -1, :])
                _sync()
                t = time.perf_counter()
                logits, kv = self.model(next_tok, use_kv_cache=True, past_key_values=kv)
                _sync()
                per_step.append(time.perf_counter() - t)

        kv_bytes = kv_cache_bytes(kv)
        median_itl = statistics.median(per_step) if per_step else float("nan")
        p99_itl = sorted(per_step)[int(len(per_step) * 0.99) - 1] if per_step else float("nan")
        # Achieved bandwidth: each decode step re-reads the full KV cache
        # (the model also re-reads weights, but those usually live in LLC
        # after warmup). KV stream is the *additive* per-step cost.
        achieved_bw_gbps = kv_bytes / median_itl / 1e9 if per_step else 0.0

        return {
            "phase": "decode",
            "prefill_ctx": self.prefill_ctx,
            "decode_steps": self.decode_steps,
            "batch_size": self.batch,
            "prefill_warm_s": prefill_time,
            "ttft_s": ttft,
            "itl_median_s": median_itl,
            "itl_p99_s": p99_itl,
            "kv_cache_bytes": kv_bytes,
            "achieved_bw_gbps": achieved_bw_gbps,
            "output_tokens_per_sec": 1.0 / median_itl if per_step else 0.0,
        }


def run_benchmark(checkpoint_path: str = None, scenario: str = "Offline",
                   prefill_ctx: int = 1792, decode_steps: int = 64,
                   batch_size: int = 1) -> dict:
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = NanoGPTWhiteBox().to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return NanoGPTDecode(
        model, prefill_ctx=prefill_ctx, decode_steps=decode_steps, batch_size=batch_size
    ).run()
