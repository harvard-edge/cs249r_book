"""
MLPerf EDU: NanoGPT-Prefill workload (Cloud Division)

Single forward pass over a long context with KV-cache disabled. Exercises
the compute-bound regime: every weight matrix is reused across `ctx_len`
tokens, giving high arithmetic intensity. Should sit on the compute side
of the roofline.

Pair with nanogpt-decode (same checkpoint) to observe the prefill-vs-decode
bottleneck split that defines modern LLM serving economics.
"""
import time
import torch

from .nanogpt_train import NanoGPTWhiteBox


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
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

    Reports prefill latency, throughput, and a static peak-activation
    estimate. The KV cache is *not* warmed; this is the cold prefill
    regime that production serving systems run once per request.
    """

    def __init__(self, model: NanoGPTWhiteBox, context_len: int = 1792, batch_size: int = 1):
        if context_len > model.config["max_seq_len"]:
            raise ValueError(
                f"context_len={context_len} exceeds model max_seq_len="
                f"{model.config['max_seq_len']}; bump NanoGPTWhiteBox(max_seq_len=) first."
            )
        self.model = model.eval()
        self.ctx_len = context_len
        self.batch = batch_size
        self.vocab = model.config["vocab_size"]

    def run(self, n_warmup: int = 3, n_iter: int = 10) -> dict:
        device = next(self.model.parameters()).device
        ids = torch.randint(0, self.vocab, (self.batch, self.ctx_len), device=device)

        with torch.no_grad():
            for _ in range(n_warmup):
                self.model(ids)
            _sync()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                self.model(ids)
            _sync()
            total = time.perf_counter() - t0

        latency = total / n_iter
        return {
            "phase": "prefill",
            "context_length": self.ctx_len,
            "batch_size": self.batch,
            "prefill_latency_s": latency,
            "prefill_tokens_per_sec": self.ctx_len * self.batch / latency,
            "peak_activation_bytes": estimate_activation_bytes(
                self.model, self.ctx_len, self.batch
            ),
        }


def run_benchmark(checkpoint_path: str = None, scenario: str = "Offline",
                   context_len: int = 1792, batch_size: int = 1) -> dict:
    """Entry point used by the CLI / smoke test."""
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = NanoGPTWhiteBox().to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return NanoGPTPrefill(model, context_len=context_len, batch_size=batch_size).run()
