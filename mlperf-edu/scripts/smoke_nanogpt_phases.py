#!/usr/bin/env python3
"""
Smoke test for the iter-3 NanoGPT prefill / decode split.

Verifies that prefill and decode hit different bottleneck regimes on
the same model with the same weights:

  Check 1 (informational): wall-clock prefill latency at ctx=1792.
  Check 2 (GATED): arithmetic-intensity ratio prefill / decode >= 5x.
                   This is the load-bearing claim -- prefill should be
                   compute-bound (intensity > 10) and decode should be
                   bandwidth-bound (intensity ~1-2) on the same weights.
  Check 3 (informational): achieved decode bandwidth in GB/s.

Acceptance from Dean's iter-3 spec.
Run: python3 scripts/smoke_nanogpt_phases.py
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from reference.cloud.nanogpt_train import NanoGPTWhiteBox
from reference.cloud.nanogpt_prefill import NanoGPTPrefill
from reference.cloud.nanogpt_decode import NanoGPTDecode


def model_param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    PREFILL_CTX = 1792
    DECODE_STEPS = 64

    model = NanoGPTWhiteBox().to(device)
    cfg = model.config
    n_params = model_param_count(model)

    # KV bytes per token: 2 (K and V) * n_layer * n_head * head_dim * 4 bytes.
    head_dim = cfg["n_embd"] // cfg["n_head"]
    kv_bytes_per_token = 2 * cfg["n_layer"] * cfg["n_head"] * head_dim * 4

    print(f"Device: {device}")
    print(f"Model: {n_params / 1e6:.2f}M params, "
          f"d_model={cfg['n_embd']}, n_head={cfg['n_head']}, n_layer={cfg['n_layer']}")
    print(f"KV bytes/token: {kv_bytes_per_token:,} ({kv_bytes_per_token/1024:.1f} KB)")
    print(f"At prefill_ctx={PREFILL_CTX}: KV stream/step = "
          f"{kv_bytes_per_token * PREFILL_CTX / 1024 / 1024:.1f} MB")
    print()

    # --- Check 1: prefill latency (informational) ---
    prefill_result = NanoGPTPrefill(model, context_len=PREFILL_CTX, batch_size=1).run()
    print(f"Prefill (ctx={PREFILL_CTX}, B=1):")
    print(f"  latency:    {prefill_result['prefill_latency_s']*1000:7.2f} ms")
    print(f"  throughput: {prefill_result['prefill_tokens_per_sec']:7.0f} tok/s")
    print(f"  peak act:   {prefill_result['peak_activation_bytes']/1024/1024:7.2f} MB (estimate)")
    print()

    # --- Check 2: arithmetic-intensity ratio (gated) ---
    decode_result = NanoGPTDecode(
        model, prefill_ctx=PREFILL_CTX, decode_steps=DECODE_STEPS, batch_size=1
    ).run()

    # Prefill: forward over ctx tokens. Compute is dominated by 2*params*ctx
    # FLOPs (one mac per (param, token) pair, times 2 for forward). Bytes
    # touched: parameters once + activations per layer.
    prefill_flops = 2 * n_params * PREFILL_CTX
    prefill_bytes = n_params * 4 + prefill_result["peak_activation_bytes"]
    intensity_prefill = prefill_flops / prefill_bytes

    # Decode: one new token, but attention reads full KV cache. Compute is
    # 2*params (one token through all weights) + attention over ctx tokens.
    decode_flops_per_step = 2 * n_params + 4 * cfg["n_layer"] * cfg["n_head"] * head_dim * PREFILL_CTX
    # Bytes: parameters (re-read) + KV stream.
    decode_bytes_per_step = n_params * 4 + kv_bytes_per_token * PREFILL_CTX
    intensity_decode = decode_flops_per_step / decode_bytes_per_step

    print(f"Decode ({DECODE_STEPS} steps from ctx={PREFILL_CTX}):")
    print(f"  TTFT:           {decode_result['ttft_s']*1000:7.2f} ms")
    print(f"  ITL (median):   {decode_result['itl_median_s']*1000:7.2f} ms")
    print(f"  ITL (p99):      {decode_result['itl_p99_s']*1000:7.2f} ms")
    print(f"  output_tok/s:   {decode_result['output_tokens_per_sec']:7.1f}")
    print(f"  KV cache:       {decode_result['kv_cache_bytes']/1024/1024:7.2f} MB")
    print(f"  achieved BW:    {decode_result['achieved_bw_gbps']:7.2f} GB/s")
    print()

    ratio = intensity_prefill / intensity_decode
    print(f"Arithmetic intensity (FLOP/byte):")
    print(f"  prefill:  {intensity_prefill:7.2f}")
    print(f"  decode:   {intensity_decode:7.2f}")
    print(f"  ratio:    {ratio:7.2f}x")
    print()

    # Visceral check: how many prefill tokens of work fit in one decode step?
    prefill_tok_per_decode_step = (decode_result["itl_median_s"]
                                    * prefill_result["prefill_tokens_per_sec"])
    print(f"Visceral: 1 decode step = {prefill_tok_per_decode_step:.0f} tokens'-worth "
          f"of prefill throughput")
    print()

    # --- Check 3: bandwidth sanity (informational) ---
    if decode_result["achieved_bw_gbps"] < 15.0:
        print("WARN: achieved decode BW < 15 GB/s; dispatch overhead may still")
        print("      dominate. Bump prefill_ctx to 3584 and re-run.")

    if ratio >= 5.0:
        print(f"PASS: intensity ratio {ratio:.1f}x (>= 5x). Compute/BW split confirmed.")
        return 0
    else:
        print(f"FAIL: intensity ratio only {ratio:.1f}x; expected >= 5x.")
        print("      Try: bump prefill_ctx (currently {}) toward 3584.".format(PREFILL_CTX))
        return 1


if __name__ == "__main__":
    sys.exit(main())
