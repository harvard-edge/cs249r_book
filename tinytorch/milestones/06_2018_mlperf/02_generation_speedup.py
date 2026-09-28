#!/usr/bin/env python3
"""
Generative Serving (2022-Present): KV Cache Generation Speedup
=============================================================================

📚 HISTORICAL CONTEXT:
When OpenAI deployed ChatGPT in late 2022, naive autoregressive generation was
catastrophically slow: generating token N required recomputing the key and value
vectors for all previous N-1 tokens through every transformer layer. The computational
cost per sequence grew cubically: O(S^3) total attention operations.

The introduction of Key-Value (KV) Caching transformed generative AI serving:
- Prefill Phase: Compute and cache keys and values for all prompt tokens once.
- Decode Phase: Compute only the single query vector for the new token, retrieve
  cached keys and values from DRAM/SRAM, and append the new token in O(1) time.
This slashes decoding cost to O(S^2) and delivers interactive token streaming!

🎯 MILESTONE 06 PART 2: MEASURE YOUR KV CACHE SPEEDUP
Replay a token sequence through YOUR TinyGPT model with and without YOUR KV Cache.
Verify numerical output invariance (identical logits) and measure the real wall-clock
speedup delivered by YOUR Module 18 implementation!

✅ REQUIRED MODULES (Run after Module 18):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR strided tensor data structure
  Module 06 (Autograd)      : YOUR no_grad inference scope
  Module 11 (Embeddings)    : YOUR Token + Learned Positional Embeddings
  Module 12 (Attention)     : YOUR Causal Multi-Head Attention
  Module 13 (Transformers)  : YOUR Stacked Transformer Blocks & TinyGPT
  Module 18 (Memoization)   : YOUR KV Cache state reuse engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Naive vs. KV Cached Autoregressive Serving):
    NAIVE (O(S^2) per step):
    Step 1: [Tok 1] ──────────────▶ Compute Q, K, V
    Step 2: [Tok 1, Tok 2] ───────▶ Recompute ALL Q, K, V  (Redundant!)
    Step 3: [Tok 1, Tok 2, Tok 3] ─▶ Recompute ALL Q, K, V  (Wasteful!)

    WITH YOUR KV CACHE (O(1) per step):
    Step 1: [Tok 1] ──────────────▶ Store K1, V1 in YOUR Cache
    Step 2: [Tok 2] ──────────────▶ Compute only Q2, K2, V2; reuse K1, V1!
    Step 3: [Tok 3] ──────────────▶ Compute only Q3, K3, V3; reuse K1, K2, V1, V2!
"""

from contextlib import nullcontext
from pathlib import Path
import sys
import time

import numpy as np
from rich.console import Console
from rich.table import Table

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# =============================================================================
# 🎓 ZONE 1: STUDENT CORE LEGO BRICKS (YOUR Modules 01, 06, 13, 18)
# =============================================================================

from tinytorch.core.tensor import Tensor  # noqa: E402
from tinytorch.core.autograd import no_grad  # noqa: E402
from tinytorch.core.transformers import GPT  # noqa: E402
from tinytorch.perf.memoization import enable_kv_cache, disable_kv_cache  # noqa: E402

def replay_prefixes(model, tokens, cache=None):
    """Return the next-token logits at every position of one fixed sequence.

    The cache cursor advances once after all layers process a token. Resetting
    at entry makes each replay an independent request, including warmup runs.
    Cached forwards run inside an explicit generation scope; ordinary forwards
    resume when the scope exits, including after an exception.
    """
    from tinytorch.core.tensor import Tensor

    if cache is not None:
        cache.reset()
    logits = []
    with cache.generation() if cache is not None else nullcontext():
        for position in range(tokens.shape[1]):
            if cache is None:
                output = model(Tensor(tokens[:, :position + 1]))
            else:
                output = model(Tensor(tokens[:, position:position + 1]),
                               start_pos=cache.seq_pos)
                cache.advance()
            logits.append(output.data[:, -1, :].copy())
    return np.stack(logits, axis=1)


def measure_replay(model, tokens, cache=None, repeats=7):
    """Warm up once, then measure independent complete replays in milliseconds."""
    replay_prefixes(model, tokens, cache)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        replay_prefixes(model, tokens, cache)
        elapsed.append((time.perf_counter() - start) * 1000)
    return float(np.median(elapsed))


def compare_cached_replay(model, tokens, repeats=7):
    """Verify equivalent outputs and time the same model with and without caching."""
    if getattr(model, '_cache_enabled', False):
        raise ValueError('Pass an uncached model; this comparison manages its cache.')
    if tokens.ndim != 2 or tokens.shape[0] != 1 or not 0 < tokens.shape[1] <= model.max_seq_len:
        raise ValueError('Use one nonempty token sequence within the model context length.')
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')

    with no_grad():
        expected = replay_prefixes(model, tokens)
        baseline_ms = measure_replay(model, tokens, repeats=repeats)
        cache = enable_kv_cache(model)
        try:
            actual = replay_prefixes(model, tokens, cache)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            cached_ms = measure_replay(model, tokens, cache, repeats)
            memory = cache.get_memory_usage()
            return {
                'baseline_ms': baseline_ms,
                'cached_ms': cached_ms,
                'speedup': baseline_ms / cached_ms,
                'max_logit_error': float(np.max(np.abs(actual - expected))),
                'cache_bytes': int(round(memory['total_mb'] * 1024 * 1024)),
                'tokens_processed': cache.seq_pos,
            }
        finally:
            disable_kv_cache(model)


def main():
    console = Console()
    console.print('[bold cyan]Milestone 06.2: KV Cache: Same Computation, Less Repetition[/bold cyan]')
    model = GPT(vocab_size=27, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=64)
    tokens = np.random.default_rng(7).integers(0, model.vocab_size, (1, 64))
    results = compare_cached_replay(model, tokens)

    table = Table(title='Fixed-sequence inference replay (median of 7 runs)')
    table.add_column('Path')
    table.add_column('Total (ms)', justify='right')
    table.add_column('Per token (ms)', justify='right')
    for label, key in [('Full causal prefixes', 'baseline_ms'), ('Cached single tokens', 'cached_ms')]:
        table.add_row(label, f"{results[key]:.3f}", f"{results[key] / tokens.shape[1]:.3f}")
    console.print(table)
    console.print(f"Logit equivalence passed; maximum difference: {results['max_logit_error']:.2e}")
    console.print(f"Measured speed ratio (baseline/cached): {results['speedup']:.2f}×")
    console.print(f"Preallocated K+V storage: {results['cache_bytes']:,} bytes")
    console.print('The cache reuses earlier projections; each new query still reads the growing prefix.')
    console.print('Small workloads can be slower with caching. Correctness is required; speedup is measured.')
    console.print('[bold green]MILESTONE 06.2 COMPLETE[/bold green]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
